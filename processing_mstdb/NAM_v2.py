"""
Small recap for myself (NAM version)

Idea
----
Instead of one model per coefficient, this script uses:
- one shared "backbone" that sees the element fractions,
- several small heads (one per target coefficient),
- plus a physics loss that ties related coefficients together
  (density, viscosities, k, cp over temperature).

The backbone is 'NAM-like': it uses simple spline-style
nonlinearities on the features, then a small residual MLP.

Targets & data
--------------
We work only on the coefficients:
  rho_a, rho_b,
  mu1_a, mu1_b,
  mu2_a, mu2_b, mu2_c,
  k_a, k_b,
  cp_a, cp_b


Lots of coefficients are missing. We do *not* drop rows.
Instead we use a mask: for each sample & each target, the loss
is computed only if the true value exists.

The multi-task loss is uncertainty-weighted (one learned weight
per target) + physics regularisation over random temperatures.

Plots
-----
For the *test* split we save in visualisation/NAM:

- Actual vs Predicted (per target)
- Residuals (pred - actual, per target)
- Absolute error distribution (per target)
- Percent error distribution (per target)
- Error-bin confusion matrix (all targets together)
"""

import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sns.set(style="whitegrid")

# --------------------------------------------------
# basic config and small helpers
# --------------------------------------------------
CSV_PATH = "/Users/meggie/Documents/MoltenSaltPropnet/data/new_mstdb_janz.csv"

# where to save evaluation CSV + plots
EVAL_BASE_DIR = "evaluate_modelperformance"
NAM_PLOT_DIR = os.path.join(EVAL_BASE_DIR, "NAM_v2")
os.makedirs(NAM_PLOT_DIR, exist_ok=True)

# where to save the trained model
MODEL_BASE_DIR = "best_models"
MODEL_SUBDIR = os.path.join(MODEL_BASE_DIR, "NAM_v2")
os.makedirs(MODEL_SUBDIR, exist_ok=True)

R_GAS = 8.314
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"


def _savefig(filename):
    path = os.path.join(NAM_PLOT_DIR, filename)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved → {path}")


def rel_mse_pct(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mse = mean_squared_error(y_true, y_pred)
    denom = np.mean(y_true ** 2)
    if denom <= 1e-12:
        denom = 1e-12
    return 100.0 * mse / denom


# --------------------------------------------------
# tiny spline-ish layer (NAM-style nonlinearity)
# --------------------------------------------------
class SplineLayer(nn.Module):
    # simple triangular basis per feature
    def __init__(self, in_dim, num_knots=16):
        super().__init__()
        self.num_knots = num_knots
        # knots placed in a reasonable range for standardized inputs
        self.knots = nn.Parameter(torch.linspace(-3.0, 3.0, num_knots))
        # one "filter" per feature
        self.weight = nn.Parameter(torch.randn(in_dim, num_knots) * 0.1)
        self.bias = nn.Parameter(torch.zeros(in_dim))

    def forward(self, x):
        # x: (B, D)
        B, D = x.shape
        # (B, D, K)
        diff = x.unsqueeze(-1) - self.knots
        # triangular basis, compact support around knots
        basis = torch.relu(1.0 - torch.abs(diff))
        # (B, D)
        out = (basis * self.weight).sum(dim=-1) + self.bias
        return out


# --------------------------------------------------
# backbone + multi-head NAM
# --------------------------------------------------
class HybridNAM(nn.Module):
    def __init__(
        self,
        in_dim,
        targets,
        hidden=128,
        num_hidden_layers=2,
        dropout=0.2,
        num_knots=16,
    ):
        super().__init__()
        self.targets = list(targets)
        self.n_targets = len(targets)

        # spline-like transform of inputs
        self.spline = SplineLayer(in_dim, num_knots=num_knots)

        # small residual MLP as shared backbone
        layers = []
        layers.append(nn.Linear(in_dim, hidden))
        layers.append(nn.SiLU())
        layers.append(nn.Dropout(dropout))
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
        self.backbone = nn.Sequential(*layers)

        # one linear head per target
        self.heads = nn.ModuleDict(
            {t: nn.Linear(hidden, 1) for t in self.targets}
        )

        # one uncertainty parameter per target (log sigma^2)
        self.log_sigma = nn.Parameter(torch.zeros(self.n_targets))

    def forward(self, x):
        # x: (B, D)
        z = self.spline(x)
        h = self.backbone(z)
        outs = []
        for t in self.targets:
            outs.append(self.heads[t](h).squeeze(-1))
        # (B, T)
        return torch.stack(outs, dim=1)


# --------------------------------------------------
# Trainer that handles missing data + physics
# --------------------------------------------------
class HybridNAMTrainer:
    def __init__(
        self,
        df,
        targets,
        derived_groups,
        physics_weight=0.1,
        temp_range=(500.0, 1200.0),
        batch_size=128,
        model_dir=MODEL_SUBDIR,
    ):
        self.df = df.copy()
        self.targets = [t for t in targets if t in self.df.columns]
        self.derived_groups = derived_groups
        self.physics_weight = physics_weight
        self.temp_range = temp_range
        self.batch_size = batch_size
        self.model_dir = model_dir
        self.best_model_path = None

        print("\nTargets used in NAM:", self.targets)

        # ---------------------------
        # Composition → element matrix
        # ---------------------------
        if "Composition" not in self.df.columns:
            raise RuntimeError("Expected a 'Composition' column with element fractions.")

        comp_df = pd.json_normalize(self.df["Composition"]).fillna(0.0)
        comp_df = comp_df.reindex(sorted(comp_df.columns), axis=1)
        self.element_cols = list(comp_df.columns)

        print("\nElement columns used in NAM:", self.element_cols)

        X_frac = comp_df.to_numpy(dtype=np.float32)
        if X_frac.shape[1] == 0:
            raise RuntimeError("Composition column did not contain any elements.")

        self.x_scaler = StandardScaler()
        X = self.x_scaler.fit_transform(X_frac).astype(np.float32)
        self.X = X
        self.n_samples, self.n_features = X.shape

        # ---------------------------
        # Target matrix + mask
        # ---------------------------
        y_list = []
        mask_list = []
        for t in self.targets:
            col = pd.to_numeric(self.df[t], errors="coerce")
            y_list.append(col.to_numpy(dtype=np.float32))
            mask_list.append(np.isfinite(col.to_numpy(dtype=float)))

        y_raw_full = np.stack(y_list, axis=1)  # (N, T) with NaNs
        mask_all = np.stack(mask_list, axis=1)  # (N, T) bool
        self.mask_all = mask_all

        # keep a copy with NaNs for metrics
        self.y_raw_full = y_raw_full.copy()

        # for training, fill NaNs (mask will control loss)
        y_filled = np.where(mask_all, y_raw_full, 0.0)
        self.y_raw = y_filled.astype(np.float32)

        # normalisation based only on observed entries
        μ = []
        σ = []
        for j, t in enumerate(self.targets):
            vals = y_raw_full[mask_all[:, j], j]
            if len(vals) == 0:
                μ.append(0.0)
                σ.append(1.0)
            else:
                μ.append(float(np.mean(vals)))
                s = float(np.std(vals))
                if s < 1e-12:
                    s = 1.0
                σ.append(s)
        self.μ = np.array(μ, dtype=np.float32)
        self.σ = np.array(σ, dtype=np.float32)

        print("\nPer-target normalisation (μ, σ):")
        for t, m, s in zip(self.targets, self.μ, self.σ):
            print(f"  {t}: mean={m:.4g}, std={s:.4g}")

        y_std = (self.y_raw - self.μ) / self.σ
        self.y_std = y_std.astype(np.float32)

        # indices
        idx_all = np.arange(self.n_samples)
        tr_idx, te_idx = train_test_split(idx_all, test_size=0.2, random_state=SEED)
        tr_idx, va_idx = train_test_split(tr_idx, test_size=0.2, random_state=SEED)
        self.tr_idx = tr_idx
        self.va_idx = va_idx
        self.te_idx = te_idx

        # torch copies of μ, σ on device
        self.mu_t = torch.tensor(self.μ, device=device)
        self.sigma_t = torch.tensor(self.σ, device=device)

        # handy mapping target → column index
        self.target_index = {t: j for j, t in enumerate(self.targets)}

        # model
        self.model = HybridNAM(
            in_dim=self.n_features,
            targets=self.targets,
            hidden=128,
            num_hidden_layers=2,
            dropout=0.2,
            num_knots=16,
        ).to(device)

        # for storing test predictions for plotting
        self.pred_vs_actual_test = None

    # ---------------------------
    # data loaders
    # ---------------------------
    def _make_loader(self, idx, shuffle):
        Xb = self.X[idx]
        yb = self.y_std[idx]
        mb = self.mask_all[idx].astype(np.float32)
        ds = TensorDataset(
            torch.tensor(Xb, dtype=torch.float32),
            torch.tensor(yb, dtype=torch.float32),
            torch.tensor(mb, dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    # ---------------------------
    # physics loss on batch
    # ---------------------------
    def physics_loss(self, pred_raw, y_raw, mask, T_tensor):
        """
        pred_raw, y_raw: (B, T)
        mask: (B, T)  bool
        T_tensor: (B,)
        """
        loss = 0.0
        n_terms = 0

        for dname, coeffs in self.derived_groups:
            idxs = [self.target_index[c] for c in coeffs if c in self.target_index]
            if len(idxs) != len(coeffs):
                continue

            # rows where all needed coeffs are present
            valid_mask = mask[:, idxs].all(dim=1)  # (B,)
            if valid_mask.sum() < 2:
                continue

            T = T_tensor[valid_mask]  # (B_valid,)
            y_c = y_raw[valid_mask][:, idxs]  # (B_valid, n_coeff)
            p_c = pred_raw[valid_mask][:, idxs]

            if dname == "rho":
                y_vals = y_c[:, 0] - y_c[:, 1] * T
                p_vals = p_c[:, 0] - p_c[:, 1] * T

            elif dname == "muA":
                # exp( E / (R T) ) style, compare in log-space
                p_mu1_a = torch.clamp(p_c[:, 0], min=1e-6)
                y_mu1_a = torch.clamp(y_c[:, 0], min=1e-6)
                p_vals = p_mu1_a * torch.exp(p_c[:, 1] / (R_GAS * T))
                y_vals = y_mu1_a * torch.exp(y_c[:, 1] / (R_GAS * T))
                p_vals = torch.log(p_vals + 1e-8)
                y_vals = torch.log(y_vals + 1e-8)

            elif dname == "muB":
                T2 = T * T
                y_vals = y_c[:, 0] + y_c[:, 1] / T + y_c[:, 2] / T2
                p_vals = p_c[:, 0] + p_c[:, 1] / T + p_c[:, 2] / T2

            elif dname == "k":
                y_vals = y_c[:, 0] + y_c[:, 1] * T
                p_vals = p_c[:, 0] + p_c[:, 1] * T

            elif dname == "cp":
                # simple linear in T for cp here (2 coeffs)
                y_vals = y_c[:, 0] + y_c[:, 1] * T
                p_vals = p_c[:, 0] + p_c[:, 1] * T

            else:
                continue

            term = nn.functional.mse_loss(p_vals, y_vals)
            loss += term
            n_terms += 1

        if n_terms == 0:
            return torch.tensor(0.0, device=device)
        return loss / n_terms

    # ---------------------------
    # training loop
    # ---------------------------
    def train(self, epochs=80, patience=20, lr=1e-3):
        print("\nstarting NAM multi-head + shared backbone training ...")

        train_loader = self._make_loader(self.tr_idx, shuffle=True)
        val_loader = self._make_loader(self.va_idx, shuffle=False)

        optim = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=epochs, eta_min=1e-4
        )

        best_val = float("inf")
        best_state = None
        wait = 0

        for ep in range(epochs):
            self.model.train()
            total_loss = 0.0
            total_mse = 0.0
            total_phys = 0.0
            n_batches = 0

            for xb, yb, mb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                mb = mb.to(device)  # float mask (0/1)
                mb_bool = mb.bool()

                B = xb.size(0)
                # random temperature per sample
                T_rand = torch.rand(B, device=device) * (
                    self.temp_range[1] - self.temp_range[0]
                ) + self.temp_range[0]

                pred_std = self.model(xb)  # (B, T)

                # multi-task uncertainty-weighted MSE in std space
                loss_mse = 0.0
                active = 0
                for j in range(len(self.targets)):
                    m_j = mb_bool[:, j]
                    if m_j.sum() < 2:
                        continue
                    diff = pred_std[m_j, j] - yb[m_j, j]
                    mse_j = torch.mean(diff ** 2)
                    w_j = torch.exp(-self.model.log_sigma[j])
                    loss_j = 0.5 * (w_j * mse_j + self.model.log_sigma[j])
                    loss_mse += loss_j
                    active += 1

                if active == 0:
                    continue

                loss_mse = loss_mse / active

                # back to raw space for physics
                pred_raw = pred_std * self.sigma_t + self.mu_t
                y_raw = yb * self.sigma_t + self.mu_t

                phys_loss = self.physics_weight * self.physics_loss(
                    pred_raw, y_raw, mb_bool, T_rand
                )

                loss = loss_mse + phys_loss

                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optim.step()

                total_loss += loss.item()
                total_mse += loss_mse.item()
                total_phys += phys_loss.item()
                n_batches += 1

            scheduler.step()

            if n_batches > 0:
                avg_total = total_loss / n_batches
                avg_mse = total_mse / n_batches
                avg_phys = total_phys / n_batches
            else:
                avg_total = float("nan")
                avg_mse = float("nan")
                avg_phys = float("nan")

            # validation (just coeff loss)
            self.model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for xb, yb, mb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    mb = mb.to(device)
                    mb_bool = mb.bool()
                    pred_std = self.model(xb)

                    loss_val = 0.0
                    active = 0
                    for j in range(len(self.targets)):
                        m_j = mb_bool[:, j]
                        if m_j.sum() < 2:
                            continue
                        diff = pred_std[m_j, j] - yb[m_j, j]
                        mse_j = torch.mean(diff ** 2)
                        w_j = torch.exp(-self.model.log_sigma[j])
                        loss_j = 0.5 * (w_j * mse_j + self.model.log_sigma[j])
                        loss_val += loss_j
                        active += 1
                    if active == 0:
                        continue
                    loss_val = loss_val / active
                    val_loss += loss_val.item()
                    val_batches += 1

            if val_batches > 0:
                val_avg = val_loss / val_batches
            else:
                val_avg = float("nan")

            print(
                f"Epoch {ep:3d} | train {avg_total:.4f} | mse {avg_mse:.4f} "
                f"| phys {avg_phys:.4f} | val {val_avg:.4f}"
            )

            # early stopping
            if val_avg < best_val - 1e-4:
                best_val = val_avg
                best_state = self.model.state_dict()
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print("Early stopping")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            # save best model
            model_path = os.path.join(self.model_dir, "HybridNAM_best.pt")
            torch.save(self.model.state_dict(), model_path)
            self.best_model_path = model_path
            print(f"\nSaved best NAM model to {model_path}")

    # ---------------------------
    # coefficient-level metrics (test)
    # ---------------------------
    def evaluate_coefficients_on_test(self):
        self.model.eval()
        idx = self.te_idx
        X_te = torch.tensor(self.X[idx], dtype=torch.float32, device=device)

        with torch.no_grad():
            pred_std = self.model(X_te)  # (Nte, T)
            pred_raw = pred_std.cpu().numpy() * self.σ + self.μ

        y_true_full = self.y_raw_full[idx]  # with NaNs
        rows = []
        print("\nCoefficient-level metrics (per target) on test split:")
        for j, t in enumerate(self.targets):
            mask = np.isfinite(y_true_full[:, j])
            if mask.sum() < 2:
                continue
            y_t = y_true_full[mask, j]
            y_p = pred_raw[mask, j]

            r2 = r2_score(y_t, y_p)
            mse = mean_squared_error(y_t, y_p)
            rm = rel_mse_pct(y_t, y_p)

            print(f"{t:8s} R²={r2:+.4f}  MSE={mse:.3g}  relMSE%={rm:.3f}")

            idx_local = np.where(mask)[0]
            for k, (yt, yp) in enumerate(zip(y_t, y_p)):
                global_idx = idx[idx_local[k]]
                rows.append(
                    {
                        "index": global_idx,
                        "target": t,
                        "actual": yt,
                        "predicted": yp,
                        "abs_error": abs(yt - yp),
                        "percent_error": 100.0 * abs(yt - yp) / (abs(yt) + 1e-12),
                    }
                )

        df_pa = pd.DataFrame(rows)
        self.pred_vs_actual_test = df_pa
        return df_pa

    # ---------------------------
    # physical property evaluation at one T
    # ---------------------------
    def evaluate_physical(self, T_eval=800.0, split="test"):
        if split == "test":
            idx = self.te_idx
        else:
            idx = self.va_idx

        self.model.eval()
        X_s = torch.tensor(self.X[idx], dtype=torch.float32, device=device)

        with torch.no_grad():
            pred_std = self.model(X_s)
            pred_raw = pred_std.cpu().numpy() * self.σ + self.μ

        y_true_full = self.y_raw_full[idx]
        ix = self.target_index
        results = {}

        # density
        if "rho_a" in ix and "rho_b" in ix:
            mask = np.isfinite(y_true_full[:, ix["rho_a"]]) & np.isfinite(
                y_true_full[:, ix["rho_b"]]
            )
            if mask.sum() >= 2:
                ya = y_true_full[mask, ix["rho_a"]] - y_true_full[mask, ix["rho_b"]] * T_eval
                yp = pred_raw[mask, ix["rho_a"]] - pred_raw[mask, ix["rho_b"]] * T_eval
                results["density"] = {"MSE%": rel_mse_pct(ya, yp), "R2": r2_score(ya, yp)}

        # muA
        if "mu1_a" in ix and "mu1_b" in ix:
            mask = np.isfinite(y_true_full[:, ix["mu1_a"]]) & np.isfinite(
                y_true_full[:, ix["mu1_b"]]
            )
            if mask.sum() >= 2:
                ya = y_true_full[mask, ix["mu1_a"]] * np.exp(
                    y_true_full[mask, ix["mu1_b"]] / (R_GAS * T_eval)
                )
                yp = pred_raw[mask, ix["mu1_a"]] * np.exp(
                    pred_raw[mask, ix["mu1_b"]] / (R_GAS * T_eval)
                )
                results["viscA"] = {"MSE%": rel_mse_pct(ya, yp), "R2": r2_score(ya, yp)}

        # muB
        if "mu2_a" in ix and "mu2_b" in ix and "mu2_c" in ix:
            mask = (
                np.isfinite(y_true_full[:, ix["mu2_a"]])
                & np.isfinite(y_true_full[:, ix["mu2_b"]])
                & np.isfinite(y_true_full[:, ix["mu2_c"]])
            )
            if mask.sum() >= 2:
                ya_log = (
                    y_true_full[mask, ix["mu2_a"]]
                    + y_true_full[mask, ix["mu2_b"]] / T_eval
                    + y_true_full[mask, ix["mu2_c"]] / T_eval**2
                )
                yp_log = (
                    pred_raw[mask, ix["mu2_a"]]
                    + pred_raw[mask, ix["mu2_b"]] / T_eval
                    + pred_raw[mask, ix["mu2_c"]] / T_eval**2
                )
                ya = 10.0 ** ya_log
                yp = 10.0 ** yp_log
                results["viscB"] = {"MSE%": rel_mse_pct(ya, yp), "R2": r2_score(ya, yp)}

        # k
        if "k_a" in ix and "k_b" in ix:
            mask = np.isfinite(y_true_full[:, ix["k_a"]]) & np.isfinite(
                y_true_full[:, ix["k_b"]]
            )
            if mask.sum() >= 2:
                ya = y_true_full[mask, ix["k_a"]] + y_true_full[mask, ix["k_b"]] * T_eval
                yp = pred_raw[mask, ix["k_a"]] + pred_raw[mask, ix["k_b"]] * T_eval
                results["k"] = {"MSE%": rel_mse_pct(ya, yp), "R2": r2_score(ya, yp)}

        print(f"\nPhysical Property Evaluation @ T = {T_eval}")
        for k, v in results.items():
            print(f"{k:8s} MSE%={v['MSE%']:.2f}   R²={v['R2']:+.3f}")

        return results

    # ---------------------------
    # plotting helpers
    # (kept here if you ever want to call them directly, but not used
    #  in this script's main; plotting is in nam_v2_plots.py)
    # ---------------------------
    def plot_actual_vs_predicted(self, df_pa, target):
        d = df_pa[df_pa["target"] == target]
        if len(d) == 0:
            print(f"No data for {target}")
            return
        plt.figure(figsize=(6, 5))
        sns.scatterplot(data=d, x="actual", y="predicted", alpha=0.6)
        lo = min(d["actual"].min(), d["predicted"].min())
        hi = max(d["actual"].max(), d["predicted"].max())
        plt.plot([lo, hi], [lo, hi], "r--")
        plt.title(f"Actual vs Predicted – {target}")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.tight_layout()
        _savefig(f"actual_vs_pred_{target}.png")
        plt.close()

    def plot_residuals(self, df_pa, target):
        d = df_pa[df_pa["target"] == target].copy()
        if len(d) == 0:
            print(f"No data for {target}")
            return
        d["residual"] = d["predicted"] - d["actual"]
        plt.figure(figsize=(6, 5))
        sns.scatterplot(data=d, x="actual", y="residual", alpha=0.6)
        plt.axhline(0, color="red", linestyle="--")
        plt.title(f"Residuals – {target}")
        plt.xlabel("Actual")
        plt.ylabel("Residual")
        plt.tight_layout()
        _savefig(f"residuals_{target}.png")
        plt.close()

    def plot_error_distribution(self, df_pa, target, percent=False):
        d = df_pa[df_pa["target"] == target]
        if len(d) == 0:
            print(f"No data for {target}")
            return
        plt.figure(figsize=(6, 4))
        values = d["percent_error"] if percent else d["abs_error"]
        sns.histplot(values, bins=30, kde=True)
        plt.xlabel("Percent Error (%)" if percent else "Absolute Error")
        plt.title(
            f"Error Distribution – {target} "
            + ("(%)" if percent else "(abs)")
        )
        plt.tight_layout()
        suffix = "pct" if percent else "abs"
        _savefig(f"error_dist_{target}_{suffix}.png")
        plt.close()

    def build_error_bin_matrix(self, df_pa, bins=(0, 5, 10, 20, 50, 100)):
        labels = []
        for i in range(len(bins) - 1):
            labels.append(f"{bins[i]}–{bins[i+1]}%")
        labels.append(f">{bins[-1]}%")

        mat = pd.DataFrame(0, index=self.targets, columns=labels)

        for t in self.targets:
            d = df_pa[df_pa["target"] == t]
            if len(d) == 0:
                continue
            errs = d["percent_error"].values
            idxs = np.digitize(errs, bins)
            for idx in idxs:
                if idx < len(bins):
                    col = labels[idx - 1] if idx > 0 else labels[0]
                else:
                    col = labels[-1]
                mat.loc[t, col] += 1
        return mat

    def plot_error_bin_matrix(self, df_pa, bins=(0, 5, 10, 20, 50, 100)):
        mat = self.build_error_bin_matrix(df_pa, bins=bins)
        plt.figure(figsize=(1.4 * len(mat.columns), 0.5 * len(mat.index) + 3))
        sns.heatmap(mat, annot=True, fmt="d", cmap="viridis", cbar_kws={"label": "Count"})
        plt.xlabel("Percent Error Bin")
        plt.ylabel("Target")
        plt.title("Error-bin Confusion Matrix (NAM)")
        plt.tight_layout()
        _savefig("error_bin_confusion_matrix.png")
        plt.close()

    def generate_all_plots(self):
        if self.pred_vs_actual_test is None:
            print("No stored test predictions. Run evaluate_coefficients_on_test() first.")
            return

        df_pa = self.pred_vs_actual_test

        print("\n=== Generating per-target NAM plots ===")
        for t in self.targets:
            d = df_pa[df_pa["target"] == t]
            if len(d) == 0:
                print(f"[{t}] – no data, skipping.")
                continue
            print(f"[{t}] – plotting ...")
            self.plot_actual_vs_predicted(df_pa, t)
            self.plot_residuals(df_pa, t)
            self.plot_error_distribution(df_pa, t, percent=False)
            self.plot_error_distribution(df_pa, t, percent=True)

        print("\n=== Generating global NAM error-bin matrix ===")
        self.plot_error_bin_matrix(df_pa, bins=(0, 5, 10, 20, 50, 100))
        print(f"\nAll NAM plots saved in '{NAM_PLOT_DIR}'.")


# --------------------------------------------------
# main
# --------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    # just to have a quick look when running
    print(df.head())
    print(df.columns)

    # if Composition is already in the CSV as JSON-like string, load it
    if isinstance(df["Composition"].iloc[0], str):
        def parse_comp(x):
            if isinstance(x, str):
                try:
                    return json.loads(x.replace("'", '"'))
                except Exception:
                    return {}
            return x
        df["Composition"] = df["Composition"].apply(parse_comp)

    TARGETS = [
        "rho_a", "rho_b",
        "mu1_a", "mu1_b",
        "mu2_a", "mu2_b", "mu2_c",
        "k_a", "k_b",
        "cp_a", "cp_b",
    ]

    DERIVED_GROUPS = [
        ("rho", ["rho_a", "rho_b"]),
        ("muA", ["mu1_a", "mu1_b"]),
        ("muB", ["mu2_a", "mu2_b", "mu2_c"]),
        ("k",   ["k_a", "k_b"]),
        ("cp",  ["cp_a", "cp_b"]),
    ]

    trainer = HybridNAMTrainer(
        df=df,
        targets=TARGETS,
        derived_groups=DERIVED_GROUPS,
        physics_weight=0.1,
        temp_range=(500.0, 1200.0),
        batch_size=128,
    )

    trainer.train(epochs=80, patience=20, lr=1e-3)

    df_pa = trainer.evaluate_coefficients_on_test()
    trainer.evaluate_physical(T_eval=800.0, split="test")

    # save csv with test predictions
    out_csv = os.path.join(NAM_PLOT_DIR, "predicted_vs_actual_NAM_test.csv")
    df_pa.to_csv(out_csv, index=False)
    print(f"\nSaved predicted_vs_actual_NAM_test.csv in {NAM_PLOT_DIR}")

    print("\nDone training & evaluation.")
    if trainer.best_model_path is not None:
        print(f"Best model saved at: {trainer.best_model_path}")
