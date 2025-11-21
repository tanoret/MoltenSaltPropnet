"""
Overview
more or less the architecture to remember

What this script does
- builds element-fraction features from the molten salt database
- trains a single shared backbone (MLP) on compositions
- on top of that, one NAM-style head per target coefficient (multi-head NAM)
- missing targets are handled with masks, so rows can have only a few labels
- physics loss nudges the model towards reasonable temperature behaviour
  (rho(T), mu(T), k(T), cp(T))
- stronger regularisation on the problematic slope-like coefficients (C2 idea)

What is a NAM (Neural Additive Model) here?
- each target is modelled by a small spline-based network
- instead of one big black box, you have per-target additive heads
- we still have a shared backbone so the heads see a good representation
"""

import os
import json
import re
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns

# make local package visible
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from processing_mstdb.processor import MSTDBProcessor

sns.set(style="whitegrid")

# -----------------------------
# basic configuration
# -----------------------------

CSV_PATH = "/Users/meggie/Documents/MoltenSaltPropnet/data/new_mstdb_janz.csv"

R_GAS = 8.314
SEED = 42

BASE_PLOT_DIR = "visualisation"
NAM_PLOT_DIR = os.path.join(BASE_PLOT_DIR, "NAM")
os.makedirs(NAM_PLOT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
np.random.seed(SEED)

# which coefficient columns we try to learn
TARGETS = [
    "rho_a", "rho_b",
    "mu1_a", "mu1_b",
    "mu2_a", "mu2_b", "mu2_c",
    "k_a", "k_b",
    "cp_a", "cp_b", "cp_c"  # cp_c might be missing; we will handle that
]

# groups for physics-based loss / evaluation
DERIVED_GROUPS = [
    ("rho", ["rho_a", "rho_b"]),
    ("muA", ["mu1_a", "mu1_b"]),                 # Arrhenius-like
    ("muB", ["mu2_a", "mu2_b", "mu2_c"]),        # VFT-ish
    ("k",   ["k_a", "k_b"]),
    ("cp",  ["cp_a", "cp_b", "cp_c"]),           # only used if cp_c exists
]


def _savefig(filename: str):
    full_path = os.path.join(NAM_PLOT_DIR, filename)
    plt.savefig(full_path, dpi=200, bbox_inches="tight")
    print(f"Saved → {full_path}")


def rel_mse_pct(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mse = mean_squared_error(y_true, y_pred)
    denom = np.mean(y_true ** 2)
    if denom <= 0:
        denom = 1e-12
    return 100.0 * mse / denom


# -----------------------------
# NAM-style spline layers
# -----------------------------

class SplineLayer(nn.Module):
    """small spline layer: triangular basis around learned knots"""

    def __init__(self, in_dim, out_dim, num_knots=16):
        super().__init__()
        self.num_knots = num_knots
        # knots shared across dimensions; simple but works
        self.knots = nn.Parameter(torch.linspace(-1.0, 1.0, num_knots))
        self.w = nn.Parameter(torch.randn(out_dim, in_dim, num_knots) * 0.1)
        self.b = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        # x: (B, D)
        B, D = x.shape
        diffs = x.unsqueeze(2) - self.knots  # (B, D, K)
        basis = torch.relu(1.0 - torch.abs(diffs * 3.0))  # triangle
        out = torch.einsum("b d k, o d k -> b o", basis, self.w)
        return out + self.b


class NAMHead(nn.Module):
    """per-target NAM head on top of the shared backbone representation"""

    def __init__(self, in_dim, hidden=32, num_knots=16):
        super().__init__()
        self.s1 = SplineLayer(in_dim, hidden, num_knots)
        self.s2 = SplineLayer(hidden, hidden, num_knots)
        self.out = nn.Linear(hidden, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.s1(x))
        x = self.act(self.s2(x))
        return self.out(x).squeeze(-1)


class Backbone(nn.Module):
    """shared encoder for compositions"""

    def __init__(self, in_dim, hidden=128, depth=2, drop=0.1):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.SiLU(), nn.Dropout(drop)]
        for _ in range(depth):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(drop))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# -----------------------------
# multi-head NAM trainer
# -----------------------------

class MultiHeadNAMTrainer:
    def __init__(
        self,
        df: pd.DataFrame,
        targets: List[str],
        derived_groups: List[Tuple[str, List[str]]],
        physics_weight: float = 0.1,
    ):
        self.df = df.copy()
        self.physics_weight = physics_weight
        self.derived_groups = derived_groups

        # --- clean target columns; some may be missing or all NaN ---
        self.present_targets: List[str] = []
        for t in targets:
            if t not in self.df.columns:
                continue
            col = self.df[t]
            col = col.replace(["----", ""], np.nan).replace(r"\*", "", regex=True)
            col = pd.to_numeric(col, errors="coerce")
            if np.isfinite(col).any():
                self.df[t] = col
                self.present_targets.append(t)

        if not self.present_targets:
            raise RuntimeError("No usable target columns found in dataframe.")

        print("\nTargets used in NAM:", self.present_targets)

        # --- composition → element fraction matrix ---
        # --- composition → element fraction matrix ---
        if "Composition" not in self.df.columns:
            raise RuntimeError("Expected 'Composition' column with element fractions.")

        # Convert python-dict-style strings ('Al':0.25) → real dicts
        import ast
        if self.df["Composition"].dtype == object:
            try:
                self.df["Composition"] = self.df["Composition"].apply(ast.literal_eval)
            except Exception:
                raise RuntimeError("Failed to parse Composition column. They must look like {'Al': 0.25}.")

        # Now normalise into element columns
        comp_df = pd.json_normalize(self.df["Composition"]).fillna(0.0)

        # Sort elements consistently
        comp_df = comp_df.reindex(sorted(comp_df.columns), axis=1)

        if comp_df.shape[1] == 0:
            raise RuntimeError("Composition column did not contain any valid elements.")


        self.element_columns = list(comp_df.columns)
        print("\nElement columns used in NAM:", self.element_columns)

        # --- features: just scaled elemental fractions (keep it simple + stable) ---
        self.x_scaler = StandardScaler()
        X_frac = comp_df.to_numpy(dtype=np.float32)
        X = self.x_scaler.fit_transform(X_frac).astype(np.float32)
        self.X = X
        self.feat_dim = X.shape[1]

        # --- target matrix + mask (handle missing) ---
        y_mat = self.df[self.present_targets].to_numpy(dtype=np.float32)
        mask_all = np.isfinite(y_mat)
        self.mask_all = mask_all

        # fill missing with 0 for the tensor, masking will handle it in loss
        y_mat_filled = np.where(mask_all, y_mat, 0.0)
        self.y_raw = y_mat_filled

        # --- train / val / test split ---
        idx_all = np.arange(len(self.X))
        tr_idx, te_idx = train_test_split(idx_all, test_size=0.20, random_state=SEED)
        tr_idx, va_idx = train_test_split(tr_idx, test_size=0.20, random_state=SEED)

        self.tr_idx = tr_idx
        self.va_idx = va_idx
        self.te_idx = te_idx
        self.idx_all = idx_all

        # --- normalise targets per-column using train split only ---
        μ = self.y_raw[tr_idx].mean(axis=0)
        σ = self.y_raw[tr_idx].std(axis=0)
        σ[σ == 0.0] = 1.0  # avoid divide by zero

        self.μ = μ
        self.σ = σ

        self.y_std = (self.y_raw - μ) / σ
        self.mask_float = self.mask_all.astype(np.float32)

        print("\nPer-target normalisation (μ, σ):")
        for j, t in enumerate(self.present_targets):
            print(f"  {t}: mean={μ[j]:.4g}, std={σ[j]:.4g}")

        # --- model pieces ---
        self.backbone = Backbone(self.feat_dim, hidden=128, depth=2, drop=0.1).to(device)
        self.heads = nn.ModuleDict(
            {t: NAMHead(in_dim=128, hidden=48, num_knots=16).to(device) for t in self.present_targets}
        )
        self.idx_map = {name: j for j, name in enumerate(self.present_targets)}

        # history for optional diagnostics
        self.history = {"train_total": [], "train_mse": [], "train_phys": [], "val_mse": []}

        # strong regularisation on some “nasty” slopes (in standardised space)
        self.reg_strength = {
            "mu1_b": 0.05,   # Arrhenius slope
            "mu2_c": 0.05,   # high-order VFT term
            "k_b":   0.02,   # slope of k(T)
            "cp_b":  0.02,   # slope of cp(T)
        }

    # ---------- helpers ----------

    def _make_loader(self, idx, batch_size=128, shuffle=True):
        Xb = self.X[idx]
        yb = self.y_std[idx]
        mb = self.mask_float[idx]
        ds = TensorDataset(
            torch.tensor(Xb, dtype=torch.float32),
            torch.tensor(yb, dtype=torch.float32),
            torch.tensor(mb, dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    def _forward_batch(self, xb: torch.Tensor) -> torch.Tensor:
        """forward through backbone and all heads; returns preds in std space (B, P)"""
        rep = self.backbone(xb)  # (B, H)
        preds = []
        for t in self.present_targets:
            preds.append(self.heads[t](rep))
        return torch.stack(preds, dim=1)  # (B, P)

    def _physics_loss(self, pred_std, y_std, mask, T_tensor):
        """
        physics loss in raw space, but only on rows where all coefficients exist for that group.
        pred_std, y_std: (B, P)
        mask: (B, P)
        T_tensor: (B,)
        """
        μ_t = torch.tensor(self.μ, device=device, dtype=torch.float32)
        σ_t = torch.tensor(self.σ, device=device, dtype=torch.float32)

        pred_raw = pred_std * σ_t + μ_t
        y_raw = y_std * σ_t + μ_t

        loss = 0.0
        valid_terms = 0

        for dname, coeffs in self.derived_groups:
            idxs = [self.idx_map[c] for c in coeffs if c in self.idx_map]
            if len(idxs) != len(coeffs):
                continue

            # rows where *all* needed coeffs are valid
            m_group = mask[:, idxs].min(dim=1).values > 0.5
            if not m_group.any():
                continue

            T = T_tensor[m_group]
            y_c = y_raw[m_group][:, idxs]
            p_c = pred_raw[m_group][:, idxs]

            if dname == "rho":
                # ρ(T) = ρ_a + ρ_b * T
                y_vals = y_c[:, 0] + y_c[:, 1] * T
                p_vals = p_c[:, 0] + p_c[:, 1] * T

            elif dname == "muA":
                # Arrhenius-like, work in log space
                A_y = torch.clamp(y_c[:, 0], min=1e-6)
                B_y = y_c[:, 1]
                A_p = torch.clamp(p_c[:, 0], min=1e-6)
                B_p = p_c[:, 1]

                exp_y = torch.clamp(B_y / (R_GAS * T), -50.0, 50.0)
                exp_p = torch.clamp(B_p / (R_GAS * T), -50.0, 50.0)

                mu_y = A_y * torch.exp(exp_y)
                mu_p = A_p * torch.exp(exp_p)

                y_vals = torch.log(mu_y + 1e-8)
                p_vals = torch.log(mu_p + 1e-8)

            elif dname == "muB":
                # simple polynomial in 1/T, 1/T² in log space
                invT = 1.0 / T
                invT2 = invT * invT
                y_vals = y_c[:, 0] + y_c[:, 1] * invT + y_c[:, 2] * invT2
                p_vals = p_c[:, 0] + p_c[:, 1] * invT + p_c[:, 2] * invT2

            elif dname == "k":
                y_vals = y_c[:, 0] + y_c[:, 1] * T
                p_vals = p_c[:, 0] + p_c[:, 1] * T

            elif dname == "cp":
                # if cp_c missing, group is skipped earlier
                invT2 = 1.0 / (T * T)
                y_vals = y_c[:, 0] + y_c[:, 1] * T + y_c[:, 2] * invT2
                p_vals = p_c[:, 0] + p_c[:, 1] * T + p_c[:, 2] * invT2

            else:
                continue

            term = nn.functional.mse_loss(p_vals, y_vals)
            loss += term
            valid_terms += 1

        if valid_terms == 0:
            return torch.tensor(0.0, device=device)
        return loss / valid_terms

    # ---------- training & evaluation ----------

    def train_joint(
        self,
        epochs: int = 80,
        batch_size: int = 128,
        lr: float = 1e-3,
        patience: int = 15,
    ):
        print("\nstarting NAM multi-head + shared backbone training ...")

        train_loader = self._make_loader(self.tr_idx, batch_size=batch_size, shuffle=True)
        val_loader = self._make_loader(self.va_idx, batch_size=256, shuffle=False)

        params = list(self.backbone.parameters())
        for h in self.heads.values():
            params += list(h.parameters())

        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)

        best_val = float("inf")
        wait = 0
        best_state = None

        for ep in range(epochs):
            self.backbone.train()
            for h in self.heads.values():
                h.train()

            total_loss = 0.0
            total_mse = 0.0
            total_phys = 0.0
            nbatches = 0

            for xb, yb, mb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                mb = mb.to(device)

                B = xb.size(0)
                T_rand = torch.rand(B, device=device) * (1200.0 - 500.0) + 500.0

                preds_std = self._forward_batch(xb)  # (B, P)

                diff = (preds_std - yb) * mb
                mse_loss = (diff ** 2).sum() / mb.sum().clamp_min(1.0)

                phys_loss = self._physics_loss(preds_std, yb, mb, T_rand)

                # strong regularisation on "bad" coefficients (in std space)
                reg_loss = 0.0
                for name, lam in self.reg_strength.items():
                    if name in self.idx_map:
                        j = self.idx_map[name]
                        col = preds_std[:, j]
                        reg_loss = reg_loss + lam * (col ** 2).mean()

                loss = mse_loss + self.physics_weight * phys_loss + reg_loss

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()

                total_loss += loss.item()
                total_mse += mse_loss.item()
                total_phys += phys_loss.item()
                nbatches += 1

            sched.step()

            avg_loss = total_loss / max(nbatches, 1)
            avg_mse = total_mse / max(nbatches, 1)
            avg_phys = total_phys / max(nbatches, 1)

            # validation
            self.backbone.eval()
            for h in self.heads.values():
                h.eval()

            val_loss = 0.0
            v_batches = 0
            with torch.no_grad():
                for xb, yb, mb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    mb = mb.to(device)

                    preds_std = self._forward_batch(xb)
                    diff = (preds_std - yb) * mb
                    mse_val = (diff ** 2).sum() / mb.sum().clamp_min(1.0)
                    val_loss += mse_val.item()
                    v_batches += 1

            val_loss /= max(v_batches, 1)

            self.history["train_total"].append(avg_loss)
            self.history["train_mse"].append(avg_mse)
            self.history["train_phys"].append(avg_phys)
            self.history["val_mse"].append(val_loss)

            print(
                f"Epoch {ep:3d} | train {avg_loss:.4f} | mse {avg_mse:.4f} "
                f"| phys {avg_phys:.4f} | val {val_loss:.4f}"
            )

            if val_loss < best_val - 1e-4:
                best_val = val_loss
                wait = 0
                best_state = {
                    "backbone": self.backbone.state_dict(),
                    "heads": {k: v.state_dict() for k, v in self.heads.items()},
                }
            else:
                wait += 1
                if wait >= patience:
                    print("Early stopping")
                    break

        if best_state is not None:
            self.backbone.load_state_dict(best_state["backbone"])
            for k, v in self.heads.items():
                if k in best_state["heads"]:
                    v.load_state_dict(best_state["heads"][k])

    def _predict_split(self, split: str = "test"):
        if split == "train":
            idx = self.tr_idx
        elif split == "val":
            idx = self.va_idx
        else:
            idx = self.te_idx

        Xs = self.X[idx]
        ys = self.y_raw[idx]
        ms = self.mask_all[idx]

        self.backbone.eval()
        for h in self.heads.values():
            h.eval()

        with torch.no_grad():
            xb = torch.tensor(Xs, dtype=torch.float32, device=device)
            preds_std = self._forward_batch(xb).cpu().numpy()

        preds_raw = preds_std * self.σ + self.μ
        return idx, ys, preds_raw, ms

    def evaluate_coefficients(self, split: str = "test"):
        idx, y_true, y_pred, mask = self._predict_split(split=split)

        print("\nCoefficient-level metrics (per target) on {} split:".format(split))
        for j, t in enumerate(self.present_targets):
            m = mask[:, j]
            if not m.any():
                continue
            yt = y_true[m, j]
            yp = y_pred[m, j]
            if not np.isfinite(yt).any():
                continue
            try:
                r2 = r2_score(yt, yp)
                mse = mean_squared_error(yt, yp)
                relm = rel_mse_pct(yt, yp)
            except ValueError:
                continue
            print(f"{t:<8s} R²={r2:+.4f}  MSE={mse:.4g}  relMSE%={relm:.3f}")

    def evaluate_physical(self, T_eval: float = 800.0, split: str = "test"):
        idx, y_true, y_pred, mask = self._predict_split(split=split)

        results = {}
        ix = self.idx_map

        def _safe_rel_r2(y_t, y_p):
            y_t = np.asarray(y_t)
            y_p = np.asarray(y_p)
            m = np.isfinite(y_t) & np.isfinite(y_p)
            if not m.any():
                return None, None
            yt = y_t[m]
            yp = y_p[m]
            return rel_mse_pct(yt, yp), r2_score(yt, yp)

        # density
        if "rho_a" in ix and "rho_b" in ix:
            j0, j1 = ix["rho_a"], ix["rho_b"]
            m = mask[:, j0] & mask[:, j1]
            if m.any():
                rho_t = y_true[m, j0] + y_true[m, j1] * T_eval
                rho_p = y_pred[m, j0] + y_pred[m, j1] * T_eval
                mse_pct, r2 = _safe_rel_r2(rho_t, rho_p)
                if mse_pct is not None:
                    results["density"] = {"MSE%": mse_pct, "R2": r2}

        # viscosity A (Arrhenius-like)
        if "mu1_a" in ix and "mu1_b" in ix:
            j0, j1 = ix["mu1_a"], ix["mu1_b"]
            m = mask[:, j0] & mask[:, j1]
            if m.any():
                A_t = y_true[m, j0]
                B_t = y_true[m, j1]
                A_p = y_pred[m, j0]
                B_p = y_pred[m, j1]

                def arrhenius(A, B):
                    exp = np.clip(B / (R_GAS * T_eval), -50, 50)
                    return np.abs(A) * np.exp(exp)

                mu_t = arrhenius(A_t, B_t)
                mu_p = arrhenius(A_p, B_p)
                mse_pct, r2 = _safe_rel_r2(mu_t, mu_p)
                if mse_pct is not None:
                    results["viscA"] = {"MSE%": mse_pct, "R2": r2}

        # viscosity B (mu2)
        if all(k in ix for k in ["mu2_a", "mu2_b", "mu2_c"]):
            j0, j1, j2 = ix["mu2_a"], ix["mu2_b"], ix["mu2_c"]
            m = mask[:, j0] & mask[:, j1] & mask[:, j2]
            if m.any():
                a_t = y_true[m, j0]
                b_t = y_true[m, j1]
                c_t = y_true[m, j2]
                a_p = y_pred[m, j0]
                b_p = y_pred[m, j1]
                c_p = y_pred[m, j2]

                def vft(a, b, c):
                    invT = 1.0 / T_eval
                    invT2 = invT * invT
                    log10_mu = a + b * invT + c * invT2
                    ex = np.clip(np.log(10.0) * log10_mu, -50, 50)
                    return np.exp(ex)

                mu_t = vft(a_t, b_t, c_t)
                mu_p = vft(a_p, b_p, c_p)
                mse_pct, r2 = _safe_rel_r2(mu_t, mu_p)
                if mse_pct is not None:
                    results["viscB"] = {"MSE%": mse_pct, "R2": r2}

        # thermal conductivity
        if "k_a" in ix and "k_b" in ix:
            j0, j1 = ix["k_a"], ix["k_b"]
            m = mask[:, j0] & mask[:, j1]
            if m.any():
                k_t = y_true[m, j0] + y_true[m, j1] * T_eval
                k_p = y_pred[m, j0] + y_pred[m, j1] * T_eval
                mse_pct, r2 = _safe_rel_r2(k_t, k_p)
                if mse_pct is not None:
                    results["k"] = {"MSE%": mse_pct, "R2": r2}

        # heat capacity (only if cp_c exists)
        if all(k in ix for k in ["cp_a", "cp_b", "cp_c"]):
            j0, j1, j2 = ix["cp_a"], ix["cp_b"], ix["cp_c"]
            m = mask[:, j0] & mask[:, j1] & mask[:, j2]
            if m.any():
                cp_t = (
                    y_true[m, j0]
                    + y_true[m, j1] * T_eval
                    + y_true[m, j2] / (T_eval ** 2)
                )
                cp_p = (
                    y_pred[m, j0]
                    + y_pred[m, j1] * T_eval
                    + y_pred[m, j2] / (T_eval ** 2)
                )
                mse_pct, r2 = _safe_rel_r2(cp_t, cp_p)
                if mse_pct is not None:
                    results["cp"] = {"MSE%": mse_pct, "R2": r2}

        print(f"\nPhysical Property Evaluation @ T = {T_eval}")
        for k, v in results.items():
            print(f"{k:10s} MSE%={v['MSE%']:.2f}   R²={v['R2']:+.3f}")

        return results

    def build_pred_vs_actual_df(self, split: str = "test") -> pd.DataFrame:
        idx, y_true, y_pred, mask = self._predict_split(split=split)

        rows = []
        for j, t in enumerate(self.present_targets):
            m = mask[:, j]
            if not m.any():
                continue
            yt = y_true[m, j]
            yp = y_pred[m, j]
            idxs = np.array(idx)[m]
            for k, (a, p) in enumerate(zip(yt, yp)):
                rows.append(
                    {
                        "index": idxs[k],
                        "target": t,
                        "actual": a,
                        "predicted": p,
                        "abs_error": abs(a - p),
                        "percent_error": 100.0 * abs(a - p) / (abs(a) + 1e-12),
                    }
                )
        return pd.DataFrame(rows)


# -----------------------------
# plotting helpers
# -----------------------------

def plot_actual_vs_predicted(df_pa, target, save=True):
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
    if save:
        _savefig(f"actual_vs_pred_{target}.png")
    plt.close()


def plot_residuals(df_pa, target, save=True):
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
    if save:
        _savefig(f"residuals_{target}.png")
    plt.close()


def plot_error_distribution(df_pa, target, percent=False, save=True):
    d = df_pa[df_pa["target"] == target]
    if len(d) == 0:
        print(f"No data for {target}")
        return

    plt.figure(figsize=(6, 4))
    values = d["percent_error"] if percent else d["abs_error"]

    # cut extremely large outliers for nicer visualisation
    if percent:
        values = np.clip(values, 0, 200)

    sns.histplot(values, bins=30, kde=True)
    plt.xlabel("Percent Error (%)" if percent else "Absolute Error")
    plt.title(f"Error Distribution – {target}")
    if percent:
        plt.xlim(0, 200)
    plt.tight_layout()
    if save:
        suffix = "pct" if percent else "abs"
        _savefig(f"error_dist_{target}_{suffix}.png")
    plt.close()


def build_error_bin_matrix(df_pa, targets, bins=(0, 10, 25, 50, 100, 200)):
    labels = []
    for i in range(len(bins) - 1):
        labels.append(f"{bins[i]}–{bins[i+1]}%")
    labels.append(f">{bins[-1]}%")

    mat = pd.DataFrame(0, index=targets, columns=labels)

    for t in targets:
        d = df_pa[df_pa["target"] == t]
        if len(d) == 0:
            continue
        errs = d["percent_error"].values
        errs = np.clip(errs, 0, 1e6)
        idxs = np.digitize(errs, bins)

        for idx in idxs:
            if idx < len(bins):
                col = labels[idx - 1] if idx > 0 else labels[0]
            else:
                col = labels[-1]
            mat.loc[t, col] += 1
    return mat


def plot_error_bin_matrix(df_pa, targets, bins=(0, 10, 25, 50, 100, 200), save=True):
    mat = build_error_bin_matrix(df_pa, targets, bins=bins)

    plt.figure(figsize=(1.4 * len(mat.columns), 0.5 * len(mat.index) + 3))
    sns.heatmap(mat, annot=True, fmt="d", cmap="viridis", cbar_kws={"label": "Count"})
    plt.xlabel("Percent Error Bin")
    plt.ylabel("Target")
    plt.title("Error-bin Confusion Matrix (% error, clipped at 200%)")
    plt.tight_layout()
    if save:
        _savefig("error_bin_confusion_matrix.png")
    plt.close()


def generate_all_plots(df_raw, pred_df, targets):
    print("\n=== Generating per-target NAM plots ===")
    for t in targets:
        d = pred_df[pred_df["target"] == t]
        if len(d) == 0:
            print(f"[{t}] – no data, skipping.")
            continue
        print(f"[{t}] – plotting ...")
        plot_actual_vs_predicted(pred_df, t, save=True)
        plot_residuals(pred_df, t, save=True)
        plot_error_distribution(pred_df, t, percent=False, save=True)
        plot_error_distribution(pred_df, t, percent=True, save=True)

    print("\n=== Generating global NAM error-bin matrix ===")
    plot_error_bin_matrix(pred_df, targets, bins=(0, 10, 25, 50, 100, 200), save=True)
    print("\nAll NAM plots saved in 'visualisation/NAM'.")


# -----------------------------
# main
# -----------------------------

if __name__ == "__main__":
    # load via MSTDBProcessor so Composition is already element fractions
    processor = MSTDBProcessor.from_csv(CSV_PATH)
    print(processor.df.head())
    processor.df.columns = processor.df.columns.str.strip()
    print(processor.df.columns)

    # if Composition not present or different, you can adapt here
    df = processor.df

    trainer = MultiHeadNAMTrainer(
        df=df,
        targets=TARGETS,
        derived_groups=DERIVED_GROUPS,
        physics_weight=0.1,   # how strong we weight physics vs pure regression
    )

    trainer.train_joint(epochs=80, batch_size=128, lr=1e-3, patience=10)

    trainer.evaluate_coefficients(split="test")
    trainer.evaluate_physical(T_eval=800.0, split="test")

    pred_vs_actual = trainer.build_pred_vs_actual_df(split="test")
    pred_vs_actual.to_csv(os.path.join(NAM_PLOT_DIR, "predicted_vs_actual_NAM_test.csv"), index=False)
    print(f"\nSaved predicted_vs_actual_NAM_test.csv in {NAM_PLOT_DIR}")

    generate_all_plots(df, pred_vs_actual, trainer.present_targets)

    print("\nDone.")
