import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from processing_mstdb.processor import MSTDBProcessor
processor = MSTDBProcessor.from_csv('/Users/meggie/Documents/MoltenSaltPropnet/data/new_mstdb_janz.csv')
print(processor.df.head())
processor.df.columns = processor.df.columns.str.strip()  # clean column names
print(processor.df.columns)

# Compute compositions
compositions = []
for idx, row in processor.df.iterrows():
    comp = processor.compute_composition(row, composition_type='elements')
    compositions.append(comp)
processor.df['Composition'] = compositions

all_elements = sorted(processor.predefined_elements)
X_composition = np.zeros((len(processor.df), len(all_elements)))
for idx, comp in enumerate(compositions):
    for el, frac in comp.items():
        if el in all_elements:
            X_composition[idx, all_elements.index(el)] = frac



# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------
def rel_mse_pct(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    denom = np.mean(y_true**2) if np.mean(y_true**2) > 0 else 1e-12
    return 100 * mse / denom

# ------------------------------------------------------------
# Simple Spline Layer
# ------------------------------------------------------------
class SplineLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_knots=12):
        super().__init__()
        self.num_knots = num_knots

        # Knot positions (learnable)
        self.knots = nn.Parameter(torch.linspace(-1, 1, num_knots))

        # Output weights
        self.w = nn.Parameter(torch.randn(out_dim, in_dim, num_knots) * 0.1)
        self.b = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        # x: (B, in_dim)
        # output: (B, out_dim)
        B, D = x.shape
        diffs = (x.unsqueeze(2) - self.knots)  # (B, D, K)
        basis = torch.relu(1 - torch.abs(diffs * 3))  # triangular basis
        out = torch.einsum("b d k, o d k -> b o", basis, self.w)
        return out + self.b

# ------------------------------------------------------------
# NAM Spline Base Model
# ------------------------------------------------------------
class NAMBase(nn.Module):
    def __init__(self, in_dim, hidden=32, num_knots=12):
        super().__init__()
        self.s1 = SplineLayer(in_dim, hidden, num_knots)
        self.s2 = SplineLayer(hidden, hidden, num_knots)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        x = torch.relu(self.s1(x))
        x = torch.relu(self.s2(x))
        return self.out(x)

# ------------------------------------------------------------
# Meta NAM (adjusts stacked base outputs)
# ------------------------------------------------------------
class NAMMeta(nn.Module):
    def __init__(self, num_props, hidden=32, num_knots=12):
        super().__init__()
        self.s1 = SplineLayer(num_props, hidden, num_knots)
        self.s2 = SplineLayer(hidden, hidden, num_knots)
        self.out = nn.Linear(hidden, num_props)

    def forward(self, x):
        x = torch.relu(self.s1(x))
        x = torch.relu(self.s2(x))
        return self.out(x)

# ------------------------------------------------------------
# Main Trainer (Only Spline NAM)
# ------------------------------------------------------------
class NAMSplineTrainer:
    def __init__(
        self,
        df: pd.DataFrame,
        targets,
        derived_props,
        degree_poly=2,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.df = df.copy()
        self.targets = [c for c in targets if c in df.columns]
        self.dprops = derived_props
        self.device = device

        # Clean targets
        for t in self.targets:
            col = df[t]
            col = col.replace(["----", ""], np.nan)
            col = col.replace(r"\*", "", regex=True)
            col = pd.to_numeric(col, errors="coerce")
            self.df[t] = col.fillna(0)

        # ----------------------------------------------------
        # Composition features from MSTDBProcessor output
        # ----------------------------------------------------
        if "Composition" not in df.columns:
            raise RuntimeError("DataFrame must contain Composition column")

        comp_df = pd.json_normalize(self.df["Composition"]).fillna(0.0)
        comp_df = comp_df.reindex(sorted(comp_df.columns), axis=1)
        self.comp_cols = comp_df.columns

        # Polynomial features
        self.poly = PolynomialFeatures(degree_poly, include_bias=False)
        X_poly = self.poly.fit_transform(comp_df)

        self.scaler = StandardScaler()
        X_poly = self.scaler.fit_transform(X_poly).astype(np.float32)

        frac = comp_df.to_numpy(np.float32)
        self.X = np.hstack([X_poly, frac])
        self.feat_dim = self.X.shape[1]

        # Targets + std
        y_raw = self.df[self.targets].to_numpy(np.float32)
        self.μ = y_raw.mean(0)
        self.σ = y_raw.std(0)
        self.σ[self.σ == 0] = 1.0

        self.y_std = (y_raw - self.μ) / self.σ

        # Mask for missing
        self.mask = np.isfinite(y_raw)

        # Train/val/test split
        idx = np.arange(len(self.X))
        tr, te = train_test_split(idx, test_size=0.20, random_state=42)
        tr, va = train_test_split(tr, test_size=0.20, random_state=42)

        self.tr_idx = tr
        self.va_idx = va
        self.te_idx = te

        # Index mapping
        self.idx_map = {n: j for j, n in enumerate(self.targets)}

        # ----------------------------------------------------
      
        self.base_nets = nn.ModuleDict(
            {t: NAMBase(self.feat_dim).to(device) for t in self.targets}
        )
        self.meta = NAMMeta(len(self.targets)).to(device)

        self.history = {
            "train_total": [],
            "train_mse": [],
            "train_phys": [],
            "val_mse": [],
        }

    # ------------------------------------------------------------
    def physics_loss(self, pred_raw, y_raw, mask, T):
        R = 8.314
        terms = 0
        loss = 0

        for name, coeffs in self.dprops:
            idxs = [self.idx_map[c] for c in coeffs if c in self.idx_map]
            if len(idxs) != len(coeffs):
                continue

            m = torch.all(mask[:, idxs], dim=1)
            if not m.any():
                continue

            p = pred_raw[m][:, idxs]
            y = y_raw[m][:, idxs]
            TT = T[m]

            if name == "rho":
                loss += nn.functional.mse_loss(p[:, 0] - p[:, 1] * TT, y[:, 0] - y[:, 1] * TT)
            elif name == "muA":
                loss += nn.functional.mse_loss(
                    torch.log(torch.abs(p[:, 0]) + 1e-9)
                    + p[:, 1] / (R * TT),
                    torch.log(torch.abs(y[:, 0]) + 1e-9)
                    + y[:, 1] / (R * TT),
                )
            elif name == "muB":
                loss += nn.functional.mse_loss(
                    p[:, 0] + p[:, 1] / TT + p[:, 2] / TT**2,
                    y[:, 0] + y[:, 1] / TT + y[:, 2] / TT**2,
                )
            elif name == "k":
                loss += nn.functional.mse_loss(
                    p[:, 0] + p[:, 1] * TT,
                    y[:, 0] + y[:, 1] * TT,
                )
            elif name == "cp":
                loss += nn.functional.mse_loss(
                    p[:, 0] + p[:, 1] * TT + p[:, 2] / TT**2,
                    y[:, 0] + y[:, 1] * TT + y[:, 2] / TT**2,
                )

            terms += 1

        return loss / max(1, terms)

    # ------------------------------------------------------------
    def train_joint(self, epochs=20, batch=128, lr=1e-3, patience=5):
        device = self.device
        trL = DataLoader(
            TensorDataset(
                torch.tensor(self.X[self.tr_idx]),
                torch.tensor(self.y_std[self.tr_idx]),
                torch.tensor(self.mask[self.tr_idx]),
            ),
            batch_size=batch,
            shuffle=True,
        )
        vaL = DataLoader(
            TensorDataset(
                torch.tensor(self.X[self.va_idx]),
                torch.tensor(self.y_std[self.va_idx]),
                torch.tensor(self.mask[self.va_idx]),
            ),
            batch_size=256,
            shuffle=False,
        )

        # Collect all parameters
        params = list(self.meta.parameters())
        for net in self.base_nets.values():
            params += list(net.parameters())

        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        best, wait = 1e9, 0

        μ_t = torch.tensor(self.μ, device=device)
        σ_t = torch.tensor(self.σ, device=device)

        for ep in range(epochs):
            for net in self.base_nets.values():
                net.train()
            self.meta.train()

            total = mse_c = phys_c = 0

            for xb, yb, mb in trL:
                xb = xb.to(device)
                yb = yb.to(device)
                mb = mb.to(device)

                T = torch.rand(len(xb), device=device) * 700 + 500

                # Forward
                base_outs = []
                for t in self.targets:
                    base_outs.append(self.base_nets[t](xb))
                base = torch.cat(base_outs, dim=1)

                pred = base + self.meta(base)

                mse_loss = ((pred - yb) ** 2 * mb).sum() / mb.sum()

                # Physics loss (de-standardize)
                p_raw = pred * σ_t + μ_t
                y_raw = yb * σ_t + μ_t
                phys_loss = self.physics_loss(p_raw, y_raw, mb, T)

                loss = mse_loss + 0.1 * phys_loss

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()

                total += loss.item()
                mse_c += mse_loss.item()
                phys_c += phys_loss.item()

            total /= len(trL)
            mse_c /= len(trL)
            phys_c /= len(trL)

            # Validation
            val_loss = 0
            with torch.no_grad():
                for net in self.base_nets.values():
                    net.eval()
                self.meta.eval()

                for xb, yb, mb in vaL:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    mb = mb.to(device)

                    base = torch.cat([self.base_nets[t](xb) for t in self.targets], dim=1)
                    pred = base + self.meta(base)

                    val_loss += ((pred - yb) ** 2 * mb).sum().item() / mb.sum().item()
                val_loss /= len(vaL)

            # Log
            self.history["train_total"].append(total)
            self.history["train_mse"].append(mse_c)
            self.history["train_phys"].append(phys_c)
            self.history["val_mse"].append(val_loss)

            print(f"Epoch {ep:3d} | train {total:.4f} | mse {mse_c:.4f} | phys {phys_c:.4f} | val {val_loss:.4f}")

            # Early stopping
            if val_loss < best - 1e-4:
                best = val_loss
                wait = 0
                best_state = {
                    "meta": self.meta.state_dict(),
                    "base": {k: n.state_dict() for k, n in self.base_nets.items()},
                }
            else:
                wait += 1
                if wait >= patience:
                    print("⇢ Early stopping")
                    break

        # Load best model
        self.meta.load_state_dict(best_state["meta"])
        for k, net in self.base_nets.items():
            net.load_state_dict(best_state["base"][k])

    # ------------------------------------------------------------
    def plot_losses(self):
        plt.figure(figsize=(10,5))
        plt.plot(self.history["train_total"], label="Train Total")
        plt.plot(self.history["train_mse"], label="Train MSE")
        plt.plot(self.history["train_phys"], label="Train Physics")
        plt.plot(self.history["val_mse"], label="Validation MSE")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        plt.show()

    # ------------------------------------------------------------
    def evaluate_physical(self, T_eval=800, split="test"):
        if split == "test":
            idx = self.te_idx
        else:
            idx = self.va_idx

        Xs = self.X[idx]
        ys = self.df[self.targets].iloc[idx].to_numpy()

        μ, σ = self.μ, self.σ

        # Predict
        preds = []
        with torch.no_grad():
            xb = torch.tensor(Xs, device=self.device)
            base = torch.cat([self.base_nets[t](xb) for t in self.targets], dim=1)
            pred_std = base + self.meta(base)
            preds = pred_std.cpu().numpy() * σ + μ

        ix = self.idx_map
        results = {}

        # Density
        if "rho_a" in ix and "rho_b" in ix:
            y_p = preds[:, ix["rho_a"]] - preds[:, ix["rho_b"]] * T_eval
            y_t = ys[:, ix["rho_a"]] - ys[:, ix["rho_b"]] * T_eval
            results["density"] = {
                "MSE%": rel_mse_pct(y_t, y_p),
                "R2": r2_score(y_t, y_p),
            }

        # Viscosity A
        if "mu1_a" in ix and "mu1_b" in ix:
            y_p = preds[:, ix["mu1_a"]] * np.exp(preds[:, ix["mu1_b"]] / (8.314 * T_eval))
            y_t = ys[:, ix["mu1_a"]] * np.exp(ys[:, ix["mu1_b"]] / (8.314 * T_eval))
            results["viscA"] = {
                "MSE%": rel_mse_pct(y_t, y_p),
                "R2": r2_score(y_t, y_p),
            }

        # Viscosity B
        if "mu2_a" in ix and "mu2_b" in ix and "mu2_c" in ix:
            y_p = 10 ** (preds[:, ix["mu2_a"]] + preds[:, ix["mu2_b"]] / T_eval + preds[:, ix["mu2_c"]] / T_eval**2)
            y_t = 10 ** (ys[:, ix["mu2_a"]] + ys[:, ix["mu2_b"]] / T_eval + ys[:, ix["mu2_c"]] / T_eval**2)
            results["viscB"] = {
                "MSE%": rel_mse_pct(y_t, y_p),
                "R2": r2_score(y_t, y_p),
            }

        # Thermal conductivity
        if "k_a" in ix and "k_b" in ix:
            y_p = preds[:, ix["k_a"]] + preds[:, ix["k_b"]] * T_eval
            y_t = ys[:, ix["k_a"]] + ys[:, ix["k_b"]] * T_eval
            results["k"] = {
                "MSE%": rel_mse_pct(y_t, y_p),
                "R2": r2_score(y_t, y_p),
            }

        # Heat capacity
        if "cp_a" in ix and "cp_b" in ix and "cp_c" in ix:
            y_p = preds[:, ix["cp_a"]] + preds[:, ix["cp_b"]] * T_eval + preds[:, ix["cp_c"]] / T_eval**2
            y_t = ys[:, ix["cp_a"]] + ys[:, ix["cp_b"]] * T_eval + ys[:, ix["cp_c"]] / T_eval**2
            results["cp"] = {
                "MSE%": rel_mse_pct(y_t, y_p),
                "R2": r2_score(y_t, y_p),
            }

        print("\nPhysical Property Evaluation @ T =", T_eval)
        for k, v in results.items():
            print(f"{k:12s} MSE%={v['MSE%']:.2f}   R²={v['R2']:+.3f}")

        return results

df = processor.df

trainer = NAMSplineTrainer(
    df=df,
    targets=["rho_a","rho_b","mu1_a","mu1_b","mu2_a","mu2_b","mu2_c","k_a","k_b","cp_a","cp_b","cp_c"],
    derived_props=[
        ("rho", ["rho_a","rho_b"]),
        ("muA", ["mu1_a","mu1_b"]),
        ("muB", ["mu2_a","mu2_b","mu2_c"]),
        ("k", ["k_a","k_b"]),
        ("cp", ["cp_a","cp_b","cp_c"])
    ]
)



# ================================================================
#                 📊 NAM MODEL EVALUATION BLOCK
# ================================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

def evaluate_model(model, X, y, label="Validation"):
    """Compute RELMSE, R², MSE for a NAM model."""
    model.eval()
    with torch.no_grad():
        y_pred = model(X).cpu().numpy()
        y_true = y.cpu().numpy()

    # RELMSE (relative mean squared error)
    relmse = np.mean(((y_pred - y_true) ** 2) / (y_true ** 2 + 1e-12))

    # Standard metrics
    mse = mean_squared_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred, multioutput="uniform_average")

    # -------- Print results --------
    print(f"\n📊 {label} Metrics")
    print("--------------------------------------------------")
    print(f"RELMSE : {relmse:.6f}")
    print(f"MSE    : {mse:.6f}")
    print(f"R²     : {r2:.6f}")

    return y_true, y_pred


def plot_actual_vs_pred(y_true, y_pred, feature_names, split="Validation"):
    """Scatter plots of actual vs predicted for each coefficient."""
    n_features = y_true.shape[1]
    plt.figure(figsize=(12, 3 * n_features))

    for i in range(n_features):
        plt.subplot(n_features, 1, i+1)
        plt.scatter(y_true[:, i], y_pred[:, i], s=16, alpha=0.6)
        minv = min(y_true[:, i].min(), y_pred[:, i].min())
        maxv = max(y_true[:, i].max(), y_pred[:, i].max())
        plt.plot([minv, maxv], [minv, maxv], 'r--', label="y = x")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"{split}: {feature_names[i]}")
        plt.grid(True)

    plt.tight_layout()
    plt.show()


# ================================================================
#                      RUN EVALUATION
# ================================================================

print("\n\n====================================================")
print("📈 Running NAM Model Evaluation")
print("====================================================")

# ------ TRAIN METRICS ------
yt_train, yp_train = evaluate_model(
    trainer.model,
    trainer.X_train,
    trainer.y_train,
    "Train"
)

# ------ VALIDATION METRICS ------
yt_val, yp_val = evaluate_model(
    trainer.model,
    trainer.X_val,
    trainer.y_val,
    "Validation"
)

# ------ PLOTS ------
feature_names = trainer.target_columns
plot_actual_vs_pred(yt_val, yp_val, feature_names, split="Validation")

print("\n🎉 Evaluation finished.\n")
trainer.train_joint()

trainer.evaluate()
trainer.evaluate_physical(800)

trainer.plot_losses()
trainer.plot_actual_vs_predicted()

example = {"Na": 0.5, "Cl": 0.5}
pred = trainer.predict(example)
print(pred)
