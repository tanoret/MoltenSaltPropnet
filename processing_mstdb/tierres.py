
"""
ResNet+Meta with hybrid tier-based imputation on TRAINING LABELS only.

- Tier 1: moderately dense, physically key props (rho, mu1)
- Tier 2: medium-sparse props (Melt, Boil, cp_a)
- Ultra-sparse props (mu2, k, cp_b/c) untouched by OT/Iterative (still masked)

Hybrid scheme on TRAINING rows:
    • Tier-1 OT for mu1_a, mu1_b (NOT rho_a/rho_b anymore)
    • KNN smoothing on Tier-1 + Tier-2 props
    • IterativeImputer refinement on Tier-2 props

Physics loss is clamped to avoid numerical explosions.

Outputs:
    - Plots & metrics: evaluate_modelperformance/resnet_hybrid/
    - Optional CV results (3-fold)
"""

import os
import math
import re
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
import sys


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Optimal transport library
import ot  # pip install pot

# Your embedding preconditioner

from processing_mstdb.embedding_preconditioner import EmbeddingPreconditioner

# ---------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------

SEED = 42
R = 8.314
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Main output dir for plots/JSON
OUTDIR = os.path.join("evaluate_modelperformance", "resnet_hybrid")
os.makedirs(OUTDIR, exist_ok=True)

TARGETS = [
    "Melt(K)", "Boil(K)",
    "rho_a", "rho_b",
    "mu1_a", "mu1_b",
    "mu2_a", "mu2_b", "mu2_c",
    "k_a", "k_b",
    "cp_a", "cp_b", "cp_c",
]

DERIVED_PROPS = [
    ("rho", ["rho_a", "rho_b"]),
    ("muA", ["mu1_a", "mu1_b"]),
    ("muB", ["mu2_a", "mu2_b", "mu2_c"]),
    ("k",   ["k_a", "k_b"]),
    ("cp",  ["cp_a", "cp_b", "cp_c"]),
]

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _rel_mse_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Relative MSE as % of ⟨y²⟩."""
    mse = mean_squared_error(y_true, y_pred)
    denom = np.mean(y_true ** 2)
    if not np.isfinite(denom) or denom == 0.0:
        denom = 1e-12
    return 100.0 * mse / denom


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _p90_rel_err(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rel = np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-12)
    return float(np.percentile(rel, 90))


# ---------------------------------------------------------------------
# Hybrid tier-based imputation (TRAIN SET ONLY)
# ---------------------------------------------------------------------

def ot_impute_column(
    X_train: np.ndarray,
    y_train: np.ndarray,
    col_idx: int,
    col_name: str,
    reg: float = 0.1,
) -> None:
    """
    In-place OT imputation on y_train[:, col_idx] using feature space X_train.

    - Only modifies TRAIN column; caller is responsible for slicing.
    - Missing values (NaN) in y_train[:, col_idx] are imputed.
    """
    y_col = y_train[:, col_idx]
    mask_obs = np.isfinite(y_col)
    mask_miss = ~mask_obs

    n_obs = int(mask_obs.sum())
    n_miss = int(mask_miss.sum())
    if n_obs == 0 or n_miss == 0:
        return

    print(f" • {col_name:<7s}: OT (obs={n_obs}, missing={n_miss})")

    Xo = X_train[mask_obs]
    Xm = X_train[mask_miss]
    yo = y_col[mask_obs]

    # Cost matrix (squared Euclidean)
    M = ot.dist(Xm, Xo, metric="euclidean") ** 2
    a = np.ones(Xm.shape[0]) / Xm.shape[0]
    b = np.ones(Xo.shape[0]) / Xo.shape[0]

    # Earth Mover's Distance transport plan
    G = ot.emd(a, b, M)
    y_imp = G @ yo
    y_col[mask_miss] = y_imp
    y_train[:, col_idx] = y_col


def hybrid_impute_tiers_train_only(
    df: pd.DataFrame,
    present_targets: List[str],
    X_features: np.ndarray,
    tr_idx: np.ndarray,
    tier1_ot_props: List[str],
    tier2_props: List[str],
) -> None:
    """
    Apply hybrid tier-based imputation ONLY on TRAINING ROWS:

    - Tier1 OT: only for tier1_ot_props (here: ['mu1_a', 'mu1_b'])
    - KNN smoothing on Tier1 + Tier2 columns
    - IterativeImputer refinement on Tier2

    df is modified in-place.
    """
    print("\n=== Hybrid Tier-based Imputation on training set ===")

    # Build training label matrix
    Y_train = df.loc[tr_idx, present_targets].to_numpy(float)

    # Feature matrix for OT/KNN (same rows as tr_idx)
    X_tr = X_features[tr_idx]

    # Map props to column indices in present_targets
    name_to_idx = {name: j for j, name in enumerate(present_targets)}

    # -----------------------------
    # Tier 1 OT (ONLY mu1_a / mu1_b)
    # -----------------------------
    tier1_ot_cols = [p for p in tier1_ot_props if p in name_to_idx]
    if tier1_ot_cols:
        print(f"Tier 1 OT props: {tier1_ot_cols}")
        for prop in tier1_ot_cols:
            j = name_to_idx[prop]
            ot_impute_column(X_tr, Y_train, j, prop)
    else:
        print("No Tier 1 OT props (check names).")

    # -----------------------------
    # KNN smoothing on Tier1+Tier2
    # -----------------------------
    tier_knn_cols = list(
        {p for p in tier1_ot_props + tier2_props if p in name_to_idx}
    )
    tier_knn_indices = [name_to_idx[p] for p in tier_knn_cols]

    if tier_knn_indices:
        print(
            f"\nKNN smoothing on Tier 1+2 columns "
            f"(indices {tier_knn_indices})"
        )
        knn = KNNImputer(n_neighbors=5, weights="distance")
        # We only KNN-impute the columns in tier_knn_indices;
        # others stay unchanged.
        Y_sub = Y_train[:, tier_knn_indices]
        Y_sub_imp = knn.fit_transform(Y_sub)
        Y_train[:, tier_knn_indices] = Y_sub_imp
    else:
        print("\nNo columns for KNN smoothing.")

    # -----------------------------
    # IterativeImputer on Tier 2
    # -----------------------------
    tier2_cols = [p for p in tier2_props if p in name_to_idx]
    if tier2_cols:
        print(
            f"\nIterativeImputer refinement for Tier 2 props: "
            f"{tier2_cols}"
        )
        it = IterativeImputer(
            random_state=SEED,
            max_iter=20,
            initial_strategy="mean",
        )
        idx2 = [name_to_idx[p] for p in tier2_cols]
        Y_sub = Y_train[:, idx2]
        Y_sub_imp = it.fit_transform(Y_sub)
        Y_train[:, idx2] = Y_sub_imp
    else:
        print("\nNo Tier 2 props for IterativeImputer.")

    # -----------------------------
    # Write imputed training labels back into df
    # -----------------------------
    df.loc[tr_idx, present_targets] = Y_train
    print("=== Hybrid imputation finished ===\n")


# ---------------------------------------------------------------------
# ResNet + Meta
# ---------------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, p_drop: float = 0.2):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.lin1(x))
        h = self.drop(h)
        h = self.lin2(h)
        return self.act(x + h)


class BaseNet(nn.Module):
    def __init__(self, d_in: int, hidden: int = 64, depth: int = 3):
        super().__init__()
        layers = [nn.Linear(d_in, hidden), nn.SiLU()]
        for _ in range(depth):
            layers.append(ResidualBlock(hidden))
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MetaNet(nn.Module):
    def __init__(self, n_props: int, hidden: int = 128, depth: int = 2):
        super().__init__()
        layers = [nn.Linear(n_props, hidden), nn.SiLU()]
        for _ in range(depth):
            layers.append(ResidualBlock(hidden))
        layers.append(nn.Linear(hidden, n_props))
        self.net = nn.Sequential(*layers)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        return self.net(p)


class ResNetMetaTrainerHybrid:
    def __init__(
        self,
        df: pd.DataFrame,
        target_columns: List[str],
        derived_props: List[Tuple[str, List[str]]],
        degree_poly: int = 3,
        embedding_method: str = "none",
        n_components: int = 10,
        apply_hybrid_impute: bool = True,
    ):
        self.df = df.copy()
        self.target_columns = target_columns
        self.derived_props = derived_props
        self.model_dir = Path("best_models/resnet_hybrid")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.device = DEVICE

        # -----------------------------
        # Clean / detect present targets
        # -----------------------------
        self.present_targets: List[str] = []
        for t in target_columns:
            if t in self.df.columns:
                self.df[t] = (
                    self.df[t]
                    .replace(["----", ""], np.nan)
                    .replace(r"\*", "", regex=True)
                )
                self.df[t] = pd.to_numeric(self.df[t], errors="coerce")
                if np.isfinite(self.df[t]).any():
                    self.present_targets.append(t)

        if not self.present_targets:
            raise RuntimeError("No valid target columns found after cleaning.")

        # -----------------------------
        # Composition → features
        # -----------------------------
        self.df["Composition"] = self.df.apply(self.row_composition, axis=1)
        self.X_comp = pd.json_normalize(self.df["Composition"]).fillna(0.0)
        self.X_comp = self.X_comp.reindex(
            sorted(self.X_comp.columns), axis=1
        )

        self.poly = PolynomialFeatures(degree_poly, include_bias=False)
        X_poly = self.poly.fit_transform(self.X_comp)
        self.scaler = StandardScaler()
        X_poly = self.scaler.fit_transform(X_poly).astype(np.float32)

        frac = self.X_comp.to_numpy(np.float32)
        self.fractions = frac
        self.X = np.hstack([X_poly, frac]).astype(np.float32)
        self.feat_dim = self.X.shape[1]

        # -----------------------------
        # Train/val/test split
        # -----------------------------
        self.idx_all = np.arange(len(self.X))
        tr_idx, te_idx = train_test_split(
            self.idx_all, test_size=0.20, random_state=SEED
        )
        tr_idx, va_idx = train_test_split(
            tr_idx, test_size=0.20, random_state=SEED
        )
        self.tr_idx = tr_idx
        self.va_idx = va_idx
        self.te_idx = te_idx

        # -----------------------------
        # Hybrid imputation on TRAIN labels only
        # -----------------------------
        if apply_hybrid_impute:
            # Tier definitions:
            #   Tier 1 OT: ONLY mu1_a / mu1_b
            #   Tier 2: medium-sparse: Melt(K), Boil(K), cp_a
            tier1_ot_props = ["mu1_a", "mu1_b"]
            tier2_props = ["Melt(K)", "Boil(K)", "cp_a"]

            hybrid_impute_tiers_train_only(
                self.df,
                self.present_targets,
                self.X,
                self.tr_idx,
                tier1_ot_props=tier1_ot_props,
                tier2_props=tier2_props,
            )

        # -----------------------------
        # Masks & raw targets
        # -----------------------------
        self.mask_all = np.isfinite(
            self.df[self.present_targets]
        ).to_numpy(bool)
        self.y_raw = self.df[self.present_targets].to_numpy(np.float32)

        # -----------------------------
        # Embedding block
        # -----------------------------
        self.embedding_method = embedding_method
        self.n_components = n_components
        self.embedder = EmbeddingPreconditioner(
            method=embedding_method,
            n_components=n_components,
        )

        self.embedder.fit(self.X[self.tr_idx])
        X_emb = self.embedder.transform(self.X)
        self.X_embedded = X_emb
        self.feat_dim = (
            self.n_components if embedding_method != "none" else self.X.shape[1]
        )

        # -----------------------------
        # Normalise targets
        # -----------------------------
        self.μ = self.y_raw[self.tr_idx].mean(0)
        self.σ = self.y_raw[self.tr_idx].std(0)
        self.σ[self.σ == 0] = 1.0
        self.y_std = (self.y_raw - self.μ) / self.σ

        # -----------------------------
        # Models
        # -----------------------------
        self.idx_map = {n: j for j, n in enumerate(self.present_targets)}
        self.base_nets = nn.ModuleDict(
            {n: BaseNet(self.feat_dim).to(self.device)
             for n in self.present_targets}
        )
        self.meta = MetaNet(len(self.present_targets)).to(self.device)

    # -----------------------------
    # Chemistry helpers
    # -----------------------------

    def row_composition(self, row: pd.Series) -> Dict[str, float]:
        comps = str(row["System"]).split("-")
        mf = str(row["Mol Frac"]).strip()
        if mf.lower() == "pure salt":
            fracs = [1.0] * len(comps)
        else:
            parts = [p for p in mf.split("-") if p.strip() != ""]
            if len(parts) == len(comps):
                fracs = list(map(float, parts))
            elif len(parts) == 1 and len(comps) == 2:
                v = float(parts[0])
                fracs = [v, 1.0 - v]
            else:
                fracs = [1.0 / len(comps)] * len(comps)
        total: Dict[str, float] = {}
        for cmpd, f in zip(comps, fracs):
            for el, cnt in re.findall(r"([A-Z][a-z]*)(\d*)", cmpd):
                total[el] = total.get(el, 0.0) + f * int(cnt or "1")
        s = sum(total.values()) or 1.0
        return {el: v / s for el, v in total.items()}

    def make_loader(self, x, y, m, bs, shuf):
        ds = TensorDataset(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(m, dtype=torch.bool),
        )
        return DataLoader(ds, batch_size=bs, shuffle=shuf, drop_last=False)

    # -----------------------------
    # Stage 1: base nets
    # -----------------------------

    def train_base(self):
        for prop in self.present_targets:
            print(f" • Training base net for {prop}")
            net = self.base_nets[prop]
            j = self.idx_map[prop]

            mask_prop = self.mask_all[:, j]
            mask_tr = mask_prop & np.isin(self.idx_all, self.tr_idx)
            mask_va = mask_prop & np.isin(self.idx_all, self.va_idx)

            # if no val data, split available prop indices
            if mask_va.sum() == 0:
                idx_prop = np.where(mask_prop)[0]
                if len(idx_prop) >= 2:
                    tr_prop, va_prop = train_test_split(
                        idx_prop, test_size=0.20, random_state=SEED
                    )
                    mask_tr = np.isin(self.idx_all, tr_prop)
                    mask_va = np.isin(self.idx_all, va_prop)
                else:
                    mask_tr = np.isin(self.idx_all, idx_prop)
                    mask_va = np.zeros_like(mask_tr, dtype=bool)

            x_tr, y_tr = (
                self.X_embedded[mask_tr],
                self.y_std[mask_tr, j],
            )
            x_va, y_va = (
                self.X_embedded[mask_va],
                self.y_std[mask_va, j],
            )

            tr_loader = DataLoader(
                TensorDataset(
                    torch.tensor(x_tr, dtype=torch.float32),
                    torch.tensor(y_tr, dtype=torch.float32),
                ),
                batch_size=64,
                shuffle=True,
            )
            va_loader = (
                DataLoader(
                    TensorDataset(
                        torch.tensor(x_va, dtype=torch.float32),
                        torch.tensor(y_va, dtype=torch.float32),
                    ),
                    batch_size=256,
                    shuffle=False,
                )
                if len(x_va) > 0
                else None
            )

            opt = torch.optim.AdamW(
                net.parameters(), lr=2e-3, weight_decay=1e-4
            )
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=200, eta_min=2e-4
            )

            best = 1e9
            patience = 0
            PAT = 25
            model_path = self.model_dir / f"base_{prop}_resnet.pth"

            for epoch in range(300):
                net.train()
                for xb, yb in tr_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    opt.zero_grad()
                    loss = nn.functional.mse_loss(net(xb), yb)
                    loss.backward()
                    opt.step()
                sched.step()

                if va_loader is not None:
                    net.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for xb, yb in va_loader:
                            xb, yb = xb.to(self.device), yb.to(self.device)
                            val_loss += nn.functional.mse_loss(
                                net(xb), yb
                            ).item()
                        val_loss /= len(va_loader)

                    if val_loss < best - 1e-4:
                        best, patience = val_loss, 0
                        torch.save(net.state_dict(), model_path)
                    else:
                        patience += 1
                        if patience >= PAT:
                            print(f" ⇢ Early stopping for {prop}")
                            break

            if va_loader is not None and model_path.exists():
                net.load_state_dict(torch.load(model_path))

    # -----------------------------
    # Stage 2: meta net with clamped physics loss
    # -----------------------------

    def train_meta(self):
        for net in self.base_nets.values():
            for p in net.parameters():
                p.requires_grad_(False)

        def base_preds_tensor(xb: torch.Tensor) -> torch.Tensor:
            return torch.stack(
                [self.base_nets[p](xb) for p in self.present_targets], dim=1
            )

        PHYSICS_WEIGHT = 0.05
        TEMP_RANGE = (500.0, 1200.0)

        trL = self.make_loader(
            self.X_embedded[self.tr_idx],
            self.y_std[self.tr_idx],
            self.mask_all[self.tr_idx],
            bs=64,
            shuf=True,
        )
        vaL = self.make_loader(
            self.X_embedded[self.va_idx],
            self.y_std[self.va_idx],
            self.mask_all[self.va_idx],
            bs=256,
            shuf=False,
        )

        opt = torch.optim.AdamW(
            self.meta.parameters(), lr=1e-3, weight_decay=1e-4
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=400, eta_min=1e-4
        )

        best = 1e9
        wait = 0
        PAT = 40
        meta_path = self.model_dir / "meta_resnet.pth"

        μ_tensor = torch.tensor(self.μ, device=self.device, dtype=torch.float32)
        σ_tensor = torch.tensor(self.σ, device=self.device, dtype=torch.float32)

        def physics_loss_clamped(
            pred_raw: torch.Tensor,
            yb_raw: torch.Tensor,
            mb: torch.Tensor,
            T: torch.Tensor,
        ) -> torch.Tensor:
            """
            Physics loss with clamping to prevent exp/log overflow.
            """
            loss = 0.0
            valid_terms = 0

            # Clamp coefficients to avoid insane values from imputation or training
            coeff_min, coeff_max = -1e3, 1e3
            pred_raw = torch.clamp(pred_raw, coeff_min, coeff_max)
            yb_raw = torch.clamp(yb_raw, coeff_min, coeff_max)

            for dprop, req_coeffs in self.derived_props:
                idxs = [self.idx_map[c] for c in req_coeffs if c in self.idx_map]
                if len(idxs) != len(req_coeffs):
                    continue

                mask = torch.all(mb[:, idxs], dim=1)
                if not mask.any():
                    continue

                y_coeffs = yb_raw[mask][:, idxs]
                p_coeffs = pred_raw[mask][:, idxs]
                Tm = T[mask]

                # Extra safety clamps
                Tm = torch.clamp(Tm, 200.0, 3000.0)

                if dprop == "rho":
                    y_vals = y_coeffs[:, 0] - y_coeffs[:, 1] * Tm
                    p_vals = p_coeffs[:, 0] - p_coeffs[:, 1] * Tm
                    term_loss = nn.functional.mse_loss(p_vals, y_vals)
                elif dprop == "muA":
                    p_mu1_a = torch.clamp(p_coeffs[:, 0], min=1e-6, max=1e6)
                    expo_p = torch.clamp(
                        p_coeffs[:, 1] / (R * Tm), min=-50.0, max=50.0
                    )
                    expo_y = torch.clamp(
                        y_coeffs[:, 1] / (R * Tm), min=-50.0, max=50.0
                    )
                    p_vals = p_mu1_a * torch.exp(expo_p)
                    y_vals = torch.clamp(y_coeffs[:, 0], 1e-6, 1e6) * torch.exp(
                        expo_y
                    )
                    p_vals = torch.clamp(p_vals, 1e-12, 1e12)
                    y_vals = torch.clamp(y_vals, 1e-12, 1e12)
                    term_loss = nn.functional.mse_loss(
                        torch.log(p_vals), torch.log(y_vals)
                    )
                elif dprop == "muB":
                    invT = torch.clamp(1.0 / Tm, 1e-4, 1e-1)
                    invT2 = invT**2
                    y_log = (
                        y_coeffs[:, 0]
                        + y_coeffs[:, 1] * invT
                        + y_coeffs[:, 2] * invT2
                    )
                    p_log = (
                        p_coeffs[:, 0]
                        + p_coeffs[:, 1] * invT
                        + p_coeffs[:, 2] * invT2
                    )
                    y_log = torch.clamp(y_log, -1e3, 1e3)
                    p_log = torch.clamp(p_log, -1e3, 1e3)
                    term_loss = nn.functional.mse_loss(p_log, y_log)
                elif dprop == "k":
                    y_vals = y_coeffs[:, 0] + y_coeffs[:, 1] * Tm
                    p_vals = p_coeffs[:, 0] + p_coeffs[:, 1] * Tm
                    y_vals = torch.clamp(y_vals, -1e6, 1e6)
                    p_vals = torch.clamp(p_vals, -1e6, 1e6)
                    term_loss = nn.functional.mse_loss(p_vals, y_vals)
                elif dprop == "cp":
                    invT2 = torch.clamp(1.0 / (Tm**2), 0.0, 1e-2)
                    y_vals = (
                        y_coeffs[:, 0]
                        + y_coeffs[:, 1] * Tm
                        + y_coeffs[:, 2] * invT2
                    )
                    p_vals = (
                        p_coeffs[:, 0]
                        + p_coeffs[:, 1] * Tm
                        + p_coeffs[:, 2] * invT2
                    )
                    y_vals = torch.clamp(y_vals, -1e6, 1e6)
                    p_vals = torch.clamp(p_vals, -1e6, 1e6)
                    term_loss = nn.functional.mse_loss(p_vals, y_vals)
                else:
                    continue

                loss = loss + term_loss
                valid_terms += 1

            if valid_terms == 0:
                return torch.tensor(0.0, device=self.device)
            return loss / valid_terms

        print("\nStage-2: Training meta net with physics regularization...")
        self.meta_train_loss = []
        self.meta_val_loss = []

        for epoch in range(600):
            self.meta.train()
            total_loss = 0.0

            for xb, yb, mb in trL:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                mb = mb.to(self.device)

                batch_size = xb.size(0)
                T = torch.rand(batch_size, device=self.device)
                T = T * (TEMP_RANGE[1] - TEMP_RANGE[0]) + TEMP_RANGE[0]

                with torch.no_grad():
                    base_out = base_preds_tensor(xb)

                pred = base_out + self.meta(base_out)

                # coefficient MSE (masked)
                loss_coeff = ((pred - yb) ** 2 * mb).sum() / mb.sum()

                # physics loss on de-standardised coefficients
                pred_raw = pred * σ_tensor + μ_tensor
                yb_raw = yb * σ_tensor + μ_tensor
                loss_phys = physics_loss_clamped(
                    pred_raw, yb_raw, mb, T
                ) * PHYSICS_WEIGHT

                loss = loss_coeff + loss_phys
                loss.backward()
                nn.utils.clip_grad_norm_(self.meta.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                total_loss += loss.item()

            sched.step()
            avg_loss = total_loss / len(trL)
            self.meta_train_loss.append(avg_loss)

            # validation
            self.meta.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb, mb in vaL:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    mb = mb.to(self.device)

                    base_out = base_preds_tensor(xb)
                    pred = base_out + self.meta(base_out)
                    val_loss += ((pred - yb) ** 2 * mb).sum().item() / mb.sum().item()
            val_loss /= len(vaL)
            self.meta_val_loss.append(val_loss)

            print(f"Epoch {epoch:3d} | Train: {avg_loss:.4f} | Val: {val_loss:.4f}")

            if val_loss < best - 1e-4 and np.isfinite(val_loss):
                best, wait = val_loss, 0
                torch.save(self.meta.state_dict(), meta_path)
            else:
                wait += 1
                if wait >= PAT:
                    print(" ⇢ Early stopping")
                    break

        if meta_path.exists():
            self.meta.load_state_dict(torch.load(meta_path))

    # -----------------------------
    # Evaluation
    # -----------------------------

    def evaluate_split(self, split: str = "val", min_n: int = 5) -> Dict:
        """
        Per-target metrics on val or test split.
        Returns dict with avg stats + per-target metrics.
        """
        if split == "val":
            idx = self.va_idx
        elif split == "test":
            idx = self.te_idx
        else:
            raise ValueError("split must be 'val' or 'test'")

        self.meta.eval()
        for net in self.base_nets.values():
            net.eval()

        μ, σ = self.μ, self.σ
        Xs = self.X_embedded[idx]
        ys = self.y_raw[idx]
        ms = self.mask_all[idx]

        with torch.no_grad():
            xb = torch.tensor(Xs, dtype=torch.float32, device=self.device)
            base_out = torch.stack(
                [self.base_nets[p](xb).cpu()
                 for p in self.present_targets],
                dim=1,
            ).numpy()
            pred_std = base_out + self.meta(
                torch.tensor(base_out, dtype=torch.float32, device=self.device)
            ).cpu().numpy()

        pred = pred_std * σ + μ

        per_target: Dict[str, Dict] = {}
        rel_mses, r2s, maes, p90s = [], [], [], []

        print(f"\n{split.capitalize()} results — relMSE (%), R², MAE, p90_rel")
        for j, prop in enumerate(self.present_targets):
            mask_j = ms[:, j]
            n_j = int(mask_j.sum())
            if n_j < min_n:
                print(f" • {prop:<8s}: [skipped: only {n_j} samples]")
                continue

            yt = ys[mask_j, j]
            yp = pred[mask_j, j]

            m_rel = _rel_mse_pct(yt, yp)
            ss_res = np.sum((yt - yp) ** 2)
            ss_tot = np.sum((yt - yt.mean()) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            mae = _mae(yt, yp)
            p90 = _p90_rel_err(yt, yp)

            per_target[prop] = {
                "relMSE_pct": float(m_rel),
                "R2": float(r2),
                "MAE": float(mae),
                "p90_rel_err": float(p90),
                "n": n_j,
            }

            print(
                f" • {prop:<8s}: {m_rel:8.2f}%   R²={r2:+.3f}   "
                f"MAE={mae:9.3g}   p90={p90:6.3f}   (n={n_j})"
            )

            rel_mses.append(m_rel)
            r2s.append(r2)
            maes.append(mae)
            p90s.append(p90)

        if rel_mses:
            return {
                "avg_relMSE_pct": float(np.mean(rel_mses)),
                "avg_R2": float(np.mean(r2s)),
                "avg_MAE": float(np.mean(maes)),
                "avg_p90_rel_err": float(np.mean(p90s)),
                "per_target": per_target,
            }
        else:
            return {
                "avg_relMSE_pct": float("nan"),
                "avg_R2": float("nan"),
                "avg_MAE": float("nan"),
                "avg_p90_rel_err": float("nan"),
                "per_target": per_target,
            }


# ---------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------

def save_plot(filename: str):
    plt.tight_layout()
    path = os.path.join(OUTDIR, filename)
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


def plot_r2_bar(val_metrics: Dict, test_metrics: Dict):
    val_pt = val_metrics["per_target"]
    test_pt = test_metrics["per_target"]

    targets = sorted(set(val_pt.keys()) & set(test_pt.keys()))
    if not targets:
        return

    val_r2 = [val_pt[t]["R2"] for t in targets]
    test_r2 = [test_pt[t]["R2"] for t in targets]

    x = np.arange(len(targets))
    w = 0.35

    plt.figure(figsize=(12, 5))
    plt.bar(x - w / 2, val_r2, w, label="Val R²")
    plt.bar(x + w / 2, test_r2, w, label="Test R²")
    plt.axhline(0, color="black")

    plt.xticks(x, targets, rotation=45, ha="right")
    plt.ylabel("R²")
    plt.title("Per-target R² (val vs test)")
    plt.legend()
    save_plot("r2_bar.png")


def plot_mae_bar(val_metrics: Dict, test_metrics: Dict):
    val_pt = val_metrics["per_target"]
    test_pt = test_metrics["per_target"]

    targets = sorted(set(val_pt.keys()) & set(test_pt.keys()))
    if not targets:
        return

    val_mae = np.array([val_pt[t]["MAE"] for t in targets], dtype=float)
    test_mae = np.array([test_pt[t]["MAE"] for t in targets], dtype=float)

    x = np.arange(len(targets))
    w = 0.35

    plt.figure(figsize=(12, 5))
    plt.bar(x - w / 2, val_mae, w, label="Val MAE")
    plt.bar(x + w / 2, test_mae, w, label="Test MAE")

    plt.xticks(x, targets, rotation=45, ha="right")
    plt.ylabel("MAE (log scale)")
    plt.title("Per-target MAE")
    plt.yscale("log")
    plt.legend()
    save_plot("mae_bar.png")


def plot_all_true_vs_pred(
    trainer: ResNetMetaTrainerHybrid, split: str = "test", min_n: int = 5
):
    if split == "test":
        idx = trainer.te_idx
    elif split == "val":
        idx = trainer.va_idx
    else:
        raise ValueError("split must be 'val' or 'test'")

    trainer.meta.eval()
    for net in trainer.base_nets.values():
        net.eval()

    Xs = trainer.X_embedded[idx]
    ys_all = trainer.y_raw[idx]
    ms_all = trainer.mask_all[idx]
    μ, σ = trainer.μ, trainer.σ

    with torch.no_grad():
        xb = torch.tensor(Xs, dtype=torch.float32, device=trainer.device)
        base_out = torch.stack(
            [trainer.base_nets[p](xb) for p in trainer.present_targets],
            dim=1,
        )
        pred_std = base_out + trainer.meta(base_out)
        pred_all = pred_std.cpu().numpy() * σ + μ

    targets = trainer.present_targets
    n_targets = len(targets)
    n_cols = 3
    n_rows = int(np.ceil(n_targets / n_cols))

    plt.figure(figsize=(5 * n_cols, 4 * n_rows))
    plot_idx = 1

    for j, prop in enumerate(targets):
        mask_j = ms_all[:, j]
        n_j = int(mask_j.sum())
        if n_j < min_n:
            continue

        yt = ys_all[mask_j, j]
        yp = pred_all[mask_j, j]

        plt.subplot(n_rows, n_cols, plot_idx)
        plt.scatter(yt, yp, alpha=0.5, s=10)
        lo = min(yt.min(), yp.min())
        hi = max(yt.max(), yp.max())
        plt.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        plt.title(prop)
        plt.xlabel("True")
        plt.ylabel("Pred")
        plot_idx += 1

    plt.suptitle(f"True vs predicted ({split} set)", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_plot(f"true_vs_pred_all_{split}.png")


def plot_learning_curve(trainer: ResNetMetaTrainerHybrid):
    if not hasattr(trainer, "meta_train_loss") or not hasattr(
        trainer, "meta_val_loss"
    ):
        print("No meta_train_loss/meta_val_loss; skipping learning curve.")
        return

    train_losses = trainer.meta_train_loss
    val_losses = trainer.meta_val_loss
    epochs = np.arange(1, len(train_losses) + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Meta network learning curve")
    plt.legend()
    save_plot("learning_curve_meta.png")


def plot_cv_r2_boxplot(cv_results: List[Dict]):
    r2_per_target: Dict[str, List[float]] = {}

    for fold_res in cv_results:
        test = fold_res.get("test", {})
        pt = test.get("per_target", {})
        for t, m in pt.items():
            r2_per_target.setdefault(t, []).append(m["R2"])

    if not r2_per_target:
        print("cv_results has no per-target R²; skipping CV boxplot.")
        return

    targets = sorted(r2_per_target.keys())
    data = [r2_per_target[t] for t in targets]

    plt.figure(figsize=(12, 5))
    plt.boxplot(data, labels=targets, showmeans=True)
    plt.axhline(0, color="black")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("R² (test across folds)")
    plt.title("Cross-validation R² stability per target")
    save_plot("cv_r2_boxplot.png")


# ---------------------------------------------------------------------
# Cross-validation wrapper
# ---------------------------------------------------------------------

def cross_validate_resnet_hybrid(
    df: pd.DataFrame,
    targets: List[str],
    derived_props: List[Tuple[str, List[str]]],
    k: int = 3,
) -> List[Dict]:
    kf = KFold(
        n_splits=k,
        shuffle=True,
        random_state=SEED,
    )

    idx_all = np.arange(len(df))
    results: List[Dict] = []

    print(f"\nRunning {k}-fold cross-validation (hybrid-impute ResNet)...")

    fold = 0
    for train_idx, test_idx in kf.split(idx_all):
        fold += 1
        print(f"\n========== Fold {fold}/{k} ==========")

        df_fold = df.copy().reset_index(drop=True)
        # For each fold, the trainer will do its own train/val split
        # inside the train_idx. We just mask the test_idx for evaluation.
        trainer = ResNetMetaTrainerHybrid(
            df_fold, targets, derived_props, apply_hybrid_impute=True
        )
        trainer.te_idx = test_idx  # override global test to fold test

        trainer.train_base()
        trainer.train_meta()

        val_metrics = trainer.evaluate_split(split="val")
        test_metrics = trainer.evaluate_split(split="test")

        results.append({"val": val_metrics, "test": test_metrics})

    return results


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    # Adjust path to your CSV
    csv_path = "/Users/meggie/Documents/MoltenSaltPropnet/data/mstdb_processed.csv"
    df = pd.read_csv(csv_path).rename(columns=str.strip)

    trainer = ResNetMetaTrainerHybrid(df, TARGETS, DERIVED_PROPS, apply_hybrid_impute=True)
    print(f"Using {len(trainer.present_targets)} properties:", ", ".join(trainer.present_targets))

    trainer.train_base()
    trainer.train_meta()

    val_metrics = trainer.evaluate_split(split="val")
    test_metrics = trainer.evaluate_split(split="test")

    # Save metrics JSON
    with open(os.path.join(OUTDIR, "val_test_metrics.json"), "w") as f:
        json.dump({"val": val_metrics, "test": test_metrics}, f, indent=4)

    # Plots
    plot_learning_curve(trainer)
    plot_r2_bar(val_metrics, test_metrics)
    plot_mae_bar(val_metrics, test_metrics)
    plot_all_true_vs_pred(trainer, split="test")

    # 3-fold CV
    cv_results = cross_validate_resnet_hybrid(df, TARGETS, DERIVED_PROPS, k=3)
    with open(os.path.join(OUTDIR, "cv_results.json"), "w") as f:
        json.dump(cv_results, f, indent=4)
    plot_cv_r2_boxplot(cv_results)

    print("\nAll results saved under:", OUTDIR)


if __name__ == "__main__":
    main()
