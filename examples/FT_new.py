"""
All-in-one Transformer Meta Trainer + Training + Evaluation + Visualisation

Version: no EmbeddingPreconditioner, uses polynomial + scaler only.

- Sequence Feature Transformer (one token per feature)
- Group towers (one tower per derived-property group)
- Meta-transformer across properties
- Physics-informed regularisation
- Masked multi-task loss for missing targets
- Training: train_base(), train_meta(), train_joint()
- Evaluation: per-target R² and relative MSE (%)
- Builds df_pa (index, target, actual, predicted, abs_error, percent_error)
- Plotting:
    * Actual vs Predicted
    * Residuals
    * Error distributions
    * Confusion-style heatmaps
    * All-target grid
    * Error-bin confusion matrix
    * Error by System
- Permutation feature importance per target
- Plots saved in: visualisation/transformer_prediction_plots/

Data path (adjust if needed):
    /Users/meggie/Documents/MoltenSaltPropnet/data/new_mstdb_janz.csv
"""

import os
import re
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns


SEED = 42
R = 8.314 
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

TARGETS = [
    "Melt(K)", "Boil(K)",
    "rho_a", "rho_b",
    "mu1_a", "mu1_b",
    "mu2_a", "mu2_b", "mu2_c",
    "k_a", "k_b",
    "cp_a", "cp_b", "cp_c"
]

DERIVED_PROPS = [
    ('rho', ['rho_a', 'rho_b']),
    ('muA', ['mu1_a', 'mu1_b']),
    ('muB', ['mu2_a', 'mu2_b', 'mu2_c']),
    ('k',   ['k_a',   'k_b']),
    ('cp',  ['cp_a',  'cp_b', 'cp_c'])
]

PLOT_DIR = "visualisation/transformer_prediction_plots"
os.makedirs(PLOT_DIR, exist_ok=True)


def _savefig(name: str):
    path = os.path.join(PLOT_DIR, name)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")



#  Metrics


def _rel_mse_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return relative MSE as a percentage of ⟨y²⟩ — unitless."""
    mse = mean_squared_error(y_true, y_pred)
    denom = np.mean(y_true ** 2) or 1e-12
    return 100.0 * mse / denom

#  Sequence Transformer over features


class SeqFeatureTransformer(nn.Module):
    """
    Treat each feature as a token:
        token_i = value_embed(x_i) + feature_embed(i)
    Then apply TransformerEncoder and mean-pool over tokens.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 8,
        num_layers: int = 4,
        dim_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        self.value_embed = nn.Linear(1, d_model)
        self.feature_embed = nn.Embedding(n_features, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, F) float
        returns latent: (B, d_model)
        """
        B, F = x.shape
        feat_idx = torch.arange(F, device=x.device).unsqueeze(0).expand(B, F)
        val = x.unsqueeze(-1)  # (B, F, 1)

        tokens = self.value_embed(val) + self.feature_embed(feat_idx)  # (B, F, d)
        h = self.encoder(tokens)                                       # (B, F, d)
        h = self.norm(h)
        latent = h.mean(dim=1)                                         # (B, d)
        return latent


#  Group towers (one tower per group of coefficients)


class GroupTower(nn.Module):
    def __init__(self, d_in: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


#  Meta-transformer over properties


class PropertyMetaTransformer(nn.Module):
    """
    Treat each property prediction as a token:
        token_j = value_embed(p_j) + prop_embed(j)
    Run a small TransformerEncoder and project back to corrections Δp_j.
    """

    def __init__(
        self,
        n_props: int,
        d_model: int = 128,
        n_heads: int = 8,
        num_layers: int = 2,
        dim_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_props = n_props
        self.d_model = d_model

        self.value_embed = nn.Linear(1, d_model)
        self.prop_embed = nn.Embedding(n_props, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, base_preds: torch.Tensor) -> torch.Tensor:
        """
        base_preds: (B, Nprop)
        returns Δpred: (B, Nprop)
        """
        B, N = base_preds.shape
        prop_idx = torch.arange(N, device=base_preds.device).unsqueeze(0).expand(B, N)
        val = base_preds.unsqueeze(-1)  # (B, N, 1)

        tokens = self.value_embed(val) + self.prop_embed(prop_idx)  # (B, N, d)
        h = self.encoder(tokens)
        h = self.norm(h)
        delta = self.out_proj(h).squeeze(-1)  # (B, N)
        return delta


#  Main trainer


class TransformerMetaTrainer:
    """
    Transformer-based analogue of SNNMetaTrainer:
      - SeqFeatureTransformer encoder
      - Group towers for related coefficients
      - Meta-transformer over properties
      - Physics-based regularisation
      - Masked multi-task loss for missing targets
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_cols: List[str],
        derived_props: List[Tuple[str, List[str]]],
        degree_poly: int = 3,
        d_model: int = 128,
        tower_hidden: int = 128,
        meta_d_model: int = 128,
    ):
        self.df = df.copy()
        self.target_columns = target_cols
        self.derived_props = derived_props
        self.device = device
        self.model_dir = Path("data/trained_models_transformer")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # ── clean targets & detect available ones ────────────────
        self.present_targets: List[str] = []
        for t in target_cols:
            if t in self.df.columns:
                self.df[t] = (
                    self.df[t]
                    .replace(["----", ""], np.nan)
                    .replace(r"\*", "", regex=True)
                    .pipe(pd.to_numeric, errors="coerce")
                )
                if np.isfinite(self.df[t]).any():
                    self.present_targets.append(t)

        if not self.present_targets:
            raise RuntimeError("No valid target columns found in DataFrame.")

        # ── composition features ────────────────────────────────
        self.df["Composition"] = self.df.apply(self._row_composition, axis=1)
        self.X_comp = pd.json_normalize(self.df["Composition"]).fillna(0.0)
        self.X_comp = self.X_comp.reindex(sorted(self.X_comp.columns), axis=1)

        # Polynomial features + scaler
        self.poly = PolynomialFeatures(degree_poly, include_bias=False)
        X_poly = self.poly.fit_transform(self.X_comp)
        self.scaler = StandardScaler()
        X_poly = self.scaler.fit_transform(X_poly).astype(np.float32)
        frac = self.X_comp.to_numpy(np.float32)
        self.X = np.hstack([X_poly, frac]).astype(np.float32)
        self.X_embedded = self.X  # no embedding preconditioner
        self.feat_dim = self.X_embedded.shape[1]
        self.composition_df = self.X_comp

        # ── targets, masks, splits ─────────────────────────────
        mask_all = np.isfinite(self.df[self.present_targets]).to_numpy(bool)
        self.df[self.present_targets] = self.df[self.present_targets].fillna(0.0)
        y_raw = self.df[self.present_targets].to_numpy(np.float32)

        idx = np.arange(len(self.X_embedded))
        tr, te = train_test_split(idx, test_size=0.20, random_state=SEED)
        tr, va = train_test_split(tr, test_size=0.20, random_state=SEED)

        μ = y_raw[tr].mean(0)
        σ = y_raw[tr].std(0)
        σ[σ == 0] = 1.0

        self.mask_all = mask_all
        self.y_raw = y_raw
        self.y_std = (y_raw - μ) / σ
        self.μ, self.σ = μ, σ

        self.tr_idx, self.va_idx, self.te_idx = tr, va, te
        self.idx_map = {n: j for j, n in enumerate(self.present_targets)}

        #  build group definitions & towers
        group_defs: List[Tuple[str, List[str]]] = []
        coeff_in_any_group = set()
        for gname, coeffs in derived_props:
            valid = [c for c in coeffs if c in self.present_targets]
            if valid:
                group_defs.append((gname, valid))
                coeff_in_any_group.update(valid)

        for t in self.present_targets:
            if t not in coeff_in_any_group:
                group_defs.append((t, [t]))

        self.group_defs = group_defs  # list[(group_name, [coeffs])]

        # instantiate nets 
        self.encoder = SeqFeatureTransformer(
            n_features=self.feat_dim,
            d_model=d_model,
            n_heads=8,
            num_layers=4,
            dim_ff=4 * d_model,
            dropout=0.1,
        ).to(device)

        self.group_towers = nn.ModuleDict()
        for gname, coeffs in self.group_defs:
            self.group_towers[gname] = GroupTower(
                d_in=d_model,
                hidden=tower_hidden,
                out_dim=len(coeffs),
            ).to(device)

        self.meta = PropertyMetaTransformer(
            n_props=len(self.present_targets),
            d_model=meta_d_model,
            n_heads=8,
            num_layers=2,
            dim_ff=4 * meta_d_model,
            dropout=0.1,
        ).to(device)

    # helpers 

    @staticmethod
    def _row_composition(row) -> Dict[str, float]:
        comps = str(row["System"]).split("-")
        fracs_str = str(row.get("Mol Frac", "1")).strip()
        if isinstance(fracs_str, str) and fracs_str == "Pure Salt":
            fracs = [1.0] * len(comps)
        else:
            fracs = list(map(float, fracs_str.split("-")))
        total = {}
        for cmp, f in zip(comps, fracs):
            for el, cnt in re.findall(r"([A-Z][a-z]*)(\d*)", cmp):
                total[el] = total.get(el, 0) + int(cnt or "1") * f
        s = sum(total.values())
        return {el: cnt / s for el, cnt in total.items()} if s > 0 else {}

    @staticmethod
    def _loader(x, y, m, bs, shuf):
        ds = TensorDataset(torch.tensor(x), torch.tensor(y), torch.tensor(m))
        return DataLoader(ds, batch_size=bs, shuffle=shuf, drop_last=False)

    #  base forward

    def _base_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """
        latent: (B, d_model) → base predictions (B, Nprop) in std space.
        """
        B = latent.shape[0]
        base = torch.zeros(B, len(self.present_targets), device=latent.device)
        for gname, coeffs in self.group_defs:
            tower = self.group_towers[gname]
            out = tower(latent)  # (B, len(coeffs))
            for k, c in enumerate(coeffs):
                j = self.idx_map[c]
                base[:, j] = out[:, k]
        return base

    def _base_preds(self, xb: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(xb)
        return self._base_from_latent(latent)

    # physics loss 

    def _physics_loss(self, pred_raw, y_raw, mask_b, T):
        """
        pred_raw, y_raw: (B, Nprop) in physical units
        mask_b: (B, Nprop) boolean mask for observed targets
        T: (B,) temperature [K]
        """
        loss, terms = 0.0, 0
        for dprop, coeffs in self.derived_props:
            idxs = [self.idx_map[c] for c in coeffs if c in self.idx_map]
            if len(idxs) != len(coeffs):
                continue
            m = torch.all(mask_b[:, idxs], dim=1)
            if not m.any():
                continue

            y = y_raw[m][:, idxs]
            p = pred_raw[m][:, idxs]
            Tm = T[m]

            if dprop == "rho":
                loss_t = nn.functional.mse_loss(
                    p[:, 0] - p[:, 1] * Tm,
                    y[:, 0] - y[:, 1] * Tm,
                )
            elif dprop == "muA":
                loss_t = nn.functional.mse_loss(
                    torch.log(torch.clamp(p[:, 0], 1e-6) * torch.exp(p[:, 1] / (R * Tm))),
                    torch.log(y[:, 0] * torch.exp(y[:, 1] / (R * Tm))),
                )
            elif dprop == "muB":
                loss_t = nn.functional.mse_loss(
                    p[:, 0] + p[:, 1] / Tm + p[:, 2] / Tm**2,
                    y[:, 0] + y[:, 1] / Tm + y[:, 2] / Tm**2,
                )
            elif dprop == "k":
                loss_t = nn.functional.mse_loss(
                    p[:, 0] + p[:, 1] * Tm,
                    y[:, 0] + y[:, 1] * Tm,
                )
            elif dprop == "cp":
                loss_t = nn.functional.mse_loss(
                    p[:, 0] + p[:, 1] * Tm + p[:, 2] / Tm**2,
                    y[:, 0] + y[:, 1] * Tm + y[:, 2] / Tm**2,
                )
            else:
                continue

            loss += loss_t
            terms += 1

        if terms == 0:
            return torch.tensor(0.0, device=self.device)
        return loss / terms

    #  base training 

    def train_base(self, epochs: int = 300, physics_weight: float = 0.05):
        """
        Train encoder + group towers only (no meta-transformer).
        """
        trL = self._loader(
            self.X_embedded[self.tr_idx],
            self.y_std[self.tr_idx],
            self.mask_all[self.tr_idx],
            64,
            True,
        )
        vaL = self._loader(
            self.X_embedded[self.va_idx],
            self.y_std[self.va_idx],
            self.mask_all[self.va_idx],
            256,
            False,
        )

        params = list(self.encoder.parameters())
        for t in self.group_towers.values():
            params += list(t.parameters())

        opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)

        best, wait, PAT = 1e9, 0, 50
        μ_t = torch.tensor(self.μ, device=self.device, dtype=torch.float32)
        σ_t = torch.tensor(self.σ, device=self.device, dtype=torch.float32)

        print("\nStage-1: Training base Transformer + group towers …")
        for ep in range(epochs):
            self.encoder.train()
            for t in self.group_towers.values():
                t.train()

            tot = 0.0
            for xb, yb, mb in trL:
                xb, yb, mb = xb.to(self.device), yb.to(self.device), mb.to(self.device)

                latent = self.encoder(xb)
                base = self._base_from_latent(latent)  # std space

                # masked MSE
                loss_coeff = ((base - yb) ** 2 * mb).sum() / mb.sum()

                # physics 
                T = torch.rand(len(xb), device=self.device) * 700 + 500  # 500–1200 K
                base_raw = base * σ_t + μ_t
                yb_raw = yb * σ_t + μ_t
                loss_phys = self._physics_loss(base_raw, yb_raw, mb, T) * physics_weight

                loss = loss_coeff + loss_phys
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                tot += loss.item()

            sched.step()
            train_loss = tot / len(trL)

            # validation
            self.encoder.eval()
            for t in self.group_towers.values():
                t.eval()

            val = 0.0
            with torch.no_grad():
                for xb, yb, mb in vaL:
                    xb, yb, mb = xb.to(self.device), yb.to(self.device), mb.to(self.device)
                    base = self._base_preds(xb)
                    val += ((base - yb) ** 2 * mb).sum().item() / mb.sum().item()
            val /= len(vaL)

            print(f"Epoch {ep:3d} | train {train_loss:.4f} | val {val:.4f}")

            if val < best - 1e-4:
                best, wait = val, 0
                torch.save(self.encoder.state_dict(), self.model_dir / "encoder_base.pth")
                for gname, tower in self.group_towers.items():
                    torch.save(tower.state_dict(), self.model_dir / f"tower_{gname}.pth")
            else:
                wait += 1
                if wait >= PAT:
                    print(" ⇢ Early stopping base")
                    break

        # reload best
        self.encoder.load_state_dict(
            torch.load(self.model_dir / "encoder_base.pth", map_location=self.device)
        )
        for gname, tower in self.group_towers.items():
            tower.load_state_dict(
                torch.load(self.model_dir / f"tower_{gname}.pth", map_location=self.device)
            )

    #  meta training 

    def train_meta(self, epochs: int = 400, physics_weight: float = 0.1):
        """
        Train meta-transformer on top of frozen base encoder + towers.
        """

        # freeze base
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        for t in self.group_towers.values():
            for p in t.parameters():
                p.requires_grad_(False)

        trL = self._loader(
            self.X_embedded[self.tr_idx],
            self.y_std[self.tr_idx],
            self.mask_all[self.tr_idx],
            64,
            True,
        )
        vaL = self._loader(
            self.X_embedded[self.va_idx],
            self.y_std[self.va_idx],
            self.mask_all[self.va_idx],
            256,
            False,
        )

        opt = torch.optim.AdamW(self.meta.parameters(), lr=8e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)

        best, wait, PAT = 1e9, 0, 80
        μ_t = torch.tensor(self.μ, device=self.device, dtype=torch.float32)
        σ_t = torch.tensor(self.σ, device=self.device, dtype=torch.float32)

        print("\nStage-2: Training meta-transformer with physics regularisation …")
        for ep in range(epochs):
            self.meta.train()
            tot = 0.0

            for xb, yb, mb in trL:
                xb, yb, mb = xb.to(self.device), yb.to(self.device), mb.to(self.device)
                T = torch.rand(len(xb), device=self.device) * 700 + 500

                with torch.no_grad():
                    base = self._base_preds(xb)  # std space

                delta = self.meta(base)
                pred = base + delta

                loss_coeff = ((pred - yb) ** 2 * mb).sum() / mb.sum()

                pred_raw = pred * σ_t + μ_t
                yb_raw = yb * σ_t + μ_t
                loss_phys = self._physics_loss(pred_raw, yb_raw, mb, T) * physics_weight

                loss = loss_coeff + loss_phys
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.meta.parameters(), 0.5)
                opt.step()
                tot += loss.item()

            sched.step()
            train_loss = tot / len(trL)

            # validation
            self.meta.eval()
            val = 0.0
            with torch.no_grad():
                for xb, yb, mb in vaL:
                    xb, yb, mb = xb.to(self.device), yb.to(self.device), mb.to(self.device)
                    base = self._base_preds(xb)
                    pred = base + self.meta(base)
                    val += ((pred - yb) ** 2 * mb).sum().item() / mb.sum().item()
            val /= len(vaL)

            print(f"Epoch {ep:3d} | train {train_loss:.4f} | val {val:.4f}")

            if val < best - 1e-4:
                best, wait = val, 0
                torch.save(self.meta.state_dict(), self.model_dir / "meta_transformer.pth")
            else:
                wait += 1
                if wait >= PAT:
                    print(" ⇢ Early stopping meta")
                    break

        self.meta.load_state_dict(
            torch.load(self.model_dir / "meta_transformer.pth", map_location=self.device)
        )

    # joint training 

    def train_joint(self, epochs: int = 400, physics_weight: float = 0.1):
        """
        Jointly train encoder + towers + meta-transformer.
        """
        trL = self._loader(
            self.X_embedded[self.tr_idx],
            self.y_std[self.tr_idx],
            self.mask_all[self.tr_idx],
            64,
            True,
        )
        vaL = self._loader(
            self.X_embedded[self.va_idx],
            self.y_std[self.va_idx],
            self.mask_all[self.va_idx],
            256,
            False,
        )

        params = list(self.encoder.parameters())
        for t in self.group_towers.values():
            params += list(t.parameters())
        params += list(self.meta.parameters())

        opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)

        best, wait, PAT = 1e9, 0, 80
        μ_t = torch.tensor(self.μ, device=self.device, dtype=torch.float32)
        σ_t = torch.tensor(self.σ, device=self.device, dtype=torch.float32)
        TEMP_RANGE = (500, 1200)

        print("\nJoint Training: encoder + towers + meta-transformer …")
        for ep in range(epochs):
            self.encoder.train()
            for t in self.group_towers.values():
                t.train()
            self.meta.train()

            tot = 0.0
            for xb, yb, mb in trL:
                xb, yb, mb = xb.to(self.device), yb.to(self.device), mb.to(self.device)
                T = torch.rand(len(xb), device=self.device) * (TEMP_RANGE[1] - TEMP_RANGE[0]) + TEMP_RANGE[0]

                latent = self.encoder(xb)
                base = self._base_from_latent(latent)
                pred = base + self.meta(base)

                loss_coeff = ((pred - yb) ** 2 * mb).sum() / mb.sum()

                pred_raw = pred * σ_t + μ_t
                yb_raw = yb * σ_t + μ_t
                loss_phys = self._physics_loss(pred_raw, yb_raw, mb, T) * physics_weight

                loss = loss_coeff + loss_phys
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                tot += loss.item()

            sched.step()
            train_loss = tot / len(trL)

            # validation
            self.encoder.eval()
            for t in self.group_towers.values():
                t.eval()
            self.meta.eval()

            val = 0.0
            with torch.no_grad():
                for xb, yb, mb in vaL:
                    xb, yb, mb = xb.to(self.device), yb.to(self.device), mb.to(self.device)
                    latent = self.encoder(xb)
                    base = self._base_from_latent(latent)
                    pred = base + self.meta(base)
                    val += ((pred - yb) ** 2 * mb).sum().item() / mb.sum().item()
            val /= len(vaL)

            print(f"Epoch {ep:3d} | train {train_loss:.4f} | val {val:.4f}")

            if val < best - 1e-4:
                best, wait = val, 0
                torch.save(self.encoder.state_dict(), self.model_dir / "encoder_joint.pth")
                for gname, tower in self.group_towers.items():
                    torch.save(tower.state_dict(), self.model_dir / f"tower_joint_{gname}.pth")
                torch.save(self.meta.state_dict(), self.model_dir / "meta_joint.pth")
            else:
                wait += 1
                if wait >= PAT:
                    print(" ⇢ Early stopping joint")
                    break

    #  evaluation

    def evaluate(self, return_dict: bool = False):
        """
        Compute per-target relative-MSE (%) + R² on the validation split.
        """
        self.encoder.eval()
        for t in self.group_towers.values():
            t.eval()
        self.meta.eval()

        per_target = {}
        rel_mses, r2s = [], []

        Xval = self.X_embedded[self.va_idx]
        yval = self.y_raw[self.va_idx]

        with torch.no_grad():
            xb = torch.tensor(Xval, device=self.device)
            base_std = self._base_preds(xb)
            pred_std = base_std + self.meta(base_std)
            pred = (pred_std.cpu().numpy() * self.σ) + self.μ

        for j, prop in enumerate(self.present_targets):
            yt = yval[:, j]
            yp = pred[:, j]
            m_rel = _rel_mse_pct(yt, yp)
            r2 = r2_score(yt, yp)
            per_target[prop] = {"MSE_pct": float(m_rel), "R2": float(r2)}
            rel_mses.append(m_rel)
            r2s.append(r2)

        avg_rel_mse = float(np.mean(rel_mses))
        avg_r2 = float(np.mean(r2s))

        print(f"\nValidation results — relative MSE (% of ⟨y²⟩) and R²")
        for p, d in per_target.items():
            print(f" • {p:<8s}: {d['MSE_pct']:6.2f}%   R²={d['R2']:+.3f}")
        print(f" ⇒ Average   : {avg_rel_mse:6.2f}%   R²={avg_r2:+.3f}")

        if return_dict:
            self.metrics_ = {
                "avg_mse_pct": avg_rel_mse,
                "avg_r2": avg_r2,
                "per_target": per_target,
            }
            return self.metrics_



    def predict_from_embedded(self, X_embedded: np.ndarray) -> np.ndarray:
        """
        Convenience: predict in physical units from embedded input.
        """
        self.encoder.eval()
        for t in self.group_towers.values():
            t.eval()
        self.meta.eval()
        xb = torch.tensor(X_embedded, device=self.device)
        with torch.no_grad():
            base_std = self._base_preds(xb)
            pred_std = base_std + self.meta(base_std)
        return pred_std.cpu().numpy() * self.σ + self.μ

    def build_prediction_analysis_df(self, split: str = "val") -> pd.DataFrame:
        """
        Build long-form df with:
          index, target, actual, predicted, abs_error, percent_error
        for a given split: 'train', 'val', or 'test'.
        """
        if split == "train":
            idxs = self.tr_idx
        elif split in ["val", "validation"]:
            idxs = self.va_idx
        elif split == "test":
            idxs = self.te_idx
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

        X_split = self.X_embedded[idxs]
        y_split = self.y_raw[idxs]
        preds = self.predict_from_embedded(X_split)

        rows = []
        for local_i, global_i in enumerate(idxs):
            for j, tgt in enumerate(self.present_targets):
                y_true = y_split[local_i, j]
                y_pred = preds[local_i, j]
                if not np.isfinite(y_true):
                    continue
                abs_err = float(y_pred - y_true)
                denom = y_true if abs(y_true) > 1e-12 else 1.0
                pct_err = float(100.0 * abs_err / denom)
                rows.append(
                    {
                        "index": int(global_i),
                        "target": tgt,
                        "actual": float(y_true),
                        "predicted": float(y_pred),
                        "abs_error": abs_err,
                        "percent_error": pct_err,
                    }
                )
        return pd.DataFrame(rows)


    def predict(self, composition: Dict[str, float]) -> Dict[str, float]:
        """
        Predict properties from composition with full preprocessing and model loading.
        """
        # load saved weights
        enc_path = self.model_dir / "encoder_base.pth"
        meta_path = self.model_dir / "meta_transformer.pth"
        if enc_path.exists():
            self.encoder.load_state_dict(
                torch.load(enc_path, map_location=self.device)
            )
        for gname, tower in self.group_towers.items():
            pth = self.model_dir / f"tower_{gname}.pth"
            if pth.exists():
                tower.load_state_dict(torch.load(pth, map_location=self.device))
        if meta_path.exists():
            self.meta.load_state_dict(
                torch.load(meta_path, map_location=self.device)
            )

        # Process composition to fraction vector
        elements = {}
        compounds = {}

        for key, value in composition.items():
            parsed = self.parse_compound(key)
            if len(parsed) > 1:  # compound
                compounds[key] = compounds.get(key, 0.0) + value
                for el, count in parsed.items():
                    elements[el] = elements.get(el, 0.0) + value * count
            else:
                el = list(parsed.keys())[0]
                elements[el] = elements.get(el, 0.0) + value

        combined = {**compounds, **elements}
        total = sum(combined.values())
        if total <= 0:
            raise ValueError("Composition must have positive total")
        normalized = {k: v / total for k, v in combined.items()}

        feat_columns = self.X_comp.columns.tolist()
        feat_vector = np.zeros(len(feat_columns), dtype=np.float32)
        for i, col in enumerate(feat_columns):
            feat_vector[i] = normalized.get(col, 0.0)

        raw_df = pd.DataFrame([feat_vector], columns=feat_columns)
        raw_poly = self.poly.transform(raw_df)
        scaled_poly = self.scaler.transform(raw_poly)
        final_feats = np.hstack([scaled_poly, feat_vector[None, :]]).astype(np.float32)

        xb = torch.tensor(final_feats, device=self.device)
        self.encoder.eval()
        for t in self.group_towers.values():
            t.eval()
        self.meta.eval()

        with torch.no_grad():
            base_std = self._base_preds(xb)
            pred_std = base_std + self.meta(base_std)
            pred = (pred_std.cpu().numpy()[0] * self.σ) + self.μ

        return {prop: float(pred[self.present_targets.index(prop)]) for prop in self.present_targets}

    @staticmethod
    def parse_compound(c: str) -> Dict[str, int]:
        """Parse compound formula into constituent elements with counts."""
        elements = {}
        for el, n in re.findall(r"([A-Z][a-z]*)(\d*)", c):
            elements[el] = elements.get(el, 0) + int(n or "1")
        return elements

    def derived(self, coeffs: Dict[str, float], T: float) -> Dict[str, float]:
        out = {}
        if {"rho_a", "rho_b"}.issubset(coeffs):
            out["rho"] = coeffs["rho_a"] - coeffs["rho_b"] * T
        if {"mu1_a", "mu1_b"}.issubset(coeffs):
            out["muA"] = coeffs["mu1_a"] * math.exp(coeffs["mu1_b"] / (R * T))
        if {"mu2_a", "mu2_b", "mu2_c"}.issubset(coeffs):
            out["muB"] = 10 ** (
                coeffs["mu2_a"]
                + coeffs["mu2_b"] / T
                + coeffs["mu2_c"] / T**2
            )
        if {"k_a", "k_b"}.issubset(coeffs):
            out["k"] = coeffs["k_a"] + coeffs["k_b"] * T
        if {"cp_a", "cp_b", "cp_c"}.issubset(coeffs):
            out["cp"] = (
                coeffs["cp_a"] + coeffs["cp_b"] * T + coeffs["cp_c"] / T**2
            )
        return out


#plots

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

    values = d["percent_error"] if percent else d["abs_error"]
    values = values[np.isfinite(values)]
    if len(values) == 0:
        print(f"No finite errors for {target}")
        return

    # Clip at 99th percentile for readability
    p99 = np.percentile(values, 99)
    values_plot = np.clip(values, None, p99)

    plt.figure(figsize=(6, 4))
    sns.histplot(values_plot, bins=30, kde=True)
    if percent:
        plt.xlabel("Percent Error (%)")
    else:
        plt.xlabel("Absolute Error (same units as target)")
    plt.title(f"Error Distribution – {target}")
    plt.tight_layout()
    if save:
        suffix = "pct" if percent else "abs"
        _savefig(f"error_dist_{target}_{suffix}.png")
    plt.close()


def plot_all_targets_grid(df_pa, targets, cols=4, save=True):
    n = len(targets)
    rows_n = (n + cols - 1) // cols
    plt.figure(figsize=(cols * 4, rows_n * 3.2))
    for i, t in enumerate(targets):
        d = df_pa[df_pa["target"] == t]
        if len(d) == 0:
            continue
        plt.subplot(rows_n, cols, i + 1)
        sns.scatterplot(data=d, x="actual", y="predicted", alpha=0.6)
        lo = min(d["actual"].min(), d["predicted"].min())
        hi = max(d["actual"].max(), d["predicted"].max())
        plt.plot([lo, hi], [lo, hi], "r--")
        plt.title(t)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
    plt.tight_layout()
    if save:
        _savefig("all_targets_grid.png")
    plt.close()


def plot_confusion_style_heatmap(df_pa, target, bins=40, save=True):
    d = df_pa[df_pa["target"] == target]
    if len(d) == 0:
        print(f"No data for {target}")
        return

    actual = d["actual"].values
    pred = d["predicted"].values

    vmin = min(actual.min(), pred.min())
    vmax = max(actual.max(), pred.max())

    plt.figure(figsize=(6, 5))
    plt.hist2d(actual, pred, bins=bins,
               range=[[vmin, vmax], [vmin, vmax]],
               cmap="viridis")
    plt.plot([vmin, vmax], [vmin, vmax], "r--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Confusion-style Heatmap – {target}")
    plt.colorbar(label="Count")
    plt.tight_layout()
    if save:
        _savefig(f"confusion_heatmap_{target}.png")
    plt.close()


def build_error_bin_matrix(df_pa, targets, bins=(0, 5, 10, 20, 50, 100)):
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
        idxs = np.digitize(errs, bins)
        for idx in idxs:
            if idx < len(bins):
                col = labels[idx-1] if idx > 0 else labels[0]
            else:
                col = labels[-1]
            mat.loc[t, col] += 1
    return mat


def plot_error_bin_matrix(df_pa, targets, bins=(0, 5, 10, 20, 50, 100), save=True):
    mat = build_error_bin_matrix(df_pa, targets, bins=bins)
    plt.figure(figsize=(1.4 * len(mat.columns), 0.5 * len(mat.index) + 3))
    sns.heatmap(mat, annot=True, fmt="d", cmap="viridis", cbar_kws={"label": "Count"})
    plt.xlabel("Percent Error Bin")
    plt.ylabel("Target")
    plt.title("Error-bin Confusion Matrix")
    plt.tight_layout()
    if save:
        _savefig("error_bin_confusion_matrix.png")
    plt.close()


def plot_error_by_system(df_raw, df_pa, target, top_n_systems=10, percent=True, save=True):
    d = df_pa[df_pa["target"] == target]
    if len(d) == 0:
        print(f"No data for {target}")
        return

    merged = d.merge(df_raw[["System"]], left_on="index", right_index=True, how="left")
    err_col = "percent_error" if percent else "abs_error"
    ylabel = "Percent Error (%)" if percent else "Absolute Error"

    top = merged["System"].value_counts().head(top_n_systems).index
    merged = merged[merged["System"].isin(top)]

    plt.figure(figsize=(max(8, 0.7 * len(top)), 4))
    sns.boxplot(data=merged, x="System", y=err_col)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Error by System – {target}")
    plt.ylabel(ylabel)
    plt.tight_layout()
    if save:
        _savefig(f"error_by_system_{target}.png")
    plt.close()


def generate_all_plots(df_raw, df_pa, targets, top_n_systems=10):
    print("\n=== Generating per-target plots ===")
    for t in targets:
        d = df_pa[df_pa["target"] == t]
        if len(d) == 0:
            print(f"[{t}] – no data, skipping.")
            continue
        print(f"[{t}] – plotting...")
        plot_actual_vs_predicted(df_pa, t, save=True)
        plot_residuals(df_pa, t, save=True)
        plot_error_distribution(df_pa, t, percent=False, save=True)
        plot_error_distribution(df_pa, t, percent=True, save=True)
        plot_confusion_style_heatmap(df_pa, t, bins=40, save=True)
        plot_error_by_system(df_raw, df_pa, t,
                             top_n_systems=top_n_systems,
                             percent=True,
                             save=True)

    print("\n=== Generating global plots (all targets) ===")
    plot_all_targets_grid(df_pa, targets, cols=4, save=True)
    plot_error_bin_matrix(df_pa, targets, bins=(0, 5, 10, 20, 50, 100), save=True)
    print(f"\nAll plots generated and saved in '{PLOT_DIR}'.")


#  Permutation Feature Importance


def compute_permutation_importance(trainer: TransformerMetaTrainer,
                                   split: str = "val",
                                   n_repeats: int = 3):
    """
    Compute permutation importance of each embedded feature for each target.
    Returns:
        dict[target] -> np.array(shape=(n_features,))
    Importance = increase in relative MSE when shuffling that feature.
    """
    if split == "train":
        idxs = trainer.tr_idx
    elif split in ["val", "validation"]:
        idxs = trainer.va_idx
    elif split == "test":
        idxs = trainer.te_idx
    else:
        raise ValueError("split must be 'train', 'val', or 'test'")

    X = trainer.X_embedded[idxs].copy()
    y = trainer.y_raw[idxs].copy()
    n_samples, n_features = X.shape
    targets = trainer.present_targets

    # baseline
    base_preds = trainer.predict_from_embedded(X)
    base_imp = {}
    for j, tgt in enumerate(targets):
        base_imp[tgt] = _rel_mse_pct(y[:, j], base_preds[:, j])

    importances = {tgt: np.zeros(n_features, dtype=np.float32) for tgt in targets}

    for f in range(n_features):
        print(f"Permutation importance: feature {f+1}/{n_features}")
        deltas = {tgt: [] for tgt in targets}
        for _ in range(n_repeats):
            X_perm = X.copy()
            perm = np.random.permutation(n_samples)
            X_perm[:, f] = X_perm[perm, f]
            preds_perm = trainer.predict_from_embedded(X_perm)
            for j, tgt in enumerate(targets):
                m = _rel_mse_pct(y[:, j], preds_perm[:, j])
                deltas[tgt].append(m - base_imp[tgt])
        for tgt in targets:
            importances[tgt][f] = float(np.mean(deltas[tgt]))
    return importances


def plot_feature_importance_permutation(importances, target, top_k=20, save=True):
    """
    importances: dict[target] -> np.array(n_features)
    """
    if target not in importances:
        print(f"No importances for target {target}")
        return
    imp = importances[target]
    idx = np.argsort(imp)[::-1][:top_k]
    plt.figure(figsize=(6, max(4, 0.25 * top_k)))
    plt.barh([f"feat_{i}" for i in idx[::-1]], imp[idx][::-1])
    plt.xlabel("Increase in relMSE (%) when shuffled")
    plt.title(f"Permutation Feature Importance – {target}")
    plt.tight_layout()
    if save:
        _savefig(f"perm_importance_{target}.png")
    plt.close()


#  Main script


def main():
    # 1. Load data
    data_path = "/Users/meggie/Documents/MoltenSaltPropnet/data/new_mstdb_janz.csv"
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()

    # 2. Instantiate trainer
    trainer = TransformerMetaTrainer(df, TARGETS, DERIVED_PROPS)

    # 3. Train
    trainer.train_base()
    trainer.train_meta()
    # Optionally, you can also call trainer.train_joint() instead or after.

    # Evaluate
    metrics = trainer.evaluate(return_dict=True)
    print("\nPer-target metrics (validation split):")
    for t, m in metrics["per_target"].items():
        print(f"{t:<8s}  relMSE = {m['MSE_pct']:6.2f}%   R² = {m['R2']:+.3f}")

    # Build df_pa for train/val/test
    df_pa_train = trainer.build_prediction_analysis_df(split="train")
    df_pa_val   = trainer.build_prediction_analysis_df(split="val")
    df_pa_test  = trainer.build_prediction_analysis_df(split="test")

    df_pa = pd.concat(
        [df_pa_train.assign(split="train"),
         df_pa_val.assign(split="val"),
         df_pa_test.assign(split="test")],
        ignore_index=True,
    )

    # Visualisations (use val+test)
    df_pa_valtest = df_pa[df_pa["split"].isin(["val", "test"])]
    generate_all_plots(df, df_pa_valtest, trainer.present_targets, top_n_systems=10)

    # Permutation feature importance (on validation split)
    print("\n=== Computing permutation feature importance (validation split) ===")
    perm_importances = compute_permutation_importance(trainer, split="val", n_repeats=3)
    for t in trainer.present_targets:
        plot_feature_importance_permutation(perm_importances, t, top_k=20, save=True)

    print(f"\nDone. All plots & importance charts are in '{PLOT_DIR}'.")


if __name__ == "__main__":
    main()
