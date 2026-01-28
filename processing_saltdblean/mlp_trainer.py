# ============================================================
# mlp_trainer.py — MLP + Meta + Physics Regularization
# EXACT MATCH to SNN workflow (A/B training split)
# Architecture: GELU + Dropout(0.2) in base nets, CLEAN meta.
# ============================================================

import re
import ast
import math
import random
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from processing_saltdblean.embedding_preconditioner import EmbeddingPreconditioner

# -----------------------------
# Global config
# -----------------------------
SEED = 42
R = 8.314
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
warnings.filterwarnings("ignore", category=FutureWarning)

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

ELEMENT_FEATURE_COLS = [
    "polarizability_element[10-24Cm3]",
    "atomic_mass_element",
    "electronegativity_element",
    "atomic_radius_element[Angstrom]",
    "ionic_radius_element[Angstrom]",
    "first_ionization_energy[kJ_per_mol]",
]

# ============================================================
# Helper: relative MSE%
# ============================================================
def _rel_mse_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Relative MSE% compared to <y^2>."""
    mse = mean_squared_error(y_true, y_pred)
    denom = float(np.mean(y_true**2)) or 1e-12
    return 100.0 * mse / denom

# ============================================================
# MLP Blocks (Base net: GELU + Dropout; Meta net: clean GELU)
# ============================================================
class MLPBase(nn.Module):
    """MLP predicting one coefficient. GELU + Dropout(0.2)."""
    def __init__(self, in_dim: int, hidden: int = 128, out_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class MLPMeta(nn.Module):
    """
    Meta correction model (CLEAN: no dropout).
    Maps N_targets → N_targets.
    """
    def __init__(self, in_dim: int, hidden: int = 128, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),

            nn.Linear(hidden, hidden),
            nn.GELU(),

            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)

# ============================================================
# MLPMetaTrainer
# full A/B support, PCA embedding, poly features, element features,
# physics-loss identical to SNN trainer.
# ============================================================

class MLPMetaTrainer:
    def __init__(
        self,
        df: pd.DataFrame,
        target_cols: List[str],
        derived_props: List[Tuple[str, List[str]]],
        element_feature_cols: Optional[List[str]] = None,
        use_element_features: bool = True,
        degree_poly: int = 3,
        embedding_method: str = "none",     # set "pca" externally
        n_components: int = 10,             # set 32 externally
        splits=None,
        model_dir=None,
    ):
        self.df = df.copy()
        self.target_columns = target_cols
        self.derived_props = derived_props
        self.ELEMENT_FEATURE_COLS = list(element_feature_cols) if element_feature_cols else []
        self.use_element_features = bool(use_element_features)
        self.device = DEVICE

        self.embedding_method = embedding_method
        self.n_components = int(n_components)

        self.model_dir = Path(model_dir) if model_dir else Path("../data/trained_mlp")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # ========== 1) Clean target columns ==========
        self.present_targets = []
        for t in target_cols:
            if t not in self.df.columns:
                continue
            self.df[t] = (
                self.df[t]
                .replace(["----", ""], np.nan)
                .replace(r"\*", "", regex=True)
            )
            self.df[t] = pd.to_numeric(self.df[t], errors="coerce")
            if np.isfinite(self.df[t]).any():
                self.present_targets.append(t)

        if not self.present_targets:
            raise RuntimeError("No valid targets found.")

        # ========== 2) Build composition_df exactly like SNN ==========
        self.df["Composition"] = self.df.apply(self._row_composition, axis=1)

        X_comp = pd.json_normalize(self.df["Composition"]).fillna(0.0)
        X_comp = X_comp.reindex(sorted(X_comp.columns), axis=1)
        self.composition_df = X_comp
        self.fractions = X_comp.to_numpy(np.float32)

        # ========== 3) Poly features ==========
        self.poly = PolynomialFeatures(degree_poly, include_bias=False)
        X_poly = self.poly.fit_transform(X_comp).astype(np.float32)
        self.poly_scaler = StandardScaler()
        X_poly = self.poly_scaler.fit_transform(X_poly).astype(np.float32)
        self.X_poly = X_poly

        # ========== 4) Element features (optional) ==========
        if self.use_element_features:
            missing = [c for c in self.ELEMENT_FEATURE_COLS if c not in self.df.columns]
            if missing:
                raise ValueError(f"Missing element feature columns: {missing}")

            for col in self.ELEMENT_FEATURE_COLS:
                self.df[col] = self.df[col].apply(self._to_dict)

            self.elem_lookup = {col: {} for col in self.ELEMENT_FEATURE_COLS}
            for col in self.ELEMENT_FEATURE_COLS:
                for d in self.df[col]:
                    if not isinstance(d, dict):
                        continue
                    for el, v in d.items():
                        try:
                            fv = float(v)
                        except Exception:
                            continue
                        if el not in self.elem_lookup[col]:
                            self.elem_lookup[col][el] = fv

            feat_vectors = []
            self.elem_feat_cols = []
            for col in self.ELEMENT_FEATURE_COLS:
                newcol = f"{col}__wmean"
                self.elem_feat_cols.append(newcol)
                self.df[newcol] = [
                    self._weighted_mean_from_dict(comp, dct)
                    for comp, dct in zip(self.df["Composition"], self.df[col])
                ]
                feat_vectors.append(self.df[newcol].to_numpy(np.float32))

            elem_feat = np.vstack(feat_vectors).T.astype(np.float32)
            self.elem_scaler = StandardScaler()
            self.elem_features = self.elem_scaler.fit_transform(elem_feat).astype(np.float32)
        else:
            self.elem_lookup = {}
            self.elem_scaler = None
            self.elem_features = None
            self.elem_feat_cols = []

        # ========== 5) Final feature matrix ==========
        if self.use_element_features:
            self.X = np.hstack([self.X_poly, self.fractions, self.elem_features]).astype(np.float32)
        else:
            self.X = np.hstack([self.X_poly, self.fractions]).astype(np.float32)

        self.idx_all = np.arange(len(self.X))

        # ========== 6) Targets & masks ==========
        self.mask_all = np.isfinite(self.df[self.present_targets]).to_numpy(bool)
        self.df[self.present_targets] = self.df[self.present_targets].fillna(0.0)
        self.y_raw = self.df[self.present_targets].to_numpy(np.float32)

        # ========== 7) Train/Val/Test splits ==========
        if splits is None:
            tr_idx, te_idx = train_test_split(self.idx_all, test_size=0.20, random_state=SEED)
            tr_idx, va_idx = train_test_split(tr_idx, test_size=0.20, random_state=SEED)
        else:
            tr_idx, va_idx, te_idx = splits

        self.tr_idx, self.va_idx, self.te_idx = tr_idx, va_idx, te_idx

        # ========== 8) PCA embedding ==========
        self.embedder = EmbeddingPreconditioner(method=self.embedding_method, n_components=self.n_components)
        self.embedder.fit(self.X[self.tr_idx])
        self.X_embedded = self.embedder.transform(self.X).astype(np.float32)
        self.feat_dim = self.X_embedded.shape[1]

        # ========== 9) Normalize targets ==========
        self.μ = self.y_raw[self.tr_idx].mean(0)
        self.σ = self.y_raw[self.tr_idx].std(0)
        self.σ[self.σ == 0] = 1.0
        self.y_std = (self.y_raw - self.μ) / self.σ

        # ========== 10) Build networks ==========
        self.idx_map = {n: j for j, n in enumerate(self.present_targets)}

        self.base_nets = nn.ModuleDict({
            n: MLPBase(self.feat_dim, hidden=128, out_dim=1).to(self.device)
            for n in self.present_targets
        })

        self.meta = MLPMeta(
            in_dim=len(self.present_targets),
            hidden=128,
            out_dim=len(self.present_targets)
        ).to(self.device)

    def _base_preds(self, xb: torch.Tensor) -> torch.Tensor:
        """
        Compute stacked base predictions for all targets.
        xb : (B, feat_dim)
        Returns: (B, n_targets)
        """
        outs = []
        for t in self.present_targets:
            net = self.base_nets[t]
            y = net(xb)            # (B,1)
            outs.append(y)
        return torch.cat(outs, dim=1)   # (B, n_targets)
    # ============================================================
    # Internal helper functions
    # ============================================================

    def _to_dict(self, x):
        """Safely parse dict-like strings into dictionaries."""
        if isinstance(x, dict):
            return x
        if isinstance(x, str):
            s = x.strip()
            if s.startswith("{") and s.endswith("}"):
                try:
                    d = ast.literal_eval(s)
                    return d if isinstance(d, dict) else {}
                except Exception:
                    return {}
        return {}

    @staticmethod
    def _row_composition(row):
        """
        EXACT match to SNN implementation.
        Converts System + Mol Frac → per-element normalized fractions.
        Example:
            System: "NaCl-KCl"
            Mol Frac: "0.5-0.5"
        """
        comps = str(row["System"]).split("-")
        mf = str(row["Mol Frac"]).strip()

        if mf == "Pure Salt":
            fracs = [1.0] * len(comps)
        else:
            try:
                parts = [float(x) for x in mf.split("-")]
                fracs = parts
            except Exception:
                # fallback → equal fractions
                fracs = [1.0 / len(comps)] * len(comps)

        total = {}
        for cmp, f in zip(comps, fracs):
            # parse e.g. "NaCl" or "LiF"
            for el, cnt in re.findall(r"([A-Z][a-z]*)(\d*)", cmp):
                cnt = int(cnt or "1")
                total[el] = total.get(el, 0.0) + cnt * f

        s = sum(total.values()) or 1.0
        return {el: v / s for el, v in total.items()}
    
    # ============================================================
    # Parse formula: EXACT match to SNN version
    # ============================================================
    def parse_compound(self, formula: str):
        """
        Convert chemical formula → {element: count}.
        Example: 'NaCl' → {'Na':1, 'Cl':1}
                 'LiF2' → {'Li':1, 'F':2}
        """
        out = {}
        parts = re.findall(r"([A-Z][a-z]*)(\d*)", formula)
        for el, num in parts:
            cnt = int(num) if num else 1
            out[el] = out.get(el, 0) + cnt
        return out



    def _weighted_mean_from_dict(self, comp: dict, prop_dict: dict):
        """Weighted mean of element-wise dictionary using composition fractions."""
        if not isinstance(prop_dict, dict):
            prop_dict = {}
        out = 0.0
        for el, frac in comp.items():
            v = prop_dict.get(el, 0.0)
            try:
                out += float(frac) * float(v)
            except Exception:
                pass
        return float(out)

    # ============================================================
    # DataLoader helper
    # ============================================================
    def _loader(self, x, y, m, batch_size, shuffle):
        ds = TensorDataset(torch.tensor(x), torch.tensor(y), torch.tensor(m))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    # ============================================================
    # Base-net training (one model per target)
    # IDENTICAL PRINTOUT STYLE TO SNN
    # ============================================================
    def train_base(self):
        print("\n==============================")
        print("TRAINING BASE MLP MODELS")
        print("==============================")

        for prop in self.present_targets:
            print(f" • Training base net for {prop}")

            net = self.base_nets[prop]
            j = self.idx_map[prop]

            mask = self.mask_all[:, j].astype(bool)

            # Train/val subset for this target
            mask_tr = mask & np.isin(self.idx_all, self.tr_idx)
            mask_va = mask & np.isin(self.idx_all, self.va_idx)

            # If no val rows exist, create internal split
            if mask_va.sum() == 0:
                all_idx = np.where(mask)[0]
                if len(all_idx) >= 4:
                    tr_sub, va_sub = train_test_split(all_idx, test_size=0.2, random_state=SEED)
                    mask_tr = np.isin(self.idx_all, tr_sub)
                    mask_va = np.isin(self.idx_all, va_sub)
                else:
                    mask_va = np.zeros_like(mask_tr, dtype=bool)

            x_tr = self.X_embedded[mask_tr]
            y_tr = self.y_std[mask_tr, j:j+1]

            x_va = self.X_embedded[mask_va]
            y_va = self.y_std[mask_va, j:j+1]

            trL = DataLoader(
                TensorDataset(torch.tensor(x_tr), torch.tensor(y_tr)),
                batch_size=64, shuffle=True
            )
            vaL = DataLoader(
                TensorDataset(torch.tensor(x_va), torch.tensor(y_va)),
                batch_size=256, shuffle=False
            ) if len(x_va) else None

            optim = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 150, 1e-4)

            best_val = 1e12
            patience = 80
            wait = 0

            model_path = self.model_dir / f"base_{prop}_mlp.pth"

            for epoch in range(200):
                net.train()
                for xb, yb in trL:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)

                    optim.zero_grad()
                    pred = net(xb)
                    loss = nn.functional.mse_loss(pred, yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    optim.step()

                sched.step()

                # Validation
                if vaL:
                    net.eval()
                    vloss = 0.0
                    with torch.no_grad():
                        for xb, yb in vaL:
                            xb = xb.to(self.device)
                            yb = yb.to(self.device)
                            vloss += nn.functional.mse_loss(net(xb), yb).item()
                    vloss /= len(vaL)

                    if vloss < best_val - 1e-4:
                        best_val = vloss
                        wait = 0
                        torch.save(net.state_dict(), model_path)
                    else:
                        wait += 1
                        if wait >= patience:
                            print(f" ⇢ Early stopping for {prop}")
                            break

            # Reload best
            if model_path.exists():
                net.load_state_dict(torch.load(model_path, map_location=self.device))

    # ============================================================
    # Physics Loss (IDENTICAL FORMULAS TO SNN TRAINER)
    # ============================================================
    def _physics_loss(self, pred_raw, y_raw, mask_b, T):
        """
        pred_raw, y_raw: un-standardized predictions/targets
        mask_b: boolean mask (batch × targets)
        T: temperature vector
        """
        loss = 0.0
        count = 0

        for name, coeffs in self.derived_props:
            idxs = [self.idx_map[c] for c in coeffs if c in self.idx_map]
            if len(idxs) != len(coeffs):
                continue

            m = torch.all(mask_b[:, idxs], dim=1)
            if not m.any():
                continue

            y = y_raw[m][:, idxs]
            p = pred_raw[m][:, idxs]

            if name == "rho":
                # rho_a - rho_b * T
                loss_t = nn.functional.mse_loss(
                    p[:, 0] - p[:, 1] * T[m],
                    y[:, 0] - y[:, 1] * T[m]
                )

            elif name == "muA":
                # μ = μ1_a * exp( μ1_b / (R*T) )
                p_mu1a = torch.clamp(p[:, 0], min=1e-8)
                y_mu1a = torch.clamp(y[:, 0], min=1e-8)

                p_val = p_mu1a * torch.exp(p[:, 1] / (R * T[m]))
                y_val = y_mu1a * torch.exp(y[:, 1] / (R * T[m]))

                loss_t = nn.functional.mse_loss(
                    torch.log(p_val + 1e-9),
                    torch.log(y_val + 1e-9)
                )

            elif name == "muB":
                # log10(viscosity)
                loss_t = nn.functional.mse_loss(
                    p[:, 0] + p[:, 1] / T[m] + p[:, 2] / T[m] ** 2,
                    y[:, 0] + y[:, 1] / T[m] + y[:, 2] / T[m] ** 2,
                )

            elif name == "k":
                loss_t = nn.functional.mse_loss(
                    p[:, 0] + p[:, 1] * T[m],
                    y[:, 0] + y[:, 1] * T[m]
                )

            elif name == "cp":
                loss_t = nn.functional.mse_loss(
                    p[:, 0] + p[:, 1] * T[m] + p[:, 2] / T[m] ** 2,
                    y[:, 0] + y[:, 1] * T[m] + y[:, 2] / T[m] ** 2
                )

            else:
                continue

            loss += loss_t
            count += 1

        return loss / count if count > 0 else torch.tensor(0.0, device=self.device)
    # ============================================================
    # Stage-2 Training: Meta-network with physics regularization
    # ============================================================
    def train_meta(self, physics_weight: float = 0.10, temp_range=(500, 1200)):
        print("\n==============================")
        print("Stage-2: Training meta model with physics regularization…")
        print("==============================")

        # Freeze base nets
        for net in self.base_nets.values():
            for p in net.parameters():
                p.requires_grad_(False)

        train_loader = self._loader(
            self.X_embedded[self.tr_idx],
            self.y_std[self.tr_idx],
            self.mask_all[self.tr_idx],
            batch_size=64,
            shuffle=True
        )

        val_loader = self._loader(
            self.X_embedded[self.va_idx],
            self.y_std[self.va_idx],
            self.mask_all[self.va_idx],
            batch_size=256,
            shuffle=False
        )

        optim = torch.optim.AdamW(self.meta.parameters(), lr=8e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 300, 1e-4)

        best_val = 1e12
        patience = 100
        wait = 0

        meta_path = self.model_dir / "meta_mlp.pth"

        μ_t = torch.tensor(self.μ, dtype=torch.float32, device=self.device)
        σ_t = torch.tensor(self.σ, dtype=torch.float32, device=self.device)

        for epoch in range(400):

            # ------------------------
            # TRAINING
            # ------------------------
            self.meta.train()
            total_loss = 0.0

            for xb, yb, mb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                mb = mb.to(self.device)

                # sample temperature for each row
                T = torch.rand(len(xb), device=self.device)
                T = T * (temp_range[1] - temp_range[0]) + temp_range[0]

                with torch.no_grad():
                    base = self._base_preds(xb)

                pred = base + self.meta(base)

                # coeff loss
                loss_coeff = ((pred - yb) ** 2 * mb).sum() / mb.sum()

                # convert back to physical scale
                pred_raw = pred * σ_t + μ_t
                yb_raw = yb * σ_t + μ_t

                # physics penalty
                loss_phys = self._physics_loss(pred_raw, yb_raw, mb, T)

                loss = loss_coeff + physics_weight * loss_phys

                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.meta.parameters(), 0.5)
                optim.step()

                total_loss += loss.item()

            sched.step()

            # ------------------------
            # VALIDATION
            # ------------------------
            self.meta.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb, mb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    mb = mb.to(self.device)

                    base = self._base_preds(xb)
                    pred = base + self.meta(base)

                    val_loss += ((pred - yb) ** 2 * mb).sum().item() / mb.sum().item()

            val_loss /= len(val_loader)

            # ------------------------
            # Early stopping
            # ------------------------
            if val_loss < best_val - 1e-4:
                best_val = val_loss
                wait = 0
                torch.save(self.meta.state_dict(), meta_path)
            else:
                wait += 1
                if wait >= patience:
                    print(" ⇢ Early stopping meta")
                    break

        # Load best weights
        if meta_path.exists():
            self.meta.load_state_dict(torch.load(meta_path, map_location=self.device))

    # ============================================================
    # Evaluation routine (SNN-style)
    # ============================================================
    def evaluate(self, split: str = "val", return_dict: bool = False):
        split_map = {"train": self.tr_idx, "val": self.va_idx, "test": self.te_idx}
        idxs = split_map[split]

        self.meta.eval()
        for net in self.base_nets.values():
            net.eval()

        Xs = self.X_embedded[idxs]
        ys = self.y_raw[idxs]
        ms = self.mask_all[idxs].astype(bool)

        with torch.no_grad():
            xb = torch.tensor(Xs, device=self.device)
            base = torch.cat([self.base_nets[p](xb) for p in self.present_targets], dim=1)
            pred_std = (base + self.meta(base)).cpu().numpy()

        pred = pred_std * self.σ + self.μ

        # Per-target statistics
        per_target = {}
        mse_list, r2_list = [], []

        for j, prop in enumerate(self.present_targets):
            mask = ms[:, j]
            if not np.any(mask):
                continue

            yt = ys[mask, j]
            yp = pred[mask, j]

            mse_pct = 100 * mean_squared_error(yt, yp) / (np.mean(yt ** 2) or 1e-12)
            r2_val = r2_score(yt, yp)

            mse_list.append(mse_pct)
            r2_list.append(r2_val)

            per_target[prop] = {
                "MSE_pct": float(mse_pct),
                "R2": float(r2_val)
            }

        out = {
            "avg_mse_pct": float(np.mean(mse_list)) if mse_list else np.nan,
            "avg_r2": float(np.mean(r2_list)) if r2_list else np.nan,
            "per_target": per_target,
        }

        print(f"\n[{split.upper()}] avg MSE%={out['avg_mse_pct']:.3f} | avg R2={out['avg_r2']:.3f}")

        if return_dict:
            return out

    # ============================================================
    # Prediction for a single composition
    # ============================================================
    def predict(self, composition: dict) -> dict:
        """
        EXACT match to SNN predict() logic.
        Input:
            { "Na":0.5, "Cl":0.5 }
        """
        # Step 1 — convert NaCl mixture → element normalized fractions
        elem_counts = {}
        for comp, frac in composition.items():
            parsed = self.parse_compound(comp)
            for el, cnt in parsed.items():
                elem_counts[el] = elem_counts.get(el, 0) + cnt * float(frac)

        s = sum(elem_counts.values())
        if s <= 0:
            raise ValueError("Composition must sum to positive.")

        elem_frac = {el: v / s for el, v in elem_counts.items()}

        # Build feature vector
        frac_vec = np.zeros(len(self.composition_df.columns), dtype=np.float32)
        for i, col in enumerate(self.composition_df.columns):
            frac_vec[i] = elem_frac.get(col, 0.0)

        poly_raw = self.poly.transform([frac_vec]).astype(np.float32)
        poly_raw = self.poly_scaler.transform(poly_raw).astype(np.float32)

        if self.use_element_features:
            elem_vals = []
            for col in self.ELEMENT_FEATURE_COLS:
                mapping = self.elem_lookup.get(col, {})
                w = 0.0
                for el, f in elem_frac.items():
                    w += f * mapping.get(el, 0.0)
                elem_vals.append(w)
            elem_vals = np.array(elem_vals, dtype=np.float32)[None, :]
            elem_vals = self.elem_scaler.transform(elem_vals).astype(np.float32)
            feats = np.hstack([poly_raw, frac_vec[None, :], elem_vals]).astype(np.float32)
        else:
            feats = np.hstack([poly_raw, frac_vec[None, :]]).astype(np.float32)

        feats = self.embedder.transform(feats).astype(np.float32)

        xb = torch.tensor(feats, device=self.device, dtype=torch.float32)

        self.meta.eval()
        for net in self.base_nets.values():
            net.eval()

        with torch.no_grad():
            base = torch.cat([self.base_nets[p](xb) for p in self.present_targets], dim=1)
            pred_std = (base + self.meta(base)).cpu().numpy()[0]

        pred_raw = pred_std * self.σ + self.μ

        return {p: float(pred_raw[i]) for i, p in enumerate(self.present_targets)}

    # ============================================================
    # Derived property computation (same as SNN)
    # ============================================================
    def derived(self, coeffs: dict, T: float) -> dict:
        out = {}
        if {"rho_a", "rho_b"}.issubset(coeffs):
            out["rho"] = coeffs["rho_a"] - coeffs["rho_b"] * T
        if {"mu1_a", "mu1_b"}.issubset(coeffs):
            out["muA"] = coeffs["mu1_a"] * math.exp(coeffs["mu1_b"] / (R * T))
        if {"mu2_a", "mu2_b", "mu2_c"}.issubset(coeffs):
            out["muB"] = 10 ** (coeffs["mu2_a"] + coeffs["mu2_b"] / T + coeffs["mu2_c"] / T**2)
        if {"k_a", "k_b"}.issubset(coeffs):
            out["k"] = coeffs["k_a"] + coeffs["k_b"] * T
        if {"cp_a", "cp_b", "cp_c"}.issubset(coeffs):
            out["cp"] = coeffs["cp_a"] + coeffs["cp_b"] * T + coeffs["cp_c"] / T**2
        return out
# ============================================================
# END OF CLASS MLPMetaTrainer
# ============================================================

# NOTE:
# Nothing below here may import mlp_trainer again.
# This avoids circular imports when the package loads.


# ============================================================
# Convenience factory to load trainer from a DataFrame
# ============================================================
def make_mlp_trainer(
    df,
    use_element_features=True,
    embedding_method="pca",
    n_components=32,
    model_dir=None,
    splits=None,
):
    """
    Simplified constructor wrapper.
    Equivalent to calling MLPMetaTrainer(...) directly.
    """
    return MLPMetaTrainer(
        df=df,
        target_cols=TARGETS,
        derived_props=DERIVED_PROPS,
        element_feature_cols=ELEMENT_FEATURE_COLS,
        use_element_features=use_element_features,
        embedding_method=embedding_method,
        n_components=n_components,
        model_dir=model_dir,
        splits=splits,
    )


__all__ = [
    "MLPMetaTrainer",
    "make_mlp_trainer",
    "TARGETS",
    "DERIVED_PROPS",
    "ELEMENT_FEATURE_COLS",
]
