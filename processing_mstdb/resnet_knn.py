#!/usr/bin/env python
import re, math, random, warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import KNNImputer  # <-- NEW
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from processing_mstdb.embedding_preconditioner import EmbeddingPreconditioner

# ----------------------------------------------------------------------
# Utilities / globals
# ----------------------------------------------------------------------

def _rel_mse_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return relative MSE as a percentage of ⟨y²⟩ — avoids unit issues."""
    mse = mean_squared_error(y_true, y_pred)
    denom = np.mean(y_true ** 2) or 1e-12           # guard /0
    return 100.0 * mse / denom

SEED = 42
R = 8.314
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
warnings.filterwarnings("ignore", category=FutureWarning)

TARGETS = [
    "Melt(K)", "Boil(K)",
    "rho_a", "rho_b",
    "mu1_a", "mu1_b",
    "mu2_a", "mu2_b", "mu2_c",
    "k_a",  "k_b",
    "cp_a", "cp_b", "cp_c"
]

DERIVED_PROPS = [
    ('rho', ['rho_a', 'rho_b']),
    ('muA', ['mu1_a', 'mu1_b']),
    ('muB', ['mu2_a', 'mu2_b', 'mu2_c']),
    ('k',   ['k_a', 'k_b']),
    ('cp',  ['cp_a', 'cp_b', 'cp_c'])
]

# ----------------------------------------------------------------------
# Network building blocks
# ----------------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, dim, p_drop=0.2):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(p_drop)

    def forward(self, x):
        h = self.act(self.lin1(x))
        h = self.drop(h)
        h = self.lin2(h)
        return self.act(x + h)

class BaseNet(nn.Module):
    def __init__(self, d_in, hidden=64, depth=3):
        super().__init__()
        layers = [nn.Linear(d_in, hidden), nn.SiLU()]
        for _ in range(depth):
            layers.append(ResidualBlock(hidden))
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

class MetaNet(nn.Module):
    def __init__(self, n_props, hidden=128, depth=2):
        super().__init__()
        layers = [nn.Linear(n_props, hidden), nn.SiLU()]
        for _ in range(depth):
            layers.append(ResidualBlock(hidden))
        layers.append(nn.Linear(hidden, n_props))
        self.net = nn.Sequential(*layers)

    def forward(self, p):
        return self.net(p)

# ----------------------------------------------------------------------
# Trainer with KNN label imputation on TRAIN ONLY
# ----------------------------------------------------------------------

class ResNetMetaTrainerKNN:
    def __init__(self,
                 df: pd.DataFrame,
                 target_columns,
                 derived_props,
                 degree_poly: int = 3,
                 embedding_method: str = 'none',
                 n_components: int = 10):

        self.df = df.copy()
        self.target_columns = target_columns
        self.derived_props = derived_props
        self.model_dir = Path("../data/trained_models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        # --------------------------------------------------------------
        # 1) Clean targets & keep those with at least one finite value
        # --------------------------------------------------------------
        self.present_targets = []
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

        # --------------------------------------------------------------
        # 2) Composition → elemental fractions
        # --------------------------------------------------------------
        self.df["Composition"] = self.df.apply(self.row_composition, axis=1)
        self.X_comp = pd.json_normalize(self.df["Composition"]).fillna(0.0)
        self.X_comp = self.X_comp.reindex(sorted(self.X_comp.columns), axis=1)
        self.composition_df = self.X_comp

        # Polynomial features + standardisation
        self.poly = PolynomialFeatures(degree_poly, include_bias=False)
        self.X_poly = self.poly.fit_transform(self.X_comp)
        self.scaler = StandardScaler()
        self.X_poly = self.scaler.fit_transform(self.X_poly).astype(np.float32)

        self.fractions = self.X_comp.to_numpy(np.float32)
        self.X = np.hstack([self.X_poly, self.fractions])
        self.feat_dim = self.X.shape[1]

        # --------------------------------------------------------------
        # 3) Target matrix with NaNs (no filling yet)
        # --------------------------------------------------------------
        y_mat = self.df[self.present_targets].to_numpy(float)   # (N, P) with NaNs

        # --------------------------------------------------------------
        # 4) Global data splits (indices)
        # --------------------------------------------------------------
        self.idx_all = np.arange(len(self.X))
        self.tr_idx, self.te_idx = train_test_split(
            self.idx_all, test_size=0.20, random_state=SEED
        )
        self.tr_idx, self.va_idx = train_test_split(
            self.tr_idx, test_size=0.20, random_state=SEED
        )

        # --------------------------------------------------------------
        # 5) KNN IMPUTATION **ONLY ON TRAINING LABELS**
        # --------------------------------------------------------------
        imputer = KNNImputer(
            n_neighbors=5,
            weights="distance",
            metric="nan_euclidean",
        )

        y_train = y_mat[self.tr_idx]                       # (N_train, P) with NaNs
        y_train_imputed = imputer.fit_transform(y_train)   # fully numeric

        # Build final y_raw: train rows imputed, val/test original
        y_raw = y_mat.copy()
        y_raw[self.tr_idx] = y_train_imputed
        self.y_raw = y_raw.astype(np.float32)

        # mask_all:
        #  - True everywhere on train (all imputed → finite)
        #  - On val/test: True only where original labels were present
        self.mask_all = np.isfinite(self.y_raw)

        # --------------------------------------------------------------
        # 6) Embedding (optional)
        # --------------------------------------------------------------
        self.embedding_method = embedding_method
        self.n_components = n_components
        self.embedder = EmbeddingPreconditioner(
            method=embedding_method,
            n_components=n_components
        )
        self.embedder.fit(self.X[self.tr_idx])
        self.X_embedded = self.embedder.transform(self.X)
        self.feat_dim = (
            self.n_components if embedding_method != 'none' else self.X.shape[1]
        )

        # --------------------------------------------------------------
        # 7) Standardise targets using TRAIN (imputed) labels
        # --------------------------------------------------------------
        self.μ = self.y_raw[self.tr_idx].mean(0)
        self.σ = self.y_raw[self.tr_idx].std(0)
        self.σ[self.σ == 0] = 1.0

        self.y_std = (self.y_raw - self.μ) / self.σ
        # optional safety: remove NaNs where mask_all is False anyway
        self.y_std = np.nan_to_num(self.y_std, nan=0.0)

        # --------------------------------------------------------------
        # 8) Models
        # --------------------------------------------------------------
        self.idx_map = {n: j for j, n in enumerate(self.present_targets)}
        self.base_nets = nn.ModuleDict(
            {n: BaseNet(self.feat_dim).to(self.device) for n in self.present_targets}
        )
        self.meta = MetaNet(len(self.present_targets)).to(self.device)

    # ------------------------------------------------------------------
    # Helper: parse a single row into elemental composition
    # ------------------------------------------------------------------
    def row_composition(self, row):
        comps = row["System"].split("-")
        mf = str(row["Mol Frac"]).strip()
        if mf == "Pure Salt":
            fracs = [1.0] * len(comps)
        else:
            fracs = list(map(float, mf.split("-")))
            if len(fracs) == 1 and len(comps) == 2:
                # interpret as x / (1-x) for binary
                x = fracs[0]
                fracs = [x, 1.0 - x]
        total = {}
        for cmp, f in zip(comps, fracs):
            for el, cnt in re.findall(r"([A-Z][a-z]*)(\d*)", cmp):
                total[el] = total.get(el, 0) + int(cnt or "1") * f
        s = sum(total.values())
        return {el: cnt / s for el, cnt in total.items()}

    # ------------------------------------------------------------------
    # DataLoader helper
    # ------------------------------------------------------------------
    def make_loader(self, x, y, m, bs, shuf):
        ds = TensorDataset(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(m, dtype=torch.bool),
        )
        return DataLoader(ds, batch_size=bs, shuffle=shuf, drop_last=False)

    # ------------------------------------------------------------------
    # Stage 1: train base nets individually
    # ------------------------------------------------------------------
    def train_base(self):
        for prop in self.present_targets:
            print(f" • Training base net for {prop}")
            net = self.base_nets[prop]
            j = self.idx_map[prop]

            mask = self.mask_all[:, j].astype(bool)
            mask_tr_glb = mask & np.isin(self.idx_all, self.tr_idx)
            mask_va_glb = mask & np.isin(self.idx_all, self.va_idx)

            # If no validation data in global split, split available data
            if mask_va_glb.sum() == 0:
                idx_prop = np.where(mask)[0]
                if len(idx_prop) >= 2:
                    tr_prop, va_prop = train_test_split(
                        idx_prop, test_size=0.20, random_state=SEED
                    )
                    mask_tr_glb = np.isin(self.idx_all, tr_prop)
                    mask_va_glb = np.isin(self.idx_all, va_prop)
                else:
                    mask_tr_glb = np.isin(self.idx_all, idx_prop)
                    mask_va_glb = np.zeros_like(mask_tr_glb, dtype=bool)

            x_tr, y_tr = self.X_embedded[mask_tr_glb], self.y_std[mask_tr_glb, j]
            x_va, y_va = self.X_embedded[mask_va_glb], self.y_std[mask_va_glb, j]

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

            opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 200, 2e-4)
            best, patience, PAT = 1e9, 0, 25
            model_path = self.model_dir / f"base_{prop}_resnet_KNN.pth"

            for epoch in range(300):
                net.train()
                for xb, yb in tr_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    opt.zero_grad()
                    nn.functional.mse_loss(net(xb), yb).backward()
                    opt.step()
                sched.step()

                if va_loader:
                    net.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for xb, yb in va_loader:
                            xb, yb = xb.to(self.device), yb.to(self.device)
                            val_loss += nn.functional.mse_loss(net(xb), yb).item()
                        val_loss /= len(va_loader)

                    if val_loss < best - 1e-4:
                        best, patience = val_loss, 0
                        torch.save(net.state_dict(), model_path)
                    else:
                        patience += 1
                        if patience >= PAT:
                            print(f" ⇢ Early stopping for {prop}")
                            break

            # Load the best model if validation was used, else keep final model
            if va_loader:
                try:
                    net.load_state_dict(torch.load(model_path))
                except Exception:
                    print(f" No best model saved for {prop}, using final model")

    # ------------------------------------------------------------------
    # Stage 2: train meta net with physics regularisation
    # ------------------------------------------------------------------
    def train_meta(self):
        for net in self.base_nets.values():
            for p in net.parameters():
                p.requires_grad_(False)

        def base_preds_tensor(xb):
            return torch.stack(
                [self.base_nets[p](xb) for p in self.present_targets], 1
            )

        def physics_loss(pred_raw, yb_raw, mb, T):
            loss = 0.0
            valid_terms = 0
            for dprop, req_coeffs in self.derived_props:
                coeff_indices = [
                    self.idx_map[rc] for rc in req_coeffs if rc in self.idx_map
                ]
                if len(coeff_indices) != len(req_coeffs):
                    continue
                mask = torch.all(mb[:, coeff_indices], dim=1)
                if not mask.any():
                    continue
                y_coeffs = yb_raw[mask][:, coeff_indices]
                p_coeffs = pred_raw[mask][:, coeff_indices]
                with torch.no_grad():
                    if dprop == 'rho':
                        y_vals = y_coeffs[:, 0] - y_coeffs[:, 1] * T[mask]
                        p_vals = p_coeffs[:, 0] - p_coeffs[:, 1] * T[mask]
                        term_loss = nn.functional.mse_loss(p_vals, y_vals)
                    elif dprop == 'muA':
                        p_mu1_a = torch.clamp(p_coeffs[:, 0], min=1e-6)
                        p_vals = p_mu1_a * torch.exp(
                            p_coeffs[:, 1] / (R * T[mask])
                        )
                        y_vals = y_coeffs[:, 0] * torch.exp(
                            y_coeffs[:, 1] / (R * T[mask])
                        )
                        term_loss = nn.functional.mse_loss(
                            torch.log(p_vals + 1e-8), torch.log(y_vals + 1e-8)
                        )
                    elif dprop == 'muB':
                        y_log = (
                            y_coeffs[:, 0]
                            + y_coeffs[:, 1] / T[mask]
                            + y_coeffs[:, 2] / T[mask] ** 2
                        )
                        p_log = (
                            p_coeffs[:, 0]
                            + p_coeffs[:, 1] / T[mask]
                            + p_coeffs[:, 2] / T[mask] ** 2
                        )
                        term_loss = nn.functional.mse_loss(p_log, y_log)
                    elif dprop == 'k':
                        y_vals = y_coeffs[:, 0] + y_coeffs[:, 1] * T[mask]
                        p_vals = p_coeffs[:, 0] + p_coeffs[:, 1] * T[mask]
                        term_loss = nn.functional.mse_loss(p_vals, y_vals)
                    elif dprop == 'cp':
                        y_vals = (
                            y_coeffs[:, 0]
                            + y_coeffs[:, 1] * T[mask]
                            + y_coeffs[:, 2] / T[mask] ** 2
                        )
                        p_vals = (
                            p_coeffs[:, 0]
                            + p_coeffs[:, 1] * T[mask]
                            + p_coeffs[:, 2] / T[mask] ** 2
                        )
                        term_loss = nn.functional.mse_loss(p_vals, y_vals)
                    else:
                        continue
                loss += term_loss
                valid_terms += 1
            return (
                loss / valid_terms
                if valid_terms
                else torch.tensor(0.0, device=self.device)
            )

        PHYSICS_WEIGHT = 0.1
        TEMP_RANGE = (500, 1200)
        trL = self.make_loader(
            self.X_embedded[self.tr_idx],
            self.y_std[self.tr_idx],
            self.mask_all[self.tr_idx],
            64,
            True,
        )
        vaL = self.make_loader(
            self.X_embedded[self.va_idx],
            self.y_std[self.va_idx],
            self.mask_all[self.va_idx],
            256,
            False,
        )

        opt = torch.optim.AdamW(self.meta.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 400, 1e-4)
        best, wait, PAT = 1e9, 0, 40
        meta_path = self.model_dir / "meta_resnet_KNN.pth"

        μ_tensor = torch.tensor(self.μ, device=self.device, dtype=torch.float32)
        σ_tensor = torch.tensor(self.σ, device=self.device, dtype=torch.float32)

        print("\nStage-2: Training meta net with physics regularization...")
        for epoch in range(600):
            self.meta.train()
            total_loss = 0.0
            for xb, yb, mb in trL:
                xb, yb, mb = xb.to(self.device), yb.to(self.device), mb.to(self.device)
                batch_size = xb.size(0)
                T = (
                    torch.rand(batch_size, device=self.device)
                    * (TEMP_RANGE[1] - TEMP_RANGE[0])
                    + TEMP_RANGE[0]
                )
                with torch.no_grad():
                    base_out = base_preds_tensor(xb)
                pred = base_out + self.meta(base_out)
                loss_coeff = ((pred - yb) ** 2 * mb).sum() / mb.sum()
                pred_raw = pred * σ_tensor + μ_tensor
                yb_raw = yb * σ_tensor + μ_tensor
                loss_phys = physics_loss(pred_raw, yb_raw, mb, T) * PHYSICS_WEIGHT
                total_loss_ = loss_coeff + loss_phys
                total_loss_.backward()
                nn.utils.clip_grad_norm_(self.meta.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                total_loss += total_loss_.item()

            sched.step()
            avg_loss = total_loss / len(trL)

            self.meta.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb, mb in vaL:
                    xb, yb, mb = xb.to(self.device), yb.to(self.device), mb.to(self.device)
                    base_out = base_preds_tensor(xb)
                    pred = base_out + self.meta(base_out)
                    val_loss += ((pred - yb) ** 2 * mb).sum().item() / mb.sum().item()
            val_loss /= len(vaL)

            print(f"Epoch {epoch:3d} | Train: {avg_loss:.4f} | Val: {val_loss:.4f}")
            if val_loss < best - 1e-4:
                best, wait = val_loss, 0
                torch.save(self.meta.state_dict(), meta_path)
            else:
                wait += 1
                if wait >= PAT:
                    print(" ⇢ Early stopping")
                    break

        self.meta.load_state_dict(torch.load(meta_path, map_location=self.device))

    # ------------------------------------------------------------------
    # Evaluation on validation split (masking NaNs)
    # ------------------------------------------------------------------
    def evaluate(self, return_dict: bool = False):
        """Compute per-target relative-MSE (%) + R² on the *validation* split."""
        self.meta.eval()
        for net in self.base_nets.values():
            net.eval()

        per_target = {}
        rel_mses, r2s = [], []

        μ, σ = self.μ, self.σ
        Xval = self.X_embedded[self.va_idx]
        yval = self.y_raw[self.va_idx]   # may contain NaNs where labels missing

        with torch.no_grad():
            xb = torch.tensor(Xval, device=self.device, dtype=torch.float32)
            base_out = torch.stack(
                [self.base_nets[p](xb).cpu() for p in self.present_targets],
                dim=1,
            ).numpy()
            pred_std = base_out + self.meta(
                torch.tensor(base_out, device=self.device, dtype=torch.float32)
            ).cpu().numpy()

        pred = pred_std * σ + μ  # de-standardise

        print(f"\nValidation results — relative MSE (% of ⟨y²⟩) and R²")
        for j, prop in enumerate(self.present_targets):
            yt = yval[:, j]
            yp = pred[:, j]

            mask = np.isfinite(yt)
            if mask.sum() < 2:
                m_rel = float("nan")
                r2 = float("nan")
            else:
                m_rel = _rel_mse_pct(yt[mask], yp[mask])
                r2 = r2_score(yt[mask], yp[mask])

            per_target[prop] = {"MSE_pct": float(m_rel), "R2": float(r2)}
            rel_mses.append(m_rel)
            r2s.append(r2)
            print(f" • {prop:<8s}: {m_rel:6.2f}%   R²={r2:+.3f}")

        avg_rel_mse = float(np.nanmean(rel_mses))
        avg_r2 = float(np.nanmean(r2s))
        print(f" ⇒ Average   : {avg_rel_mse:6.2f}%   R²={avg_r2:+.3f}")

        if return_dict:
            self.metrics_ = {
                "avg_mse_pct": avg_rel_mse,
                "avg_r2": avg_r2,
                "per_target": per_target,
            }
            return self.metrics_

    # ------------------------------------------------------------------
    # Predict from a composition dict
    # ------------------------------------------------------------------
    def predict(self, composition: Dict[str, float]) -> Dict[str, float]:
        """Predict properties from composition with proper model loading and ordering."""

        model_dir = self.model_dir

        # Load base networks in alphabetical order
        sorted_targets = sorted(self.present_targets)
        for prop in sorted_targets:
            model_path = model_dir / f"base_{prop}_resnet_KNN.pth"
            if model_path.exists():
                self.base_nets[prop].load_state_dict(
                    torch.load(model_path, map_location=self.device)
                )
            else:
                raise FileNotFoundError(
                    f"Base model for {prop} not found at {model_path}"
                )

        # Load meta network
        meta_path = model_dir / "meta_resnet_KNN.pth"
        if meta_path.exists():
            self.meta.load_state_dict(
                torch.load(meta_path, map_location=self.device)
            )
        else:
            raise FileNotFoundError(f"Meta model not found at {meta_path}")

        # 2. Process composition (compound decomposition + normalization)
        elements = {}
        compounds = {}

        for key, value in composition.items():
            parsed = self.parse_compound(key)
            if len(parsed) > 1:  # Compound
                compounds[key] = compounds.get(key, 0.0) + value
                for el, count in parsed.items():
                    elements[el] = elements.get(el, 0.0) + value * count
            else:  # Element
                el = list(parsed.keys())[0]
                elements[el] = elements.get(el, 0.0) + value

        combined = {**compounds, **elements}
        total = sum(combined.values())
        if total <= 0:
            raise ValueError("Composition must have positive total")
        normalized = {k: v / total for k, v in combined.items()}

        # 3. Create input tensor with proper feature order
        frac = np.zeros(len(self.X_comp.columns), dtype=np.float32)
        for i, col in enumerate(self.X_comp.columns):
            frac[i] = normalized.get(col, 0.0)

        # 4. Generate predictions
        raw_df = pd.DataFrame([frac], columns=self.X_comp.columns).fillna(0.0)
        raw = self.poly.transform(raw_df)
        feats = np.hstack([self.scaler.transform(raw), frac[None, :]]).astype(
            np.float32
        )
        if self.embedding_method != 'none':
            feats = self.embedder.transform(feats)
        xb = torch.tensor(feats, device=self.device, dtype=torch.float32)

        with torch.no_grad():
            base_outputs = []
            for prop in sorted_targets:
                base_outputs.append(self.base_nets[prop](xb))
            base_out = torch.stack(base_outputs, dim=1)
            pred_std = base_out + self.meta(base_out)

        pred = pred_std.cpu().numpy()[0]

        return {
            prop: float(pred[i] * self.σ[i] + self.μ[i])
            for i, prop in enumerate(self.present_targets)
        }

    # ------------------------------------------------------------------
    # Helpers for predicting / saving / loading
    # ------------------------------------------------------------------
    @staticmethod
    def parse_compound(c: str) -> Dict[str, int]:
        out = {}
        for el, n in re.findall(r"([A-Z][a-z]*)(\d*)", c):
            out[el] = out.get(el, 0) + int(n or "1")
        return out

    def derived(self, coeffs: Dict[str, float], T: float) -> Dict[str, float]:
        out = {}
        if {'rho_a', 'rho_b'}.issubset(coeffs):
            out['rho'] = coeffs['rho_a'] - coeffs['rho_b'] * T
        if {'mu1_a', 'mu1_b'}.issubset(coeffs):
            out['muA'] = coeffs['mu1_a'] * math.exp(coeffs['mu1_b'] / (R * T))
        if {'mu2_a', 'mu2_b', 'mu2_c'}.issubset(coeffs):
            out['muB'] = 10 ** (
                coeffs['mu2_a'] + coeffs['mu2_b'] / T + coeffs['mu2_c'] / T ** 2
            )
        if {'k_a', 'k_b'}.issubset(coeffs):
            out['k'] = coeffs['k_a'] + coeffs['k_b'] * T
        if {'cp_a', 'cp_b', 'cp_c'}.issubset(coeffs):
            out['cp'] = (
                coeffs['cp_a'] + coeffs['cp_b'] * T + coeffs['cp_c'] / T ** 2
            )
        return out

    def save(self, path: str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for prop, net in self.base_nets.items():
            torch.save(net.state_dict(), path / f"base_{prop}_resnet_KNN.pth")
        torch.save(self.meta.state_dict(), path / "meta_resnet_KNN.pth")
        np.save(path / "μ_resnet_KNN.npy", self.μ)
        np.save(path / "σ_resnet_KNN.npy", self.σ)
        pd.to_pickle(self.poly, path / "poly_resnet_KNN.pkl")
        pd.to_pickle(self.scaler, path / "scaler_resnet_KNN.pkl")
        pd.to_pickle(self.X_comp.columns.tolist(), path / "elements_resnet_KNN.pkl")

    def load(self, path: str):
        path = Path(path)
        for prop in self.present_targets:
            self.base_nets[prop].load_state_dict(
                torch.load(path / f"base_{prop}_resnet_KNN.pth", map_location=self.device)
            )
        self.meta.load_state_dict(
            torch.load(path / "meta_resnet_KNN.pth", map_location=self.device)
        )
        self.μ = np.load(path / "μ_resnet_KNN.npy")
        self.σ = np.load(path / "σ_resnet_KNN.npy")
        self.poly = pd.read_pickle(path / "poly_resnet_KNN.pkl")
        self.scaler = pd.read_pickle(path / "scaler_resnet_KNN.pkl")
        self.X_comp.columns = pd.read_pickle(path / "elements_resnet_KNN.pkl")

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

if __name__ == "__main__":
    df = pd.read_csv(
        "/Users/meggie/Documents/MoltenSaltPropnet/data/new_mstdb_janz.csv"
    ).rename(columns=str.strip)

    trainer = ResNetMetaTrainerKNN(df, TARGETS, DERIVED_PROPS)
    print(
        f"Using {len(trainer.present_targets)} properties: "
        + ", ".join(trainer.present_targets)
    )

    trainer.train_base()
    trainer.train_meta()
    trainer.evaluate()

    # Example usage:
    # coeff = trainer.predict({'Na': 0.5, 'Cl': 0.5})
    # print("\nPredicted coefficients for 50-50 NaCl (KNN version):")
    # for k, v in coeff.items(): print(f"{k:7s}: {v:11.4f}")
    # print("\nDerived properties @ 900K:")
    # deriv = trainer.derived(coeff, 900)
    # for k, v in deriv.items(): print(f"{k:4s}: {v:11.4f}")
