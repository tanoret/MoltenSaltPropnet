# ============================================================
# FULL, CLEAN, REPRODUCIBLE A/B EXPERIMENT SCRIPT
# - Run A: WITHOUT element features
# - Run B: WITH element features
# - same splits, same seed, same training loops
# - produces side-by-side plots (A vs B)
# ============================================================

import os
import sys
import re
import ast
import math
import random
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Local import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from processing_saltdblean.processor import SALTDBLEANProcessor
from processing_saltdblean.embedding_preconditioner import EmbeddingPreconditioner


# ============================================================
# CONFIG
# ============================================================
DATA_PATH = "/Users/meggie/Documents/MoltenSaltPropnet/data/new_mstdb_janz_with_ionic_polarizability.csv"

BASE_OUTDIR = os.path.join("data", "resnet_compare")
PLOT_DIR = os.path.join(BASE_OUTDIR, "plots")
MODELS_A_DIR = os.path.join(BASE_OUTDIR, "models_A_without_elements")
MODELS_B_DIR = os.path.join(BASE_OUTDIR, "models_B_with_elements")

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODELS_A_DIR, exist_ok=True)
os.makedirs(MODELS_B_DIR, exist_ok=True)

SEED = 42
R = 8.314
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
warnings.filterwarnings("ignore", category=FutureWarning)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEMPERATURE = 900
MAX_TARGETS_PER_FIG = 8
ELEMENT_FILTERS = ["Cl", "F"]

SINGLE_COMPOSITIONS = {
    "NaCl (50-50 atoms)": {"Na": 0.5, "Cl": 0.5},
}

TARGETS = ["Melt(K)", "Boil(K)",
           "rho_a", "rho_b",
           "mu1_a", "mu1_b",
           "mu2_a", "mu2_b", "mu2_c",
           "k_a",  "k_b",
           "cp_a", "cp_b", "cp_c"]

DERIVED_PROPS = [
    ('rho', ['rho_a', 'rho_b']),
    ('muA', ['mu1_a', 'mu1_b']),
    ('muB', ['mu2_a', 'mu2_b', 'mu2_c']),
    ('k',   ['k_a', 'k_b']),
    ('cp',  ['cp_a', 'cp_b', 'cp_c'])
]

# NOTE: If you decided to drop covalent radius, comment it out here.
ELEMENT_FEATURE_COLS = [
    "polarizability_element[10-24Cm3]",
    "atomic_mass_element",
    "electronegativity_element",
    "atomic_radius_element[Angstrom]",
    "ionic_radius_element[Angstrom]",
    # "covalent_radius_element[Angstrom]",   # optional: drop if missingness is an issue
    "first_ionization_energy[kJ_per_mol]",
]


# ============================================================
# Utilities
# ============================================================
def _rel_mse_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Relative MSE as % of <y^2>."""
    mse = mean_squared_error(y_true, y_pred)
    denom = np.mean(y_true ** 2) or 1e-12
    return 100.0 * mse / denom


# ============================================================
# Networks
# ============================================================
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


# ============================================================
# Trainer (A/B capable)
# ============================================================
class ResNetMetaTrainer:
    def __init__(
        self,
        df: pd.DataFrame,
        target_columns: List[str],
        derived_props: List[Tuple[str, List[str]]],
        element_feature_cols: List[str],
        degree_poly: int = 3,
        embedding_method: str = 'none',
        n_components: int = 10,
        use_element_features: bool = True,
        splits=None,
        model_dir=None,
    ):
        self.df = df.copy()
        self.target_columns = target_columns
        self.derived_props = derived_props
        self.ELEMENT_FEATURE_COLS = element_feature_cols
        self.use_element_features = bool(use_element_features)

        self.device = DEVICE

        # model_dir
        self.model_dir = Path(model_dir) if model_dir is not None else Path("../data/trained_models")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # ---------- Clean and identify valid target columns ----------
        self.present_targets = []
        for t in target_columns:
            if t in self.df.columns:
                self.df[t] = self.df[t].replace(["----", ""], np.nan).replace(r"\*", "", regex=True)
                self.df[t] = pd.to_numeric(self.df[t], errors="coerce")
                if np.isfinite(self.df[t]).any():
                    self.present_targets.append(t)

        if not self.present_targets:
            raise RuntimeError("No valid target columns found after cleaning.")

        # ---------- Composition normalization ----------
        self.df["Composition"] = self.df.apply(self.row_composition, axis=1)

        # Element-fraction matrix
        self.X_comp = pd.json_normalize(self.df["Composition"]).fillna(0.0)
        self.X_comp = self.X_comp.reindex(sorted(self.X_comp.columns), axis=1)
        self.composition_df = self.X_comp
        self.fractions = self.X_comp.to_numpy(np.float32)

        # ---------- Poly features ----------
        self.poly = PolynomialFeatures(degree_poly, include_bias=False)
        X_poly = self.poly.fit_transform(self.X_comp).astype(np.float32)
        self.poly_scaler = StandardScaler()
        self.X_poly = self.poly_scaler.fit_transform(X_poly).astype(np.float32)

        # ---------- Element feature cols ----------
        if self.use_element_features:
            missing = [c for c in self.ELEMENT_FEATURE_COLS if c not in self.df.columns]
            if missing:
                raise ValueError(f"Missing ELEMENT_FEATURE_COLS in df: {missing}")

            for col in self.ELEMENT_FEATURE_COLS:
                self.df[col] = self.df[col].apply(self._to_dict)

            # lookup for predict()
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

            # weighted means
            elem_feat_mat = []
            self.elem_feat_cols = []
            for col in self.ELEMENT_FEATURE_COLS:
                new_col = f"{col}__wmean"
                self.elem_feat_cols.append(new_col)
                self.df[new_col] = [
                    self._weighted_mean_from_dict(comp, dct)
                    for comp, dct in zip(self.df["Composition"], self.df[col])
                ]
                elem_feat_mat.append(self.df[new_col].to_numpy(dtype=np.float32))

            element_features = np.vstack(elem_feat_mat).T.astype(np.float32)
            self.elem_scaler = StandardScaler()
            self.element_features = self.elem_scaler.fit_transform(element_features).astype(np.float32)
        else:
            self.elem_lookup = {}
            self.elem_scaler = None
            self.element_features = None
            self.elem_feat_cols = []

        # ---------- Final feature matrix ----------
        if self.use_element_features:
            self.X = np.hstack([self.X_poly, self.fractions, self.element_features]).astype(np.float32)
        else:
            self.X = np.hstack([self.X_poly, self.fractions]).astype(np.float32)

        self.idx_all = np.arange(len(self.X))

        # ---------- Targets + masks ----------
        self.mask_all = np.isfinite(self.df[self.present_targets]).to_numpy(bool)
        self.df[self.present_targets] = self.df[self.present_targets].fillna(0.0)
        self.y_raw = self.df[self.present_targets].to_numpy(np.float32)

        # ---------- Splits ----------
        if splits is None:
            self.tr_idx, self.te_idx = train_test_split(self.idx_all, test_size=0.20, random_state=SEED)
            self.tr_idx, self.va_idx = train_test_split(self.tr_idx, test_size=0.20, random_state=SEED)
        else:
            self.tr_idx, self.va_idx, self.te_idx = splits

        # ---------- Embedding ----------
        self.embedding_method = embedding_method
        self.n_components = n_components
        self.embedder = EmbeddingPreconditioner(method=embedding_method, n_components=n_components)
        self.embedder.fit(self.X[self.tr_idx])
        self.X_embedded = self.embedder.transform(self.X)

        self.feat_dim = self.X_embedded.shape[1]

        # ---------- Normalize targets ----------
        self.μ = self.y_raw[self.tr_idx].mean(0)
        self.σ = self.y_raw[self.tr_idx].std(0)
        self.σ[self.σ == 0] = 1.0
        self.y_std = (self.y_raw - self.μ) / self.σ

        # ---------- Models ----------
        self.idx_map = {n: j for j, n in enumerate(self.present_targets)}
        self.base_nets = nn.ModuleDict({n: BaseNet(self.feat_dim).to(self.device) for n in self.present_targets})
        self.meta = MetaNet(len(self.present_targets)).to(self.device)

    # ----------------- helpers -----------------
    def _to_dict(self, x):
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

    def _weighted_mean_from_dict(self, comp: dict, prop_dict: dict) -> float:
        s = 0.0
        if not isinstance(prop_dict, dict):
            prop_dict = {}
        for el, frac in comp.items():
            v = prop_dict.get(el, 0.0)
            try:
                s += float(frac) * float(v)
            except Exception:
                pass
        return float(s)

    def row_composition(self, row):
        comps = str(row["System"]).split("-")
        mf = str(row["Mol Frac"]).strip()
        fracs = [1.0] * len(comps) if mf == "Pure Salt" else list(map(float, mf.split("-")))
        total = {}
        for cmp, f in zip(comps, fracs):
            for el, cnt in re.findall(r"([A-Z][a-z]*)(\d*)", cmp):
                total[el] = total.get(el, 0) + int(cnt or "1") * f
        s = sum(total.values())
        return {el: cnt / s for el, cnt in total.items()}

    @staticmethod
    def parse_compound(c: str) -> Dict[str, int]:
        out = {}
        for el, n in re.findall(r"([A-Z][a-z]*)(\d*)", str(c)):
            out[el] = out.get(el, 0) + int(n or "1")
        return out

    def make_loader(self, x, y, m, bs, shuf):
        ds = TensorDataset(torch.tensor(x), torch.tensor(y), torch.tensor(m))
        return DataLoader(ds, batch_size=bs, shuffle=shuf, drop_last=False)

    # ----------------- train -----------------
    def train_base(self):
        for prop in self.present_targets:
            print(f" • Training base net for {prop}")
            net = self.base_nets[prop]
            j = self.idx_map[prop]

            mask = self.mask_all[:, j].astype(bool)
            mask_tr = mask & np.isin(self.idx_all, self.tr_idx)
            mask_va = mask & np.isin(self.idx_all, self.va_idx)

            if mask_va.sum() == 0:
                idx_prop = np.where(mask)[0]
                if len(idx_prop) >= 2:
                    tr_prop, va_prop = train_test_split(idx_prop, test_size=0.20, random_state=SEED)
                    mask_tr = np.isin(self.idx_all, tr_prop)
                    mask_va = np.isin(self.idx_all, va_prop)
                else:
                    mask_tr = np.isin(self.idx_all, idx_prop)
                    mask_va = np.zeros_like(mask_tr, dtype=bool)

            x_tr, y_tr = self.X_embedded[mask_tr], self.y_std[mask_tr, j]
            x_va, y_va = self.X_embedded[mask_va], self.y_std[mask_va, j]

            trL = DataLoader(TensorDataset(torch.tensor(x_tr), torch.tensor(y_tr)), batch_size=64, shuffle=True)
            vaL = DataLoader(TensorDataset(torch.tensor(x_va), torch.tensor(y_va)), batch_size=256, shuffle=False) if len(x_va) else None

            opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 200, 2e-4)
            best, patience, PAT = 1e9, 0, 25
            model_path = self.model_dir / f"base_{prop}_resnet.pth"

            for epoch in range(300):
                net.train()
                for xb, yb in trL:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    opt.zero_grad()
                    nn.functional.mse_loss(net(xb), yb).backward()
                    opt.step()
                sched.step()

                if vaL:
                    net.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for xb, yb in vaL:
                            xb, yb = xb.to(self.device), yb.to(self.device)
                            val_loss += nn.functional.mse_loss(net(xb), yb).item()
                        val_loss /= len(vaL)

                    if val_loss < best - 1e-4:
                        best, patience = val_loss, 0
                        torch.save(net.state_dict(), model_path)
                    else:
                        patience += 1
                        if patience >= PAT:
                            print(f" ⇢ Early stopping for {prop}")
                            break

            if vaL and model_path.exists():
                net.load_state_dict(torch.load(model_path, map_location=self.device))

    def train_meta(self):
        for net in self.base_nets.values():
            for p in net.parameters():
                p.requires_grad_(False)

        def base_preds_tensor(xb):
            return torch.stack([self.base_nets[p](xb) for p in self.present_targets], 1)

        def physics_loss(pred_raw, yb_raw, mb, T):
            loss = 0.0
            valid_terms = 0
            for dprop, req_coeffs in self.derived_props:
                coeff_indices = [self.idx_map[rc] for rc in req_coeffs if rc in self.idx_map]
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
                        p_vals = p_mu1_a * torch.exp(p_coeffs[:, 1] / (R * T[mask]))
                        y_vals = y_coeffs[:, 0] * torch.exp(y_coeffs[:, 1] / (R * T[mask]))
                        term_loss = nn.functional.mse_loss(torch.log(p_vals + 1e-8), torch.log(y_vals + 1e-8))
                    elif dprop == 'muB':
                        y_log = y_coeffs[:, 0] + y_coeffs[:, 1]/T[mask] + y_coeffs[:, 2]/T[mask]**2
                        p_log = p_coeffs[:, 0] + p_coeffs[:, 1]/T[mask] + p_coeffs[:, 2]/T[mask]**2
                        term_loss = nn.functional.mse_loss(p_log, y_log)
                    elif dprop == 'k':
                        y_vals = y_coeffs[:, 0] + y_coeffs[:, 1] * T[mask]
                        p_vals = p_coeffs[:, 0] + p_coeffs[:, 1] * T[mask]
                        term_loss = nn.functional.mse_loss(p_vals, y_vals)
                    elif dprop == 'cp':
                        y_vals = y_coeffs[:, 0] + y_coeffs[:, 1] * T[mask] + y_coeffs[:, 2]/T[mask]**2
                        p_vals = p_coeffs[:, 0] + p_coeffs[:, 1] * T[mask] + p_coeffs[:, 2]/T[mask]**2
                        term_loss = nn.functional.mse_loss(p_vals, y_vals)
                    else:
                        continue
                loss += term_loss
                valid_terms += 1
            return loss / valid_terms if valid_terms else torch.tensor(0.0, device=self.device)

        PHYSICS_WEIGHT = 0.1
        TEMP_RANGE = (500, 1200)

        trL = self.make_loader(self.X_embedded[self.tr_idx], self.y_std[self.tr_idx], self.mask_all[self.tr_idx], 64, True)
        vaL = self.make_loader(self.X_embedded[self.va_idx], self.y_std[self.va_idx], self.mask_all[self.va_idx], 256, False)

        opt = torch.optim.AdamW(self.meta.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 400, 1e-4)
        best, wait, PAT = 1e9, 0, 40
        meta_path = self.model_dir / "meta_resnet.pth"

        μ_tensor = torch.tensor(self.μ, device=self.device, dtype=torch.float32)
        σ_tensor = torch.tensor(self.σ, device=self.device, dtype=torch.float32)

        print("\nStage-2: Training meta net with physics regularization...")
        for epoch in range(600):
            self.meta.train()
            total_loss = 0.0
            for xb, yb, mb in trL:
                xb, yb, mb = xb.to(self.device), yb.to(self.device), mb.to(self.device)
                batch_size = xb.size(0)
                T = torch.rand(batch_size, device=self.device) * (TEMP_RANGE[1] - TEMP_RANGE[0]) + TEMP_RANGE[0]

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

            self.meta.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb, mb in vaL:
                    xb, yb, mb = xb.to(self.device), yb.to(self.device), mb.to(self.device)
                    base_out = base_preds_tensor(xb)
                    pred = base_out + self.meta(base_out)
                    val_loss += ((pred - yb) ** 2 * mb).sum().item() / mb.sum().item()
            val_loss /= len(vaL)

            if val_loss < best - 1e-4:
                best, wait = val_loss, 0
                torch.save(self.meta.state_dict(), meta_path)
            else:
                wait += 1
                if wait >= PAT:
                    print(" ⇢ Early stopping")
                    break

        if meta_path.exists():
            self.meta.load_state_dict(torch.load(meta_path, map_location=self.device))

    # ----------------- predict / derived / evaluate -----------------
    def predict(self, composition: Dict[str, float]) -> Dict[str, float]:
        # composition -> element fractions
        elements = {}
        for key, value in composition.items():
            parsed = self.parse_compound(key)
            for el, count in parsed.items():
                elements[el] = elements.get(el, 0.0) + float(value) * float(count)

        total = sum(elements.values())
        if total <= 0:
            raise ValueError("Composition must have positive total")
        normalized = {k: v / total for k, v in elements.items()}

        frac = np.zeros(len(self.X_comp.columns), dtype=np.float32)
        for i, col in enumerate(self.X_comp.columns):
            frac[i] = normalized.get(col, 0.0)

        raw_df = pd.DataFrame([frac], columns=self.X_comp.columns).fillna(0.0)
        raw_poly = self.poly.transform(raw_df).astype(np.float32)
        raw_poly = self.poly_scaler.transform(raw_poly).astype(np.float32)

        if self.use_element_features:
            elem_vec = []
            for col in self.ELEMENT_FEATURE_COLS:
                prop_map = self.elem_lookup.get(col, {})
                s = 0.0
                for el, f in normalized.items():
                    s += float(f) * float(prop_map.get(el, 0.0))
                elem_vec.append(s)
            elem_vec = np.array(elem_vec, dtype=np.float32)[None, :]
            elem_vec = self.elem_scaler.transform(elem_vec).astype(np.float32)

            feats = np.hstack([raw_poly, frac[None, :], elem_vec]).astype(np.float32)
        else:
            feats = np.hstack([raw_poly, frac[None, :]]).astype(np.float32)

        feats = self.embedder.transform(feats) if self.embedding_method != 'none' else feats
        xb = torch.tensor(feats, device=self.device)

        self.meta.eval()
        for net in self.base_nets.values():
            net.eval()

        with torch.no_grad():
            base_out = torch.stack([self.base_nets[p](xb) for p in self.present_targets], dim=1)
            pred_std = (base_out + self.meta(base_out)).cpu().numpy()[0]

        pred_raw = pred_std * self.σ + self.μ
        return {prop: float(pred_raw[i]) for i, prop in enumerate(self.present_targets)}

    def derived(self, coeffs: Dict[str, float], T: float) -> Dict[str, float]:
        out = {}
        if {'rho_a', 'rho_b'}.issubset(coeffs):
            out['rho'] = coeffs['rho_a'] - coeffs['rho_b'] * T
        if {'mu1_a', 'mu1_b'}.issubset(coeffs):
            out['muA'] = coeffs['mu1_a'] * math.exp(coeffs['mu1_b'] / (R * T))
        if {'mu2_a', 'mu2_b', 'mu2_c'}.issubset(coeffs):
            out['muB'] = 10 ** (coeffs['mu2_a'] + coeffs['mu2_b']/T + coeffs['mu2_c']/T**2)
        if {'k_a', 'k_b'}.issubset(coeffs):
            out['k'] = coeffs['k_a'] + coeffs['k_b'] * T
        if {'cp_a', 'cp_b', 'cp_c'}.issubset(coeffs):
            out['cp'] = coeffs['cp_a'] + coeffs['cp_b'] * T + coeffs['cp_c']/T**2
        return out

    def evaluate(self, split="val", return_dict=False):
        split_map = {"train": self.tr_idx, "val": self.va_idx, "test": self.te_idx}
        idxs = split_map[split]

        self.meta.eval()
        Xs = self.X_embedded[idxs]
        ys = self.y_raw[idxs]
        ms = self.mask_all[idxs].astype(bool)

        with torch.no_grad():
            xb = torch.tensor(Xs, device=self.device)
            base_out = torch.stack([self.base_nets[p](xb) for p in self.present_targets], dim=1)
            pred_std = (base_out + self.meta(base_out)).cpu().numpy()

        pred = pred_std * self.σ + self.μ

        per_target = {}
        rel_mses, r2s = [], []
        for j, prop in enumerate(self.present_targets):
            m = ms[:, j]
            if not np.any(m):
                continue
            yt = ys[m, j]
            yp = pred[m, j]
            m_rel = _rel_mse_pct(yt, yp)
            r2 = r2_score(yt, yp)
            per_target[prop] = {"MSE_pct": float(m_rel), "R2": float(r2)}
            rel_mses.append(m_rel)
            r2s.append(r2)

        out = {
            "avg_mse_pct": float(np.mean(rel_mses)) if rel_mses else np.nan,
            "avg_r2": float(np.mean(r2s)) if r2s else np.nan,
            "per_target": per_target
        }

        print(f"\n[{split.upper()}] avg MSE%={out['avg_mse_pct']:.3f} | avg R2={out['avg_r2']:.3f}")
        if return_dict:
            return out


# ============================================================
# Plot helpers
# ============================================================
def predict_all_embedded(trainer: ResNetMetaTrainer, X_embedded: np.ndarray) -> np.ndarray:
    for net in trainer.base_nets.values():
        net.eval()
    trainer.meta.eval()
    with torch.no_grad():
        xb = torch.tensor(X_embedded, dtype=torch.float32, device=trainer.device)
        base_out = torch.stack([trainer.base_nets[p](xb) for p in trainer.present_targets], dim=1)
        pred_std = (base_out + trainer.meta(base_out)).cpu().numpy()
        return pred_std * trainer.σ + trainer.μ

def indices_with_element(trainer: ResNetMetaTrainer, element: str, split: str = "test", min_frac: float = 1e-12) -> np.ndarray:
    split_map = {"train": trainer.tr_idx, "val": trainer.va_idx, "test": trainer.te_idx}
    idxs = split_map[split]
    frac = trainer.composition_df.loc[idxs, element].to_numpy(dtype=float)
    return idxs[frac > float(min_frac)]

def plot_coeffs_A_vs_B(trA, trB, idxs, title_prefix, outdir, fname_prefix, max_targets_per_fig=8):
    if len(idxs) == 0:
        print(f"[WARN] No rows for: {title_prefix}. Skipping.")
        return

    y_true = trA.y_raw[idxs]
    mask = trA.mask_all[idxs].astype(bool)

    predA = predict_all_embedded(trA, trA.X_embedded[idxs])
    predB = predict_all_embedded(trB, trB.X_embedded[idxs])

    targets = trA.present_targets
    n = len(targets)

    for start in range(0, n, max_targets_per_fig):
        chunk = targets[start:start+max_targets_per_fig]
        rows = len(chunk)

        fig, axes = plt.subplots(rows, 2, figsize=(13, 4.2 * rows))
        if rows == 1:
            axes = np.array([axes])

        for r, t in enumerate(chunk):
            j = trA.idx_map[t]
            m = mask[:, j]
            axA = axes[r, 0]
            axB = axes[r, 1]

            if not np.any(m):
                axA.set_axis_off()
                axB.set_axis_off()
                continue

            x = y_true[m, j]
            yA = predA[m, j]
            yB = predB[m, j]

            axA.scatter(x, yA, alpha=0.65)
            mn = min(x.min(), yA.min()); mx = max(x.max(), yA.max())
            axA.plot([mn, mx], [mn, mx], "k--", linewidth=1)
            axA.set_title(f"{t} | WITHOUT element feats")
            axA.set_xlabel("Actual"); axA.set_ylabel("Predicted")
            axA.grid(True, alpha=0.25)

            axB.scatter(x, yB, alpha=0.65)
            mn = min(x.min(), yB.min()); mx = max(x.max(), yB.max())
            axB.plot([mn, mx], [mn, mx], "k--", linewidth=1)
            axB.set_title(f"{t} | WITH element feats")
            axB.set_xlabel("Actual"); axB.set_ylabel("Predicted")
            axB.grid(True, alpha=0.25)

        fig.suptitle(f"{title_prefix}\nCoefficients: Actual vs Predicted — A vs B", y=1.01, fontsize=14)
        fig.tight_layout()

        fname = f"{fname_prefix}_coeff_A_vs_B_{start}_{start+len(chunk)-1}.png"
        fig.savefig(os.path.join(outdir, fname), dpi=170, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {fname}")

def plot_derived_A_vs_B(trA, trB, idxs, title_prefix, outdir, fname, T, props=("rho","muA","muB","k","cp")):
    if len(idxs) == 0:
        print(f"[WARN] No rows for: {title_prefix}. Skipping.")
        return

    predA = predict_all_embedded(trA, trA.X_embedded[idxs])
    predB = predict_all_embedded(trB, trB.X_embedded[idxs])

    actual_vals = {p: [] for p in props}
    predA_vals = {p: [] for p in props}
    predB_vals = {p: [] for p in props}

    for kpos, idx in enumerate(idxs):
        row = trA.df.iloc[idx]
        mask_row = trA.mask_all[idx]

        actual_coeffs = {}
        for j, col in enumerate(trA.present_targets):
            if mask_row[j]:
                actual_coeffs[col] = float(row[col])

        aprops = trA.derived(actual_coeffs, T)

        coeffA = dict(zip(trA.present_targets, predA[kpos]))
        coeffB = dict(zip(trB.present_targets, predB[kpos]))
        pA = trA.derived(coeffA, T)
        pB = trB.derived(coeffB, T)

        for p in props:
            a = aprops.get(p)
            va = pA.get(p)
            vb = pB.get(p)
            if a is None or va is None or vb is None:
                continue
            if not (np.isfinite(a) and np.isfinite(va) and np.isfinite(vb)):
                continue
            if abs(a) <= 1e-12:
                continue
            actual_vals[p].append(a)
            predA_vals[p].append(va)
            predB_vals[p].append(vb)

    rows = len(props)
    fig, axes = plt.subplots(rows, 2, figsize=(13, 4.0 * rows))
    if rows == 1:
        axes = np.array([axes])

    for r, p in enumerate(props):
        axA = axes[r, 0]
        axB = axes[r, 1]

        if len(actual_vals[p]) == 0:
            axA.set_axis_off(); axB.set_axis_off()
            continue

        x = np.array(actual_vals[p], float)
        yA = np.array(predA_vals[p], float)
        yB = np.array(predB_vals[p], float)

        axA.scatter(x, yA, alpha=0.65)
        mn = min(x.min(), yA.min()); mx = max(x.max(), yA.max())
        axA.plot([mn, mx], [mn, mx], "k--", linewidth=1)
        axA.set_title(f"{p} @ {T}K | WITHOUT element feats")
        axA.set_xlabel("Actual"); axA.set_ylabel("Predicted")
        axA.grid(True, alpha=0.25)

        axB.scatter(x, yB, alpha=0.65)
        mn = min(x.min(), yB.min()); mx = max(x.max(), yB.max())
        axB.plot([mn, mx], [mn, mx], "k--", linewidth=1)
        axB.set_title(f"{p} @ {T}K | WITH element feats")
        axB.set_xlabel("Actual"); axB.set_ylabel("Predicted")
        axB.grid(True, alpha=0.25)

    fig.suptitle(f"{title_prefix}\nDerived properties: Actual vs Predicted — A vs B", y=1.01, fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, fname), dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")

def plot_single_composition_A_vs_B(trA, trB, name, comp, outdir, T):
    coeffA = trA.predict(comp)
    coeffB = trB.predict(comp)

    targets = trA.present_targets
    n = len(targets)

    for start in range(0, n, MAX_TARGETS_PER_FIG):
        chunk = targets[start:start+MAX_TARGETS_PER_FIG]
        rows = len(chunk)
        fig, axes = plt.subplots(rows, 1, figsize=(11, 3.2 * rows))
        if rows == 1:
            axes = np.array([axes])

        for r, t in enumerate(chunk):
            ax = axes[r]
            a = coeffA.get(t, np.nan)
            b = coeffB.get(t, np.nan)
            ax.bar(["WITHOUT elem-feats", "WITH elem-feats"], [a, b], alpha=0.85)
            ax.set_title(t)
            ax.grid(True, axis="y", alpha=0.25)

        fig.suptitle(f"Single composition: {name}\nCoefficients — A vs B", y=1.01, fontsize=14)
        fig.tight_layout()

        fname = f"single_{name.replace(' ', '_')}_coeffs_A_vs_B_{start}_{start+len(chunk)-1}.png"
        fig.savefig(os.path.join(outdir, fname), dpi=170, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {fname}")

    props = ["rho","muA","muB","k","cp"]
    dA = trA.derived(coeffA, T)
    dB = trB.derived(coeffB, T)

    fig, axes = plt.subplots(len(props), 1, figsize=(11, 3.0 * len(props)))
    if len(props) == 1:
        axes = np.array([axes])

    for r, p in enumerate(props):
        ax = axes[r]
        a = dA.get(p, np.nan)
        b = dB.get(p, np.nan)
        ax.bar(["WITHOUT elem-feats", "WITH elem-feats"], [a, b], alpha=0.85)
        ax.set_title(f"{p} @ {T}K")
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle(f"Single composition: {name}\nDerived — A vs B", y=1.01, fontsize=14)
    fig.tight_layout()
    fname = f"single_{name.replace(' ', '_')}_derived_A_vs_B_T{int(T)}K.png"
    fig.savefig(os.path.join(outdir, fname), dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")


# ============================================================
# MAIN: load, split once, train A & B, plot comparisons
# ============================================================
if __name__ == "__main__":
    print("Loading dataset...")
    processor = SALTDBLEANProcessor.from_csv(DATA_PATH)
    processor.df.columns = processor.df.columns.str.strip()
    df = processor.df

    # Create shared splits using baseline (no element feats). This is crucial.
    print("Creating shared splits...")
    tmp = ResNetMetaTrainer(df, TARGETS, DERIVED_PROPS, ELEMENT_FEATURE_COLS, use_element_features=False, model_dir=MODELS_A_DIR)
    splits = (tmp.tr_idx, tmp.va_idx, tmp.te_idx)

    # Train A
    print("\n==============================")
    print("TRAIN RUN A: WITHOUT element features")
    print("==============================")
    trA = ResNetMetaTrainer(
        df, TARGETS, DERIVED_PROPS, ELEMENT_FEATURE_COLS,
        use_element_features=False,
        splits=splits,
        model_dir=MODELS_A_DIR
    )
    trA.train_base()
    trA.train_meta()
    trA.evaluate(split="val")
    trA.evaluate(split="test")

    # Train B
    print("\n==============================")
    print("TRAIN RUN B: WITH element features")
    print("==============================")
    trB = ResNetMetaTrainer(
        df, TARGETS, DERIVED_PROPS, ELEMENT_FEATURE_COLS,
        use_element_features=True,
        splits=splits,
        model_dir=MODELS_B_DIR
    )
    trB.train_base()
    trB.train_meta()
    trB.evaluate(split="val")
    trB.evaluate(split="test")

    print(f"\nSaving comparison plots under: {PLOT_DIR}")

    # Element-filtered plots (test split)
    for el in ELEMENT_FILTERS:
        idxs = indices_with_element(trA, element=el, split="test")
        title = f"Element filter: {el} > 0 | split=test | N={len(idxs)}"
        prefix = f"{el}_test"

        plot_coeffs_A_vs_B(trA, trB, idxs, title, PLOT_DIR, prefix, MAX_TARGETS_PER_FIG)
        plot_derived_A_vs_B(trA, trB, idxs, title, PLOT_DIR, f"{el}_derived_A_vs_B_T{int(TEMPERATURE)}K.png", TEMPERATURE)

    # Single-composition plots
    for name, comp in SINGLE_COMPOSITIONS.items():
        plot_single_composition_A_vs_B(trA, trB, name, comp, PLOT_DIR, TEMPERATURE)

    print("\nDone.")
    print("Plots written to:", PLOT_DIR)
    print("Models A written to:", MODELS_A_DIR)
    print("Models B written to:", MODELS_B_DIR)
