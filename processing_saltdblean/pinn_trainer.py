"""
pinn_trainer.py — Improved Physics-Informed Conditional VAE (A/B + plots + element-filter metrics)

Stability + reproducibility improvements:
  - Deterministic seeding + deterministic DataLoader generators
  - Lower default LR support + ReduceLROnPlateau scheduler
  - Safer heteroscedastic Gaussian NLL (clamped logvar/var)
  - Skip optimizer step on non-finite losses (prevents NaN cascade)
  - Optional grad clipping (kept) + anomaly guards
  - Physics loss uses relative error, optional, and uses a seeded generator

Features:
  - Train/eval splits (A/B share exact split)
  - Evaluate relMSE[%] and R² (per target + averages)
  - Element-filter evaluation (e.g., Cl/F subsets)
  - Save metrics to a text file in plot folder
  - Make standard actual-vs-pred plots (coeff + derived)

NOTE:
  This is a "physics-informed CVAE" model, not a classic PINN PDE solver.
  Your "physics" is enforced via derived-property consistency losses.
"""

import os
import re
import ast
import math
import copy
import random
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from processing_saltdblean.embedding_preconditioner import EmbeddingPreconditioner

warnings.filterwarnings("ignore", category=FutureWarning)

# ────────────────────────────────────────────────────────────────
# Globals / config
# ────────────────────────────────────────────────────────────────
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
R_GAS = 8.314

TARGETS: List[str] = [
    "Melt(K)", "Boil(K)",
    "rho_a", "rho_b",
    "mu1_a", "mu1_b",
    "mu2_a", "mu2_b", "mu2_c",
    "k_a",  "k_b",
    "cp_a", "cp_b", "cp_c",
]

DERIVED_PROPS: List[Tuple[str, List[str]]] = [
    ("rho", ["rho_a", "rho_b"]),
    ("muA", ["mu1_a", "mu1_b"]),
    ("muB", ["mu2_a", "mu2_b", "mu2_c"]),
    ("k",   ["k_a", "k_b"]),
    ("cp",  ["cp_a", "cp_b", "cp_c"]),
]

# Same element feature dict columns you used in ResNet/SNN
ELEMENT_FEATURE_COLS: List[str] = [
    "polarizability_element[10-24Cm3]",
    "atomic_mass_element",
    "electronegativity_element",
    "atomic_radius_element[Angstrom]",
    "ionic_radius_element[Angstrom]",
    "first_ionization_energy[kJ_per_mol]",
]

# Scaling to stabilize exponentials/powers
SCALE_MAP: Dict[str, float] = {
    "mu1_b": 1000.0,
    "mu2_b": 100.0,
    "mu2_c": 100.0,
    "cp_b":  100.0,
}

# Enable physics losses safely by default
PHYSICS_ENABLED_DEFAULT = {
    "rho": True,
    "k": True,
    "muA": False,
    "muB": False,
    "cp": False,
}


# ────────────────────────────────────────────────────────────────
# Reproducibility
# ────────────────────────────────────────────────────────────────
def set_global_seed(seed: int = SEED):
    global SEED
    SEED = int(seed)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


# ────────────────────────────────────────────────────────────────
# Metrics
# ────────────────────────────────────────────────────────────────
def _rel_mse_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    mse = float(np.mean((y_pred - y_true) ** 2))
    denom = float(np.mean(y_true ** 2))
    if denom <= 0:
        return float("nan")
    return 100.0 * mse / denom


def _r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    ym = float(y_true.mean())
    ss_res = float(np.sum((y_pred - y_true) ** 2))
    ss_tot = float(np.sum((y_true - ym) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


# ────────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────────
class PINNDataset(Dataset):
    def __init__(self, X: np.ndarray, Y_std: np.ndarray, M: np.ndarray):
        # ensure finite X/Y_std to prevent silent NaN propagation
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        Y_std = np.nan_to_num(Y_std, nan=0.0, posinf=0.0, neginf=0.0)
        M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)

        self.X = torch.from_numpy(X.astype(np.float32))
        self.Y = torch.from_numpy(Y_std.astype(np.float32))
        self.M = torch.from_numpy(M.astype(np.float32))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx], self.M[idx]


# ────────────────────────────────────────────────────────────────
# Model blocks
# ────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int], out_dim: int, dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.GELU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PhysicsCVAE(nn.Module):
    """
    Conditional VAE with:
      - Posterior: q(z | X, Y, M)
      - Prior:     p(z | X)
      - Decoder:   p(Y | z, X) (heteroscedastic: mean + logvar per target)
    """
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        latent_dim: int = 16,
        hidden_dim: int = 256,
        num_hidden_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.latent_dim = latent_dim

        enc_in = x_dim + y_dim + y_dim
        enc_hidden = [hidden_dim] * num_hidden_layers
        self.encoder = MLP(enc_in, enc_hidden, out_dim=2 * latent_dim, dropout=dropout)

        self.prior = MLP(x_dim, enc_hidden, out_dim=2 * latent_dim, dropout=dropout)

        dec_in = x_dim + latent_dim
        dec_hidden = [hidden_dim] * num_hidden_layers
        self.decoder = MLP(dec_in, dec_hidden, out_dim=2 * y_dim, dropout=dropout)

    def encode_posterior(self, x: torch.Tensor, y: torch.Tensor, m: torch.Tensor):
        h = self.encoder(torch.cat([x, y, m], dim=1))
        mu, logvar = torch.chunk(h, 2, dim=1)
        # mild clamp helps avoid rare blowups
        logvar = torch.clamp(logvar, min=-12.0, max=8.0)
        return mu, logvar

    def encode_prior(self, x: torch.Tensor):
        h = self.prior(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        logvar = torch.clamp(logvar, min=-12.0, max=8.0)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, x: torch.Tensor, z: torch.Tensor):
        out = self.decoder(torch.cat([x, z], dim=1))
        mu, logvar = torch.chunk(out, 2, dim=1)
        logvar = torch.clamp(logvar, min=-12.0, max=6.0)  # tighter clamp for decoder variance
        return mu, logvar


# ────────────────────────────────────────────────────────────────
# Losses
# ────────────────────────────────────────────────────────────────
def kl_diag_gauss_gauss(mu_q, logvar_q, mu_p, logvar_p) -> torch.Tensor:
    """
    KL( N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2) ) averaged over batch.
    """
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    kl = 0.5 * (
        (logvar_p - logvar_q)
        + (var_q + (mu_q - mu_p) ** 2) / (var_p + 1e-8)
        - 1.0
    )
    return kl.sum(dim=1).mean()


def masked_gaussian_nll(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    y: torch.Tensor,
    m: torch.Tensor,
    w: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Masked heteroscedastic Gaussian NLL:
      0.5 * [ (y-mu)^2 / var + logvar ]
    averaged over observed entries; optionally weighted per-target.

    Stability:
      - clamp var away from 0
    """
    logvar = torch.clamp(logvar, min=-12.0, max=6.0)
    var = torch.exp(logvar).clamp(min=1e-6, max=1e6)

    nll = 0.5 * ((y - mu) ** 2 / var + logvar)

    if w is not None:
        nll = nll * w[None, :]

    nll = nll * m
    denom = (m if w is None else (m * w[None, :])).sum()
    if denom <= 0:
        return torch.tensor(0.0, device=mu.device)
    return nll.sum() / denom


def build_target_index_map(target_cols: List[str]) -> Dict[str, int]:
    return {c: j for j, c in enumerate(target_cols)}


def physics_loss_relative(
    y_pred_phys: torch.Tensor,
    y_true_phys: torch.Tensor,
    mask: torch.Tensor,
    target_index: Dict[str, int],
    derived_groups: List[Tuple[str, List[str]]],
    enabled: Dict[str, bool],
    temp_range: Tuple[float, float] = (500.0, 1200.0),
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Physics loss on derived quantities, using relative squared error:
        (pred-true)^2 / (true^2 + eps)

    Only applies on samples where all required coefficients are observed.
    Uses a generator for deterministic temperature draws.
    """
    device = y_pred_phys.device
    B = y_pred_phys.shape[0]
    eps = 1e-8

    if rng is None:
        T = torch.rand(B, device=device) * (temp_range[1] - temp_range[0]) + temp_range[0]
    else:
        # torch.rand supports generator; for CUDA generator, create it on CUDA in trainer
        T = torch.rand(B, device=device, generator=rng) * (temp_range[1] - temp_range[0]) + temp_range[0]

    total = 0.0
    n_terms = 0

    for tag, coeffs in derived_groups:
        if not enabled.get(tag, False):
            continue

        idxs = [target_index[c] for c in coeffs if c in target_index]
        if len(idxs) != len(coeffs):
            continue

        idxs_t = torch.tensor(idxs, device=device, dtype=torch.long)
        ok = mask[:, idxs_t].bool().all(dim=1)
        if not ok.any():
            continue

        yt = y_true_phys[ok][:, idxs_t]
        yp = y_pred_phys[ok][:, idxs_t]
        Tt = T[ok]

        try:
            if tag == "rho":
                true_val = yt[:, 0] - yt[:, 1] * Tt
                pred_val = yp[:, 0] - yp[:, 1] * Tt
            elif tag == "k":
                true_val = yt[:, 0] + yt[:, 1] * Tt
                pred_val = yp[:, 0] + yp[:, 1] * Tt
            elif tag == "muA":
                # optional; disabled by default
                mu1a_t = torch.clamp(yt[:, 0], min=1e-10, max=1e10)
                mu1a_p = torch.clamp(yp[:, 0], min=1e-10, max=1e10)
                arg_t = torch.clamp(yt[:, 1] / (R_GAS * Tt), min=-50.0, max=50.0)
                arg_p = torch.clamp(yp[:, 1] / (R_GAS * Tt), min=-50.0, max=50.0)
                true_val = mu1a_t * torch.exp(arg_t)
                pred_val = mu1a_p * torch.exp(arg_p)
            elif tag == "muB":
                # optional; disabled by default
                a_t, b_t, c_t = yt[:, 0], yt[:, 1], yt[:, 2]
                a_p, b_p, c_p = yp[:, 0], yp[:, 1], yp[:, 2]
                exp_t = torch.clamp(a_t + b_t / Tt + c_t / (Tt ** 2), min=-20.0, max=20.0)
                exp_p = torch.clamp(a_p + b_p / Tt + c_p / (Tt ** 2), min=-20.0, max=20.0)
                true_val = 10.0 ** exp_t
                pred_val = 10.0 ** exp_p
            elif tag == "cp":
                # optional; disabled by default
                true_val = yt[:, 0] + yt[:, 1] * Tt + yt[:, 2] / (Tt ** 2)
                pred_val = yp[:, 0] + yp[:, 1] * Tt + yp[:, 2] / (Tt ** 2)
            else:
                continue

            rel_sq = (pred_val - true_val) ** 2 / (true_val ** 2 + eps)
            rel_sq = torch.nan_to_num(rel_sq, nan=0.0, posinf=0.0, neginf=0.0)
            total = total + torch.mean(rel_sq)
            n_terms += 1
        except Exception:
            # if anything explodes numerically, skip that term
            continue

    if n_terms == 0:
        return torch.tensor(0.0, device=device)
    return torch.clamp(total / n_terms, max=50.0)


# ────────────────────────────────────────────────────────────────
# Trainer
# ────────────────────────────────────────────────────────────────
class PINNMetaTrainer:
    """
    Improved physics-informed CVAE trainer with A/B element-feature support + embedding.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        target_cols: List[str],
        derived_props: List[Tuple[str, List[str]]],
        element_feature_cols: Optional[List[str]] = None,
        use_element_features: bool = True,
        degree_poly: int = 3,
        embedding_method: str = "pca",
        n_components: int = 32,
        splits=None,                      # (tr_idx, va_idx, te_idx)
        model_dir: Optional[str] = None,
        latent_dim: int = 16,
        hidden_dim: int = 256,
        num_hidden_layers: int = 3,
        dropout: float = 0.05,
        physics_enabled: Optional[Dict[str, bool]] = None,
        seed: int = SEED,
    ):
        set_global_seed(seed)
        self.seed = int(seed)

        self.df = df.copy()
        self.target_cols = list(target_cols)
        self.derived_props = list(derived_props)

        self.ELEMENT_FEATURE_COLS = list(element_feature_cols) if element_feature_cols else []
        self.use_element_features = bool(use_element_features)

        self.embedding_method = str(embedding_method)
        self.n_components = int(n_components)

        self.device = DEVICE
        self.model_dir = Path(model_dir) if model_dir is not None else Path("../data/trained_models_pinn")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.physics_enabled = dict(PHYSICS_ENABLED_DEFAULT)
        if physics_enabled is not None:
            self.physics_enabled.update(physics_enabled)

        # ----------- targets cleaning + presence -----------
        self.present_targets: List[str] = []
        for t in self.target_cols:
            if t in self.df.columns:
                s = self.df[t].replace(["----", ""], np.nan).replace(r"\*", "", regex=True)
                vals = pd.to_numeric(s, errors="coerce")
                if np.isfinite(vals).any():
                    self.present_targets.append(t)
                    self.df[t] = vals

        if not self.present_targets:
            raise RuntimeError("No valid target columns found after cleaning.")

        # ----------- composition fractions -----------
        self.df["Composition"] = self.df.apply(self._row_composition, axis=1)
        self.X_comp = pd.json_normalize(self.df["Composition"]).fillna(0.0)
        self.X_comp = self.X_comp.reindex(sorted(self.X_comp.columns), axis=1)
        self.fractions = self.X_comp.to_numpy(np.float32)

        # ----------- polynomial features -----------
        self.poly = PolynomialFeatures(degree_poly, include_bias=False)
        X_poly = self.poly.fit_transform(self.X_comp).astype(np.float32)
        self.poly_scaler = StandardScaler()
        X_poly = self.poly_scaler.fit_transform(X_poly).astype(np.float32)

        # ----------- optional element features -----------
        if self.use_element_features:
            if not self.ELEMENT_FEATURE_COLS:
                raise ValueError("use_element_features=True but element_feature_cols is empty.")
            missing = [c for c in self.ELEMENT_FEATURE_COLS if c not in self.df.columns]
            if missing:
                raise ValueError(f"Missing ELEMENT_FEATURE_COLS in df: {missing}")

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

            feat_mat = []
            self.elem_feat_cols = []
            for col in self.ELEMENT_FEATURE_COLS:
                new_col = f"{col}__wmean"
                self.elem_feat_cols.append(new_col)
                self.df[new_col] = [
                    self._weighted_mean_from_dict(comp, dct)
                    for comp, dct in zip(self.df["Composition"], self.df[col])
                ]
                feat_mat.append(self.df[new_col].to_numpy(dtype=np.float32))

            element_features = np.vstack(feat_mat).T.astype(np.float32)
            self.elem_scaler = StandardScaler()
            element_features = self.elem_scaler.fit_transform(element_features).astype(np.float32)
            self.element_features = element_features
        else:
            self.elem_lookup = {}
            self.elem_scaler = None
            self.element_features = None
            self.elem_feat_cols = []

        # ----------- final X (before embedding) -----------
        if self.use_element_features:
            self.X = np.hstack([X_poly, self.fractions, self.element_features]).astype(np.float32)
        else:
            self.X = np.hstack([X_poly, self.fractions]).astype(np.float32)

        self.X = np.nan_to_num(self.X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        self.idx_all = np.arange(len(self.X))

        # ----------- raw targets with scale factors -----------
        self.scale_factors = np.ones(len(self.present_targets), dtype=np.float32)
        Y_raw = np.full((len(self.df), len(self.present_targets)), np.nan, dtype=np.float32)

        for j, col in enumerate(self.present_targets):
            vals = self.df[col].to_numpy(np.float32)
            s = float(SCALE_MAP.get(col, 1.0))
            self.scale_factors[j] = s
            if s != 1.0:
                vals = vals / s
            Y_raw[:, j] = vals

        self.mask_all = np.isfinite(Y_raw)
        Y_raw_filled = Y_raw.copy()
        Y_raw_filled[~self.mask_all] = 0.0

        self.y_raw_scaled = Y_raw              # scaled, may contain NaN
        self.y_raw_scaled_filled = Y_raw_filled

        # ----------- splits -----------
        if splits is None:
            tr_idx, te_idx = train_test_split(self.idx_all, test_size=0.20, random_state=self.seed)
            tr_idx, va_idx = train_test_split(tr_idx, test_size=0.20, random_state=self.seed)
        else:
            tr_idx, va_idx, te_idx = splits

        self.tr_idx, self.va_idx, self.te_idx = np.array(tr_idx), np.array(va_idx), np.array(te_idx)

        # ----------- standardize targets using train only -----------
        self.y_mean = np.zeros(len(self.present_targets), dtype=np.float32)
        self.y_std = np.ones(len(self.present_targets), dtype=np.float32)

        for j in range(len(self.present_targets)):
            ok = self.mask_all[self.tr_idx, j]
            if not np.any(ok):
                continue
            col = self.y_raw_scaled[self.tr_idx[ok], j]
            m = float(np.mean(col))
            s = float(np.std(col))
            self.y_mean[j] = m
            self.y_std[j] = s if s > 0 else 1.0

        Y_std = (self.y_raw_scaled_filled - self.y_mean[None, :]) / self.y_std[None, :]
        Y_std[~self.mask_all] = 0.0
        self.y_std_all = np.nan_to_num(Y_std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # ----------- per-target weights (frequency balancing) -----------
        obs_counts = self.mask_all[self.tr_idx].sum(axis=0).astype(np.float32)
        self.target_weights = 1.0 / (obs_counts + 10.0)
        self.target_weights = self.target_weights / np.mean(self.target_weights)

        # ----------- embedding -----------
        self.embedder = EmbeddingPreconditioner(method=self.embedding_method, n_components=self.n_components)
        self.embedder.fit(self.X[self.tr_idx])
        self.X_embedded = self.embedder.transform(self.X).astype(np.float32)
        self.X_embedded = np.nan_to_num(self.X_embedded, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        self.x_dim = self.X_embedded.shape[1]
        self.y_dim = len(self.present_targets)

        # ----------- model -----------
        self.model = PhysicsCVAE(
            x_dim=self.x_dim,
            y_dim=self.y_dim,
            latent_dim=int(latent_dim),
            hidden_dim=int(hidden_dim),
            num_hidden_layers=int(num_hidden_layers),
            dropout=float(dropout),
        ).to(self.device)

        self.target_index = build_target_index_map(self.present_targets)

        # dataset
        self.dataset = PINNDataset(self.X_embedded, self.y_std_all, self.mask_all.astype(np.float32))

        # deterministic generator for DataLoader shuffles + physics temperatures
        self._cpu_gen = torch.Generator(device="cpu")
        self._cpu_gen.manual_seed(self.seed)

        if self.device.startswith("cuda"):
            self._dev_gen = torch.Generator(device="cuda")
            self._dev_gen.manual_seed(self.seed)
        else:
            self._dev_gen = torch.Generator(device="cpu")
            self._dev_gen.manual_seed(self.seed)

    # ───────────────────────── helpers ─────────────────────────
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
            return 0.0
        for el, frac in comp.items():
            try:
                v = float(prop_dict.get(el, 0.0))
                s += float(frac) * v
            except Exception:
                pass
        return float(s)

    @staticmethod
    def _row_composition(row):
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

    def _indices_with_element(self, idxs: np.ndarray, element: str) -> np.ndarray:
        element = str(element)
        keep = []
        # use Composition dict (already computed) => safest
        comps = self.df.loc[idxs, "Composition"].values
        for k, comp in zip(idxs, comps):
            try:
                if isinstance(comp, dict) and float(comp.get(element, 0.0)) > 0.0:
                    keep.append(int(k))
            except Exception:
                pass
        return np.array(keep, dtype=int)

    # ───────────────────────── training ─────────────────────────
    def train(
        self,
        batch_size: int = 128,
        num_epochs: int = 400,
        lr: float = 3e-4,               # LOWER default (stability)
        weight_decay: float = 1e-4,
        beta_kl: float = 2e-3,
        lambda_phys: float = 0.05,
        kl_warmup_epochs: int = 50,     # slower warmup is gentler
        phys_warmup_epochs: int = 50,
        patience_limit: int = 60,
        temp_range: Tuple[float, float] = (500.0, 1200.0),
        grad_clip: float = 1.0,
        min_lr: float = 1e-5,
        lr_patience: int = 10,
        lr_factor: float = 0.5,
    ):
        train_loader = DataLoader(
            Subset(self.dataset, self.tr_idx),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            generator=self._cpu_gen,
            num_workers=0,
        )
        val_loader = DataLoader(
            Subset(self.dataset, self.va_idx),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

        opt = torch.optim.AdamW(self.model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=float(lr_factor), patience=int(lr_patience), min_lr=float(min_lr)
        )

        y_mean_t = torch.tensor(self.y_mean, device=self.device, dtype=torch.float32)
        y_std_t = torch.tensor(self.y_std, device=self.device, dtype=torch.float32)
        scale_t = torch.tensor(self.scale_factors, device=self.device, dtype=torch.float32)
        w_t = torch.tensor(self.target_weights, device=self.device, dtype=torch.float32)

        best_val = float("inf")
        best_state = None
        patience = 0

        for epoch in range(1, int(num_epochs) + 1):
            kl_factor = float(beta_kl) * min(1.0, epoch / float(kl_warmup_epochs))
            phys_factor = float(lambda_phys) * min(1.0, epoch / float(phys_warmup_epochs))

            # ---- train ----
            self.model.train()
            tr_tot = tr_rec = tr_kl = tr_phys = 0.0
            nb = 0
            skipped = 0

            for xb, yb, mb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                mb = mb.to(self.device)

                opt.zero_grad(set_to_none=True)

                mu_q, logvar_q = self.model.encode_posterior(xb, yb, mb)
                mu_p, logvar_p = self.model.encode_prior(xb)

                z = self.model.reparameterize(mu_q, logvar_q)
                y_mu, y_logvar = self.model.decode(xb, z)

                loss_rec = masked_gaussian_nll(y_mu, y_logvar, yb, mb, w=w_t)
                loss_kl = kl_diag_gauss_gauss(mu_q, logvar_q, mu_p, logvar_p)

                y_true_scaled = yb * y_std_t + y_mean_t
                y_pred_scaled = y_mu * y_std_t + y_mean_t
                y_true_phys = y_true_scaled * scale_t
                y_pred_phys = y_pred_scaled * scale_t

                loss_phys = physics_loss_relative(
                    y_pred_phys=y_pred_phys,
                    y_true_phys=y_true_phys,
                    mask=mb,
                    target_index=self.target_index,
                    derived_groups=self.derived_props,
                    enabled=self.physics_enabled,
                    temp_range=temp_range,
                    rng=self._dev_gen,
                )

                loss = loss_rec + kl_factor * loss_kl + phys_factor * loss_phys

                # if loss is not finite, skip update (prevents NaN cascade)
                if not torch.isfinite(loss):
                    skipped += 1
                    continue

                loss.backward()
                if grad_clip is not None and float(grad_clip) > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), float(grad_clip))
                opt.step()

                tr_tot += float(loss.item())
                tr_rec += float(loss_rec.item())
                tr_kl += float(loss_kl.item())
                tr_phys += float(loss_phys.item())
                nb += 1

            tr_tot /= max(1, nb)
            tr_rec /= max(1, nb)
            tr_kl /= max(1, nb)
            tr_phys /= max(1, nb)

            # ---- val ----
            self.model.eval()
            va_tot = va_rec = va_kl = va_phys = 0.0
            nbv = 0
            with torch.no_grad():
                for xb, yb, mb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    mb = mb.to(self.device)

                    mu_q, logvar_q = self.model.encode_posterior(xb, yb, mb)
                    mu_p, logvar_p = self.model.encode_prior(xb)

                    z = mu_q  # deterministic eval path
                    y_mu, y_logvar = self.model.decode(xb, z)

                    loss_rec = masked_gaussian_nll(y_mu, y_logvar, yb, mb, w=w_t)
                    loss_kl = kl_diag_gauss_gauss(mu_q, logvar_q, mu_p, logvar_p)

                    y_true_scaled = yb * y_std_t + y_mean_t
                    y_pred_scaled = y_mu * y_std_t + y_mean_t
                    y_true_phys = y_true_scaled * scale_t
                    y_pred_phys = y_pred_scaled * scale_t

                    loss_phys = physics_loss_relative(
                        y_pred_phys=y_pred_phys,
                        y_true_phys=y_true_phys,
                        mask=mb,
                        target_index=self.target_index,
                        derived_groups=self.derived_props,
                        enabled=self.physics_enabled,
                        temp_range=temp_range,
                        rng=self._dev_gen,
                    )

                    loss = loss_rec + kl_factor * loss_kl + phys_factor * loss_phys
                    if not torch.isfinite(loss):
                        continue

                    va_tot += float(loss.item())
                    va_rec += float(loss_rec.item())
                    va_kl += float(loss_kl.item())
                    va_phys += float(loss_phys.item())
                    nbv += 1

            va_tot /= max(1, nbv)
            va_rec /= max(1, nbv)
            va_kl /= max(1, nbv)
            va_phys /= max(1, nbv)

            # LR schedule on validation loss
            scheduler.step(va_tot)
            cur_lr = opt.param_groups[0]["lr"]

            print(
                f"Epoch {epoch:03d} | "
                f"train total={tr_tot:.4f} (rec={tr_rec:.4f}, kl={tr_kl:.4f}, phys={tr_phys:.4f}) | "
                f"val total={va_tot:.4f} (rec={va_rec:.4f}, kl={va_kl:.4f}, phys={va_phys:.4f}) | "
                f"lr={cur_lr:.2e} | skipped={skipped}"
            )

            if va_tot < best_val - 1e-4:
                best_val = va_tot
                best_state = copy.deepcopy(self.model.state_dict())
                patience = 0
                self._save_checkpoint()
            else:
                patience += 1
                if patience >= int(patience_limit):
                    print("⇢ Early stopping")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self._save_checkpoint()

    # ───────────────────────── inference ─────────────────────────
    @torch.no_grad()
    def predict_batch(self, X_input: np.ndarray) -> np.ndarray:
        """
        Predict in physical units for a batch X_input (already embedded).
        Uses prior p(z|X), deterministic z=mu_prior.
        Returns: (N, P) physical units.
        """
        self.model.eval()
        X_input = np.nan_to_num(X_input, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        xb = torch.tensor(X_input, device=self.device, dtype=torch.float32)

        mu_p, _ = self.model.encode_prior(xb)
        z = mu_p
        y_mu, _ = self.model.decode(xb, z)

        y_mean_t = torch.tensor(self.y_mean, device=self.device, dtype=torch.float32)
        y_std_t = torch.tensor(self.y_std, device=self.device, dtype=torch.float32)
        scale_t = torch.tensor(self.scale_factors, device=self.device, dtype=torch.float32)

        y_scaled = y_mu * y_std_t + y_mean_t
        y_phys = y_scaled * scale_t
        return y_phys.detach().cpu().numpy()

    def predict(self, composition: Dict[str, float]) -> Dict[str, float]:
        """
        Predict coefficients in physical units for a single composition dict.
        """
        elements = {}
        for key, value in composition.items():
            parsed = self.parse_compound(key)
            for el, count in parsed.items():
                elements[el] = elements.get(el, 0.0) + float(value) * float(count)

        total = sum(elements.values())
        if total <= 0:
            raise ValueError("Composition must have positive total.")
        normalized = {k: v / total for k, v in elements.items()}

        frac = np.zeros(len(self.X_comp.columns), dtype=np.float32)
        for i, col in enumerate(self.X_comp.columns):
            frac[i] = float(normalized.get(col, 0.0))

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

        feats_emb = self.embedder.transform(feats).astype(np.float32)
        pred = self.predict_batch(feats_emb)[0]
        return {t: float(pred[j]) for j, t in enumerate(self.present_targets)}

    # ───────────────────────── derived props ─────────────────────────
    def derived(self, coeffs: Dict[str, float], T: float) -> Dict[str, float]:
        out = {}
        if {"rho_a", "rho_b"}.issubset(coeffs):
            out["rho"] = coeffs["rho_a"] - coeffs["rho_b"] * T
        if {"k_a", "k_b"}.issubset(coeffs):
            out["k"] = coeffs["k_a"] + coeffs["k_b"] * T
        if {"cp_a", "cp_b", "cp_c"}.issubset(coeffs):
            out["cp"] = coeffs["cp_a"] + coeffs["cp_b"] * T + coeffs["cp_c"] / (T ** 2)
        if {"mu1_a", "mu1_b"}.issubset(coeffs):
            arg = coeffs["mu1_b"] / (R_GAS * T)
            arg = float(np.clip(arg, -50.0, 50.0))
            out["muA"] = coeffs["mu1_a"] * math.exp(arg)
        if {"mu2_a", "mu2_b", "mu2_c"}.issubset(coeffs):
            expv = coeffs["mu2_a"] + coeffs["mu2_b"] / T + coeffs["mu2_c"] / (T ** 2)
            expv = float(np.clip(expv, -20.0, 20.0))
            out["muB"] = 10 ** expv
        return out

    # ───────────────────────── evaluation ─────────────────────────
    def _true_phys_matrix(self, idxs: np.ndarray) -> np.ndarray:
        Y_true_scaled = self.y_raw_scaled[idxs]  # scaled may have NaN
        Y_true_phys = Y_true_scaled.copy()
        for j in range(len(self.present_targets)):
            Y_true_phys[:, j] = Y_true_phys[:, j] * self.scale_factors[j]
        return Y_true_phys

    def evaluate(self, split: str = "val", return_dict: bool = False) -> Optional[Dict[str, Any]]:
        split_map = {"train": self.tr_idx, "val": self.va_idx, "test": self.te_idx}
        idxs = split_map[split]

        Xs = self.X_embedded[idxs]
        M = self.mask_all[idxs].astype(bool)

        Y_true_phys = self._true_phys_matrix(idxs)
        Y_pred_phys = self.predict_batch(Xs)

        per_target = {}
        rels, r2s = [], []
        for j, t in enumerate(self.present_targets):
            m = M[:, j]
            if not np.any(m):
                continue
            yt = Y_true_phys[m, j].astype(float)
            yp = Y_pred_phys[m, j].astype(float)
            rel = _rel_mse_pct(yt, yp)
            r2 = _r2_score_np(yt, yp)
            per_target[t] = {"MSE_pct": float(rel), "R2": float(r2), "N": int(np.sum(m))}
            if not np.isnan(rel):
                rels.append(rel)
            r2s.append(r2)

        out = {
            "avg_mse_pct": float(np.mean(rels)) if rels else float("nan"),
            "avg_r2": float(np.nanmean(r2s)) if r2s else float("nan"),
            "per_target": per_target,
            "split": split,
            "n_rows": int(len(idxs)),
        }

        print(f"\n[{split.upper()}] avg MSE%={out['avg_mse_pct']:.3f} | avg R2={out['avg_r2']:.3f} | N={out['n_rows']}")
        return out if return_dict else None

    def evaluate_by_element(
        self,
        split: str,
        element: str,
        return_dict: bool = True,
    ) -> Dict[str, Any]:
        split_map = {"train": self.tr_idx, "val": self.va_idx, "test": self.te_idx}
        base = split_map[split]
        idxs = self._indices_with_element(base, element=str(element))

        if idxs.size == 0:
            return {
                "split": split, "element": element, "n_rows": 0,
                "avg_mse_pct": float("nan"), "avg_r2": float("nan"),
                "per_target": {}
            }

        Xs = self.X_embedded[idxs]
        M = self.mask_all[idxs].astype(bool)

        Y_true_phys = self._true_phys_matrix(idxs)
        Y_pred_phys = self.predict_batch(Xs)

        per_target = {}
        rels, r2s = [], []
        for j, t in enumerate(self.present_targets):
            m = M[:, j]
            if not np.any(m):
                continue
            yt = Y_true_phys[m, j].astype(float)
            yp = Y_pred_phys[m, j].astype(float)
            rel = _rel_mse_pct(yt, yp)
            r2 = _r2_score_np(yt, yp)
            per_target[t] = {"MSE_pct": float(rel), "R2": float(r2), "N": int(np.sum(m))}
            if not np.isnan(rel):
                rels.append(rel)
            r2s.append(r2)

        out = {
            "split": split,
            "element": element,
            "n_rows": int(len(idxs)),
            "avg_mse_pct": float(np.mean(rels)) if rels else float("nan"),
            "avg_r2": float(np.nanmean(r2s)) if r2s else float("nan"),
            "per_target": per_target,
        }
        return out

    # ───────────────────────── plots ─────────────────────────
    def make_plots(
        self,
        plot_dir: str,
        temperature: float = 900.0,
        splits: Tuple[str, ...] = ("train", "test"),
        derived_to_compare: Optional[List[str]] = None,
    ):
        """
        Saves per-target coefficient plots and derived-property plots:
          - actual_vs_predicted_coeff_<target>_<split>.png
          - actual_vs_predicted_property_<prop>_<split>.png
        """
        import matplotlib.pyplot as plt

        os.makedirs(plot_dir, exist_ok=True)
        if derived_to_compare is None:
            derived_to_compare = ["rho", "k"]

        split_map = {"train": self.tr_idx, "val": self.va_idx, "test": self.te_idx}

        for split_name in splits:
            idxs = split_map[split_name]
            Xs = self.X_embedded[idxs]
            M = self.mask_all[idxs].astype(bool)

            y_true = self._true_phys_matrix(idxs)
            y_pred = self.predict_batch(Xs)

            # ---- coefficient plots ----
            for j, target in enumerate(self.present_targets):
                m = M[:, j]
                if not np.any(m):
                    continue
                yt = y_true[m, j]
                yp = y_pred[m, j]
                if yt.size < 5:
                    continue

                plt.figure(figsize=(6, 6))
                plt.scatter(yt, yp, alpha=0.7)
                mn = float(min(np.min(yt), np.min(yp)))
                mx = float(max(np.max(yt), np.max(yp)))
                plt.plot([mn, mx], [mn, mx], "k--", linewidth=1.0)
                plt.title(f"{target} ({split_name} set)")
                plt.xlabel("Actual")
                plt.ylabel("Predicted")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                fname = f"actual_vs_predicted_coeff_{target}_{split_name}.png"
                plt.savefig(os.path.join(plot_dir, fname), dpi=160)
                plt.close()

            # ---- derived property plots ----
            actual_vals = {p: [] for p in derived_to_compare}
            pred_vals = {p: [] for p in derived_to_compare}

            for row_i in range(len(idxs)):
                coeff_true = {t: float(y_true[row_i, jj]) for jj, t in enumerate(self.present_targets)}
                coeff_pred = {t: float(y_pred[row_i, jj]) for jj, t in enumerate(self.present_targets)}

                a_props = self.derived(coeff_true, temperature)
                p_props = self.derived(coeff_pred, temperature)

                for p in derived_to_compare:
                    av = a_props.get(p, None)
                    pv = p_props.get(p, None)
                    if av is None or pv is None:
                        continue
                    if not np.isfinite(av) or not np.isfinite(pv):
                        continue
                    if abs(av) < 1e-12:
                        continue
                    actual_vals[p].append(av)
                    pred_vals[p].append(pv)

            for p in derived_to_compare:
                if len(actual_vals[p]) < 10:
                    continue
                yt = np.asarray(actual_vals[p], dtype=float)
                yp = np.asarray(pred_vals[p], dtype=float)

                plt.figure(figsize=(6, 6))
                plt.scatter(yt, yp, alpha=0.7)
                mn = float(min(np.min(yt), np.min(yp)))
                mx = float(max(np.max(yt), np.max(yp)))
                plt.plot([mn, mx], [mn, mx], "k--", linewidth=1.0)
                plt.title(f"{p} at {temperature:.0f} K ({split_name} set)")
                plt.xlabel("Actual")
                plt.ylabel("Predicted")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                fname = f"actual_vs_predicted_property_{p}_{split_name}.png"
                plt.savefig(os.path.join(plot_dir, fname), dpi=160)
                plt.close()

        print(f"\nAll plots saved in: {plot_dir}")

    # ───────────────────────── metrics saving ─────────────────────────
    @staticmethod
    def _format_metrics_block(title: str, metrics: Dict[str, Any]) -> str:
        lines = []
        lines.append(title)
        lines.append("-" * len(title))
        lines.append(f"split={metrics.get('split')} element={metrics.get('element','ALL')} n_rows={metrics.get('n_rows')}")
        lines.append(f"avg MSE%={metrics.get('avg_mse_pct'):.4f} | avg R2={metrics.get('avg_r2'):.4f}")
        lines.append("")
        lines.append("Per-target:")
        for t, d in metrics.get("per_target", {}).items():
            lines.append(f"  {t:10s} | MSE%={d['MSE_pct']:10.4f} | R2={d['R2']:+8.4f} | N={d.get('N',0)}")
        lines.append("")
        return "\n".join(lines)

    def save_metrics_text(
        self,
        out_path: str,
        metrics_main: Dict[str, Any],
        metrics_by_element: Optional[List[Dict[str, Any]]] = None,
        header: Optional[str] = None,
    ):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        blocks = []
        if header:
            blocks.append(header)
            blocks.append("")
        blocks.append(self._format_metrics_block("Overall metrics", metrics_main))
        if metrics_by_element:
            for m in metrics_by_element:
                blocks.append(self._format_metrics_block(f"Element-filter metrics ({m.get('element')})", m))
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(blocks))

    # ───────────────────────── persistence ─────────────────────────
    def _save_checkpoint(self):
        ckpt = {
            "model_state": self.model.state_dict(),
            "present_targets": self.present_targets,
            "x_comp_cols": self.X_comp.columns.tolist(),
            "scale_factors": self.scale_factors,
            "y_mean": self.y_mean,
            "y_std": self.y_std,
            "target_weights": self.target_weights,
            "use_element_features": self.use_element_features,
            "element_feature_cols": self.ELEMENT_FEATURE_COLS,
            "physics_enabled": self.physics_enabled,
            "embedding_method": self.embedding_method,
            "n_components": self.n_components,
            "embedder": self.embedder,
            "poly": self.poly,
            "poly_scaler": self.poly_scaler,
            "elem_scaler": self.elem_scaler,
            "elem_lookup": self.elem_lookup,
            "seed": self.seed,
        }
        torch.save(ckpt, self.model_dir / "pinn_checkpoint.pt")

    def load(self, path: Optional[str] = None):
        p = Path(path) if path is not None else (self.model_dir / "pinn_checkpoint.pt")
        ckpt = torch.load(p, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device)
        self.model.eval()
