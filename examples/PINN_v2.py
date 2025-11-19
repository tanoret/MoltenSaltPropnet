"""
Physics-informed VAE with 5-fold cross-validation.

Key features:
    - Physics loss only on "safe" groups (rho, k by default)
    - KL and physics warmup over a configurable number of epochs
    - Fully deterministic (seeded) training and CV
    - relMSE [%] and R² per target for Train / Val / Test
    - 5-fold CV with inner train/val split and early stopping
"""

from typing import Dict, List, Tuple, Any

import random
import copy

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.model_selection import KFold, train_test_split

#

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_global_seed(seed: int = 42):
    """
    Set seeds and deterministic flags to make training as reproducible as possible.
    """
    global SEED
    SEED = seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Make cuDNN deterministic (may reduce speed but improves reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For PyTorch 2.0+ deterministic algorithms
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


# ================================================================
#  Targets and physics configuration
# ================================================================

TARGET_COLUMNS: List[str] = [
    "Melt(K)", "Boil(K)",
    "rho_a", "rho_b",
    "mu1_a", "mu1_b",
    "mu2_a", "mu2_b", "mu2_c",
    "k_a",  "k_b",
    "cp_a", "cp_b", "cp_c",
]

# Groups of coefficients that define derived physical properties
DERIVED_GROUPS: List[Tuple[str, List[str]]] = [
    ("rho", ["rho_a", "rho_b"]),
    ("muA", ["mu1_a", "mu1_b"]),
    ("muB", ["mu2_a", "mu2_b", "mu2_c"]),
    ("k",   ["k_a", "k_b"]),
    ("cp",  ["cp_a", "cp_b", "cp_c"]),
]

R_GAS = 8.314  # J/(mol*K)

# Columns that need numeric scaling (to avoid exponentials blowing up)
SCALE_MAP: Dict[str, float] = {
    "mu1_b": 1000.0,
    "mu2_b": 100.0,
    "mu2_c": 100.0,
    "cp_b":  100.0,
}

# We only enable physics where the relationships are simple and stable
PHYSICS_ENABLED: Dict[str, bool] = {
    "rho": True,
    "muA": False,   # exponential, disabled by default in this stable version
    "muB": False,   # 10^x, also disabled by default
    "k":  True,
    "cp": False,
}


# ================================================================
#  Data utilities
# ================================================================

def extract_raw_targets_with_scaling(
    df: pd.DataFrame,
    target_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a raw (possibly NaN) target matrix Y_raw with per-column scaling.

    Y_raw[i,j] is the scaled value for sample i, target j:
        - strings like "----", "" are treated as NaN
        - non-numeric junk is coerced to NaN
        - each column j is divided by SCALE_MAP.get(col, 1.0)

    Returns:
        Y_raw        : (N,P) float32, scaled but not standardised, may contain NaNs
        mask         : (N,P) bool indicating finite entries
        scale_factors: (P,) float32, the divisors applied per column
    """
    N = len(df)
    P = len(target_cols)

    Y_raw = np.full((N, P), np.nan, dtype=np.float32)
    scale_factors = np.ones(P, dtype=np.float32)

    for j, col in enumerate(target_cols):
        if col not in df.columns:
            continue

        series = (
            df[col]
            .replace(["", "----"], np.nan)
            .replace(r"\*", "", regex=True)
        )
        vals = pd.to_numeric(series, errors="coerce").to_numpy(np.float32)

        s = SCALE_MAP.get(col, 1.0)
        scale_factors[j] = float(s)
        if s != 1.0:
            vals = vals / s

        Y_raw[:, j] = vals

    mask = np.isfinite(Y_raw)
    return Y_raw, mask, scale_factors.astype(np.float32)


def standardise_with_train_indices(
    Y_raw: np.ndarray,
    mask: np.ndarray,
    train_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardise each target using mean/std computed from training indices only.

    This is important for cross-validation: statistics must be derived from
    the training set to avoid information leak.

    Returns:
        Y_std   : (N,P) float32; missing entries filled with 0.0
        y_mean  : (P,) float32
        y_std   : (P,) float32 (no zeros; we replace 0 with 1.0)
    """
    N, P = Y_raw.shape
    y_mean = np.zeros(P, dtype=np.float32)
    y_std = np.ones(P, dtype=np.float32)

    for j in range(P):
        # Use only training rows where the value is finite
        valid_train = mask[train_indices, j]
        if not np.any(valid_train):
            # no training data for this target; leave mean=0, std=1
            continue
        col_train = Y_raw[train_indices[valid_train], j]
        m = col_train.mean()
        s = col_train.std()
        y_mean[j] = m
        y_std[j] = s if s > 0 else 1.0

    Y_std = (Y_raw - y_mean[None, :]) / y_std[None, :]
    Y_std[~mask] = 0.0

    return Y_std.astype(np.float32), y_mean, y_std


class PhysicsVAEDataset(Dataset):
    """
    Thin Dataset wrapper to hold:
        - X: composition features (N, D)
        - Y: standardised targets (N, P)
        - M: mask of observed targets (N, P), as float (0 or 1)
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, M: np.ndarray):
        assert X.shape[0] == Y.shape[0] == M.shape[0]
        self.X = torch.from_numpy(X.astype(np.float32))
        self.Y = torch.from_numpy(Y.astype(np.float32))
        self.M = torch.from_numpy(M.astype(np.float32))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx], self.M[idx]


# ================================================================
#  Model definition
# ================================================================

class MLP(nn.Module):
    """
    Simple MLP with GELU activations; used for encoder and decoder.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int],
        out_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PhysicsVAE(nn.Module):
    """
    Conditional Variational Autoencoder:

        Encoder input: [X_comp, Y_std, M]
        Decoder input: [z, X_comp]

    X_comp is the composition encoding (your element fractions).
    Y_std is the standardised target vector.
    M is the mask (1 where Y is observed, 0 where missing).
    """

    def __init__(
        self,
        comp_dim: int,
        target_dim: int,
        latent_dim: int = 16,
        hidden_dim: int = 256,
        num_hidden_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.comp_dim = comp_dim
        self.target_dim = target_dim
        self.latent_dim = latent_dim

        enc_in_dim = comp_dim + target_dim + target_dim
        enc_hidden = [hidden_dim] * num_hidden_layers
        self.encoder = MLP(enc_in_dim, enc_hidden, out_dim=2 * latent_dim, dropout=dropout)

        dec_in_dim = latent_dim + comp_dim
        dec_hidden = [hidden_dim] * num_hidden_layers
        self.decoder = MLP(dec_in_dim, dec_hidden, out_dim=target_dim, dropout=dropout)

    def encode(self, x_comp: torch.Tensor, y_std: torch.Tensor, mask: torch.Tensor):
        enc_in = torch.cat([x_comp, y_std, mask], dim=1)
        h = self.encoder(enc_in)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, x_comp: torch.Tensor) -> torch.Tensor:
        dec_in = torch.cat([z, x_comp], dim=1)
        return self.decoder(dec_in)

    def forward(self, x_comp: torch.Tensor, y_std: torch.Tensor, mask: torch.Tensor):
        mu, logvar = self.encode(x_comp, y_std, mask)
        z = self.reparameterize(mu, logvar)
        y_pred_std = self.decode(z, x_comp)
        return y_pred_std, mu, logvar


# ================================================================
#  Losses
# ================================================================

def masked_mse(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Mean squared error averaged over observed entries only (mask==1).
    """
    diff2 = (y_pred - y_true) ** 2 * mask
    denom = mask.sum()
    if denom <= 0:
        return torch.tensor(0.0, device=y_pred.device)
    return diff2.sum() / denom


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    KL divergence between N(mu, sigma^2) and N(0, I) per sample, averaged over batch.
    """
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl.mean()


def build_target_index_map(target_columns: List[str]) -> Dict[str, int]:
    return {name: j for j, name in enumerate(target_columns)}


def physics_loss(
    y_pred_phys: torch.Tensor,
    y_true_phys: torch.Tensor,
    mask: torch.Tensor,
    target_index: Dict[str, int],
    derived_groups: List[Tuple[str, List[str]]],
    temp_range: Tuple[float, float] = (500.0, 1200.0),
    enabled: Dict[str, bool] = PHYSICS_ENABLED,
) -> torch.Tensor:
    """
    Physics-based regularisation on derived quantities.
    Only 'rho' and 'k' are enabled by default in this stable version.

    For each enabled group:
        - Identify samples where all required coefficients are observed.
        - Compute derived value from true and predicted coefficients at a
          random temperature T in [temp_range[0], temp_range[1]].
        - Penalise MSE of the derived values.
    """
    device = y_pred_phys.device
    batch_size = y_pred_phys.shape[0]
    T = torch.rand(batch_size, device=device) * (temp_range[1] - temp_range[0]) + temp_range[0]

    total = 0.0
    n_groups = 0

    for tag, coeff_names in derived_groups:
        if not enabled.get(tag, False):
            continue

        idxs = [target_index[c] for c in coeff_names if c in target_index]
        if len(idxs) != len(coeff_names):
            continue

        idxs_t = torch.tensor(idxs, device=device, dtype=torch.long)
        mask_group = mask[:, idxs_t].bool().all(dim=1)
        if not mask_group.any():
            continue

        y_true_g = y_true_phys[mask_group][:, idxs_t]
        y_pred_g = y_pred_phys[mask_group][:, idxs_t]
        T_g = T[mask_group]

        if tag == "rho":
            rho_a_true, rho_b_true = y_true_g[:, 0], y_true_g[:, 1]
            rho_a_pred, rho_b_pred = y_pred_g[:, 0], y_pred_g[:, 1]
            rho_true = rho_a_true - rho_b_true * T_g
            rho_pred = rho_a_pred - rho_b_pred * T_g
            loss_g = torch.mean((rho_pred - rho_true) ** 2)

        elif tag == "k":
            k_a_true, k_b_true = y_true_g[:, 0], y_true_g[:, 1]
            k_a_pred, k_b_pred = y_pred_g[:, 0], y_pred_g[:, 1]
            k_true = k_a_true + k_b_true * T_g
            k_pred = k_a_pred + k_b_pred * T_g
            loss_g = torch.mean((k_pred - k_true) ** 2)

        else:
            continue

        total = total + loss_g
        n_groups += 1

    if n_groups == 0:
        return torch.tensor(0.0, device=device)

    loss_out = total / n_groups
    # Optional clamp to avoid rare physics spikes dominating
    return torch.clamp(loss_out, max=50.0)


# ================================================================
#  Metrics and evaluation
# ================================================================

def _rel_mse_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Relative MSE (percent of mean(y_true^2)).
    """
    if y_true.size == 0:
        return float("nan")
    mse = np.mean((y_pred - y_true) ** 2)
    denom = np.mean(y_true ** 2)
    if denom <= 0:
        return float("nan")
    return 100.0 * mse / denom


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Simple R² = 1 - SS_res / SS_tot.
    """
    if y_true.size == 0:
        return float("nan")
    y_mean = y_true.mean()
    ss_res = np.sum((y_pred - y_true) ** 2)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


@torch.no_grad()
def evaluate_split(
    model: PhysicsVAE,
    dataset: PhysicsVAEDataset,
    indices: np.ndarray,
    stats: Dict[str, Any],
    batch_size: int = 256,
    device: str = DEVICE,
) -> Dict[str, Dict[str, float]]:
    """
    Compute relMSE [%] and R² per target for a given subset of indices.
    All metrics are computed in physical units.
    """
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size,
                        shuffle=False, drop_last=False)

    y_mean = stats["y_mean"]
    y_std = stats["y_std"]
    scale_factors = stats["scale_factors"]
    target_cols = stats["target_cols"]

    y_mean_t = torch.from_numpy(y_mean).to(device=device, dtype=torch.float32)
    y_std_t  = torch.from_numpy(y_std).to(device=device, dtype=torch.float32)
    scale_t  = torch.from_numpy(scale_factors).to(device=device, dtype=torch.float32)

    model.eval()

    P = len(target_cols)
    per_target_true: List[List[float]] = [[] for _ in range(P)]
    per_target_pred: List[List[float]] = [[] for _ in range(P)]

    for x_b, y_std_b, m_b in loader:
        x_b = x_b.to(device)
        y_std_b = y_std_b.to(device)
        m_b = m_b.to(device)

        y_pred_std_b, _, _ = model(x_b, y_std_b, m_b)

        y_true_scaled = y_std_b * y_std_t + y_mean_t
        y_pred_scaled = y_pred_std_b * y_std_t + y_mean_t
        y_true_phys = y_true_scaled * scale_t
        y_pred_phys = y_pred_scaled * scale_t

        y_true_np = y_true_phys.cpu().numpy()
        y_pred_np = y_pred_phys.cpu().numpy()
        m_np = m_b.cpu().numpy().astype(bool)

        for j in range(P):
            mj = m_np[:, j]
            if not np.any(mj):
                continue
            per_target_true[j].extend(y_true_np[mj, j].tolist())
            per_target_pred[j].extend(y_pred_np[mj, j].tolist())

    results: Dict[str, Dict[str, float]] = {}
    for j, name in enumerate(target_cols):
        y_t = np.asarray(per_target_true[j], dtype=float)
        y_p = np.asarray(per_target_pred[j], dtype=float)
        if y_t.size == 0:
            rel = float("nan")
            r2 = float("nan")
        else:
            rel = _rel_mse_pct(y_t, y_p)
            r2 = _r2_score(y_t, y_p)
        results[name] = {"relMSE": float(rel), "R2": float(r2)}

    return results


def summarize_results(res_dict: Dict[str, Dict[str, float]], name: str):
    """
    Pretty-print per-target metrics + average summary for a single split.
    """
    rels = []
    r2s = []

    print(f"\n{name} split — per-target PI-VAE metrics")
    for t, d in res_dict.items():
        m_rel = d["relMSE"]
        r2    = d["R2"]
        rel_str = "nan" if np.isnan(m_rel) else f"{m_rel:8.2f}%"
        r2_str  = "nan" if np.isnan(r2)    else f"{r2:+.3f}"
        print(f"  {t:8s}: relMSE={rel_str}   R²={r2_str}")
        if not np.isnan(m_rel):
            rels.append(m_rel)
        r2s.append(r2)

    if rels:
        print(f"  ⇒ {name} avg : relMSE={np.mean(rels):8.2f}%   R²={np.nanmean(r2s):+.3f}")
    else:
        print(f"  ⇒ {name} avg : no finite metrics")


def summarize_cv_average(
    all_fold_metrics: List[Dict[str, Dict[str, Dict[str, float]]]],
    target_cols: List[str],
):
    """
    Aggregate metrics across folds and print 5-fold averages per split.
    all_fold_metrics is a list over folds, each item like:
        {"train": {...}, "val": {...}, "test": {...}}.
    """
    splits = ["train", "val", "test"]

    for split in splits:
        agg: Dict[str, Dict[str, float]] = {}
        for t in target_cols:
            rel_list = []
            r2_list = []
            for fold_m in all_fold_metrics:
                m = fold_m[split][t]
                rel_list.append(m["relMSE"])
                r2_list.append(m["R2"])
            rel_arr = np.asarray(rel_list, dtype=float)
            r2_arr  = np.asarray(r2_list, dtype=float)
            agg[t] = {
                "relMSE": float(np.nanmean(rel_arr)),
                "R2": float(np.nanmean(r2_arr)),
            }

        print(f"\n===== {split} split — 5-fold average metrics =====")
        summarize_results(agg, f"{split} (5-fold avg)")


# ================================================================
#  Training with 5-fold CV
# ================================================================

def train_physics_vae_kfold(
    df: pd.DataFrame,
    X_comp: np.ndarray,
    target_cols: List[str] = TARGET_COLUMNS,
    n_splits: int = 5,
    batch_size: int = 128,
    latent_dim: int = 16,
    hidden_dim: int = 256,
    num_hidden_layers: int = 3,
    dropout: float = 0.0,
    num_epochs: int = 300,
    beta_kl: float = 1e-3,
    lambda_phys: float = 0.05,
    kl_warmup_epochs: int = 20,      # your choice B
    phys_warmup_epochs: int = 20,    # your choice B
    learning_rate: float = 1e-3,
    patience_limit: int = 30,
    seed: int = SEED,
    device: str = DEVICE,
):
    """
    5-fold cross-validation for the Physics-informed VAE.

    For each fold:
        - Use KFold to define test indices.
        - Split the remaining indices into train/val (e.g. 80/20).
        - Standardise targets using *training* indices only.
        - Train VAE with KL and physics warmup, early stopping on val loss.
        - Evaluate Train / Val / Test metrics (relMSE, R²).
    """
    set_global_seed(seed)

    # Y_raw, mask, scale_factors do NOT depend on fold; only standardisation does.
    Y_raw, mask, scale_factors = extract_raw_targets_with_scaling(df, target_cols)
    N = len(df)
    dataset_dummy = None  # we will rebuild a dataset each fold (because Y_std changes)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_models: List[PhysicsVAE] = []
    fold_stats: List[Dict[str, Any]] = []
    fold_histories: List[Dict[str, List[float]]] = []
    fold_metrics: List[Dict[str, Dict[str, Dict[str, float]]]] = []
    fold_indices: List[Dict[str, np.ndarray]] = []

    target_index = build_target_index_map(target_cols)

    fold_id = 0
    for trainval_idx, test_idx in kf.split(np.arange(N)):
        fold_id += 1
        print("\n" + "=" * 70)
        print(f"Starting fold {fold_id}/{n_splits}")
        print("=" * 70)

        # Split trainval into train and val
        rng = np.random.RandomState(seed + fold_id)
        train_idx, val_idx = train_test_split(
            trainval_idx,
            test_size=0.20,
            random_state=rng,
        )

        # Standardise using *train_idx* only
        Y_std, y_mean, y_std = standardise_with_train_indices(Y_raw, mask, train_idx)

        # Dataset for this fold
        dataset = PhysicsVAEDataset(X_comp, Y_std, mask)

        # DataLoaders
        train_loader = DataLoader(Subset(dataset, train_idx),
                                  batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader   = DataLoader(Subset(dataset, val_idx),
                                  batch_size=batch_size, shuffle=False, drop_last=False)
        test_loader  = DataLoader(Subset(dataset, test_idx),
                                  batch_size=batch_size, shuffle=False, drop_last=False)

        comp_dim = X_comp.shape[1]
        target_dim = Y_std.shape[1]

        model = PhysicsVAE(
            comp_dim=comp_dim,
            target_dim=target_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            dropout=dropout,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

        # Stats as tensors for this fold
        y_mean_t = torch.from_numpy(y_mean).to(device=device, dtype=torch.float32)
        y_std_t  = torch.from_numpy(y_std).to(device=device, dtype=torch.float32)
        scale_t  = torch.from_numpy(scale_factors).to(device=device, dtype=torch.float32)

        # History for this fold
        history = {
            "epoch": [],
            "train_total": [],
            "train_recon": [],
            "train_kl": [],
            "train_phys": [],
            "val_total": [],
            "val_recon": [],
            "val_kl": [],
            "val_phys": [],
        }

        best_val = float("inf")
        best_state = None
        patience = 0

        for epoch in range(1, num_epochs + 1):
            # Warmup schedules for KL and physics
            kl_factor = beta_kl * min(1.0, epoch / float(kl_warmup_epochs))
            phys_factor = lambda_phys * min(1.0, epoch / float(phys_warmup_epochs))

            # -------------- Training --------------
            model.train()
            run_tot = run_rec = run_kl = run_phys = 0.0
            n_batches = 0

            for xb, yb, mb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                mb = mb.to(device)

                optimizer.zero_grad()

                y_pred_std, mu_b, logvar_b = model(xb, yb, mb)

                loss_recon = masked_mse(y_pred_std, yb, mb)
                loss_kl    = kl_divergence(mu_b, logvar_b)

                # back to physical units for physics regularisation
                y_true_scaled = yb  * y_std_t + y_mean_t
                y_pred_scaled = y_pred_std * y_std_t + y_mean_t
                y_true_phys   = y_true_scaled * scale_t
                y_pred_phys   = y_pred_scaled * scale_t

                loss_phys = physics_loss(
                    y_pred_phys=y_pred_phys,
                    y_true_phys=y_true_phys,
                    mask=mb,
                    target_index=target_index,
                    derived_groups=DERIVED_GROUPS,
                )

                loss = loss_recon + kl_factor * loss_kl + phys_factor * loss_phys
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                run_tot  += loss.item()
                run_rec  += loss_recon.item()
                run_kl   += loss_kl.item()
                run_phys += loss_phys.item()
                n_batches += 1

            train_total = run_tot / n_batches
            train_recon = run_rec / n_batches
            train_kl    = run_kl / n_batches
            train_phys  = run_phys / n_batches

            # -------------- Validation --------------
            model.eval()
            run_tot = run_rec = run_kl = run_phys = 0.0
            n_batches = 0

            with torch.no_grad():
                for xb, yb, mb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    mb = mb.to(device)

                    y_pred_std, mu_b, logvar_b = model(xb, yb, mb)

                    loss_recon = masked_mse(y_pred_std, yb, mb)
                    loss_kl    = kl_divergence(mu_b, logvar_b)

                    y_true_scaled = yb  * y_std_t + y_mean_t
                    y_pred_scaled = y_pred_std * y_std_t + y_mean_t
                    y_true_phys   = y_true_scaled * scale_t
                    y_pred_phys   = y_pred_scaled * scale_t

                    loss_phys = physics_loss(
                        y_pred_phys=y_pred_phys,
                        y_true_phys=y_true_phys,
                        mask=mb,
                        target_index=target_index,
                        derived_groups=DERIVED_GROUPS,
                    )

                    loss = loss_recon + kl_factor * loss_kl + phys_factor * loss_phys

                    run_tot  += loss.item()
                    run_rec  += loss_recon.item()
                    run_kl   += loss_kl.item()
                    run_phys += loss_phys.item()
                    n_batches += 1

            val_total = run_tot / n_batches
            val_recon = run_rec / n_batches
            val_kl    = run_kl / n_batches
            val_phys  = run_phys / n_batches

            history["epoch"].append(epoch)
            history["train_total"].append(train_total)
            history["train_recon"].append(train_recon)
            history["train_kl"].append(train_kl)
            history["train_phys"].append(train_phys)
            history["val_total"].append(val_total)
            history["val_recon"].append(val_recon)
            history["val_kl"].append(val_kl)
            history["val_phys"].append(val_phys)

            print(
                f"Fold {fold_id:2d} | Epoch {epoch:3d} | "
                f"train_total={train_total:.4f} (rec={train_recon:.4f}, KL={train_kl:.4f}, phys={train_phys:.4f}) | "
                f"val_total={val_total:.4f} (rec={val_recon:.4f}, KL={val_kl:.4f}, phys={val_phys:.4f})"
            )

            # Early stopping on total validation loss
            if val_total < best_val - 1e-4:
                best_val = val_total
                best_state = copy.deepcopy(model.state_dict())
                patience = 0
            else:
                patience += 1
                if patience >= patience_limit:
                    print(f" ⇢ Early stopping fold {fold_id} at epoch {epoch}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # Stats dict for this fold
        stats = {
            "y_mean": y_mean,
            "y_std": y_std,
            "scale_factors": scale_factors,
            "target_cols": target_cols,
        }

        # Evaluate metrics for this fold
        res_train = evaluate_split(model, dataset, train_idx, stats, device=device)
        res_val   = evaluate_split(model, dataset, val_idx,   stats, device=device)
        res_test  = evaluate_split(model, dataset, test_idx,  stats, device=device)

        print(f"\n=== Metrics for fold {fold_id} ===")
        summarize_results(res_train, "Train")
        summarize_results(res_val,   "Val")
        summarize_results(res_test,  "Test")

        fold_models.append(model)
        fold_stats.append(stats)
        fold_histories.append(history)
        fold_metrics.append({"train": res_train, "val": res_val, "test": res_test})
        fold_indices.append({"train": train_idx, "val": val_idx, "test": test_idx})

    # 5-fold average metrics
    summarize_cv_average(fold_metrics, target_cols)

    return fold_models, fold_stats, fold_histories, fold_metrics, fold_indices


# ================================================================
#  Imputation helper
# ================================================================

@torch.no_grad()
def impute_properties(
    model: PhysicsVAE,
    comp_vec: Dict[str, float],
    element_order: List[str],
    stats: Dict[str, Any],
    device: str = DEVICE,
) -> Dict[str, float]:
    """
    Use a trained PhysicsVAE to predict all target coefficients for a single composition.

    comp_vec: dict, e.g. {"Na":0.5, "Cl":0.5}
    element_order: the same element ordering used for X_comp during training.
    stats: dictionary with "y_mean", "y_std", "scale_factors", "target_cols".
    """
    model.eval()

    y_mean = torch.from_numpy(stats["y_mean"]).to(device=device, dtype=torch.float32)
    y_std  = torch.from_numpy(stats["y_std"]).to(device=device, dtype=torch.float32)
    scale  = torch.from_numpy(stats["scale_factors"]).to(device=device, dtype=torch.float32)
    target_cols = stats["target_cols"]

    x_arr = np.zeros(len(element_order), dtype=np.float32)
    for i, el in enumerate(element_order):
        x_arr[i] = float(comp_vec.get(el, 0.0))

    x_comp = torch.from_numpy(x_arr).unsqueeze(0).to(device)

    # For imputation we don't have targets or masks; they are just zeros.
    y_std_dummy = torch.zeros((1, len(target_cols)), device=device)
    mask_dummy  = torch.zeros_like(y_std_dummy)

    y_pred_std, _, _ = model(x_comp, y_std_dummy, mask_dummy)

    y_pred_scaled = y_pred_std * y_std + y_mean
    y_pred_phys   = (y_pred_scaled * scale).squeeze(0).cpu().numpy()

    return {name: float(val) for name, val in zip(target_cols, y_pred_phys)}
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from processing_mstdb.processor import MSTDBProcessor
set_global_seed(42)

processor = MSTDBProcessor.from_csv("/Users/meggie/Documents/MoltenSaltPropnet/data/new_mstdb_janz.csv")
processor.df.columns = processor.df.columns.str.strip()

# Build compositions exactly like your existing code
compositions = []
for _, row in processor.df.iterrows():
    comp = processor.compute_composition(row, composition_type="elements")
    compositions.append(comp)
processor.df["Composition"] = compositions

all_elements = sorted(processor.predefined_elements)
X_composition = np.zeros((len(processor.df), len(all_elements)), dtype=np.float32)
for idx, comp in enumerate(compositions):
    for el, frac in comp.items():
        if el in all_elements:
            X_composition[idx, all_elements.index(el)] = frac

df = processor.df

fold_models, fold_stats, fold_histories, fold_metrics, fold_indices = train_physics_vae_kfold(
    df=df,
    X_comp=X_composition,
)

# Example imputation for 50–50 NaCl using fold 0
preds = impute_properties(
    model=fold_models[0],
    comp_vec={'Na': 0.5, 'Cl': 0.5},
    element_order=all_elements,
    stats=fold_stats[0],
)
print(preds)
