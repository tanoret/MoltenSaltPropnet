import os
import sys
import math
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------------------------------------------------
#  Basic config
# ------------------------------------------------------------------
SEED = 42
R = 8.314
torch.manual_seed(SEED)
np.random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------
#  Imports from your project
# ------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from processing_mstdb.processor import MSTDBProcessor

# ------------------------------------------------------------------
#  Targets (same as SNN)
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
#  Metrics utility
# ------------------------------------------------------------------
def rel_mse_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Relative MSE [% of mean(y^2)], guarded against tiny denominators."""
    mse = mean_squared_error(y_true, y_pred)
    denom = np.mean(y_true ** 2)
    if denom < 1e-8:
        # if variance is effectively zero, relMSE is meaningless
        return float("nan")
    return 100.0 * mse / denom


# ------------------------------------------------------------------
#  FNO building blocks
# ------------------------------------------------------------------
class SpectralConv1d(nn.Module):
    """
    1D Fourier layer: performs convolution in Fourier space.
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # number of low-frequency modes to keep

        # Complex weights for Fourier modes: (out_channels, in_channels, modes)
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, modes, dtype=torch.cfloat)
            * (1.0 / math.sqrt(in_channels * max(modes, 1)))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, L)
        returns: (B, C_out, L)
        """
        B, C_in, L = x.shape
        # FFT along last dimension
        x_ft = torch.fft.rfft(x, dim=-1)  # (B, C_in, L_ft)
        L_ft = x_ft.shape[-1]
        m = min(self.modes, L_ft)

        out_ft = torch.zeros(
            B, self.out_channels, L_ft,
            device=x.device, dtype=torch.cfloat
        )

        if m > 0:
            out_ft[:, :, :m] = torch.einsum(
                "bcl, ocm -> bol",
                x_ft[:, :, :m],
                self.weight[:, :, :m]
            )

        # Inverse FFT
        x_out = torch.fft.irfft(out_ft, n=L, dim=-1)  # (B, C_out, L)
        return x_out


class FNO1dLayer(nn.Module):
    """SpectralConv1d + pointwise Conv1d + GELU."""

    def __init__(self, width: int, modes: int):
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.pointwise = nn.Conv1d(width, width, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.spectral(x)
        z = self.pointwise(x)
        return self.act(y + z)


class FNOModel(nn.Module):
    """
    1D FNO mapping composition signal (over elements) to coefficients.

    Input:  (B, L)          — composition fractions per element
    Output: (B, n_targets)  — standardised coefficients
    """

    def __init__(
        self,
        n_elements: int,
        n_targets: int,
        modes: int = 16,
        width: int = 64,
        n_layers: int = 4,
    ):
        super().__init__()
        self.n_elements = n_elements
        self.n_targets = n_targets
        self.width = width

        # Lift 1 channel → width channels
        self.input_proj = nn.Conv1d(1, width, kernel_size=1)

        self.layers = nn.ModuleList(
            [FNO1dLayer(width, modes) for _ in range(n_layers)]
        )

        # Project width channels → n_targets at each element position
        self.output_proj = nn.Conv1d(width, n_targets, kernel_size=1)

        # Optional small head after global pooling
        self.head = nn.Sequential(
            nn.Linear(n_targets, n_targets),
            nn.GELU(),
            nn.Linear(n_targets, n_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L) or (B, 1, L)
        returns: (B, n_targets)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, L)

        x = self.input_proj(x)  # (B, width, L)

        for layer in self.layers:
            x = layer(x)  # (B, width, L)

        x = self.output_proj(x)  # (B, n_targets, L)

        # Global average pool over element dimension
        x = x.mean(dim=-1)  # (B, n_targets)

        x = self.head(x)    # (B, n_targets)
        return x


# ------------------------------------------------------------------
#  Physics-based derived properties (helper)
# ------------------------------------------------------------------
def derived_properties(coeffs: Dict[str, float], T: float) -> Dict[str, float]:
    out = {}
    if {'rho_a', 'rho_b'}.issubset(coeffs):
        out['rho'] = coeffs['rho_a'] - coeffs['rho_b'] * T
    if {'mu1_a', 'mu1_b'}.issubset(coeffs):
        out['muA'] = coeffs['mu1_a'] * math.exp(coeffs['mu1_b'] / (R * T))
    if {'mu2_a', 'mu2_b', 'mu2_c'}.issubset(coeffs):
        out['muB'] = 10 ** (coeffs['mu2_a'] + coeffs['mu2_b'] / T + coeffs['mu2_c'] / T ** 2)
    if {'k_a', 'k_b'}.issubset(coeffs):
        out['k'] = coeffs['k_a'] + coeffs['k_b'] * T
    if {'cp_a', 'cp_b', 'cp_c'}.issubset(coeffs):
        out['cp'] = coeffs['cp_a'] + coeffs['cp_b'] * T + coeffs['cp_c'] / T ** 2
    return out


# ------------------------------------------------------------------
#  Physics loss like in your SNN trainer
# ------------------------------------------------------------------
def physics_loss(
    pred_raw: torch.Tensor,
    y_raw: torch.Tensor,
    mask_b: torch.Tensor,
    T: torch.Tensor,
    present_targets: List[str],
) -> torch.Tensor:
    idx_map = {name: i for i, name in enumerate(present_targets)}
    loss = 0.0
    terms = 0

    for dprop, coeffs in DERIVED_PROPS:
        idxs = [idx_map[c] for c in coeffs if c in idx_map]
        if len(idxs) != len(coeffs):
            continue

        m = torch.all(mask_b[:, idxs], dim=1)
        if not m.any():
            continue

        y = y_raw[m][:, idxs]
        p = pred_raw[m][:, idxs]

        if dprop == 'rho':
            loss_t = nn.functional.mse_loss(
                p[:, 0] - p[:, 1] * T[m],
                y[:, 0] - y[:, 1] * T[m],
            )
        elif dprop == 'muA':
            loss_t = nn.functional.mse_loss(
                torch.log(torch.clamp(p[:, 0], 1e-6) * torch.exp(p[:, 1] / (R * T[m]))),
                torch.log(torch.clamp(y[:, 0], 1e-6) * torch.exp(y[:, 1] / (R * T[m]))),
            )
        elif dprop == 'muB':
            loss_t = nn.functional.mse_loss(
                p[:, 0] + p[:, 1] / T[m] + p[:, 2] / T[m] ** 2,
                y[:, 0] + y[:, 1] / T[m] + y[:, 2] / T[m] ** 2,
            )
        elif dprop == 'k':
            loss_t = nn.functional.mse_loss(
                p[:, 0] + p[:, 1] * T[m],
                y[:, 0] + y[:, 1] * T[m],
            )
        elif dprop == 'cp':
            loss_t = nn.functional.mse_loss(
                p[:, 0] + p[:, 1] * T[m] + p[:, 2] / T[m] ** 2,
                y[:, 0] + y[:, 1] * T[m] + y[:, 2] / T[m] ** 2,
            )
        else:
            continue

        loss += loss_t
        terms += 1

    if terms == 0:
        return torch.tensor(0.0, device=pred_raw.device)
    return loss / terms


# ------------------------------------------------------------------
#  Helper: parse compound to elements
# ------------------------------------------------------------------
def parse_compound(c: str) -> Dict[str, int]:
    out = {}
    for el, n in re.findall(r"([A-Z][a-z]*)(\d*)", c):
        out[el] = out.get(el, 0) + int(n or "1")
    return out


# ==================================================================
#  MAIN SCRIPT
# ==================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------
    # 1. Load CSV via MSTDBProcessor (your usual start)
    # ------------------------------------------------------------
    processor = MSTDBProcessor.from_csv(
        "/Users/meggie/Documents/MoltenSaltPropnet/data/new_mstdb_janz.csv"
    )
    print(processor.df.head())
    processor.df.columns = processor.df.columns.str.strip()
    print(processor.df.columns)

    # ------------------------------------------------------------
    # 2. Compute compositions (elements) → X_composition
    # ------------------------------------------------------------
    compositions = []
    for idx, row in processor.df.iterrows():
        comp = processor.compute_composition(row, composition_type="elements")
        compositions.append(comp)
    processor.df["Composition"] = compositions

    all_elements = sorted(processor.predefined_elements)
    X_composition = np.zeros((len(processor.df), len(all_elements)), dtype=np.float32)
    for idx, comp in enumerate(compositions):
        for el, frac in comp.items():
            if el in all_elements:
                X_composition[idx, all_elements.index(el)] = frac

    # Ensure features have no NaNs
    X_composition = np.nan_to_num(X_composition, nan=0.0)

    # ------------------------------------------------------------
    # 3. Prepare targets + masks, handling NaNs carefully
    # ------------------------------------------------------------
    df = processor.df

    present_targets: List[str] = []
    cleaned_targets = {}
    for t in TARGETS:
        if t not in df.columns:
            continue
        col = (
            df[t]
            .replace(["----", ""], np.nan)
            .replace(r"\*", "", regex=True)
        )
        col = pd.to_numeric(col, errors="coerce")
        if np.isfinite(col).any():
            present_targets.append(t)
            cleaned_targets[t] = col

    if not present_targets:
        raise RuntimeError("No valid target columns found.")

    y_df = pd.DataFrame(cleaned_targets)
    mask_all = np.isfinite(y_df.to_numpy(np.float32))  # where data exists
    y_df_filled = y_df.fillna(0.0)
    y_raw = y_df_filled.to_numpy(np.float32)

    N = X_composition.shape[0]
    idx_all = np.arange(N)
    tr_idx, te_idx = train_test_split(idx_all, test_size=0.20, random_state=SEED)
    tr_idx, va_idx = train_test_split(tr_idx, test_size=0.20, random_state=SEED)

    # Standardise targets on train only
    μ = y_raw[tr_idx].mean(axis=0)
    σ = y_raw[tr_idx].std(axis=0)
    σ[σ == 0] = 1.0
    y_std = (y_raw - μ) / σ

    idx_map = {name: i for i, name in enumerate(present_targets)}

    # ------------------------------------------------------------
    # 4. DataLoaders
    # ------------------------------------------------------------
    def make_loader(idxs, batch_size=64, shuffle=True):
        x = X_composition[idxs]
        y = y_std[idxs]
        m = mask_all[idxs]
        ds = TensorDataset(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(m, dtype=torch.bool),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    train_loader = make_loader(tr_idx, batch_size=64, shuffle=True)
    val_loader   = make_loader(va_idx, batch_size=256, shuffle=False)
    test_loader  = make_loader(te_idx, batch_size=256, shuffle=False)

    # ------------------------------------------------------------
    # 5. Create FNO model
    # ------------------------------------------------------------
    n_elements = X_composition.shape[1]
    n_targets  = len(present_targets)
    print(f"\nTraining FNO on {n_elements} elements → {n_targets} targets")
    print("Targets:", present_targets)

    model = FNOModel(
        n_elements=n_elements,
        n_targets=n_targets,
        modes=16,
        width=64,
        n_layers=4,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400, eta_min=1e-4)

    # You can tune this; if physics hurts, set to 0.0
    PHYS_WEIGHT = 0.05
    TEMP_RANGE = (500.0, 1200.0)

    train_loss_hist = []
    val_loss_hist   = []

    best_val = 1e9
    wait = 0
    PATIENCE = 80
    model_dir = Path("../data/trained_models_fno")
    model_dir.mkdir(parents=True, exist_ok=True)
    best_path = model_dir / "fno_coeffs.pth"

    μ_t = torch.tensor(μ, device=device)
    σ_t = torch.tensor(σ, device=device)

    # ------------------------------------------------------------
    # 6. Training loop
    # ------------------------------------------------------------
    print("\nTraining FNO model …")
    print(f"{'Epoch':>6s} | {'Train Loss':>12s} | {'Val Loss':>12s}")

    for epoch in range(400):
        model.train()
        tot = 0.0

        for xb, yb, mb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            mb = mb.to(device)

            # random temperature for physics regularisation
            T = torch.rand(len(xb), device=device) * (TEMP_RANGE[1] - TEMP_RANGE[0]) + TEMP_RANGE[0]

            optimizer.zero_grad()
            pred_std = model(xb)  # (B, n_targets)

            # masked MSE in standardised space
            loss_coeff = ((pred_std - yb) ** 2 * mb).sum() / mb.sum()

            # physics loss in physical units
            pred_raw = pred_std * σ_t + μ_t
            y_raw_t = yb * σ_t + μ_t
            loss_phys = physics_loss(pred_raw, y_raw_t, mb, T, present_targets) * PHYS_WEIGHT

            loss = loss_coeff + loss_phys
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tot += loss.item()

        scheduler.step()
        train_loss = tot / len(train_loader)

        # ---- validation ----
        model.eval()
        val = 0.0
        with torch.no_grad():
            for xb, yb, mb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                mb = mb.to(device)
                pred_std = model(xb)
                val += ((pred_std - yb) ** 2 * mb).sum().item() / mb.sum().item()
        val /= len(val_loader)

        train_loss_hist.append(train_loss)
        val_loss_hist.append(val)

        print(f"{epoch:6d} | {train_loss:12.6f} | {val:12.6f}")

        # early stopping
        if val < best_val - 1e-4:
            best_val = val
            wait = 0
            torch.save(model.state_dict(), best_path)
        else:
            wait += 1
            if wait >= PATIENCE:
                print(" ⇢ Early stopping FNO")
                break

    # load best model
    model.load_state_dict(torch.load(best_path, map_location=device))
    print("\nBest validation loss:", best_val)

    # ------------------------------------------------------------
    # 7. Evaluation: relMSE and R² per target on each split
    # ------------------------------------------------------------
    def eval_split(name: str, loader):
        model.eval()
        all_y_true = []
        all_y_pred = []

        with torch.no_grad():
            for xb, yb, mb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred_std = model(xb).cpu().numpy()
                y_true = yb.cpu().numpy()

                # de-standardise
                y_pred_phys = pred_std * σ + μ
                y_true_phys = y_true * σ + μ

                all_y_true.append(y_true_phys)
                all_y_pred.append(y_pred_phys)

        y_true = np.vstack(all_y_true)
        y_pred = np.vstack(all_y_pred)

        print(f"\n{name} split — per-target metrics")
        rels = []
        r2s  = []
        for j, tname in enumerate(present_targets):
            yt = y_true[:, j]
            yp = y_pred[:, j]

            mask = np.isfinite(yt)
            if not np.any(mask):
                print(f"  {tname:8s}: skipped (no finite data)")
                continue

            yt_m = yt[mask]
            yp_m = yp[mask]

            m_rel = rel_mse_pct(yt_m, yp_m)
            if np.isnan(m_rel):
                rel_str = "nan"
            else:
                rels.append(m_rel)
                rel_str = f"{m_rel:8.2f}%"

            try:
                r2 = r2_score(yt_m, yp_m)
            except ValueError:
                r2 = float("nan")
            r2s.append(r2)

            print(f"  {tname:8s}: relMSE={rel_str:>10s}   R²={r2:+.3f}")

        if rels:
            print(f"  ⇒ {name} avg : relMSE={np.mean(rels):8.2f}%   R²={np.nanmean(r2s):+.3f}")

    eval_split("Train", train_loader)
    eval_split("Val",   val_loader)
    eval_split("Test",  test_loader)

    # ------------------------------------------------------------
    # 8. Example: predict for a new composition (Na0.5 Cl0.5)
    # ------------------------------------------------------------
    def vector_from_composition(comp: Dict[str, float], element_list: List[str]) -> np.ndarray:
        """
        comp: dict like {"Na": 0.5, "Cl": 0.5} or {"NaCl": 1.0}
        element_list: the all_elements list used in training
        """
        elements = {}
        for key, value in comp.items():
            parsed = parse_compound(key)
            if len(parsed) > 1:  # compound
                for el, cnt in parsed.items():
                    elements[el] = elements.get(el, 0.0) + value * cnt
            else:
                el = list(parsed.keys())[0]
                elements[el] = elements.get(el, 0.0) + value

        total = sum(elements.values())
        if total <= 0:
            raise ValueError("Composition must have positive total.")
        for k in elements:
            elements[k] /= total

        vec = np.zeros(len(element_list), dtype=np.float32)
        for i, el in enumerate(element_list):
            vec[i] = elements.get(el, 0.0)
        return vec

    example_comp = {"Na": 0.5, "Cl": 0.5}
    x_ex = vector_from_composition(example_comp, all_elements)
    xb_ex = torch.tensor(x_ex[None, :], dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        pred_std_ex = model(xb_ex).cpu().numpy()[0]

    coeffs_ex = {
        tname: float(pred_std_ex[idx_map[tname]] * σ[idx_map[tname]] + μ[idx_map[tname]])
        for tname in present_targets
    }

    print("\nFNO prediction for 50-50 NaCl (coefficients):")
    for k, v in coeffs_ex.items():
        print(f"  {k:8s}: {v:11.4f}")

    print("\nDerived properties at 900 K (if enough coeffs present):")
    deriv_ex = derived_properties(coeffs_ex, 900.0)
    for k, v in deriv_ex.items():
        print(f"  {k:4s}: {v:11.4f}")
