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
#  Targets (same list as for SNN)
# ------------------------------------------------------------------
TARGETS = [
    "Melt(K)", "Boil(K)",
    "rho_a", "rho_b",
    "mu1_a", "mu1_b",
    "mu2_a", "mu2_b", "mu2_c",
    "k_a", "k_b",
    "cp_a", "cp_b", "cp_c"
]

# ------------------------------------------------------------------
#  Metrics utility
# ------------------------------------------------------------------
def rel_mse_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Relative MSE [% of mean(y^2)], guarded against tiny denominators."""
    mse = mean_squared_error(y_true, y_pred)
    denom = np.mean(y_true ** 2)
    if denom < 1e-8:
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
            torch.randn(out_channels, in_channels, max(modes, 1), dtype=torch.cfloat)
            * (1.0 / math.sqrt(in_channels * max(modes, 1)))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, L)
        returns: (B, C_out, L)
        """
        B, C_in, L = x.shape
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
    1D FNO mapping composition signal (over elements) to a single coefficient.

    Input:  (B, L)        — composition fractions per element
    Output: (B, 1)        — standardised coefficient
    """

    def __init__(
        self,
        n_elements: int,
        modes: int = 16,
        width: int = 64,
        n_layers: int = 4,
    ):
        super().__init__()
        self.n_elements = n_elements
        self.width = width

        # Lift 1 channel → width channels
        self.input_proj = nn.Conv1d(1, width, kernel_size=1)

        self.layers = nn.ModuleList(
            [FNO1dLayer(width, modes) for _ in range(n_layers)]
        )

        # Project width channels → 1 output channel at each element position
        self.output_proj = nn.Conv1d(width, 1, kernel_size=1)

        # Small head after global pooling
        self.head = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L) or (B, 1, L)
        returns: (B, 1)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, L)

        x = self.input_proj(x)  # (B, width, L)

        for layer in self.layers:
            x = layer(x)  # (B, width, L)

        x = self.output_proj(x)  # (B, 1, L)

        # Global average pool over element dimension
        x = x.mean(dim=-1)  # (B, 1)

        x = self.head(x)    # (B, 1)
        return x


# ------------------------------------------------------------------
#  Helper: parse compound to elements (for prediction example)
# ------------------------------------------------------------------
def parse_compound(c: str) -> Dict[str, int]:
    out = {}
    for el, n in re.findall(r"([A-Z][a-z]*)(\d*)", c):
        out[el] = out.get(el, 0) + int(n or "1")
    return out


# ==================================================================
#  MAIN SCRIPT: train one FNO per target
# ==================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------
    # 1. Load CSV via MSTDBProcessor (your usual start)
    # ------------------------------------------------------------
    processor = MSTDBProcessor.from_csv(
        "/Users/meggie/Documents/MoltenSaltPropnet/data/new_mstdb_janz.csv"
    )
    print(processor.df.head())
    processor.df.columns = processor.df.columns.str.strip()  # clean column names
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

    # Ensure no NaNs in features
    X_composition = np.nan_to_num(X_composition, nan=0.0)

    df = processor.df

    # Directory to store per-target FNOs
    model_dir = Path("../data/trained_models_fno_per_target")
    model_dir.mkdir(parents=True, exist_ok=True)

    # For example prediction at the end
    per_target_scalers = {}    # tname -> (mu, sigma)
    trained_targets = []       # list of targets successfully trained

    # ------------------------------------------------------------
    # 3. Loop over targets and train one FNO per target
    # ------------------------------------------------------------
    for t in TARGETS:
        if t not in df.columns:
            print(f"\n=== Skipping {t}: not in DataFrame ===")
            continue

        print(f"\n{'='*60}")
        print(f"Training FNO for target: {t}")
        print(f"{'='*60}")

        # --- clean the column, handle NaNs / strange strings ---
        col = (
            df[t]
            .replace(["----", ""], np.nan)
            .replace(r"\*", "", regex=True)
        )
        col = pd.to_numeric(col, errors="coerce")  # non-numeric → NaN

        # mask of finite values = rows where this target exists
        mask_valid = np.isfinite(col.to_numpy(np.float32))
        idx_valid = np.where(mask_valid)[0]

        if len(idx_valid) < 30:
            print(f"Not enough data points for {t} (only {len(idx_valid)}), skipping.")
            continue

        y_raw = col.to_numpy(np.float32)
        y_raw = y_raw[mask_valid]  # only valid rows
        X_t   = X_composition[mask_valid]  # corresponding features

        # Train/val/test split on valid rows only
        idx_all = np.arange(len(y_raw))
        tr_idx, te_idx = train_test_split(idx_all, test_size=0.20, random_state=SEED)
        tr_idx, va_idx = train_test_split(tr_idx, test_size=0.20, random_state=SEED)

        # Standardise on train only
        mu = y_raw[tr_idx].mean()
        sigma = y_raw[tr_idx].std()
        if sigma == 0:
            print(f"Target {t} has zero variance, skipping.")
            continue
        y_std = (y_raw - mu) / sigma

        per_target_scalers[t] = (mu, sigma)
        trained_targets.append(t)

        def make_loader(idxs, batch_size=64, shuffle=True):
            x = X_t[idxs]
            y = y_std[idxs]
            ds = TensorDataset(
                torch.tensor(x, dtype=torch.float32),
                torch.tensor(y[:, None], dtype=torch.float32),
            )
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

        train_loader = make_loader(tr_idx, batch_size=64, shuffle=True)
        val_loader   = make_loader(va_idx, batch_size=256, shuffle=False)
        test_loader  = make_loader(te_idx, batch_size=256, shuffle=False)

        # --- build model for this target ---
        n_elements = X_composition.shape[1]
        model = FNOModel(
            n_elements=n_elements,
            modes=16,
            width=64,
            n_layers=4,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=1e-4)

        best_val = 1e9
        wait = 0
        PATIENCE = 60
        model_path = model_dir / f"base_{t}_fno.pth"

        print("\nTraining FNO for", t)
        print(f"{'Epoch':>6s} | {'Train Loss':>12s} | {'Val Loss':>12s}")

        for epoch in range(300):
            # ----- train -----
            model.train()
            tot = 0.0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad()
                pred = model(xb)                # (B,1)
                loss = nn.functional.mse_loss(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                tot += loss.item()

            scheduler.step()
            train_loss = tot / len(train_loader)

            # ----- validation -----
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    pred = model(xb)
                    val_loss += nn.functional.mse_loss(pred, yb).item()
            val_loss /= len(val_loader)

            print(f"{epoch:6d} | {train_loss:12.6f} | {val_loss:12.6f}")

            # early stopping
            if val_loss < best_val - 1e-5:
                best_val = val_loss
                wait = 0
                torch.save(model.state_dict(), model_path)
            else:
                wait += 1
                if wait >= PATIENCE:
                    print(f" ⇢ Early stopping for {t}")
                    break

        # load best
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"\nBest val loss for {t}: {best_val:.6f}")

        # --------------------------------------------------------
        # Evaluation: relMSE + R² on train/val/test splits
        # --------------------------------------------------------
        def eval_split(name: str, loader, idxs):
            model.eval()
            all_y_true = []
            all_y_pred = []
            with torch.no_grad():
                for xb, yb in loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    pred = model(xb).cpu().numpy().ravel()
                    y_true = yb.cpu().numpy().ravel()
                    # de-standardise
                    y_pred_phys = pred * sigma + mu
                    y_true_phys = y_true * sigma + mu
                    all_y_true.append(y_true_phys)
                    all_y_pred.append(y_pred_phys)

            y_true = np.concatenate(all_y_true)
            y_pred = np.concatenate(all_y_pred)

            # metrics
            mask = np.isfinite(y_true)
            y_t = y_true[mask]
            y_p = y_pred[mask]
            if len(y_t) == 0:
                print(f"{name} split {t}: no finite data.")
                return

            m_rel = rel_mse_pct(y_t, y_p)
            try:
                r2 = r2_score(y_t, y_p)
            except ValueError:
                r2 = float("nan")

            rel_str = "nan" if np.isnan(m_rel) else f"{m_rel:8.2f}%"
            print(f"{name} split {t:8s}: relMSE={rel_str}   R²={r2:+.3f}")

        eval_split("Train", train_loader, tr_idx)
        eval_split("Val",   val_loader,   va_idx)
        eval_split("Test",  test_loader,  te_idx)

    # ------------------------------------------------------------
    # 4. Example: predict for a new composition (Na0.5 Cl0.5)
    # ------------------------------------------------------------
    if trained_targets:
        print("\n" + "="*60)
        print("Example prediction for 50-50 NaCl with per-target FNOs")
        print("="*60)

        def vector_from_composition(comp: Dict[str, float], element_list: List[str]) -> np.ndarray:
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

        x_ex = vector_from_composition({"Na": 0.5, "Cl": 0.5}, all_elements)
        xb_ex = torch.tensor(x_ex[None, :], dtype=torch.float32, device=device)

        coeffs_ex = {}
        for t in trained_targets:
            mu, sigma = per_target_scalers[t]
            model_path = model_dir / f"base_{t}_fno.pth"
            # rebuild model
            model_t = FNOModel(
                n_elements=X_composition.shape[1],
                modes=16,
                width=64,
                n_layers=4,
            ).to(device)
            model_t.load_state_dict(torch.load(model_path, map_location=device))
            model_t.eval()
            with torch.no_grad():
                pred_std = model_t(xb_ex).cpu().numpy().ravel()[0]
            coeffs_ex[t] = float(pred_std * sigma + mu)

        print("\nPer-target FNO coefficients for 50-50 NaCl:")
        for k in trained_targets:
            print(f"  {k:8s}: {coeffs_ex[k]:11.4f}")
    else:
        print("\nNo targets were trained (not enough data / zero variance).")
