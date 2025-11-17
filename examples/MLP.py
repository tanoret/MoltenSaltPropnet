import os
import sys
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
torch.manual_seed(SEED)
np.random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------
#  Make local package importable
# ------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from processing_mstdb.processor import MSTDBProcessor

# ------------------------------------------------------------------
#  Targets (same as your SNN)
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
#  Metrics
# ------------------------------------------------------------------
def rel_mse_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Relative MSE [% of mean(y^2)], guard against tiny denominators."""
    mse = mean_squared_error(y_true, y_pred)
    denom = np.mean(y_true ** 2)
    if denom < 1e-8:
        return float("nan")
    return 100.0 * mse / denom


# ------------------------------------------------------------------
#  Deep MLP (large) — one scalar output
# ------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H)
        y = self.fc1(x)
        y = self.act(y)
        y = self.dropout(y)
        y = self.fc2(y)
        return self.norm(x + y)


class DeepMLP(nn.Module):
    """
    Large deep MLP for one scalar target:
    - input_dim → 512
    - 4 residual blocks (each with 2 linear layers)
    - output 1 scalar (standardised target)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        n_blocks: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout=dropout) for _ in range(n_blocks)]
        )
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D)
        h = self.input(x)
        h = self.input_norm(h)
        for block in self.blocks:
            h = block(h)
        out = self.output(h)
        return out  # (B,1)


# ==================================================================
#  MAIN SCRIPT
# ==================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------
    # 1. Load CSV via MSTDBProcessor (your usual start)
    # ------------------------------------------------------------
    csv_path = "/Users/meggie/Documents/MoltenSaltPropnet/data/new_mstdb_janz.csv"

    processor = MSTDBProcessor.from_csv(csv_path)
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

    # Make sure no NaNs in features
    X_composition = np.nan_to_num(X_composition, nan=0.0)

    df = processor.df

    # Where we store trained models
    model_dir = Path("../data/trained_models_mlp_ensemble")
    model_dir.mkdir(parents=True, exist_ok=True)

    # We’ll store per-target trained models and scalers in memory
    ensembles: Dict[str, List[DeepMLP]] = {}
    scalers: Dict[str, Dict[str, float]] = {}  # t -> {"mu": float, "sigma": float}

    N_ENSEMBLES = 10
    MAX_EPOCHS = 300

    # For reporting at the end
    results_train = {}
    results_val = {}
    results_test = {}

    # ------------------------------------------------------------
    # 3. Loop over targets and train a 10-model ensemble per target
    # ------------------------------------------------------------
    for t in TARGETS:
        if t not in df.columns:
            print(f"\n=== Skipping {t}: not in DataFrame ===")
            continue

        print("\n" + "=" * 60)
        print(f"Training Deep MLP ensemble for target: {t}")
        print("=" * 60)

        # Clean column: handle weird strings and NaNs
        col = (
            df[t]
            .replace(["----", ""], np.nan)
            .replace(r"\*", "", regex=True)
        )
        col = pd.to_numeric(col, errors="coerce")

        # valid mask for this target
        y_all = col.to_numpy(np.float32)
        mask_valid = np.isfinite(y_all)
        idx_valid = np.where(mask_valid)[0]

        if len(idx_valid) < 40:
            print(f"Not enough valid data points for {t} (only {len(idx_valid)}), skipping.")
            continue

        X_t = X_composition[mask_valid]
        y_t = y_all[mask_valid]

        # Split this target’s valid rows into train/val/test
        idx_local = np.arange(len(y_t))
        idx_train, idx_test = train_test_split(idx_local, test_size=0.20, random_state=SEED)
        idx_train, idx_val = train_test_split(idx_train, test_size=0.20, random_state=SEED)

        # Standardize on training subset
        mu = y_t[idx_train].mean()
        sigma = y_t[idx_train].std()
        if sigma == 0:
            print(f"Target {t} has zero variance, skipping.")
            continue

        y_std = (y_t - mu) / sigma
        scalers[t] = {"mu": float(mu), "sigma": float(sigma)}

        def make_loader(indices, batch_size=64, shuffle=True):
            x = X_t[indices]
            y = y_std[indices]
            ds = TensorDataset(
                torch.tensor(x, dtype=torch.float32),
                torch.tensor(y[:, None], dtype=torch.float32),
            )
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

        train_loader = make_loader(idx_train, batch_size=64, shuffle=True)
        val_loader   = make_loader(idx_val,   batch_size=256, shuffle=False)
        test_loader  = make_loader(idx_test,  batch_size=256, shuffle=False)

        # Storage for ensemble models for this target
        ensembles[t] = []

        # --------------------------------------------------------
        # 3.1 Train N_ENSEMBLES MLPs independently
        # --------------------------------------------------------
        for m in range(N_ENSEMBLES):
            print(f"\n--- Training ensemble member {m+1}/{N_ENSEMBLES} for {t} ---")

            # Different seed per ensemble member
            torch.manual_seed(SEED + m)
            np.random.seed(SEED + m)

            model = DeepMLP(input_dim=X_t.shape[1], hidden_dim=512, n_blocks=4, dropout=0.1).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-4)

            best_val = 1e9
            best_state = None
            patience = 60
            wait = 0

            print(f"{'Epoch':>6s} | {'Train Loss':>12s} | {'Val Loss':>12s}")

            for epoch in range(MAX_EPOCHS):
                # ----- Train -----
                model.train()
                tot = 0.0
                for xb, yb in train_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)

                    optimizer.zero_grad()
                    pred = model(xb)  # (B,1)
                    loss = nn.functional.mse_loss(pred, yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    tot += loss.item()

                scheduler.step()
                train_loss = tot / len(train_loader)

                # ----- Val -----
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

                # Early stopping
                if val_loss < best_val - 1e-5:
                    best_val = val_loss
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        print(f" ⇢ Early stopping member {m+1} for {t}")
                        break

            # Load best state into model
            if best_state is not None:
                model.load_state_dict(best_state)
            ensembles[t].append(model)

            # Optionally save each ensemble member to disk
            model_path = model_dir / f"{t}_mlp_member_{m}.pth"
            torch.save(model.state_dict(), model_path)

        # --------------------------------------------------------
        # 3.2 Evaluation for this target (ensemble predictions)
        # --------------------------------------------------------
        def eval_split(name: str, loader, indices):
            """Evaluate ensemble on given split, return metrics + predictions."""
            all_true = []
            all_pred_ens = []

            # Build list of predictions per model, then average
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.numpy().ravel()  # standardized true

                preds_std_members = []
                for model in ensembles[t]:
                    model.eval()
                    with torch.no_grad():
                        pred_std = model(xb).cpu().numpy().ravel()
                    preds_std_members.append(pred_std)

                preds_std_members = np.stack(preds_std_members, axis=0)  # (M, B)
                pred_std_mean = preds_std_members.mean(axis=0)  # (B,)

                # de-standardise
                mu = scalers[t]["mu"]
                sigma = scalers[t]["sigma"]
                y_true_phys = yb * sigma + mu
                y_pred_phys = pred_std_mean * sigma + mu

                all_true.append(y_true_phys)
                all_pred_ens.append(y_pred_phys)

            y_true = np.concatenate(all_true)
            y_pred = np.concatenate(all_pred_ens)

            mask = np.isfinite(y_true)
            y_t = y_true[mask]
            y_p = y_pred[mask]
            if len(y_t) == 0:
                print(f"{name} split {t}: no finite data.")
                return {"relMSE": float("nan"), "R2": float("nan")}

            m_rel = rel_mse_pct(y_t, y_p)
            try:
                r2 = r2_score(y_t, y_p)
            except ValueError:
                r2 = float("nan")

            rel_str = "nan" if np.isnan(m_rel) else f"{m_rel:8.2f}%"
            print(f"{name} split {t:8s}: relMSE={rel_str}   R²={r2:+.3f}")
            return {"relMSE": float(m_rel), "R2": float(r2)}

        print(f"\nMetrics for target {t} (ensemble of {N_ENSEMBLES} models):")
        res_tr = eval_split("Train", train_loader, idx_train)
        res_va = eval_split("Val",   val_loader,   idx_val)
        res_te = eval_split("Test",  test_loader,  idx_test)

        results_train[t] = res_tr
        results_val[t]   = res_va
        results_test[t]  = res_te

    # ------------------------------------------------------------
    # 4. Global summary across all trained targets
    # ------------------------------------------------------------
    def summarize_results(res_dict: Dict[str, Dict[str, float]], name: str):
        rels = []
        r2s = []
        print(f"\n{name} split — per-target ensemble metrics")
        for t, d in res_dict.items():
            m_rel = d["relMSE"]
            r2 = d["R2"]
            rel_str = "nan" if np.isnan(m_rel) else f"{m_rel:8.2f}%"
            print(f"  {t:8s}: relMSE={rel_str}   R²={r2:+.3f}")
            if not np.isnan(m_rel):
                rels.append(m_rel)
            r2s.append(r2)
        if rels:
            print(f"  ⇒ {name} avg : relMSE={np.mean(rels):8.2f}%   R²={np.nanmean(r2s):+.3f}")
        else:
            print(f"  ⇒ {name} avg : no finite metrics")

    summarize_results(results_train, "Train")
    summarize_results(results_val,   "Val")
    summarize_results(results_test,  "Test")

    # ------------------------------------------------------------
    # 5. Example: predict for 50-50 NaCl using ensemble
    # ------------------------------------------------------------
    if ensembles:
        print("\n" + "="*60)
        print("Ensemble prediction for 50-50 NaCl")
        print("="*60)

        def vector_from_composition(comp: Dict[str, float],
                                   element_list: List[str],
                                   processor: MSTDBProcessor) -> np.ndarray:
            """
            comp: dict like {"Na":0.5, "Cl":0.5} or {"NaCl":1.0}
            """
            elements = {}
            for key, value in comp.items():
                parsed = processor.parse_compound(key)
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

        x_ex = vector_from_composition({"Na": 0.5, "Cl": 0.5}, all_elements, processor)
        xb_ex = torch.tensor(x_ex[None, :], dtype=torch.float32, device=device)

        coeffs_ex = {}
        for t, models in ensembles.items():
            mu = scalers[t]["mu"]
            sigma = scalers[t]["sigma"]

            preds_std_members = []
            for model in models:
                model.eval()
                with torch.no_grad():
                    pred_std = model(xb_ex).cpu().numpy().ravel()[0]
                preds_std_members.append(pred_std)

            pred_std_mean = np.mean(preds_std_members)
            coeffs_ex[t] = float(pred_std_mean * sigma + mu)

        print("\nEnsemble coefficients for 50-50 NaCl:")
        for t in sorted(coeffs_ex.keys()):
            print(f"  {t:8s}: {coeffs_ex[t]:11.4f}")
    else:
        print("\nNo targets were trained (not enough data / zero variance).")
