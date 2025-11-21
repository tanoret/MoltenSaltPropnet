import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
R = 8.314 

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from processing_mstdb.processor import MSTDBProcessor


TARGETS = [
    "Melt(K)", "Boil(K)",
    "rho_a", "rho_b",
    "mu1_a", "mu1_b",
    "mu2_a", "mu2_b", "mu2_c",
    "k_a", "k_b",
    "cp_a", "cp_b", "cp_c",
]

DERIVED_PROPS: List[Tuple[str, List[str]]] = [
    ("rho", ["rho_a", "rho_b"]),
    ("muA", ["mu1_a", "mu1_b"]),
    ("muB", ["mu2_a", "mu2_b", "mu2_c"]),
    ("k",   ["k_a",   "k_b"]),
    ("cp",  ["cp_a",  "cp_b", "cp_c"]),
]

def rel_mse_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Relative MSE [% of mean(y^2)], guard against tiny denominators."""
    mse = mean_squared_error(y_true, y_pred)
    denom = np.mean(y_true ** 2)
    if denom < 1e-8:
        return float("nan")
    return 100.0 * mse / denom



#  Deep Multi-Task MLP with per-target heads

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc1(x)
        y = self.act(y)
        y = self.dropout(y)
        y = self.fc2(y)
        return self.norm(x + y)


class TargetHead(nn.Module):
    """Small head for a single target: outputs mean_std and log_var."""
    def __init__(self, hidden_dim: int = 512, head_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 2),  # mean_std, log_var
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)  # (B,2)


class MultiTaskPhysMLP(nn.Module):
    """
    Deep MLP with shared backbone and per-target heads.
    Each target head outputs [mean_std, log_var] in standardised space.
    """

    def __init__(
        self,
        input_dim: int,
        target_names: List[str],
        hidden_dim: int = 512,
        n_blocks: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.target_names = target_names
        self.n_targets = len(target_names)

        # shared backbone
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.norm_in = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout=dropout) for _ in range(n_blocks)]
        )

        # per-target heads
        self.heads = nn.ModuleDict(
            {
                name: TargetHead(hidden_dim=hidden_dim, head_dim=128, dropout=dropout)
                for name in target_names
            }
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B,D)
        returns:
            mean_std: (B,T)
            log_var : (B,T)
        """
        h = self.fc_in(x)
        h = self.norm_in(h)
        for block in self.blocks:
            h = block(h)  # (B,H)

        means = []
        logvars = []
        for name in self.target_names:
            out = self.heads[name](h)  # (B,2)
            m, lv = out[:, 0:1], out[:, 1:2]
            means.append(m)
            logvars.append(lv)

        mean_std = torch.cat(means, dim=1)   # (B,T)
        log_var  = torch.cat(logvars, dim=1) # (B,T)
        return mean_std, log_var


#  Losses: Gaussian NLL + physics
def gaussian_nll(mean_std: torch.Tensor,
                 log_var: torch.Tensor,
                 target_std: torch.Tensor,
                 mask: torch.Tensor) -> torch.Tensor:
    """
    mean_std, log_var, target_std: (B,T)
    mask: (B,T) bool
    """
    mask_f = mask.float()
    if mask_f.sum() == 0:
        return torch.tensor(0.0, device=mean_std.device)

    inv_var = torch.exp(-log_var)  # (B,T)
    sq = (target_std - mean_std) ** 2
    loss = 0.5 * (sq * inv_var + log_var)  # (B,T)
    loss = (loss * mask_f).sum() / mask_f.sum()
    return loss


def physics_loss(
    pred_phys: torch.Tensor,
    y_phys: torch.Tensor,
    mask_b: torch.Tensor,
    T: torch.Tensor,
    present_targets: List[str],
    derived_props: List[Tuple[str, List[str]]],
    device: str,
) -> torch.Tensor:
    """
    pred_phys, y_phys: (B,T) in physical units
    mask_b           : (B,T) bool
    T                : (B,)
    """
    loss = torch.tensor(0.0, device=device)
    terms = 0

    idx_map = {name: j for j, name in enumerate(present_targets)}

    for dprop, coeffs in derived_props:
        idxs = [idx_map[c] for c in coeffs if c in idx_map]
        if len(idxs) != len(coeffs):
            continue

        m = torch.all(mask_b[:, idxs], dim=1)
        if not m.any():
            continue

        y = y_phys[m][:, idxs]
        p = pred_phys[m][:, idxs]
        Tm = T[m]

        if dprop == "rho":
            loss_t = nn.functional.mse_loss(
                p[:, 0] - p[:, 1] * Tm,
                y[:, 0] - y[:, 1] * Tm,
            )
        elif dprop == "muA":
            p_mu = torch.clamp(p[:, 0], min=1e-8) * torch.exp(p[:, 1] / (R * Tm))
            y_mu = torch.clamp(y[:, 0], min=1e-8) * torch.exp(y[:, 1] / (R * Tm))
            loss_t = nn.functional.mse_loss(torch.log(p_mu), torch.log(y_mu))
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
        return torch.tensor(0.0, device=device)
    return loss / terms



#  MAIN SCRIPT

if __name__ == "__main__":

    csv_path = "/Users/meggie/Documents/MoltenSaltPropnet/data/new_mstdb_janz.csv"

    processor = MSTDBProcessor.from_csv(csv_path)
    print(processor.df.head())
    processor.df.columns = processor.df.columns.str.strip()
    print(processor.df.columns)
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

    X_composition = np.nan_to_num(X_composition, nan=0.0)
    df = processor.df

    present_targets: List[str] = []
    target_values: Dict[str, np.ndarray] = {}

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
            df[t] = col
            present_targets.append(t)
            target_values[t] = col.to_numpy(np.float32)
        else:
            print(f"Skipping {t}: no finite values")

    if not present_targets:
        raise RuntimeError("No valid target columns with numeric data.")

    print("\nUsing targets:", ", ".join(present_targets))
    Tgt = len(present_targets)

    y_mat_raw = np.stack([target_values[t] for t in present_targets], axis=1)  # (N,T)
    mask_all = np.isfinite(y_mat_raw)  # (N,T) bool
    y_mat = np.nan_to_num(y_mat_raw, nan=0.0).astype(np.float32)

    N, D = X_composition.shape

    # Train/val/test split by rows
    idx_all = np.arange(N)
    idx_train, idx_test = train_test_split(idx_all, test_size=0.20, random_state=SEED)
    idx_train, idx_val = train_test_split(idx_train, test_size=0.20, random_state=SEED)

    # Per-target standardisation (μ, σ) on train subset
   
    μ = np.zeros(Tgt, dtype=np.float32)
    σ = np.ones(Tgt, dtype=np.float32)

    for j in range(Tgt):
        m_tr = mask_all[idx_train, j]
        vals = y_mat_raw[idx_train, j][m_tr]
        if len(vals) == 0:
            μ[j] = 0.0
            σ[j] = 1.0
        else:
            μ[j] = vals.mean()
            s = vals.std()
            σ[j] = s if s > 0 else 1.0

    y_std = (y_mat - μ) / σ

   
    # DataLoaders with masks
   
    def make_loader(idx_set, batch_size=64, shuffle=True):
        x = X_composition[idx_set]
        y = y_std[idx_set]
        m = mask_all[idx_set]
        ds = TensorDataset(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(m, dtype=torch.bool),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    train_loader = make_loader(idx_train, batch_size=64, shuffle=True)
    val_loader   = make_loader(idx_val,   batch_size=256, shuffle=False)
    test_loader  = make_loader(idx_test,  batch_size=256, shuffle=False)


    # Ensemble of MultiTaskPhysMLP with physics loss

    N_ENSEMBLES = 5              # 10 where actually as well nice working but afraid of overfitting was there
    MAX_EPOCHS = 300
    PHYS_LAMBDA = 0.1

    model_dir = Path("../data/trained_models_mlp_multitask_phys_deep")
    model_dir.mkdir(parents=True, exist_ok=True)

    ensembles: List[MultiTaskPhysMLP] = []

    μ_t = torch.tensor(μ, dtype=torch.float32, device=device)
    σ_t = torch.tensor(σ, dtype=torch.float32, device=device)

    print("\nTraining multi-target deep MLP ensemble with physics regularisation …")

    for m_idx in range(N_ENSEMBLES):
        print("\n" + "=" * 60)
        print(f"Ensemble member {m_idx+1}/{N_ENSEMBLES}")
        print("=" * 60)

        torch.manual_seed(SEED + m_idx)
        np.random.seed(SEED + m_idx)

        model = MultiTaskPhysMLP(
            input_dim=D,
            target_names=present_targets,
            hidden_dim=512,
            n_blocks=6,
            dropout=0.1,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-4)

        best_val = 1e9
        best_state = None
        patience = 80
        wait = 0

        print(f"{'Epoch':>6s} | {'Train Loss':>12s} | {'Val Loss':>12s}")

        for epoch in range(MAX_EPOCHS):
            # ----- train -----
            model.train()
            total = 0.0
            count = 0

            for xb, yb, mb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                mb = mb.to(device)

                optimizer.zero_grad()
                mean_std, log_var = model(xb)

                # coeff loss in std space (aleatoric NLL)
                coeff_loss = gaussian_nll(mean_std, log_var, yb, mb)

                # physics loss in physical units, using mean predictions only
                pred_phys = mean_std * σ_t + μ_t
                y_phys    = yb       * σ_t + μ_t
                T_batch = torch.rand(len(xb), device=device) * 700.0 + 500.0

                phys_l = physics_loss(pred_phys, y_phys, mb, T_batch,
                                      present_targets, DERIVED_PROPS, device)

                loss = coeff_loss + PHYS_LAMBDA * phys_l
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total += loss.item()
                count += 1

            scheduler.step()
            train_loss = total / max(count, 1)

            # validattion
            model.eval()
            val_total = 0.0
            val_count = 0
            with torch.no_grad():
                for xb, yb, mb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    mb = mb.to(device)
                    mean_std, log_var = model(xb)
                    coeff_loss = gaussian_nll(mean_std, log_var, yb, mb)

                    pred_phys = mean_std * σ_t + μ_t
                    y_phys    = yb       * σ_t + μ_t
                    T_batch = torch.rand(len(xb), device=device) * 700.0 + 500.0
                    phys_l = physics_loss(pred_phys, y_phys, mb, T_batch,
                                          present_targets, DERIVED_PROPS, device)

                    val_loss = coeff_loss + PHYS_LAMBDA * phys_l
                    val_total += val_loss.item()
                    val_count += 1

            val_loss = val_total / max(val_count, 1)
            print(f"{epoch:6d} | {train_loss:12.6f} | {val_loss:12.6f}")

            if val_loss < best_val - 1e-5:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f" ⇢ Early stopping member {m_idx+1}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        ensembles.append(model)

        torch.save(model.state_dict(), model_dir / f"mlp_multitask_phys_deep_member_{m_idx}.pth")

    print(f"\nBest validation loss among ensemble members: {best_val}")

    
    #  Evaluation: Train / Val / Test with ensemble (using means)
    def evaluate_split(name: str, idx_set, loader):
        print(f"\n{name} split — per-target ensemble metrics")

        # get ensemble predictions in std space
        all_pred_std_members = []  # list of (N_split,T)
        for model in ensembles:
            model.eval()
            preds_model = []
            with torch.no_grad():
                for xb, yb, mb in loader:
                    xb = xb.to(device)
                    mean_std, log_var = model(xb)
                    preds_model.append(mean_std.cpu().numpy())
            preds_model = np.concatenate(preds_model, axis=0)
            all_pred_std_members.append(preds_model)

        pred_std_stack = np.stack(all_pred_std_members, axis=0)  # (M,N_split,T)
        pred_std_mean = pred_std_stack.mean(axis=0)              # (N_split,T)
        pred_phys = pred_std_mean * σ + μ                       # (N_split,T)

        y_true_raw = y_mat_raw[idx_set]                          # (N_split,T)
        mask_split = mask_all[idx_set]

        rels = []
        r2s = []

        for j, t in enumerate(present_targets):
            m = mask_split[:, j]
            yt = y_true_raw[:, j][m]
            yp = pred_phys[:, j][m]
            if len(yt) == 0:
                print(f"  {t:8s}: no data")
                continue

            m_rel = rel_mse_pct(yt, yp)
            try:
                r2 = r2_score(yt, yp)
            except ValueError:
                r2 = float("nan")

            rel_str = "nan" if np.isnan(m_rel) else f"{m_rel:8.2f}%"
            print(f"  {t:8s}: relMSE={rel_str}   R²={r2:+.3f}")
            if not np.isnan(m_rel):
                rels.append(m_rel)
            r2s.append(r2)

        if rels:
            print(f"  ⇒ {name} avg : relMSE={np.mean(rels):8.2f}%   R²={np.nanmean(r2s):+.3f}")
        else:
            print(f"  ⇒ {name} avg : no finite metrics")

    evaluate_split("Train", idx_train, train_loader)
    evaluate_split("Val",   idx_val,   val_loader)
    evaluate_split("Test",  idx_test,  test_loader)

    #  Example: predict for 50-50 NaCl using ensemble
    
    if ensembles:
        print("\n" + "="*60)
        print("Ensemble multi-target deep MLP prediction for 50-50 NaCl")
        print("="*60)

        def vector_from_composition(comp: Dict[str, float],
                                   element_list: List[str],
                                   processor: MSTDBProcessor) -> np.ndarray:
            elements = {}
            for key, value in comp.items():
                parsed = processor.parse_compound(key)
                if len(parsed) > 1:  # compound like NaCl
                    for el, cnt in parsed.items():
                        elements[el] = elements.get(el, 0.0) + value * cnt
                else:                # element
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

        preds_std_members = []
        for model in ensembles:
            model.eval()
            with torch.no_grad():
                mean_std, log_var = model(xb_ex)
                preds_std_members.append(mean_std.cpu().numpy()[0])

        pred_std_mean = np.mean(preds_std_members, axis=0)
        pred_phys_ex = pred_std_mean * σ + μ

        print("\nEnsemble deep MLP coefficients for 50-50 NaCl:")
        for j, t in enumerate(present_targets):
            print(f"  {t:8s}: {pred_phys_ex[j]:11.4f}")
    else:
        print("\nNo ensemble models trained.")
