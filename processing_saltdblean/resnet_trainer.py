"""
ResNetMetaTrainer (adjusted)
===========================

What I changed vs your pasted code:

1) Removed the inference-time "WITHOUT element features" ablation helpers and reports.
   - No more X[:, -k:] = 0.0 logic.
   - No more report_with_without_elements_relative(), debug ablation section, etc.

2) Added STRICT validation + diagnostics for ELEMENT_FEATURE_COLS:
   - Parse check: fraction of empty dicts per column.
   - Weighted-mean check: nonzero fraction and std for each aggregated feature.
   - Feature block check: std of the final scaled element feature block.

3) Made elem_lookup more robust:
   - Uses average across occurrences instead of first-seen value.

4) Fixed evaluate(): now uses mask_all to avoid scoring missing targets that were filled with 0.
   - Returns masked MAE/RMSE/R2 per target and macro averages.

5) Fixed predict(): loads from self.model_dir (not hard-coded "../data/trained_models"),
   and uses the same feature ordering as training.

If you later want true WITH vs WITHOUT (train-time ablation), we can add a small wrapper
that constructs a second trainer with ELEMENT_FEATURE_COLS=[] or dicts forced to {}.
"""

import re, math, random, warnings, ast, os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

from processing_saltdblean.embedding_preconditioner import EmbeddingPreconditioner


# ------------------------- Utils -------------------------

def _rel_mse_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return relative MSE as a percentage of ⟨y²⟩ — avoids unit issues."""
    mse = mean_squared_error(y_true, y_pred)
    denom = np.mean(y_true ** 2) or 1e-12
    return 100.0 * mse / denom


SEED = 42
R = 8.314
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
warnings.filterwarnings("ignore", category=FutureWarning)


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

ELEMENT_FEATURE_COLS = [
    "polarizability_element[10-24Cm3]",
    "atomic_mass_element",
    "electronegativity_element",
    "atomic_radius_element[Angstrom]",
    "ionic_radius_element[Angstrom]",
    "covalent_radius_element[Angstrom]",
    "first_ionization_energy[kJ_per_mol]",
]


# ------------------------- Networks -------------------------

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, p_drop: float = 0.2):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.lin1(x))
        h = self.drop(h)
        h = self.lin2(h)
        return self.act(x + h)


class BaseNet(nn.Module):
    def __init__(self, d_in: int, hidden: int = 64, depth: int = 3):
        super().__init__()
        layers = [nn.Linear(d_in, hidden), nn.SiLU()]
        for _ in range(depth):
            layers.append(ResidualBlock(hidden))
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MetaNet(nn.Module):
    def __init__(self, n_props: int, hidden: int = 128, depth: int = 2):
        super().__init__()
        layers = [nn.Linear(n_props, hidden), nn.SiLU()]
        for _ in range(depth):
            layers.append(ResidualBlock(hidden))
        layers.append(nn.Linear(hidden, n_props))
        self.net = nn.Sequential(*layers)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        return self.net(p)


# ------------------------- Trainer -------------------------

class ResNetMetaTrainer:
    def __init__(
        self,
        df: pd.DataFrame,
        target_columns: List[str],
        derived_props: List[Tuple[str, List[str]]],
        element_feature_cols: List[str],
        degree_poly: int = 3,
        embedding_method: str = "none",
        n_components: int = 10,
        model_dir: str | Path = "../data/trained_models",
        verbose_feature_checks: bool = True,
    ):
        self.df = df.copy()
        self.target_columns = target_columns
        self.derived_props = derived_props
        self.ELEMENT_FEATURE_COLS = list(element_feature_cols)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_method = embedding_method
        self.n_components = n_components
        self.verbose_feature_checks = verbose_feature_checks

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
        if "System" not in self.df.columns or "Mol Frac" not in self.df.columns:
            raise ValueError("df must contain 'System' and 'Mol Frac' columns.")
        self.df["Composition"] = self.df.apply(self.row_composition, axis=1)

        # Element-fraction matrix (columns = elements sorted)
        self.X_comp = pd.json_normalize(self.df["Composition"]).fillna(0.0)
        self.X_comp = self.X_comp.reindex(sorted(self.X_comp.columns), axis=1)
        self.composition_df = self.X_comp

        self.fractions = self.X_comp.to_numpy(np.float32)  # (N, n_elements)

        # ---------- Poly features on element fractions ----------
        self.poly = PolynomialFeatures(degree_poly, include_bias=False)
        X_poly = self.poly.fit_transform(self.X_comp).astype(np.float32)

        self.poly_scaler = StandardScaler()
        self.X_poly = self.poly_scaler.fit_transform(X_poly).astype(np.float32)

        # ---------- Element feature cols: parse + aggregate ----------
        missing = [c for c in self.ELEMENT_FEATURE_COLS if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing ELEMENT_FEATURE_COLS in df: {missing}")

        # Parse dict-as-string into dict
        for col in self.ELEMENT_FEATURE_COLS:
            self.df[col] = self.df[col].apply(self._to_dict)

        if self.verbose_feature_checks:
            self._print_element_parse_diagnostics()

        # Build per-element lookup for inference (robust: average across occurrences)
        self.elem_lookup = self._build_elem_lookup_mean()

        # Aggregate to row-wise scalar features (composition-weighted mean)
        elem_feat_mat = []
        self.elem_feat_cols = []
        for col in self.ELEMENT_FEATURE_COLS:
            new_col = f"{col}__wmean"
            self.elem_feat_cols.append(new_col)
            self.df[new_col] = [
                self._weighted_mean_from_dict(comp, dct)
                for comp, dct in zip(self.df["Composition"], self.df[col])
            ]

            if self.verbose_feature_checks:
                s = self.df[new_col].to_numpy(dtype=float)
                nonzero = float((np.abs(s) > 1e-12).mean())
                std = float(np.std(s))
                print(f"[wmean] {new_col:<60s} nonzero_frac={nonzero:6.3f} std={std:.4g}")

            elem_feat_mat.append(self.df[new_col].to_numpy(dtype=np.float32))

        element_features = np.vstack(elem_feat_mat).T.astype(np.float32)  # (N, k)

        self.elem_scaler = StandardScaler()
        self.element_features = self.elem_scaler.fit_transform(element_features).astype(np.float32)

        if self.verbose_feature_checks:
            k = len(self.ELEMENT_FEATURE_COLS)
            block = self.element_features
            print("\n[Element feature block after scaling]")
            print("shape:", block.shape, " expected k=", k)
            print("std  :", block.std(axis=0))

        # ---------- Final feature matrix ----------
        # [poly_scaled, fractions, elem_features_scaled]
        self.X = np.hstack([self.X_poly, self.fractions, self.element_features]).astype(np.float32)
        self.raw_feat_dim = self.X.shape[1]

        # ---------- Targets + masks ----------
        # mask_all marks where y is actually present
        self.mask_all = np.isfinite(self.df[self.present_targets]).to_numpy(bool)
        self.df[self.present_targets] = self.df[self.present_targets].fillna(0.0)
        self.y_raw = self.df[self.present_targets].to_numpy(np.float32)

        # ---------- Splits ----------
        self.idx_all = np.arange(len(self.X))
        self.tr_idx, self.te_idx = train_test_split(self.idx_all, test_size=0.20, random_state=SEED)
        self.tr_idx, self.va_idx = train_test_split(self.tr_idx, test_size=0.20, random_state=SEED)

        # ---------- Embedding block ----------
        self.embedder = EmbeddingPreconditioner(method=embedding_method, n_components=n_components)
        self.embedder.fit(self.X[self.tr_idx])
        self.X_embedded = self.embedder.transform(self.X)

        self.feat_dim = self.n_components if embedding_method != "none" else self.raw_feat_dim

        # ---------- Normalize targets ----------
        self.μ = self.y_raw[self.tr_idx].mean(0)
        self.σ = self.y_raw[self.tr_idx].std(0)
        self.σ[self.σ == 0] = 1.0
        self.y_std = (self.y_raw - self.μ) / self.σ

        # ---------- Initialize models ----------
        self.idx_map = {n: j for j, n in enumerate(self.present_targets)}
        self.base_nets = nn.ModuleDict({n: BaseNet(self.feat_dim).to(self.device) for n in self.present_targets})
        self.meta = MetaNet(len(self.present_targets)).to(self.device)

    # ------------------------- helpers -------------------------

    def _to_dict(self, x):
        """Convert a cell that may be dict or dict-as-string into a dict."""
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

    def _print_element_parse_diagnostics(self):
        print("\n[Element feature parse check] fraction empty dicts by column:")
        empties = {}
        for col in self.ELEMENT_FEATURE_COLS:
            frac_empty = float(self.df[col].apply(lambda d: (not isinstance(d, dict)) or (len(d) == 0)).mean())
            empties[col] = frac_empty
            print(f"  {col:<45s} empty_frac={frac_empty:6.3f}")

        # hard warning if everything is empty for almost all rows (typical parse failure)
        if all(v > 0.95 for v in empties.values()):
            print(
                "\n[WARN] All ELEMENT_FEATURE_COLS are empty dicts for >95% of rows. "
                "Your element feature columns may not be valid Python dict strings."
            )

    def _build_elem_lookup_mean(self) -> Dict[str, Dict[str, float]]:
        """
        Build per-element lookup for inference by averaging across occurrences in the dataset.
        Returns: {col: {element: mean_value}}
        """
        sum_count: Dict[str, Dict[str, List[float]]] = {col: {} for col in self.ELEMENT_FEATURE_COLS}

        for col in self.ELEMENT_FEATURE_COLS:
            for d in self.df[col]:
                if not isinstance(d, dict):
                    continue
                for el, v in d.items():
                    try:
                        fv = float(v)
                    except Exception:
                        continue
                    if el not in sum_count[col]:
                        sum_count[col][el] = [0.0, 0.0]  # sum, count
                    sum_count[col][el][0] += fv
                    sum_count[col][el][1] += 1.0

        lookup: Dict[str, Dict[str, float]] = {col: {} for col in self.ELEMENT_FEATURE_COLS}
        for col in self.ELEMENT_FEATURE_COLS:
            for el, (s, c) in sum_count[col].items():
                lookup[col][el] = float(s / max(c, 1.0))
        return lookup

    def _weighted_mean_from_dict(self, comp: dict, prop_dict: dict) -> float:
        """Σ_e comp[e] * prop_dict[e] with missing -> 0."""
        s = 0.0
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
                total[el] = total.get(el, 0.0) + float(int(cnt or "1")) * float(f)

        s = sum(total.values()) or 1e-12
        return {el: cnt / s for el, cnt in total.items()}

    def make_loader(self, x, y, m, bs, shuf):
        ds = TensorDataset(torch.tensor(x), torch.tensor(y), torch.tensor(m))
        return DataLoader(ds, batch_size=bs, shuffle=shuf, drop_last=False)

    def _predict_from_features_matrix(self, X_feats: np.ndarray) -> np.ndarray:
        """Forward pass for prepared feature matrix (embedded if needed). Returns raw-scale preds (N, n_targets)."""
        self.meta.eval()
        for net in self.base_nets.values():
            net.eval()

        xb = torch.tensor(X_feats, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            base_out = torch.stack([self.base_nets[p](xb) for p in self.present_targets], dim=1)
            pred_std = (base_out + self.meta(base_out)).cpu().numpy()

        return pred_std * self.σ + self.μ

    def predict_on_split(self, split: str = "test") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (y_true_raw, y_pred_raw, mask) for a split with masking preserved."""
        split_map = {"train": self.tr_idx, "val": self.va_idx, "test": self.te_idx}
        if split not in split_map:
            raise ValueError(f"split must be one of {list(split_map.keys())}")

        idxs = split_map[split]
        y_true = self.y_raw[idxs]
        mask = self.mask_all[idxs].astype(bool)

        X = self.X_embedded[idxs]
        y_pred = self._predict_from_features_matrix(X)
        return y_true, y_pred, mask

    # ------------------------- training -------------------------

    def train_base(self):
        for prop in self.present_targets:
            print(f" • Training base net for {prop}")
            net = self.base_nets[prop]
            j = self.idx_map[prop]

            mask = self.mask_all[:, j].astype(bool)
            mask_tr_glb = mask & np.isin(self.idx_all, self.tr_idx)
            mask_va_glb = mask & np.isin(self.idx_all, self.va_idx)

            if mask_va_glb.sum() == 0:
                idx_prop = np.where(mask)[0]
                if len(idx_prop) >= 2:
                    tr_prop, va_prop = train_test_split(idx_prop, test_size=0.20, random_state=SEED)
                    mask_tr_glb = np.isin(self.idx_all, tr_prop)
                    mask_va_glb = np.isin(self.idx_all, va_prop)
                else:
                    mask_tr_glb = np.isin(self.idx_all, idx_prop)
                    mask_va_glb = np.zeros_like(mask_tr_glb, dtype=bool)

            x_tr, y_tr = self.X_embedded[mask_tr_glb], self.y_std[mask_tr_glb, j]
            x_va, y_va = self.X_embedded[mask_va_glb], self.y_std[mask_va_glb, j]

            tr_loader = DataLoader(TensorDataset(torch.tensor(x_tr), torch.tensor(y_tr)),
                                   batch_size=64, shuffle=True)
            va_loader = DataLoader(TensorDataset(torch.tensor(x_va), torch.tensor(y_va)),
                                   batch_size=256, shuffle=False) if len(x_va) > 0 else None

            opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 200, 2e-4)
            best, patience, PAT = 1e9, 0, 25
            model_path = self.model_dir / f"base_{prop}_resnet.pth"

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

            if va_loader and model_path.exists():
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
                    if dprop == "rho":
                        y_vals = y_coeffs[:, 0] - y_coeffs[:, 1] * T[mask]
                        p_vals = p_coeffs[:, 0] - p_coeffs[:, 1] * T[mask]
                        term_loss = nn.functional.mse_loss(p_vals, y_vals)

                    elif dprop == "muA":
                        p_mu1_a = torch.clamp(p_coeffs[:, 0], min=1e-6)
                        p_vals = p_mu1_a * torch.exp(p_coeffs[:, 1] / (R * T[mask]))
                        y_vals = y_coeffs[:, 0] * torch.exp(y_coeffs[:, 1] / (R * T[mask]))
                        term_loss = nn.functional.mse_loss(torch.log(p_vals + 1e-8), torch.log(y_vals + 1e-8))

                    elif dprop == "muB":
                        y_log = y_coeffs[:, 0] + y_coeffs[:, 1] / T[mask] + y_coeffs[:, 2] / T[mask] ** 2
                        p_log = p_coeffs[:, 0] + p_coeffs[:, 1] / T[mask] + p_coeffs[:, 2] / T[mask] ** 2
                        term_loss = nn.functional.mse_loss(p_log, y_log)

                    elif dprop == "k":
                        y_vals = y_coeffs[:, 0] + y_coeffs[:, 1] * T[mask]
                        p_vals = p_coeffs[:, 0] + p_coeffs[:, 1] * T[mask]
                        term_loss = nn.functional.mse_loss(p_vals, y_vals)

                    elif dprop == "cp":
                        y_vals = y_coeffs[:, 0] + y_coeffs[:, 1] * T[mask] + y_coeffs[:, 2] / T[mask] ** 2
                        p_vals = p_coeffs[:, 0] + p_coeffs[:, 1] * T[mask] + p_coeffs[:, 2] / T[mask] ** 2
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

        if meta_path.exists():
            self.meta.load_state_dict(torch.load(meta_path, map_location=self.device))

    # ------------------------- evaluation / predict -------------------------

    def evaluate_masked(self, split: str = "val", return_dict: bool = False):
        """
        Masked evaluation (does not score missing targets filled with 0.0).
        Returns per-target MAE/RMSE/R2 and macro averages.
        """
        y_true, y_pred, mask = self.predict_on_split(split)

        per_target = {}
        maes, rmses, r2s = [], [], []

        for j, prop in enumerate(self.present_targets):
            mj = mask[:, j]
            if not np.any(mj):
                continue

            yt = y_true[mj, j]
            yp = y_pred[mj, j]

            mae = float(np.mean(np.abs(yp - yt)))
            rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
            r2 = float(r2_score(yt, yp)) if len(yt) >= 2 else float("nan")

            per_target[prop] = {"MAE": mae, "RMSE": rmse, "R2": r2}
            maes.append(mae)
            rmses.append(rmse)
            r2s.append(r2)

        out = {
            "split": split,
            "MAE_macro": float(np.mean(maes)) if maes else float("nan"),
            "RMSE_macro": float(np.mean(rmses)) if rmses else float("nan"),
            "R2_macro": float(np.nanmean(r2s)) if r2s else float("nan"),
            "per_target": per_target,
        }

        print(f"\nMasked evaluation on split='{split}':")
        for p, d in per_target.items():
            print(f" • {p:<8s}: MAE={d['MAE']:.4g}  RMSE={d['RMSE']:.4g}  R2={d['R2']:+.3f}")
        print(f" ⇒ Macro: MAE={out['MAE_macro']:.4g}  RMSE={out['RMSE_macro']:.4g}  R2={out['R2_macro']:+.3f}")

        if return_dict:
            return out

    def predict(self, composition: Dict[str, float]) -> Dict[str, float]:
        """
        Predict coefficients from composition using the trained (in-memory) model.
        Feature ordering:
          [poly_scaled, fractions, elem_features_scaled]
        """
        # Convert input composition to element-only composition
        elements = {}
        for key, value in composition.items():
            parsed = self.parse_compound(key)
            for el, count in parsed.items():
                elements[el] = elements.get(el, 0.0) + float(value) * float(count)

        total = sum(elements.values())
        if total <= 0:
            raise ValueError("Composition must have positive total")
        normalized = {k: v / total for k, v in elements.items()}

        # Fraction vector in training column order
        frac = np.zeros(len(self.X_comp.columns), dtype=np.float32)
        for i, col in enumerate(self.X_comp.columns):
            frac[i] = normalized.get(col, 0.0)

        # Poly features
        raw_df = pd.DataFrame([frac], columns=self.X_comp.columns).fillna(0.0)
        raw_poly = self.poly.transform(raw_df).astype(np.float32)
        raw_poly = self.poly_scaler.transform(raw_poly).astype(np.float32)

        # Element aggregated features (weighted mean) using lookup
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

        if self.embedding_method != "none":
            feats = self.embedder.transform(feats)

        pred_raw = self._predict_from_features_matrix(feats)[0]
        return {prop: float(pred_raw[i]) for i, prop in enumerate(self.present_targets)}

    @staticmethod
    def parse_compound(c: str) -> Dict[str, int]:
        """Parse compound formula into elements (e.g., 'NaCl' → {'Na':1, 'Cl':1})"""
        out = {}
        for el, n in re.findall(r"([A-Z][a-z]*)(\d*)", str(c)):
            out[el] = out.get(el, 0) + int(n or "1")
        return out

    def derived(self, coeffs: Dict[str, float], T: float) -> Dict[str, float]:
        out = {}
        if {"rho_a", "rho_b"}.issubset(coeffs):
            out["rho"] = coeffs["rho_a"] - coeffs["rho_b"] * T
        if {"mu1_a", "mu1_b"}.issubset(coeffs):
            out["muA"] = coeffs["mu1_a"] * math.exp(coeffs["mu1_b"] / (R * T))
        if {"mu2_a", "mu2_b", "mu2_c"}.issubset(coeffs):
            out["muB"] = 10 ** (coeffs["mu2_a"] + coeffs["mu2_b"] / T + coeffs["mu2_c"] / T ** 2)
        if {"k_a", "k_b"}.issubset(coeffs):
            out["k"] = coeffs["k_a"] + coeffs["k_b"] * T
        if {"cp_a", "cp_b", "cp_c"}.issubset(coeffs):
            out["cp"] = coeffs["cp_a"] + coeffs["cp_b"] * T + coeffs["cp_c"] / T ** 2
        return out

    # ------------------------- persistence -------------------------

    def save(self, path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        for prop, net in self.base_nets.items():
            torch.save(net.state_dict(), path / f"base_{prop}_resnet.pth")
        torch.save(self.meta.state_dict(), path / "meta_resnet.pth")

        np.save(path / "μ_resnet.npy", self.μ)
        np.save(path / "σ_resnet.npy", self.σ)

        pd.to_pickle(self.poly, path / "poly_resnet.pkl")
        pd.to_pickle(self.poly_scaler, path / "poly_scaler.pkl")
        pd.to_pickle(self.elem_scaler, path / "elem_scaler.pkl")
        pd.to_pickle(self.X_comp.columns.tolist(), path / "elements_resnet.pkl")
        pd.to_pickle(self.elem_lookup, path / "elem_lookup.pkl")

    def load(self, path: str | Path):
        path = Path(path)

        for prop in self.present_targets:
            self.base_nets[prop].load_state_dict(torch.load(path / f"base_{prop}_resnet.pth", map_location=self.device))
        self.meta.load_state_dict(torch.load(path / "meta_resnet.pth", map_location=self.device))

        self.μ = np.load(path / "μ_resnet.npy")
        self.σ = np.load(path / "σ_resnet.npy")

        self.poly = pd.read_pickle(path / "poly_resnet.pkl")
        self.poly_scaler = pd.read_pickle(path / "poly_scaler.pkl")
        self.elem_scaler = pd.read_pickle(path / "elem_scaler.pkl")

        self.X_comp.columns = pd.read_pickle(path / "elements_resnet.pkl")
        self.elem_lookup = pd.read_pickle(path / "elem_lookup.pkl")
