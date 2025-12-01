
import os
import re
import math
import random
import warnings
from pathlib import Path
from typing import Dict, Sequence, Tuple, List, Optional
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

from processing_mstdb.embedding_preconditioner import EmbeddingPreconditioner
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

SEED = 42
R = 8.314
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
warnings.filterwarnings("ignore", category=FutureWarning)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _rel_mse_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Relative MSE as percentage of <y^2> (scale-invariant; stable as long as y not ~0)."""
    mse = mean_squared_error(y_true, y_pred)
    denom = np.mean(y_true ** 2)
    if not np.isfinite(denom) or denom == 0.0:
        denom = 1e-12
    return 100.0 * mse / denom


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _p90_rel_err(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rel = np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-12)
    return float(np.percentile(rel, 90))


# ----------------- network blocks -----------------


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


# ----------------- main ResNet + Meta trainer -----------------


class ResNetMetaTrainerv2:
    """
    Improved multi-target ResNet + Meta network trainer.

    Key improvements over the original:
      * missing labels are handled with masks (no fillna(0) abuse)
      * µ, σ computed only over *training* labels, ignoring missing
      * loss and metrics respect masks
      * supports embeddings via EmbeddingPreconditioner
      * per-target metrics: relMSE, R², MAE, p90_rel, sample count
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_columns: Sequence[str],
        derived_props: Sequence[Tuple[str, Sequence[str]]],
        degree_poly: int = 2,
        embedding_method: str = "none",
        n_components: int = 10,
        model_dir: str = "../data/trained_models_resnet",
        tr_idx: Optional[np.ndarray] = None,
        va_idx: Optional[np.ndarray] = None,
        te_idx: Optional[np.ndarray] = None,
        min_samples_per_target: int = 10,
    ):
        self.df = df.copy()
        self.target_columns = list(target_columns)
        self.derived_props = list(derived_props)
        self.device = DEVICE
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.min_samples_per_target = min_samples_per_target

        # ----- 1. clean targets, keep NaNs, track present_targets -----
        self.present_targets: List[str] = []
        for t in self.target_columns:
            if t not in self.df.columns:
                continue
            col = (
                self.df[t]
                .replace(["----", ""], np.nan)
                .replace(r"\*", "", regex=True)
            )
            col = pd.to_numeric(col, errors="coerce")
            self.df[t] = col
            if np.isfinite(col).any():
                self.present_targets.append(t)

        if not self.present_targets:
            raise RuntimeError("No valid target columns found after cleaning.")

        # ----- 2. composition → element fractions (features) -----
        self.df["Composition"] = self.df.apply(self._row_composition, axis=1)
        self.X_comp = pd.json_normalize(self.df["Composition"]).fillna(0.0)
        self.X_comp = self.X_comp.reindex(sorted(self.X_comp.columns), axis=1)

        self.idx_all = np.arange(len(self.df))

        # ----- 3. split train/val/test (or use provided indices) -----
        if tr_idx is None or va_idx is None or te_idx is None:
            tr, te = train_test_split(self.idx_all, test_size=0.20, random_state=SEED)
            tr, va = train_test_split(tr, test_size=0.20, random_state=SEED)
            self.tr_idx, self.va_idx, self.te_idx = np.array(tr), np.array(va), np.array(te)
        else:
            self.tr_idx = np.array(tr_idx)
            self.va_idx = np.array(va_idx)
            self.te_idx = np.array(te_idx)

        # ----- 4. polynomial features + scaler (fit only on training) -----
        self.poly = PolynomialFeatures(degree_poly, include_bias=False)
        X_poly_tr = self.poly.fit_transform(self.X_comp.iloc[self.tr_idx])
        X_poly_all = self.poly.transform(self.X_comp)

        self.scaler = StandardScaler()
        X_poly_tr = self.scaler.fit_transform(X_poly_tr)
        X_poly_all = self.scaler.transform(X_poly_all).astype(np.float32)

        # concat polynomial features + raw fractions
        self.fractions = self.X_comp.to_numpy(np.float32)
        self.X = np.hstack([X_poly_all, self.fractions])
        self.feat_dim = self.X.shape[1]

        # ----- 5. apply embedding if requested -----
        self.embedding_method = embedding_method
        self.n_components = n_components
        self.embedder = EmbeddingPreconditioner(
            method=embedding_method, n_components=n_components
        )
        self.embedder.fit(self.X[self.tr_idx])
        self.X_embedded = self.embedder.transform(self.X)
        self.feat_dim = (
            self.n_components if embedding_method != "none" else self.X.shape[1]
        )

        # ----- 6. targets + missing-data masks -----
        Y_df = self.df[self.present_targets]
        self.y_raw = Y_df.to_numpy(dtype=np.float32)  # contains NaNs
        self.mask_all = np.isfinite(self.y_raw)       # True where label exists

        # drop ultra-sparse targets
        keep_t, keep_idx = [], []
        for j, t in enumerate(self.present_targets):
            n_valid = int(self.mask_all[:, j].sum())
            if n_valid >= self.min_samples_per_target:
                keep_t.append(t)
                keep_idx.append(j)
            else:
                print(f"[WARN] Dropping target {t} (only {n_valid} valid samples).")
        if not keep_t:
            raise RuntimeError(
                "No targets have enough samples; lower min_samples_per_target."
            )

        self.present_targets = keep_t
        self.mask_all = self.mask_all[:, keep_idx]
        self.y_raw = self.y_raw[:, keep_idx]

        # ----- 7. standardization: µ, σ on training labels only -----
        y_tr = self.y_raw[self.tr_idx]
        mask_tr = self.mask_all[self.tr_idx]

        with np.errstate(all="ignore"):
            μ = np.nanmean(np.where(mask_tr, y_tr, np.nan), axis=0)
            σ = np.nanstd(np.where(mask_tr, y_tr, np.nan), axis=0)

        σ[~np.isfinite(σ)] = 1.0
        σ[σ == 0.0] = 1.0

        self.μ = μ.astype(np.float32)
        self.σ = σ.astype(np.float32)

        # standardized targets; missing entries set to 0 (ignored by mask)
        self.y_std = (self.y_raw - self.μ) / self.σ
        self.y_std[~self.mask_all] = 0.0

        # ----- 8. models -----
        self.idx_map: Dict[str, int] = {name: j for j, name in enumerate(self.present_targets)}
        self.base_nets = nn.ModuleDict(
            {name: BaseNet(self.feat_dim).to(self.device) for name in self.present_targets}
        )
        self.meta = MetaNet(len(self.present_targets)).to(self.device)

    # ----------------- utilities -----------------

    def _row_composition(self, row) -> Dict[str, float]:
        """Convert System + Mol Frac → normalized element fractions."""
        system = str(row["System"])
        comps = system.split("-")
        mol_frac = str(row["Mol Frac"]).strip()
        if mol_frac == "Pure Salt":
            fracs = [1.0] * len(comps)
        else:
            fracs = list(map(float, mol_frac.split("-")))

        total: Dict[str, float] = {}
        for cmp, f in zip(comps, fracs):
            for el, cnt in re.findall(r"([A-Z][a-z]*)(\d*)", cmp):
                total[el] = total.get(el, 0.0) + int(cnt or "1") * f
        s = sum(total.values()) or 1.0
        return {el: cnt / s for el, cnt in total.items()}

    def _make_loader(
        self,
        idx: np.ndarray,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        x = self.X_embedded[idx]
        y = self.y_std[idx]
        m = self.mask_all[idx]
        ds = TensorDataset(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(m, dtype=torch.bool),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    # ----------------- training base nets -----------------

    def train_base(self, max_epochs: int = 300):
        """Train one BaseNet per target using only rows where that target exists."""
        for prop in self.present_targets:
            print(f" • Training base net for {prop}")
            net = self.base_nets[prop]
            j = self.idx_map[prop]

            mask = self.mask_all[:, j]
            mask_tr_glb = mask & np.isin(self.idx_all, self.tr_idx)
            mask_va_glb = mask & np.isin(self.idx_all, self.va_idx)

            # if no validation data for this prop, re-split its valid indices
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

            opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=2e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max_epochs, eta_min=2e-4
            )
            best, patience, PAT = 1e9, 0, 25
            model_path = self.model_dir / f"base_{prop}_resnet_clean.pth"

            for epoch in range(max_epochs):
                net.train()
                total_loss = 0.0
                for xb, yb in tr_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    opt.zero_grad()
                    pred = net(xb)
                    loss = nn.functional.mse_loss(pred, yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    opt.step()
                    total_loss += loss.item()
                sched.step()

                if va_loader:
                    net.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for xb, yb in va_loader:
                            xb, yb = xb.to(self.device), yb.to(self.device)
                            pred = net(xb)
                            loss = nn.functional.mse_loss(pred, yb)
                            val_loss += loss.item()
                    val_loss /= len(va_loader)

                    # simple early stopping
                    if val_loss < best - 1e-4:
                        best, patience = val_loss, 0
                        torch.save(net.state_dict(), model_path)
                    else:
                        patience += 1
                        if patience >= PAT:
                            print(f" ⇢ Early stopping for {prop}")
                            break

            if va_loader and model_path.exists():
                net.load_state_dict(
                    torch.load(model_path, map_location=self.device)
                )

    # ----------------- physics loss -----------------

    def _physics_loss(
        self,
        pred_raw: torch.Tensor,
        y_raw: torch.Tensor,
        mb: torch.Tensor,
        T: torch.Tensor,
    ) -> torch.Tensor:
        """Physics-regularization loss on raw coefficients, masked."""
        loss = 0.0
        valid_terms = 0
        for dprop, req_coeffs in self.derived_props:
            coeff_indices = [self.idx_map[rc] for rc in req_coeffs if rc in self.idx_map]
            if len(coeff_indices) != len(req_coeffs):
                continue
            mask_rows = torch.all(mb[:, coeff_indices], dim=1)
            if not mask_rows.any():
                continue
            y_coeffs = y_raw[mask_rows][:, coeff_indices]
            p_coeffs = pred_raw[mask_rows][:, coeff_indices]

            with torch.no_grad():
                if dprop == "rho":
                    y_vals = y_coeffs[:, 0] - y_coeffs[:, 1] * T[mask_rows]
                    p_vals = p_coeffs[:, 0] - p_coeffs[:, 1] * T[mask_rows]
                    term_loss = nn.functional.mse_loss(p_vals, y_vals)
                elif dprop == "muA":
                    p_mu1_a = torch.clamp(p_coeffs[:, 0], min=1e-6)
                    p_vals = p_mu1_a * torch.exp(p_coeffs[:, 1] / (R * T[mask_rows]))
                    y_vals = y_coeffs[:, 0] * torch.exp(
                        y_coeffs[:, 1] / (R * T[mask_rows])
                    )
                    term_loss = nn.functional.mse_loss(
                        torch.log(p_vals + 1e-8), torch.log(y_vals + 1e-8)
                    )
                elif dprop == "muB":
                    y_log = (
                        y_coeffs[:, 0]
                        + y_coeffs[:, 1] / T[mask_rows]
                        + y_coeffs[:, 2] / T[mask_rows] ** 2
                    )
                    p_log = (
                        p_coeffs[:, 0]
                        + p_coeffs[:, 1] / T[mask_rows]
                        + p_coeffs[:, 2] / T[mask_rows] ** 2
                    )
                    term_loss = nn.functional.mse_loss(p_log, y_log)
                elif dprop == "k":
                    y_vals = y_coeffs[:, 0] + y_coeffs[:, 1] * T[mask_rows]
                    p_vals = p_coeffs[:, 0] + p_coeffs[:, 1] * T[mask_rows]
                    term_loss = nn.functional.mse_loss(p_vals, y_vals)
                elif dprop == "cp":
                    y_vals = (
                        y_coeffs[:, 0]
                        + y_coeffs[:, 1] * T[mask_rows]
                        + y_coeffs[:, 2] / T[mask_rows] ** 2
                    )
                    p_vals = (
                        p_coeffs[:, 0]
                        + p_coeffs[:, 1] * T[mask_rows]
                        + p_coeffs[:, 2] / T[mask_rows] ** 2
                    )
                    term_loss = nn.functional.mse_loss(p_vals, y_vals)
                else:
                    continue
            loss += term_loss
            valid_terms += 1

        if valid_terms == 0:
            return torch.tensor(0.0, device=self.device)
        return loss / valid_terms

    # ----------------- training meta net -----------------

    def train_meta(
        self,
        max_epochs: int = 600,
        physics_weight: float = 0.1,
        temp_range: Tuple[float, float] = (500.0, 1200.0),
    ):
        """Train MetaNet on top of frozen BaseNets with masked + physics loss."""
        for net in self.base_nets.values():
            for p in net.parameters():
                p.requires_grad_(False)

        def base_preds_tensor(xb: torch.Tensor) -> torch.Tensor:
            return torch.stack([self.base_nets[p](xb) for p in self.present_targets], dim=1)

        trL = self._make_loader(self.tr_idx, batch_size=64, shuffle=True)
        vaL = self._make_loader(self.va_idx, batch_size=256, shuffle=False)

        opt = torch.optim.AdamW(self.meta.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max_epochs, 1e-4)
        best, wait, PAT = 1e9, 0, 40
        meta_path = self.model_dir / "meta_resnet_clean.pth"

        μ_tensor = torch.tensor(self.μ, device=self.device, dtype=torch.float32)
        σ_tensor = torch.tensor(self.σ, device=self.device, dtype=torch.float32)

        print("\nStage-2: Training meta net with physics regularization...")
        for epoch in range(max_epochs):
            self.meta.train()
            total_loss = 0.0
            for xb, yb, mb in trL:
                xb, yb, mb = xb.to(self.device), yb.to(self.device), mb.to(self.device)
                batch_size = xb.size(0)
                T = torch.rand(batch_size, device=self.device) * (
                    temp_range[1] - temp_range[0]
                ) + temp_range[0]

                with torch.no_grad():
                    base_out = base_preds_tensor(xb)
                pred = base_out + self.meta(base_out)

                # coefficient loss (masked)
                loss_coeff = ((pred - yb) ** 2 * mb).sum() / mb.sum()

                # physics loss (on raw coefficients)
                pred_raw = pred * σ_tensor + μ_tensor
                yb_raw = yb * σ_tensor + μ_tensor
                loss_phys = self._physics_loss(pred_raw, yb_raw, mb, T) * physics_weight

                total = loss_coeff + loss_phys
                total.backward()
                nn.utils.clip_grad_norm_(self.meta.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                total_loss += total.item()

            sched.step()
            avg_loss = total_loss / len(trL)

            # validation
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

    # ----------------- evaluation -----------------

    def _eval_on_indices(
        self,
        idx: np.ndarray,
        split_name: str = "val",
        min_n: int = 5,
    ):
        """Masked metrics on given indices (val or test)."""
        self.meta.eval()
        for net in self.base_nets.values():
            net.eval()

        μ, σ = self.μ, self.σ
        Xs = self.X_embedded[idx]
        ys = self.y_raw[idx]
        ms = self.mask_all[idx]

        with torch.no_grad():
            xb = torch.tensor(Xs, device=self.device, dtype=torch.float32)
            base_out = torch.stack(
                [self.base_nets[p](xb).cpu() for p in self.present_targets],
                dim=1,
            ).numpy()
            pred_std = base_out + self.meta(
                torch.tensor(base_out, device=self.device, dtype=torch.float32)
            ).cpu().numpy()
        pred = pred_std * σ + μ  # de-standardise

        per_target: Dict[str, Dict[str, float]] = {}
        rel_mses, r2s, maes, p90s = [], [], [], []

        print(f"\n{split_name.capitalize()} results — relMSE (%), R², MAE, p90 rel err")
        for j, prop in enumerate(self.present_targets):
            mask_j = ms[:, j]
            n_j = int(mask_j.sum())
            if n_j < min_n:
                print(f" • {prop:<8s}: [skipped: only {n_j} samples]")
                continue

            yt = ys[mask_j, j]
            yp = pred[mask_j, j]

            m_rel = _rel_mse_pct(yt, yp)
            r2 = r2_score(yt, yp)
            mae = _mae(yt, yp)
            p90 = _p90_rel_err(yt, yp)

            per_target[prop] = {
                "relMSE_pct": float(m_rel),
                "R2": float(r2),
                "MAE": float(mae),
                "p90_rel_err": float(p90),
                "n": n_j,
            }

            rel_mses.append(m_rel)
            r2s.append(r2)
            maes.append(mae)
            p90s.append(p90)

            print(
                f" • {prop:<8s}: {m_rel:8.2f}%   R²={r2:+.3f}   "
                f"MAE={mae:9.3g}   p90_rel={p90:6.3f}   (n={n_j})"
            )

        if rel_mses:
            avg_rel_mse = float(np.mean(rel_mses))
            avg_r2 = float(np.mean(r2s))
            avg_mae = float(np.mean(maes))
            avg_p90 = float(np.mean(p90s))
            print(
                f" ⇒ Average over {len(rel_mses)} targets: "
                f"relMSE={avg_rel_mse:8.2f}%   R²={avg_r2:+.3f}   "
                f"MAE={avg_mae:9.3g}   p90_rel={avg_p90:6.3f}"
            )
        else:
            avg_rel_mse = avg_r2 = avg_mae = avg_p90 = float("nan")

        return {
            "avg_relMSE_pct": avg_rel_mse,
            "avg_R2": avg_r2,
            "avg_MAE": avg_mae,
            "avg_p90_rel_err": avg_p90,
            "per_target": per_target,
        }

    def evaluate(self, split: str = "val", min_n: int = 5):
        """Evaluate on 'val' or 'test' split with masked metrics."""
        if split == "val":
            idx = self.va_idx
        elif split == "test":
            idx = self.te_idx
        else:
            raise ValueError("split must be 'val' or 'test'")
        return self._eval_on_indices(idx, split_name=split, min_n=min_n)

    # ----------------- save/load -----------------

    def save(self, path: str):
        path = Path(path)
        os.makedirs(path, exist_ok=True)
        for prop, net in self.base_nets.items():
            torch.save(net.state_dict(), path / f"base_{prop}_resnet_clean.pth")
        torch.save(self.meta.state_dict(), path / "meta_resnet_clean.pth")
        np.save(path / "μ_resnet_clean.npy", self.μ)
        np.save(path / "σ_resnet_clean.npy", self.σ)
        pd.to_pickle(self.poly, path / "poly_resnet_clean.pkl")
        pd.to_pickle(self.scaler, path / "scaler_resnet_clean.pkl")
        pd.to_pickle(self.X_comp.columns.tolist(), path / "elements_resnet_clean.pkl")

    def load(self, path: str):
        path = Path(path)
        for prop in self.present_targets:
            self.base_nets[prop].load_state_dict(
                torch.load(path / f"base_{prop}_resnet_clean.pth", map_location=self.device)
            )
        self.meta.load_state_dict(
            torch.load(path / "meta_resnet_clean.pth", map_location=self.device)
        )
        self.μ = np.load(path / "μ_resnet_clean.npy")
        self.σ = np.load(path / "σ_resnet_clean.npy")
        self.poly = pd.read_pickle(path / "poly_resnet_clean.pkl")
        self.scaler = pd.read_pickle(path / "scaler_resnet_clean.pkl")
        self.X_comp.columns = pd.read_pickle(path / "elements_resnet_clean.pkl")


# ----------------- K-fold wrapper -----------------


def cross_validate_resnet_clean(
    df: pd.DataFrame,
    target_columns: Sequence[str],
    derived_props: Sequence[Tuple[str, Sequence[str]]],
    k: int = 5,
    random_state: int = SEED,
    degree_poly: int = 2,
    embedding_method: str = "none",
    n_components: int = 10,
    min_samples_per_target: int = 10,
) -> List[Dict]:
    """
    Run k-fold CV for ResNetMetaTrainer.

    For each fold:
      - use train indices for fitting,
      - split held-out indices into internal val/test (50/50),
      - train base + meta,
      - evaluate on both val and test (masked metrics).
    """
    idx_all = np.arange(len(df))
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    fold_results: List[Dict] = []

    for fold, (train_idx, heldout_idx) in enumerate(kf.split(idx_all), start=1):
        va_idx, te_idx = train_test_split(
            heldout_idx, test_size=0.5, random_state=random_state
        )

        print(f"\n================ Fold {fold}/{k} ================")
        print(f"Train: {len(train_idx)} | Val: {len(va_idx)} | Test: {len(te_idx)}")

        trainer = ResNetMetaTrainerv2(
            df=df,
            target_columns=target_columns,
            derived_props=derived_props,
            degree_poly=degree_poly,
            embedding_method=embedding_method,
            n_components=n_components,
            model_dir=f"../data/trained_models_resnet_fold{fold}",
            tr_idx=train_idx,
            va_idx=va_idx,
            te_idx=te_idx,
            min_samples_per_target=min_samples_per_target,
        )
        trainer.train_base()
        trainer.train_meta()

        val_metrics = trainer.evaluate(split="val")
        test_metrics = trainer.evaluate(split="test")

        fold_results.append(
            {
                "fold": fold,
                "n_train": len(train_idx),
                "n_val": len(va_idx),
                "n_test": len(te_idx),
                "val": val_metrics,
                "test": test_metrics,
            }
        )

    return fold_results
""" 
if __name__ == "__main__":
    # Example: load your processed MSTDB
    df = pd.read_csv("/Users/meggie/Documents/MoltenSaltPropnet/data/new_mstdb_janz.csv").rename(columns=str.strip)

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

    print("\n==== Single-split training ====")
    trainer = ResNetMetaTrainerv2(
        df=df,
        target_columns=TARGETS,
        derived_props=DERIVED_PROPS,
        degree_poly=2,
        embedding_method="none", 
        n_components=10,
        min_samples_per_target=10,
    )
    trainer.train_base()
    trainer.train_meta()
    trainer.evaluate(split="val")
    trainer.evaluate(split="test")

    print("\n==== K-fold cross-validation ====")
    cv_results = cross_validate_resnet_clean(
        df=df,
        target_columns=TARGETS,
        derived_props=DERIVED_PROPS,
        k=5,
        embedding_method="none",
        n_components=10,
        min_samples_per_target=10,
    )
 """