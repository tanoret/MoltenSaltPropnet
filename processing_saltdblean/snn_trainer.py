"""
snn_trainer.py — SNN + Meta + Physics-Regularized trainer (A/B-ready)
--------------------------------------------------------------------
Supports A/B experiments:
- Run A: use_element_features=False
- Run B: use_element_features=True
Same splits, separate model_dir, comparable plots.

Usage:
    from processing_saltdblean.snn_trainer import SNNMetaTrainer, TARGETS, DERIVED_PROPS, ELEMENT_FEATURE_COLS
"""

import re
import ast
import math
import random
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

import snntorch as snn
from snntorch import surrogate

from processing_saltdblean.embedding_preconditioner import EmbeddingPreconditioner


# ────────────────────────────────────────────────────────────────
#  Global config
# ────────────────────────────────────────────────────────────────
SEED = 42
R = 8.314
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
warnings.filterwarnings("ignore", category=FutureWarning)


TARGETS = [
    "Melt(K)", "Boil(K)",
    "rho_a", "rho_b",
    "mu1_a", "mu1_b",
    "mu2_a", "mu2_b", "mu2_c",
    "k_a", "k_b",
    "cp_a", "cp_b", "cp_c"
]

DERIVED_PROPS = [
    ("rho", ["rho_a", "rho_b"]),
    ("muA", ["mu1_a", "mu1_b"]),
    ("muB", ["mu2_a", "mu2_b", "mu2_c"]),
    ("k",   ["k_a", "k_b"]),
    ("cp",  ["cp_a", "cp_b", "cp_c"]),
]

# Same dict-feature columns as ResNet
ELEMENT_FEATURE_COLS = [
    "polarizability_element[10-24Cm3]",
    "atomic_mass_element",
    "electronegativity_element",
    "atomic_radius_element[Angstrom]",
    "ionic_radius_element[Angstrom]",
    # "covalent_radius_element[Angstrom]",  # optional
    "first_ionization_energy[kJ_per_mol]",
]


def _rel_mse_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Relative MSE as % of <y^2>."""
    mse = mean_squared_error(y_true, y_pred)
    denom = np.mean(y_true ** 2) or 1e-12
    return 100.0 * mse / denom


# ────────────────────────────────────────────────────────────────
#  Low-level SNN blocks (ARCHITECTURE UNCHANGED)
# ────────────────────────────────────────────────────────────────
class SNNBase(nn.Module):
    """Spiking base net predicting one coefficient."""
    def __init__(self, in_dim: int, hidden: int = 64, out_dim: int = 1, steps: int = 10):
        super().__init__()
        self.steps = steps
        sg = surrogate.fast_sigmoid(slope=25)

        self.fc1     = nn.Linear(in_dim, hidden)
        self.lif1    = snn.Leaky(beta=0.9, spike_grad=sg)

        self.fc2     = nn.Linear(hidden, hidden)
        self.lif2    = snn.Leaky(beta=0.9, spike_grad=sg)

        self.fc_out  = nn.Linear(hidden, out_dim)
        self.lif_out = snn.Leaky(beta=0.9, spike_grad=sg, output=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        memo = self.lif_out.init_leaky()

        rec = []
        for _ in range(self.steps):
            spk1, mem1 = self.lif1(self.fc1(x), mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            _,   memo  = self.lif_out(self.fc_out(spk2), memo)
            rec.append(memo)
        return torch.stack(rec, 0).mean(0)  # (B, out_dim)


class SNNMeta(nn.Module):
    """Meta-correction net working on stacked base predictions."""
    def __init__(self, in_dim: int, hidden: int = 64, out_dim: int = 1, steps: int = 10):
        super().__init__()
        self.steps = steps
        sg = surrogate.fast_sigmoid(slope=25)

        self.fc1     = nn.Linear(in_dim, hidden)
        self.lif1    = snn.Leaky(beta=0.9, spike_grad=sg)

        self.fc2     = nn.Linear(hidden, hidden)
        self.lif2    = snn.Leaky(beta=0.9, spike_grad=sg)

        self.fc_out  = nn.Linear(hidden, out_dim)
        self.lif_out = snn.Leaky(beta=0.9, spike_grad=sg, output=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        memo = self.lif_out.init_leaky()

        rec = []
        for _ in range(self.steps):
            spk1, mem1 = self.lif1(self.fc1(x), mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            _,   memo  = self.lif_out(self.fc_out(spk2), memo)
            rec.append(memo)
        return torch.stack(rec, 0).mean(0)  # (B, out_dim)


# ────────────────────────────────────────────────────────────────
#  Trainer (A/B capable with element features + PCA/embedding)
# ────────────────────────────────────────────────────────────────
class SNNMetaTrainer:
    def __init__(
        self,
        df: pd.DataFrame,
        target_cols: List[str],
        derived_props: List[Tuple[str, List[str]]],
        element_feature_cols: Optional[List[str]] = None,
        use_element_features: bool = True,
        degree_poly: int = 3,
        embedding_method: str = "none",   # set to "pca" in your run script
        n_components: int = 10,           # set to 32 in your run script
        splits=None,
        model_dir=None,
        snn_steps_base: int = 10,
        snn_steps_meta: int = 10,
    ):
        self.df = df.copy()
        self.target_columns = target_cols
        self.derived_props = derived_props
        self.ELEMENT_FEATURE_COLS = list(element_feature_cols) if element_feature_cols is not None else []
        self.use_element_features = bool(use_element_features)
        self.device = DEVICE

        self.embedding_method = str(embedding_method)
        self.n_components = int(n_components)

        self.snn_steps_base = int(snn_steps_base)
        self.snn_steps_meta = int(snn_steps_meta)

        # model_dir
        self.model_dir = Path(model_dir) if model_dir is not None else Path("../data/trained_models_snn")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # ---------- Clean and identify valid target columns ----------
        self.present_targets = []
        for t in target_cols:
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
            raise RuntimeError("No valid target columns found in DataFrame.")

        # ---------- Composition normalization ----------
        self.df["Composition"] = self.df.apply(self._row_composition, axis=1)

        # Element-fraction matrix
        self.X_comp = pd.json_normalize(self.df["Composition"]).fillna(0.0)
        self.X_comp = self.X_comp.reindex(sorted(self.X_comp.columns), axis=1)
        self.composition_df = self.X_comp
        self.fractions = self.X_comp.to_numpy(np.float32)

        # ---------- Poly features ----------
        self.poly = PolynomialFeatures(degree_poly, include_bias=False)
        X_poly = self.poly.fit_transform(self.X_comp).astype(np.float32)
        self.poly_scaler = StandardScaler()
        X_poly = self.poly_scaler.fit_transform(X_poly).astype(np.float32)
        self.X_poly = X_poly

        # ---------- Optional: element feature cols ----------
        if self.use_element_features:
            if not self.ELEMENT_FEATURE_COLS:
                raise ValueError("use_element_features=True but element_feature_cols is empty.")
            missing = [c for c in self.ELEMENT_FEATURE_COLS if c not in self.df.columns]
            if missing:
                raise ValueError(f"Missing ELEMENT_FEATURE_COLS in df: {missing}")

            # parse dict columns
            for col in self.ELEMENT_FEATURE_COLS:
                self.df[col] = self.df[col].apply(self._to_dict)

            # lookup tables for predict()
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

            # weighted means per row
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

            element_features = np.vstack(feat_mat).T.astype(np.float32)  # (N, n_elem_feats)
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
            tr_idx, te_idx = train_test_split(self.idx_all, test_size=0.20, random_state=SEED)
            tr_idx, va_idx = train_test_split(tr_idx, test_size=0.20, random_state=SEED)
        else:
            tr_idx, va_idx, te_idx = splits
        self.tr_idx, self.va_idx, self.te_idx = tr_idx, va_idx, te_idx

        # ---------- Embedding (PCA here) ----------
        self.embedder = EmbeddingPreconditioner(method=self.embedding_method, n_components=self.n_components)
        self.embedder.fit(self.X[self.tr_idx])
        self.X_embedded = self.embedder.transform(self.X).astype(np.float32)
        self.feat_dim = self.X_embedded.shape[1]

        # ---------- Normalize targets ----------
        self.μ = self.y_raw[self.tr_idx].mean(0)
        self.σ = self.y_raw[self.tr_idx].std(0)
        self.σ[self.σ == 0] = 1.0
        self.y_std = (self.y_raw - self.μ) / self.σ

        # ---------- Models ----------
        self.idx_map = {n: j for j, n in enumerate(self.present_targets)}
        self.base_nets = nn.ModuleDict({
            n: SNNBase(self.feat_dim, steps=self.snn_steps_base).to(self.device)
            for n in self.present_targets
        })
        self.meta = SNNMeta(
            in_dim=len(self.present_targets),
            hidden=64,
            out_dim=len(self.present_targets),
            steps=self.snn_steps_meta
        ).to(self.device)

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

    def _loader(self, x, y, m, bs, shuf):
        ds = TensorDataset(torch.tensor(x), torch.tensor(y), torch.tensor(m))
        return DataLoader(ds, batch_size=bs, shuffle=shuf, drop_last=False)

    # ----------------- base training -----------------
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
                    tr_p, va_p = train_test_split(idx_prop, test_size=0.2, random_state=SEED)
                    mask_tr = np.isin(self.idx_all, tr_p)
                    mask_va = np.isin(self.idx_all, va_p)
                else:
                    mask_va = np.zeros_like(mask_tr, bool)

            x_tr, y_tr = self.X_embedded[mask_tr], self.y_std[mask_tr, j:j+1]
            x_va, y_va = self.X_embedded[mask_va], self.y_std[mask_va, j:j+1]

            trL = DataLoader(TensorDataset(torch.tensor(x_tr), torch.tensor(y_tr)), batch_size=64, shuffle=True)
            vaL = DataLoader(TensorDataset(torch.tensor(x_va), torch.tensor(y_va)), batch_size=256, shuffle=False) if len(x_va) else None

            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 150, 1e-4)
            best, wait, PAT = 1e9, 0, 80
            model_path = self.model_dir / f"base_{prop}_snn.pth"

            for epoch in range(200):
                net.train()
                for xb, yb in trL:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    opt.zero_grad()
                    pred = net(xb)
                    loss = nn.functional.mse_loss(pred, yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
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
                        best, wait = val_loss, 0
                        torch.save(net.state_dict(), model_path)
                    else:
                        wait += 1
                        if wait >= PAT:
                            print(f" ⇢ Early stopping for {prop}")
                            break

            if model_path.exists():
                net.load_state_dict(torch.load(model_path, map_location=self.device))

    # ----------------- meta training -----------------
    def _base_preds(self, xb: torch.Tensor) -> torch.Tensor:
        outs = [self.base_nets[p](xb) for p in self.present_targets]  # each (B,1)
        return torch.cat(outs, dim=1)  # (B, Nprop)

    def _physics_loss(self, pred_raw, y_raw, mask_b, T):
        loss, terms = 0.0, 0
        for dprop, coeffs in self.derived_props:
            idxs = [self.idx_map[c] for c in coeffs if c in self.idx_map]
            if len(idxs) != len(coeffs):
                continue
            m = torch.all(mask_b[:, idxs], dim=1)
            if not m.any():
                continue
            y = y_raw[m][:, idxs]
            p = pred_raw[m][:, idxs]

            if dprop == "rho":
                loss_t = nn.functional.mse_loss(p[:, 0] - p[:, 1] * T[m], y[:, 0] - y[:, 1] * T[m])
            elif dprop == "muA":
                p_mu1_a = torch.clamp(p[:, 0], min=1e-6)
                y_mu1_a = torch.clamp(y[:, 0], min=1e-6)
                p_vals = p_mu1_a * torch.exp(p[:, 1] / (R * T[m]))
                y_vals = y_mu1_a * torch.exp(y[:, 1] / (R * T[m]))
                loss_t = nn.functional.mse_loss(torch.log(p_vals + 1e-8), torch.log(y_vals + 1e-8))
            elif dprop == "muB":
                loss_t = nn.functional.mse_loss(
                    p[:, 0] + p[:, 1] / T[m] + p[:, 2] / T[m] ** 2,
                    y[:, 0] + y[:, 1] / T[m] + y[:, 2] / T[m] ** 2,
                )
            elif dprop == "k":
                loss_t = nn.functional.mse_loss(p[:, 0] + p[:, 1] * T[m], y[:, 0] + y[:, 1] * T[m])
            elif dprop == "cp":
                loss_t = nn.functional.mse_loss(
                    p[:, 0] + p[:, 1] * T[m] + p[:, 2] / T[m] ** 2,
                    y[:, 0] + y[:, 1] * T[m] + y[:, 2] / T[m] ** 2,
                )
            else:
                continue

            loss += loss_t
            terms += 1
        return loss / terms if terms else torch.tensor(0.0, device=self.device)

    def train_meta(self, physics_weight: float = 0.1, temp_range=(500, 1200)):
        # freeze bases
        for net in self.base_nets.values():
            for p in net.parameters():
                p.requires_grad_(False)

        trL = self._loader(self.X_embedded[self.tr_idx], self.y_std[self.tr_idx], self.mask_all[self.tr_idx], 64, True)
        vaL = self._loader(self.X_embedded[self.va_idx], self.y_std[self.va_idx], self.mask_all[self.va_idx], 256, False)

        opt = torch.optim.AdamW(self.meta.parameters(), lr=8e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 300, 1e-4)
        best, wait, PAT = 1e9, 0, 100

        meta_path = self.model_dir / "meta_snn.pth"

        μ_t = torch.tensor(self.μ, device=self.device, dtype=torch.float32)
        σ_t = torch.tensor(self.σ, device=self.device, dtype=torch.float32)

        print("\nStage-2: Training meta net with physics regularization…")
        for ep in range(400):
            self.meta.train()
            tot = 0.0

            for xb, yb, mb in trL:
                xb, yb, mb = xb.to(self.device), yb.to(self.device), mb.to(self.device)
                T = torch.rand(len(xb), device=self.device) * (temp_range[1] - temp_range[0]) + temp_range[0]

                with torch.no_grad():
                    base = self._base_preds(xb)

                pred = base + self.meta(base)
                loss_coeff = ((pred - yb) ** 2 * mb).sum() / mb.sum()

                pred_raw = pred * σ_t + μ_t
                yb_raw = yb * σ_t + μ_t
                loss_phys = self._physics_loss(pred_raw, yb_raw, mb, T) * float(physics_weight)

                loss = loss_coeff + loss_phys
                loss.backward()
                nn.utils.clip_grad_norm_(self.meta.parameters(), 0.5)
                opt.step()
                opt.zero_grad()
                tot += loss.item()

            sched.step()

            # validation
            self.meta.eval()
            val = 0.0
            with torch.no_grad():
                for xb, yb, mb in vaL:
                    xb, yb, mb = xb.to(self.device), yb.to(self.device), mb.to(self.device)
                    base = self._base_preds(xb)
                    pred = base + self.meta(base)
                    val += ((pred - yb) ** 2 * mb).sum().item() / mb.sum().item()
            val /= len(vaL)

            if val < best - 1e-4:
                best, wait = val, 0
                torch.save(self.meta.state_dict(), meta_path)
            else:
                wait += 1
                if wait >= PAT:
                    print(" ⇢ Early stopping meta")
                    break

        if meta_path.exists():
            self.meta.load_state_dict(torch.load(meta_path, map_location=self.device))

    # ----------------- evaluation -----------------
    def evaluate(self, split="val", return_dict: bool = False):
        split_map = {"train": self.tr_idx, "val": self.va_idx, "test": self.te_idx}
        idxs = split_map[split]

        self.meta.eval()
        for net in self.base_nets.values():
            net.eval()

        Xs = self.X_embedded[idxs]
        ys = self.y_raw[idxs]
        ms = self.mask_all[idxs].astype(bool)

        with torch.no_grad():
            xb = torch.tensor(Xs, device=self.device)
            base_out = torch.cat([self.base_nets[p](xb) for p in self.present_targets], dim=1)
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
            "per_target": per_target,
        }

        print(f"\n[{split.upper()}] avg MSE%={out['avg_mse_pct']:.3f} | avg R2={out['avg_r2']:.3f}")
        if return_dict:
            return out

    # ----------------- predict -----------------
    def predict(self, composition: Dict[str, float]) -> Dict[str, float]:
        """
        Predict coefficients for a single composition.
        Assumes models are already trained/loaded in this instance.
        """
        # composition -> element fractions
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

        feats = self.embedder.transform(feats).astype(np.float32)
        xb = torch.tensor(feats, device=self.device, dtype=torch.float32)

        self.meta.eval()
        for net in self.base_nets.values():
            net.eval()

        with torch.no_grad():
            base_out = torch.cat([self.base_nets[p](xb) for p in self.present_targets], dim=1)
            pred_std = (base_out + self.meta(base_out)).cpu().numpy()[0]

        pred_raw = pred_std * self.σ + self.μ
        return {prop: float(pred_raw[i]) for i, prop in enumerate(self.present_targets)}

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
