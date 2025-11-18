"""
snn_trainer.py  — Spiking‐NN + Meta + Physics-Regularized trainer
-----------------------------------------------------------------
Clean, self-contained implementation ready for import:

    from snn_trainer import SNNMetaTrainer, TARGETS, DERIVED_PROPS
"""

import os
import re
import math
import random
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

import snntorch as snn
from snntorch import surrogate

from processing_saltdblean.embedding_preconditioner import EmbeddingPreconditioner

from sklearn.metrics import mean_squared_error, r2_score

def _rel_mse_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return relative MSE as a percentage of ⟨y²⟩ — avoids unit issues."""
    mse = mean_squared_error(y_true, y_pred)
    denom = np.mean(y_true ** 2) or 1e-12           # guard /0
    return 100.0 * mse / denom

# ────────────────────────────────────────────────────────────────
#  Global config
# ────────────────────────────────────────────────────────────────
SEED = 42
R = 8.314
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
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
    ('rho', ['rho_a', 'rho_b']),
    ('muA', ['mu1_a', 'mu1_b']),
    ('muB', ['mu2_a', 'mu2_b', 'mu2_c']),
    ('k',   ['k_a',   'k_b']),
    ('cp',  ['cp_a',  'cp_b', 'cp_c'])
]

# ────────────────────────────────────────────────────────────────
#  Low-level SNN blocks
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
        mem1, mem2, mem_o = (self.lif1.init_leaky(),
                             self.lif2.init_leaky(),
                             self.lif_out.init_leaky())
        rec = []
        for _ in range(self.steps):
            spk1, mem1 = self.lif1(self.fc1(x), mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            _,   mem_o = self.lif_out(self.fc_out(spk2), mem_o)
            rec.append(mem_o)
        return torch.stack(rec, 0).mean(0)          # (B, out_dim)

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
        mem1, mem2, mem_o = (self.lif1.init_leaky(),
                             self.lif2.init_leaky(),
                             self.lif_out.init_leaky())
        rec = []
        for _ in range(self.steps):
            spk1, mem1 = self.lif1(self.fc1(x), mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            _,   mem_o = self.lif_out(self.fc_out(spk2), mem_o)
            rec.append(mem_o)
        return torch.stack(rec, 0).mean(0)          # (B, out_dim)

# ────────────────────────────────────────────────────────────────
#  Trainer
# ────────────────────────────────────────────────────────────────
class SNNMetaTrainer:
    """Full trainer: base SNNs + meta-net + physics regularization."""
    # ──────────────────────── init ───────────────────────────────
    def __init__(self,
                 df: pd.DataFrame,
                 target_cols,
                 derived_props,
                 degree_poly: int = 3,
                 embedding_method = 'none',
                 n_components: int = 10):
        self.df             = df.copy()
        self.target_columns = target_cols
        self.derived_props  = derived_props
        self.device         = device
        self.model_dir      = Path("../data/trained_models")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # ── clean targets & detect available ones ────────────────
        self.present_targets = []
        for t in target_cols:
            if t in self.df.columns:
                self.df[t] = (self.df[t]
                              .replace(["----", ""], np.nan)
                              .replace(r"\*", "", regex=True)
                              .pipe(pd.to_numeric, errors="coerce"))
                if np.isfinite(self.df[t]).any():
                    self.present_targets.append(t)

        if not self.present_targets:
            raise RuntimeError("No valid target columns found in DataFrame.")

        # ── composition features ────────────────────────────────
        self.df["Composition"] = self.df.apply(self._row_composition, axis=1)
        self.X_comp = pd.json_normalize(self.df["Composition"]).fillna(0.0)
        self.X_comp = self.X_comp.reindex(sorted(self.X_comp.columns), axis=1)
        self.poly   = PolynomialFeatures(degree_poly, include_bias=False)
        X_poly      = self.poly.fit_transform(self.X_comp)
        self.scaler = StandardScaler()
        X_poly      = self.scaler.fit_transform(X_poly).astype(np.float32)
        frac        = self.X_comp.to_numpy(np.float32)
        self.X      = np.hstack([X_poly, frac])
        self.feat_dim = self.X.shape[1]

        self.composition_df = self.X_comp

        # ── target matrix, masks, splits ────────────────────────
        mask_all = np.isfinite(self.df[self.present_targets]).to_numpy(bool)
        # self.df[self.present_targets] = self.df[self.present_targets]\
        #                                     .fillna(self.df[self.present_targets].mean())
        self.df[self.present_targets] = (self.df[self.present_targets]).fillna(0.0)
        y_raw = self.df[self.present_targets].to_numpy(np.float32)

        idx = np.arange(len(self.X))
        tr, te = train_test_split(idx, test_size=0.20, random_state=SEED)
        tr, va = train_test_split(tr, test_size=0.20, random_state=SEED)

        # Embedding Block
        self.embedding_method = embedding_method
        self.n_components = n_components
        self.embedder = EmbeddingPreconditioner(method=embedding_method, n_components=n_components)
        self.embedder.fit(self.X[tr])
        self.X_embedded = self.embedder.transform(self.X)
        self.feat_dim = self.n_components if embedding_method != 'none' else self.X.shape[1]

        μ = y_raw[tr].mean(0)
        σ = y_raw[tr].std(0)
        σ[σ == 0] = 1.0

        self.mask_all = mask_all
        self.y_raw    = y_raw
        self.y_std    = (y_raw - μ) / σ
        self.μ, self.σ = μ, σ
        self.tr_idx, self.va_idx, self.te_idx = tr, va, te
        self.idx_map = {n: j for j, n in enumerate(self.present_targets)}

        # ── instantiate nets ────────────────────────────────────
        self.base_nets = nn.ModuleDict({
            n: SNNBase(self.feat_dim).to(device) for n in self.present_targets
        })
        self.meta = SNNMeta(len(self.present_targets),
                            hidden=64,
                            out_dim=len(self.present_targets)).to(device)

    # ──────────────────────── helpers ───────────────────────────
    @staticmethod
    def _row_composition(row):
        comps = row["System"].split("-")
        fracs = ([1.0] * len(comps)
                 if row["Mol Frac"].strip() == "Pure Salt"
                 else list(map(float, row["Mol Frac"].split("-"))))
        total = {}
        for cmp, f in zip(comps, fracs):
            for el, cnt in re.findall(r"([A-Z][a-z]*)(\d*)", cmp):
                total[el] = total.get(el, 0) + int(cnt or "1") * f
        s = sum(total.values())
        return {el: cnt / s for el, cnt in total.items()}

    @staticmethod
    def _loader(x, y, m, bs, shuf):
        ds = TensorDataset(torch.tensor(x),
                           torch.tensor(y),
                           torch.tensor(m))
        return DataLoader(ds, batch_size=bs, shuffle=shuf, drop_last=False)

    # ─────────────────────── base training ──────────────────────
    def train_base(self):
        """Train one independent SNN per present property."""
        for prop in self.present_targets:
            print(f" • Training base net for {prop}")
            net  = self.base_nets[prop]
            j    = self.idx_map[prop]

            mask = self.mask_all[:, j]
            mask_tr = mask & np.isin(np.arange(len(self.X)), self.tr_idx)
            mask_va = mask & np.isin(np.arange(len(self.X)), self.va_idx)

            # handle edge case: no validation points
            if mask_va.sum() == 0:
                idx_prop = np.where(mask)[0]
                if len(idx_prop) >= 2:
                    tr_p, va_p = train_test_split(idx_prop, test_size=0.2, random_state=SEED)
                    mask_tr = np.isin(np.arange(len(self.X)), tr_p)
                    mask_va = np.isin(np.arange(len(self.X)), va_p)
                else:
                    mask_va = np.zeros_like(mask_tr, bool)

            x_tr, y_tr = self.X_embedded[mask_tr], self.y_std[mask_tr, j:j+1]
            x_va, y_va = self.X_embedded[mask_va], self.y_std[mask_va, j:j+1]

            trL = DataLoader(TensorDataset(torch.tensor(x_tr), torch.tensor(y_tr)),
                             batch_size=64, shuffle=True)
            vaL = (DataLoader(TensorDataset(torch.tensor(x_va), torch.tensor(y_va)),
                              batch_size=256, shuffle=False)
                   if len(x_va) else None)

            opt   = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 150, 1e-4)
            best, wait, PAT = 1e9, 0, 100
            model_path = self.model_dir / f"base_{prop}_snn.pth"

            for epoch in range(200):
                net.train()
                for xb, yb in trL:
                    xb, yb = xb.to(device), yb.to(device)
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
                            xb, yb = xb.to(device), yb.to(device)
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

            try:
                net.load_state_dict(torch.load(model_path, map_location=device))
            except:
                pass

    # ─────────────────────── meta training ──────────────────────
    def _base_preds(self, xb: torch.Tensor) -> torch.Tensor:
        """Concatenate base predictions (B, Nprop)."""
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
            if dprop == 'rho':
                loss_t = nn.functional.mse_loss(p[:,0]-p[:,1]*T[m], y[:,0]-y[:,1]*T[m])
            elif dprop == 'muA':
                loss_t = nn.functional.mse_loss(
                    torch.log(torch.clamp(p[:,0],1e-6)*torch.exp(p[:,1]/(R*T[m]))),
                    torch.log(y[:,0]*torch.exp(y[:,1]/(R*T[m]))))
            elif dprop == 'muB':
                loss_t = nn.functional.mse_loss(
                    p[:,0]+p[:,1]/T[m]+p[:,2]/T[m]**2,
                    y[:,0]+y[:,1]/T[m]+y[:,2]/T[m]**2)
            elif dprop == 'k':
                loss_t = nn.functional.mse_loss(p[:,0]+p[:,1]*T[m],
                                                y[:,0]+y[:,1]*T[m])
            elif dprop == 'cp':
                loss_t = nn.functional.mse_loss(
                    p[:,0]+p[:,1]*T[m]+p[:,2]/T[m]**2,
                    y[:,0]+y[:,1]*T[m]+y[:,2]/T[m]**2)
            else:
                continue
            loss += loss_t
            terms += 1
        return loss/terms if terms else torch.tensor(0., device=device)

    def train_meta(self):
        """Train meta network with physics regularization."""
        # freeze bases
        for net in self.base_nets.values():
            for p in net.parameters(): p.requires_grad_(False)

        trL = self._loader(self.X_embedded[self.tr_idx],
                           self.y_std[self.tr_idx],
                           self.mask_all[self.tr_idx],
                           64, True)
        vaL = self._loader(self.X_embedded[self.va_idx],
                           self.y_std[self.va_idx],
                           self.mask_all[self.va_idx],
                           256, False)

        opt   = torch.optim.AdamW(self.meta.parameters(), lr=8e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 300, 1e-4)
        best, wait, PAT = 1e9, 0, 100
        pth = self.model_dir / "meta_snn.pth"
        μ_t, σ_t = (torch.tensor(self.μ, device=device),
                    torch.tensor(self.σ, device=device))

        print("\nStage-2: Training meta net with physics regularization…")
        for ep in range(400):
            self.meta.train()
            tot = 0.0
            for xb, yb, mb in trL:
                xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                T = torch.rand(len(xb), device=device)*700 + 500  # 500–1200 K
                with torch.no_grad():
                    base = self._base_preds(xb)
                pred = base + self.meta(base)
                loss_coeff = ((pred - yb)**2 * mb).sum() / mb.sum()
                loss_phys  = self._physics_loss(pred*σ_t+μ_t, yb*σ_t+μ_t, mb, T)*0.1
                loss = loss_coeff + loss_phys
                loss.backward()
                nn.utils.clip_grad_norm_(self.meta.parameters(), .5)
                opt.step(); opt.zero_grad()
                tot += loss.item()
            sched.step()

            # ── validation ──
            self.meta.eval()
            val = 0.0
            with torch.no_grad():
                for xb, yb, mb in vaL:
                    xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                    base = self._base_preds(xb)
                    pred = base + self.meta(base)
                    val += ((pred - yb)**2 * mb).sum().item() / mb.sum().item()
            val /= len(vaL)
            print(f"Epoch {ep:3d} | train {tot/len(trL):.4f} | val {val:.4f}")

            if val < best - 1e-4:
                best, wait = val, 0
                torch.save(self.meta.state_dict(), pth)
            else:
                wait += 1
                if wait >= PAT:
                    print(" ⇢ Early stopping meta")
                    break

        self.meta.load_state_dict(torch.load(pth, map_location=device))


    # ─────────────────────── train joint net ─────────────────
    def train_joint(self):
        """Train both base SNNs and the meta SNN network jointly with a combined MSE and physics loss."""
        # Prepare data loaders for training and validation
        trL = self._loader(self.X_embedded[self.tr_idx], self.y_std[self.tr_idx], self.mask_all[self.tr_idx], 64, True)
        vaL = self._loader(self.X_embedded[self.va_idx], self.y_std[self.va_idx], self.mask_all[self.va_idx], 256, False)

        # Collect all parameters from base and meta networks
        all_params = list(self.meta.parameters())
        for net in self.base_nets.values():
            all_params += list(net.parameters())
        opt = torch.optim.AdamW(all_params, lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=400, eta_min=1e-4)

        # Training hyperparameters
        PHYSICS_WEIGHT = 0.1  # Weight for physics-based regularization
        TEMP_RANGE = (500, 1200)  # Temperature range for physics loss
        best, wait, PAT = 1e9, 0, 100  # Early stopping parameters

        # Convert mean and std to tensors for physics loss
        μ_t = torch.tensor(self.μ, device=self.device, dtype=torch.float32)
        σ_t = torch.tensor(self.σ, device=self.device, dtype=torch.float32)

        print("\nJoint Training: Optimizing base and meta SNN networks together...")
        for ep in range(400):
            # Set all networks to training mode
            for net in self.base_nets.values():
                net.train()
            self.meta.train()
            tot = 0.0

            for xb, yb, mb in trL:
                xb, yb, mb = xb.to(self.device), yb.to(self.device), mb.to(self.device)
                T = torch.rand(len(xb), device=self.device) * (TEMP_RANGE[1] - TEMP_RANGE[0]) + TEMP_RANGE[0]

                # Forward pass: base predictions adjusted by meta network
                base = self._base_preds(xb)
                pred = base + self.meta(base)

                # Compute MSE loss
                loss_coeff = ((pred - yb) ** 2 * mb).sum() / mb.sum()

                # Compute physics loss
                pred_raw = pred * σ_t + μ_t
                yb_raw = yb * σ_t + μ_t
                loss_phys = self._physics_loss(pred_raw, yb_raw, mb, T) * PHYSICS_WEIGHT

                # Total loss
                loss = loss_coeff + loss_phys
                loss.backward()
                nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                opt.step()
                opt.zero_grad()
                tot += loss.item()

            sched.step()
            avg_train_loss = tot / len(trL)

            # Validation phase
            for net in self.base_nets.values():
                net.eval()
            self.meta.eval()
            val = 0.0
            with torch.no_grad():
                for xb, yb, mb in vaL:
                    xb, yb, mb = xb.to(self.device), yb.to(self.device), mb.to(self.device)
                    base = self._base_preds(xb)
                    pred = base + self.meta(base)
                    val += ((pred - yb) ** 2 * mb).sum().item() / mb.sum().item()
            val /= len(vaL)

            print(f"Epoch {ep:3d} | Train: {avg_train_loss:.4f} | Val: {val:.4f}")

            # Early stopping and model saving
            if val < best - 1e-4:
                best, wait = val, 0
                for prop, net in self.base_nets.items():
                    torch.save(net.state_dict(), self.model_dir / f"base_{prop}_snn.pth")
                torch.save(self.meta.state_dict(), self.model_dir / "meta_snn.pth")
            else:
                wait += 1
                if wait >= PAT:
                    print(" ⇢ Early stopping triggered")
                    break

        # Load the best models
        for prop, net in self.base_nets.items():
            net.load_state_dict(torch.load(self.model_dir / f"base_{prop}_snn.pth", map_location=self.device))
        self.meta.load_state_dict(torch.load(self.model_dir / "meta_snn.pth", map_location=self.device))

    # ─────────────────────── evaluation & utils ─────────────────
    def evaluate(self, return_dict: bool = False):
        """Compute per-target relative-MSE (%) + R² on the *validation* split."""
        self.meta.eval()                                # or pass for KAN/SNN before meta
        per_target = {}
        rel_mses, r2s = [], []

        # ---- forward pass over the full validation set --------------------------
        μ, σ = self.μ, self.σ                           # already on CPU here
        Xval  = self.X_embedded[self.va_idx]
        yval  = self.y_raw[self.va_idx]

        # build standardised preds (base + meta)
        with torch.no_grad():
            xb = torch.tensor(Xval, device=self.device)
            base_out = torch.cat(                       # (B, Nprop)
                [self.base_nets[p](xb) for p in self.present_targets], dim=1
            )
            pred_std = (base_out + self.meta(base_out)).cpu().numpy()
        pred = pred_std * σ + μ                         # de-standardise

        # ---- per-property metrics ----------------------------------------------
        for j, prop in enumerate(self.present_targets):
            yt = yval[:, j]
            yp = pred[:, j]
            m_rel = _rel_mse_pct(yt, yp)
            r2    = r2_score(yt, yp)
            per_target[prop] = {"MSE_pct": float(m_rel), "R2": float(r2)}
            rel_mses.append(m_rel);  r2s.append(r2)

        avg_rel_mse = float(np.mean(rel_mses))
        avg_r2      = float(np.mean(r2s))

        # ---- pretty print -------------------------------------------------------
        print(f"\nValidation results — relative MSE (% of ⟨y²⟩) and R²")
        for p, d in per_target.items():
            print(f" • {p:<8s}: {d['MSE_pct']:6.2f}%   R²={d['R2']:+.3f}")
        print(f" ⇒ Average   : {avg_rel_mse:6.2f}%   R²={avg_r2:+.3f}")

        if return_dict:
            self.metrics_ = {"avg_mse_pct": avg_rel_mse,
                            "avg_r2"     : avg_r2,
                            "per_target" : per_target}
            return self.metrics_

    # ─────────────────────── public API ─────────────────────────
    def predict(self, composition: Dict[str, float]) -> Dict[str, float]:
        """Predict properties from composition with full preprocessing and model loading"""
        # 1. Load pretrained models (sorted alphabetically)
        model_dir = Path("../data/trained_models")
        sorted_targets = sorted(self.present_targets)

        # Load base SNNs
        for prop in sorted_targets:
            model_path = model_dir / f"base_{prop}_snn.pth"
            self.base_nets[prop].load_state_dict(
                torch.load(model_path, map_location=device))

        # Load meta SNN
        meta_path = model_dir / "meta_snn.pth"
        self.meta.load_state_dict(torch.load(meta_path, map_location=device))

        # 2. Process composition (compound decomposition + normalization)
        elements = {}
        compounds = {}

        # Parse compounds and elements
        for key, value in composition.items():
            parsed = self.parse_compound(key)
            if len(parsed) > 1:  # Compound
                compounds[key] = compounds.get(key, 0.0) + value
                for el, count in parsed.items():
                    elements[el] = elements.get(el, 0.0) + value * count
            else:  # Element
                el = list(parsed.keys())[0]
                elements[el] = elements.get(el, 0.0) + value

        # Combine and normalize to sum=1
        combined = {**compounds, **elements}
        total = sum(combined.values())
        if total <= 0:
            raise ValueError("Composition must have positive total")
        normalized = {k: v/total for k, v in combined.items()}

        # 3. Create aligned input features (using original training columns)
        feat_columns = self.X_comp.columns.tolist()  # Preserved order from training
        feat_vector = np.zeros(len(feat_columns), dtype=np.float32)

        # Fill features in original training order
        for i, col in enumerate(feat_columns):
            feat_vector[i] = normalized.get(col, 0.0)

        # 4. Generate polynomial features (same pipeline as training)
        raw_df = pd.DataFrame([feat_vector], columns=feat_columns)
        raw_poly = self.poly.transform(raw_df)
        scaled_poly = self.scaler.transform(raw_poly)

        # Combine with original fractions
        final_feats = np.hstack([scaled_poly, feat_vector[None, :]]).astype(np.float32)
        if self.embedding_method != 'none':
            final_feats = self.embedder.transform(final_feats)

        # 5. Convert to tensor and make prediction
        xb = torch.tensor(final_feats, device=device)

        with torch.no_grad():  # Disable gradient tracking
            # Get base predictions in sorted order
            base_outputs = [self.base_nets[prop](xb) for prop in sorted_targets]
            base_out = torch.cat(base_outputs, dim=1)  # (1, num_properties)

            # Apply meta network
            meta_out = self.meta(base_out)

            # Combine and convert to numpy
            pred = (base_out + meta_out).cpu().numpy()[0]  # (num_properties,)

        # 6. Return predictions with original target order and denormalize
        return {
            prop: (pred[sorted_targets.index(prop)] * self.σ[self.present_targets.index(prop)]
                + self.μ[self.present_targets.index(prop)])
            for prop in self.present_targets
        }

    @staticmethod
    def parse_compound(c: str) -> Dict[str, int]:
        """Parse compound formula into constituent elements with counts"""
        elements = {}
        for el, n in re.findall(r"([A-Z][a-z]*)(\d*)", c):
            elements[el] = elements.get(el, 0) + int(n or "1")
        return elements

    def derived(self, coeffs: Dict[str,float], T: float) -> Dict[str,float]:
        out = {}
        if {'rho_a','rho_b'}.issubset(coeffs):
            out['rho'] = coeffs['rho_a'] - coeffs['rho_b']*T
        if {'mu1_a','mu1_b'}.issubset(coeffs):
            out['muA'] = coeffs['mu1_a']*math.exp(coeffs['mu1_b']/(R*T))
        if {'mu2_a','mu2_b','mu2_c'}.issubset(coeffs):
            out['muB'] = 10**(coeffs['mu2_a']+coeffs['mu2_b']/T+coeffs['mu2_c']/T**2)
        if {'k_a','k_b'}.issubset(coeffs):
            out['k'] = coeffs['k_a'] + coeffs['k_b']*T
        if {'cp_a','cp_b','cp_c'}.issubset(coeffs):
            out['cp'] = coeffs['cp_a'] + coeffs['cp_b']*T + coeffs['cp_c']/T**2
        return out

    # ─────────────── persistence (save / load) ──────────────────
    def save(self, path: str):
        p = Path(path); p.mkdir(parents=True, exist_ok=True)
        for k,n in self.base_nets.items():
            torch.save(n.state_dict(), p/f"base_{k}_snn.pth")
        torch.save(self.meta.state_dict(), p/"meta_snn.pth")
        np.save(p/"μ_snn.npy", self.μ)
        np.save(p/"σ_snn.npy", self.σ)
        pd.to_pickle(self.poly,   p/"poly_snn.pkl")
        pd.to_pickle(self.scaler, p/"scaler_snn.pkl")
        pd.to_pickle(self.X_comp.columns.tolist(), p/"elements_snn.pkl")

    def load(self, path: str):
        p = Path(path)
        for k in self.present_targets:
            self.base_nets[k].load_state_dict(torch.load(p/f"base_{k}_snn.pth", map_location=device))
        self.meta.load_state_dict(torch.load(p/"meta_snn.pth", map_location=device))
        self.μ  = np.load(p/"μ_snn.npy");  self.σ = np.load(p/"σ_snn.npy")
        self.poly   = pd.read_pickle(p/"poly_snn.pkl")
        self.scaler = pd.read_pickle(p/"scaler_snn.pkl")
        self.X_comp.columns = pd.read_pickle(p/"elements_snn.pkl")


# if __name__ == "__main__":
#     df = pd.read_csv("saltdblean_processed.csv").rename(columns=str.strip)
#     trainer = SNNMetaTrainer(df, TARGETS, DERIVED_PROPS)
#     print(f"Using {len(trainer.present_targets)} properties:", ", ".join(trainer.present_targets))
#     trainer.train_base()
#     trainer.train_meta()
#     trainer.evaluate()
#     coeff = trainer.predict({'Na': 0.5, 'Cl': 0.5})
#     print("\nPredicted coefficients for 50-50 NaCl:")
#     for k, v in coeff.items():
#         print(f"{k:7s}: {v:11.4f}")
#     print("\nDerived properties @ 900K:")
#     deriv = trainer.derived(coeff, 900)
#     for k, v in deriv.items():
#         print(f"{k:4s}: {v:11.4f}")
