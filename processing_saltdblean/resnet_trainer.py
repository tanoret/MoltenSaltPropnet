import re, math, random, warnings
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict
import os

from processing_saltdblean.embedding_preconditioner import EmbeddingPreconditioner

from sklearn.metrics import mean_squared_error, r2_score

def _rel_mse_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return relative MSE as a percentage of ⟨y²⟩ — avoids unit issues."""
    mse = mean_squared_error(y_true, y_pred)
    denom = np.mean(y_true ** 2) or 1e-12           # guard /0
    return 100.0 * mse / denom

SEED = 42
R = 8.314
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
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

class ResidualBlock(nn.Module):
    def __init__(self, dim, p_drop=0.2):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(p_drop)

    def forward(self, x):
        h = self.act(self.lin1(x))
        h = self.drop(h)
        h = self.lin2(h)
        return self.act(x + h)

class BaseNet(nn.Module):
    def __init__(self, d_in, hidden=64, depth=3):
        super().__init__()
        layers = [nn.Linear(d_in, hidden), nn.SiLU()]
        for _ in range(depth):
            layers.append(ResidualBlock(hidden))
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

class MetaNet(nn.Module):
    def __init__(self, n_props, hidden=128, depth=2):
        super().__init__()
        layers = [nn.Linear(n_props, hidden), nn.SiLU()]
        for _ in range(depth):
            layers.append(ResidualBlock(hidden))
        layers.append(nn.Linear(hidden, n_props))
        self.net = nn.Sequential(*layers)

    def forward(self, p):
        return self.net(p)

class ResNetMetaTrainer:
    def __init__(self, df, target_columns, derived_props, degree_poly=3,
                 embedding_method='none', n_components=10):
        self.df = df.copy()
        self.target_columns = target_columns
        self.derived_props = derived_props
        self.model_dir = Path("../data/trained_models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Clean and identify valid target columns
        self.present_targets = []
        for t in target_columns:
            if t in self.df.columns:
                self.df[t] = self.df[t].replace(["----", ""], np.nan).replace(r"\*", "", regex=True)
                self.df[t] = pd.to_numeric(self.df[t], errors="coerce")
                if np.isfinite(self.df[t]).any():
                    self.present_targets.append(t)

        if not self.present_targets:
            raise RuntimeError("No valid target columns found after cleaning.")

        # Composition normalization and feature engineering
        self.df["Composition"] = self.df.apply(self.row_composition, axis=1)
        self.X_comp = pd.json_normalize(self.df["Composition"]).fillna(0.0)
        self.X_comp = self.X_comp.reindex(sorted(self.X_comp.columns), axis=1)
        self.composition_df = self.X_comp

        self.poly = PolynomialFeatures(degree_poly, include_bias=False)
        self.X_poly = self.poly.fit_transform(self.X_comp)
        self.scaler = StandardScaler()
        self.X_poly = self.scaler.fit_transform(self.X_poly).astype(np.float32)

        self.fractions = self.X_comp.to_numpy(np.float32)
        self.X = np.hstack([self.X_poly, self.fractions])
        self.feat_dim = self.X.shape[1]

        # Prepare target matrix and handle missing data
        self.mask_all = np.isfinite(self.df[self.present_targets]).to_numpy(bool)
        # self.df[self.present_targets] = self.df[self.present_targets].fillna(
        #     self.df[self.present_targets].mean()
        # )
        self.df[self.present_targets] = self.df[self.present_targets].fillna(0.0)
        self.y_raw = self.df[self.present_targets].to_numpy(np.float32)

        # Data split
        self.idx_all = np.arange(len(self.X))
        self.tr_idx, self.te_idx = train_test_split(self.idx_all, test_size=0.20, random_state=SEED)
        self.tr_idx, self.va_idx = train_test_split(self.tr_idx, test_size=0.20, random_state=SEED)

        # Embedding Block
        self.embedding_method = embedding_method
        self.n_components = n_components
        self.embedder = EmbeddingPreconditioner(method=embedding_method, n_components=n_components)
        self.embedder.fit(self.X[self.tr_idx])
        self.X_embedded = self.embedder.transform(self.X)
        self.feat_dim = self.n_components if embedding_method != 'none' else self.X.shape[1]

        # Normalize targets
        self.μ = self.y_raw[self.tr_idx].mean(0)
        self.σ = self.y_raw[self.tr_idx].std(0)
        self.σ[self.σ == 0] = 1.0
        self.y_std = (self.y_raw - self.μ) / self.σ

        # Initialize models
        self.idx_map = {n: j for j, n in enumerate(self.present_targets)}
        self.base_nets = nn.ModuleDict({n: BaseNet(self.feat_dim).to(device) for n in self.present_targets})
        self.meta = MetaNet(len(self.present_targets)).to(device)

    def row_composition(self, row):
        comps = row["System"].split("-")
        fracs = [1.0] * len(comps) if row["Mol Frac"].strip() == "Pure Salt" else list(map(float, row["Mol Frac"].split("-")))
        total = {}
        for cmp, f in zip(comps, fracs):
            for el, cnt in re.findall(r"([A-Z][a-z]*)(\d*)", cmp):
                total[el] = total.get(el, 0) + int(cnt or "1") * f
        s = sum(total.values())
        return {el: cnt / s for el, cnt in total.items()}

    def make_loader(self, x, y, m, bs, shuf):
        ds = TensorDataset(torch.tensor(x), torch.tensor(y), torch.tensor(m))
        return DataLoader(ds, batch_size=bs, shuffle=shuf, drop_last=False)

    def train_base(self):
        for prop in self.present_targets:
            print(f" • Training base net for {prop}")
            net = self.base_nets[prop]
            j = self.idx_map[prop]

            mask = self.mask_all[:, j].astype(bool)
            mask_tr_glb = mask & np.isin(self.idx_all, self.tr_idx)
            mask_va_glb = mask & np.isin(self.idx_all, self.va_idx)

            # If no validation data in global split, split available data
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
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()
                    nn.functional.mse_loss(net(xb), yb).backward()
                    opt.step()
                sched.step()

                if va_loader:
                    net.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for xb, yb in va_loader:
                            xb, yb = xb.to(device), yb.to(device)
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

            # Load the best model if validation was used, else keep final model
            if va_loader:
                try:
                    net.load_state_dict(torch.load(model_path))
                except:
                    print(f" No best model saved for {prop}, using final model")

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
                if len(coeff_indices) != len(req_coeffs): continue
                mask = torch.all(mb[:, coeff_indices], dim=1)
                if not mask.any(): continue
                y_coeffs = yb_raw[mask][:, coeff_indices]
                p_coeffs = pred_raw[mask][:, coeff_indices]
                with torch.no_grad():
                    if dprop == 'rho':
                        y_vals = y_coeffs[:, 0] - y_coeffs[:, 1] * T[mask]
                        p_vals = p_coeffs[:, 0] - p_coeffs[:, 1] * T[mask]
                        term_loss = nn.functional.mse_loss(p_vals, y_vals)
                    elif dprop == 'muA':
                        p_mu1_a = torch.clamp(p_coeffs[:, 0], min=1e-6)
                        p_vals = p_mu1_a * torch.exp(p_coeffs[:, 1] / (R * T[mask]))
                        y_vals = y_coeffs[:, 0] * torch.exp(y_coeffs[:, 1] / (R * T[mask]))
                        term_loss = nn.functional.mse_loss(torch.log(p_vals + 1e-8), torch.log(y_vals + 1e-8))
                    elif dprop == 'muB':
                        y_log = y_coeffs[:, 0] + y_coeffs[:, 1]/T[mask] + y_coeffs[:, 2]/T[mask]**2
                        p_log = p_coeffs[:, 0] + p_coeffs[:, 1]/T[mask] + p_coeffs[:, 2]/T[mask]**2
                        term_loss = nn.functional.mse_loss(p_log, y_log)
                    elif dprop == 'k':
                        y_vals = y_coeffs[:, 0] + y_coeffs[:, 1] * T[mask]
                        p_vals = p_coeffs[:, 0] + p_coeffs[:, 1] * T[mask]
                        term_loss = nn.functional.mse_loss(p_vals, y_vals)
                    elif dprop == 'cp':
                        y_vals = y_coeffs[:, 0] + y_coeffs[:, 1] * T[mask] + y_coeffs[:, 2]/T[mask]**2
                        p_vals = p_coeffs[:, 0] + p_coeffs[:, 1] * T[mask] + p_coeffs[:, 2]/T[mask]**2
                        term_loss = nn.functional.mse_loss(p_vals, y_vals)
                    else:
                        continue
                loss += term_loss
                valid_terms += 1
            return loss / valid_terms if valid_terms else torch.tensor(0.0, device=device)

        PHYSICS_WEIGHT = 0.1
        TEMP_RANGE = (500, 1200)
        trL = self.make_loader(self.X_embedded[self.tr_idx], self.y_std[self.tr_idx], self.mask_all[self.tr_idx], 64, True)
        vaL = self.make_loader(self.X_embedded[self.va_idx], self.y_std[self.va_idx], self.mask_all[self.va_idx], 256, False)

        opt = torch.optim.AdamW(self.meta.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 400, 1e-4)
        best, wait, PAT = 1e9, 0, 40
        meta_path = self.model_dir / "meta_resnet.pth"

        μ_tensor = torch.tensor(self.μ, device=device, dtype=torch.float32)
        σ_tensor = torch.tensor(self.σ, device=device, dtype=torch.float32)

        print("\nStage-2: Training meta net with physics regularization...")
        for epoch in range(600):
            self.meta.train()
            total_loss = 0.0
            for xb, yb, mb in trL:
                xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                batch_size = xb.size(0)
                T = torch.rand(batch_size, device=device) * (TEMP_RANGE[1] - TEMP_RANGE[0]) + TEMP_RANGE[0]
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
                    xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
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

        self.meta.load_state_dict(torch.load(meta_path))

    def train_joint(self):
        """Train both base networks and meta network together."""
        # Prepare data loaders
        trL = self.make_loader(self.X_embedded[self.tr_idx], self.y_std[self.tr_idx], self.mask_all[self.tr_idx], 64, True)
        vaL = self.make_loader(self.X_embedded[self.va_idx], self.y_std[self.va_idx], self.mask_all[self.va_idx], 256, False)

        # Collect all parameters for joint optimization
        all_params = list(self.meta.parameters())
        for net in self.base_nets.values():
            all_params += list(net.parameters())
        opt = torch.optim.AdamW(all_params, lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 400, 1e-4)

        # Training settings
        PHYSICS_WEIGHT = 0.1
        TEMP_RANGE = (500, 1200)
        best, wait, PAT = 1e9, 0, 40  # Early stopping parameters

        # Training loop
        for epoch in range(600):
            # Set all networks to training mode
            for net in self.base_nets.values():
                net.train()
            self.meta.train()
            total_loss = 0.0

            for xb, yb, mb in trL:
                xb, yb, mb = xb.to(self.device), yb.to(self.device), mb.to(self.device)
                batch_size = xb.size(0)
                T = torch.rand(batch_size, device=self.device) * (TEMP_RANGE[1] - TEMP_RANGE[0]) + TEMP_RANGE[0]

                # Forward pass through base nets and meta net
                base_out = torch.stack([self.base_nets[p](xb) for p in self.present_targets], dim=1)
                pred = base_out + self.meta(base_out)

                # Compute MSE loss
                loss_mse = ((pred - yb) ** 2 * mb).sum() / mb.sum()

                # Convert to raw (unstandardized) values for physics loss
                pred_raw = pred * torch.tensor(self.σ, device=self.device) + torch.tensor(self.μ, device=self.device)
                yb_raw = yb * torch.tensor(self.σ, device=self.device) + torch.tensor(self.μ, device=self.device)

                # Compute physics loss
                loss_phys = 0.0
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
                        if dprop == 'rho':
                            y_vals = y_coeffs[:, 0] - y_coeffs[:, 1] * T[mask]
                            p_vals = p_coeffs[:, 0] - p_coeffs[:, 1] * T[mask]
                            term_loss = nn.functional.mse_loss(p_vals, y_vals)
                        elif dprop == 'muA':
                            p_mu1_a = torch.clamp(p_coeffs[:, 0], min=1e-6)
                            p_vals = p_mu1_a * torch.exp(p_coeffs[:, 1] / (8.314 * T[mask]))
                            y_vals = y_coeffs[:, 0] * torch.exp(y_coeffs[:, 1] / (8.314 * T[mask]))
                            term_loss = nn.functional.mse_loss(torch.log(p_vals + 1e-8), torch.log(y_vals + 1e-8))
                        elif dprop == 'muB':
                            y_log = y_coeffs[:, 0] + y_coeffs[:, 1]/T[mask] + y_coeffs[:, 2]/T[mask]**2
                            p_log = p_coeffs[:, 0] + p_coeffs[:, 1]/T[mask] + p_coeffs[:, 2]/T[mask]**2
                            term_loss = nn.functional.mse_loss(p_log, y_log)
                        elif dprop == 'k':
                            y_vals = y_coeffs[:, 0] + y_coeffs[:, 1] * T[mask]
                            p_vals = p_coeffs[:, 0] + p_coeffs[:, 1] * T[mask]
                            term_loss = nn.functional.mse_loss(p_vals, y_vals)
                        elif dprop == 'cp':
                            y_vals = y_coeffs[:, 0] + y_coeffs[:, 1] * T[mask] + y_coeffs[:, 2]/T[mask]**2
                            p_vals = p_coeffs[:, 0] + p_coeffs[:, 1] * T[mask] + p_coeffs[:, 2]/T[mask]**2
                            term_loss = nn.functional.mse_loss(p_vals, y_vals)
                        else:
                            continue
                    loss_phys += term_loss
                    valid_terms += 1
                loss_phys = loss_phys / valid_terms if valid_terms else torch.tensor(0.0, device=self.device)

                # Total loss
                total_loss_ = loss_mse + PHYSICS_WEIGHT * loss_phys
                total_loss_.backward()
                nn.utils.clip_grad_norm_(all_params, 1.0)
                opt.step()
                opt.zero_grad()
                total_loss += total_loss_.item()

            sched.step()
            avg_loss = total_loss / len(trL)

            # Validation
            for net in self.base_nets.values():
                net.eval()
            self.meta.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb, mb in vaL:
                    xb, yb, mb = xb.to(self.device), yb.to(self.device), mb.to(self.device)
                    base_out = torch.stack([self.base_nets[p](xb) for p in self.present_targets], dim=1)
                    pred = base_out + self.meta(base_out)
                    val_loss += ((pred - yb) ** 2 * mb).sum().item() / mb.sum().item()
            val_loss /= len(vaL)

            print(f"Epoch {epoch:3d} | Train: {avg_loss:.4f} | Val: {val_loss:.4f}")

            # Early stopping and model saving
            if val_loss < best - 1e-4:
                best, wait = val_loss, 0
                for prop, net in self.base_nets.items():
                    torch.save(net.state_dict(), self.model_dir / f"base_{prop}_resnet.pth")
                torch.save(self.meta.state_dict(), self.model_dir / "meta_resnet.pth")
            else:
                wait += 1
                if wait >= PAT:
                    print(" ⇢ Early stopping")
                    break

        # Load the best models
        for prop, net in self.base_nets.items():
            net.load_state_dict(torch.load(self.model_dir / f"base_{prop}_resnet.pth"))
        self.meta.load_state_dict(torch.load(self.model_dir / "meta_resnet.pth"))


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
            base_out = torch.stack(
                [self.base_nets[p](xb).cpu() for p in self.present_targets], dim=1
            ).numpy()                                   # shape (Nva, P)
            pred_std = base_out + self.meta(torch.tensor(base_out, device=self.device)).cpu().numpy()
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

    def predict(self, composition: Dict[str, float]) -> Dict[str, float]:
        """Predict properties from composition with proper model loading and ordering"""
        # 1. Load pretrained models (sorted alphabetically)
        model_dir = Path("../data/trained_models")

        # Load base networks in alphabetical order
        sorted_targets = sorted(self.present_targets)
        for prop in sorted_targets:
            model_path = model_dir / f"base_{prop}_resnet.pth"
            if model_path.exists():
                self.base_nets[prop].load_state_dict(torch.load(model_path))
            else:
                raise FileNotFoundError(f"Base model for {prop} not found at {model_path}")

        # Load meta network
        meta_path = model_dir / "meta_resnet.pth"
        if meta_path.exists():
            self.meta.load_state_dict(torch.load(meta_path))
        else:
            raise FileNotFoundError(f"Meta model not found at {meta_path}")

        # 2. Process composition (compound decomposition + normalization)
        elements = {}
        compounds = {}

        for key, value in composition.items():
            parsed = self.parse_compound(key)
            if len(parsed) > 1:  # Compound
                compounds[key] = compounds.get(key, 0.0) + value
                for el, count in parsed.items():
                    elements[el] = elements.get(el, 0.0) + value * count
            else:  # Element
                el = list(parsed.keys())[0]
                elements[el] = elements.get(el, 0.0) + value

        # Combine and normalize
        combined = {**compounds, **elements}
        total = sum(combined.values())
        if total <= 0:
            raise ValueError("Composition must have positive total")
        normalized = {k: v/total for k, v in combined.items()}

        # 3. Create input tensor with proper feature order
        frac = np.zeros(len(self.X_comp.columns), dtype=np.float32)
        for i, col in enumerate(self.X_comp.columns):  # Columns are sorted alphabetically
            frac[i] = normalized.get(col, 0.0)

        # 4. Generate predictions
        raw_df = pd.DataFrame([frac], columns=self.X_comp.columns).fillna(0.0)
        raw = self.poly.transform(raw_df)
        feats = np.hstack([self.scaler.transform(raw), frac[None, :]]).astype(np.float32)
        if self.embedding_method != 'none':
            feats = self.embedder.transform(feats)
        xb = torch.tensor(feats, device=device)

        with torch.no_grad():
            # Process base networks in alphabetical order
            base_outputs = []
            for prop in sorted_targets:
                base_outputs.append(self.base_nets[prop](xb))
            base_out = torch.stack(base_outputs, dim=1)

            # Apply meta network
            pred = (base_out + self.meta(base_out)).cpu().numpy()[0]

        # Return predictions with original target order
        return {prop: (pred[i] * self.σ[i] + self.μ[i])
                for i, prop in enumerate(self.present_targets)}

    @staticmethod
    def parse_compound(c: str) -> Dict[str, int]:
        """Parse compound formula into elements (e.g., 'NaCl' → {'Na':1, 'Cl':1})"""
        out = {}
        for el, n in re.findall(r"([A-Z][a-z]*)(\d*)", c):
            out[el] = out.get(el, 0) + int(n or "1")
        return out

    def derived(self, coeffs: Dict[str, float], T: float) -> Dict[str, float]:
        out = {}
        if {'rho_a', 'rho_b'}.issubset(coeffs):
            out['rho'] = coeffs['rho_a'] - coeffs['rho_b'] * T
        if {'mu1_a', 'mu1_b'}.issubset(coeffs):
            out['muA'] = coeffs['mu1_a'] * math.exp(coeffs['mu1_b'] / (R * T))
        if {'mu2_a', 'mu2_b', 'mu2_c'}.issubset(coeffs):
            out['muB'] = 10 ** (coeffs['mu2_a'] + coeffs['mu2_b']/T + coeffs['mu2_c']/T**2)
        if {'k_a', 'k_b'}.issubset(coeffs):
            out['k'] = coeffs['k_a'] + coeffs['k_b'] * T
        if {'cp_a', 'cp_b', 'cp_c'}.issubset(coeffs):
            out['cp'] = coeffs['cp_a'] + coeffs['cp_b'] * T + coeffs['cp_c']/T**2
        return out

    def save(self, path: str):
        path = Path(path)
        os.makedirs(path, exist_ok=True)
        for prop, net in self.base_nets.items():
            torch.save(net.state_dict(), path / f"base_{prop}_resnet.pth")
        torch.save(self.meta.state_dict(), path / "meta_resnet.pth")
        np.save(path / "μ_resnet.npy", self.μ)
        np.save(path / "σ_resnet.npy", self.σ)
        pd.to_pickle(self.poly, path / "poly_resnet.pkl")
        pd.to_pickle(self.scaler, path / "scaler_resnet.pkl")
        pd.to_pickle(self.X_comp.columns.tolist(), path / "elements_resnet.pkl")

    def load(self, path: str):
        path = Path(path)
        for prop in self.present_targets:
            self.base_nets[prop].load_state_dict(torch.load(path / f"base_{prop}_resnet.pth"))
        self.meta.load_state_dict(torch.load(path / "meta_resnet.pth"))
        self.μ = np.load(path / "μ_resnet.npy")
        self.σ = np.load(path / "σ_resnet.npy")
        self.poly = pd.read_pickle(path / "poly_resnet.pkl")
        self.scaler = pd.read_pickle(path / "scaler_resnet.pkl")
        self.X_comp.columns = pd.read_pickle(path / "elements_resnet.pkl")

# if __name__ == "__main__":
#     df = pd.read_csv("saltdblean_processed.csv").rename(columns=str.strip)
#     trainer = ResNetMetaTrainer(df, TARGETS, DERIVED_PROPS)
#     print(f"Using {len(trainer.present_targets)} properties:", ", ".join(trainer.present_targets))
#     trainer.train_base()
#     trainer.train_meta()
#     trainer.evaluate()
#     coeff = trainer.predict({'Na': 0.5, 'Cl': 0.5})
#     print("\nPredicted coefficients for 50-50 NaCl:")
#     for k, v in coeff.items(): print(f"{k:7s}: {v:11.4f}")
#     print("\nDerived properties @ 900K:")
#     deriv = trainer.derived(coeff, 900)
#     for k, v in deriv.items(): print(f"{k:4s}: {v:11.4f}")
