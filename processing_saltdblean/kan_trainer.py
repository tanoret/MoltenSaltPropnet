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

class KANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_grids=5, grid_range=[-2, 2]):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_grids = num_grids
        self.grid_range = grid_range
        self.grid = torch.linspace(grid_range[0], grid_range[1], num_grids)
        self.register_buffer('grid_points', self.grid)
        self.coeff = nn.Parameter(torch.randn(output_dim, input_dim, num_grids) * 0.1)
        self.scale = nn.Parameter(torch.ones(output_dim, input_dim))
        self.base_weight = nn.Parameter(torch.randn(output_dim, input_dim) * 0.1)

    def forward(self, x):
        batch_size, input_dim = x.shape
        x = torch.clamp(x, self.grid_range[0], self.grid_range[1])
        positions = (x - self.grid_range[0]) / (self.grid_range[1] - self.grid_range[0])
        positions = positions * (self.num_grids - 1)
        left_idx = torch.floor(positions).clamp(0, self.num_grids-2).long()
        right_idx = left_idx + 1
        right_weight = positions - left_idx
        left_weight = 1 - right_weight
        left_idx_ = left_idx.permute(1, 0).unsqueeze(0).expand(self.output_dim, -1, -1)
        right_idx_ = right_idx.permute(1, 0).unsqueeze(0).expand(self.output_dim, -1, -1)
        left_coeff = torch.gather(self.coeff, 2, left_idx_)
        right_coeff = torch.gather(self.coeff, 2, right_idx_)
        left_weight_ = left_weight.permute(1, 0).unsqueeze(0)
        right_weight_ = right_weight.permute(1, 0).unsqueeze(0)
        interp = (left_weight_ * left_coeff + right_weight_ * right_coeff)
        output = (interp * self.scale.unsqueeze(-1)).sum(dim=1).permute(1, 0) + x @ self.base_weight.T
        return output

class KANBase(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, depth=2):
        super().__init__()
        layers = [KANLayer(input_dim, hidden_dim)]
        for _ in range(depth-1):
            layers.append(nn.SiLU())
            layers.append(KANLayer(hidden_dim, hidden_dim))
        layers.append(KANLayer(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

class KANMeta(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, depth=2):
        super().__init__()
        layers = [KANLayer(input_dim, hidden_dim)]
        for _ in range(depth-1):
            layers.append(nn.SiLU())
            layers.append(KANLayer(hidden_dim, hidden_dim))
        layers.append(KANLayer(hidden_dim, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class KANMetaTrainer:
    def __init__(self, df, target_columns, derived_props, degree_poly=3,
                 embedding_method='none', n_components=10):
        self.df = df.copy()
        self.target_columns = target_columns
        self.derived_props = derived_props
        self.model_dir = Path("../data/trained_models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.present_targets = []
        for t in target_columns:
            if t in self.df.columns:
                self.df[t] = self.df[t].replace(["----", ""], np.nan).replace(r"\*", "", regex=True)
                self.df[t] = pd.to_numeric(self.df[t], errors="coerce")
                if np.isfinite(self.df[t]).any():
                    self.present_targets.append(t)

        if not self.present_targets:
            raise RuntimeError("No valid target columns found after cleaning.")

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

        self.mask_all = np.isfinite(self.df[self.present_targets]).to_numpy(bool)
        # self.df[self.present_targets] = self.df[self.present_targets].fillna(
        #     self.df[self.present_targets].mean()
        # )
        self.df[self.present_targets] = self.df[self.present_targets].fillna(0.0)
        self.y_raw = self.df[self.present_targets].to_numpy(np.float32)

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

        self.μ = self.y_raw[self.tr_idx].mean(0)
        self.σ = self.y_raw[self.tr_idx].std(0)
        self.σ[self.σ == 0] = 1.0
        self.y_std = (self.y_raw - self.μ) / self.σ

        self.idx_map = {n: j for j, n in enumerate(self.present_targets)}
        self.base_nets = nn.ModuleDict({n: KANBase(self.feat_dim).to(device) for n in self.present_targets})
        self.meta = KANMeta(len(self.present_targets)).to(device)

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
        print("\nStage-1: Training base KANs...")
        for j, prop in enumerate(self.present_targets):
            net = self.base_nets[prop]
            print(f" • Training base net for {prop}")

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
                                  batch_size=256, shuffle=False) if len(x_va) else None

            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 150, 1e-4)
            best, patience, PAT = 1e9, 0, 50
            model_path = self.model_dir / f"base_{prop}_kan.pth"

            for epoch in range(200):
                net.train()
                for xb, yb in tr_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    opt.zero_grad()
                    loss = nn.functional.mse_loss(net(xb), yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
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

            try:
                net.load_state_dict(torch.load(model_path))
            except:
                pass

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

        opt = torch.optim.AdamW(self.meta.parameters(), lr=8e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 300, 1e-4)
        best, wait, PAT = 1e9, 0, 35
        meta_path = self.model_dir / "meta_kan.pth"

        μ_tensor = torch.tensor(self.μ, device=device, dtype=torch.float32)
        σ_tensor = torch.tensor(self.σ, device=device, dtype=torch.float32)

        print("\nStage-2: Training meta net with physics regularization...")
        for epoch in range(400):
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
                nn.utils.clip_grad_norm_(self.meta.parameters(), 0.5)
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
        """Train both base KAN networks and the meta KAN network jointly with a combined MSE and physics loss."""
        # Prepare data loaders for training and validation
        tr_loader = self.make_loader(self.X_embedded[self.tr_idx], self.y_std[self.tr_idx], self.mask_all[self.tr_idx], 64, True)
        va_loader = self.make_loader(self.X_embedded[self.va_idx], self.y_std[self.va_idx], self.mask_all[self.va_idx], 256, False)

        # Collect all parameters from base and meta networks for joint optimization
        all_params = list(self.meta.parameters())
        for net in self.base_nets.values():
            all_params += list(net.parameters())
        optimizer = torch.optim.AdamW(all_params, lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400, eta_min=1e-4)

        # Training hyperparameters
        PHYSICS_WEIGHT = 0.1  # Weight for physics-based regularization loss
        TEMP_RANGE = (500, 1200)  # Temperature range for physics loss
        best_val_loss = float('inf')
        patience, PATIENCE_LIMIT = 0, 40  # Early stopping parameters

        # Training loop
        print("\nJoint Training: Optimizing base and meta KAN networks together...")
        for epoch in range(600):
            # Set all networks to training mode
            for net in self.base_nets.values():
                net.train()
            self.meta.train()
            total_train_loss = 0.0

            for xb, yb, mb in tr_loader:
                xb, yb, mb = xb.to(self.device), yb.to(self.device), mb.to(self.device)
                batch_size = xb.size(0)
                T = torch.rand(batch_size, device=self.device) * (TEMP_RANGE[1] - TEMP_RANGE[0]) + TEMP_RANGE[0]

                # Forward pass: base predictions adjusted by meta network
                base_out = torch.stack([self.base_nets[p](xb) for p in self.present_targets], dim=1)
                pred = base_out + self.meta(base_out)

                # Compute MSE loss
                mse_loss = ((pred - yb) ** 2 * mb).sum() / mb.sum()

                # Convert predictions and targets to raw (unstandardized) values for physics loss
                pred_raw = pred * torch.tensor(self.σ, device=self.device) + torch.tensor(self.μ, device=self.device)
                yb_raw = yb * torch.tensor(self.σ, device=self.device) + torch.tensor(self.μ, device=self.device)

                # Compute physics-based regularization loss
                physics_loss = 0.0
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
                    physics_loss += term_loss
                    valid_terms += 1
                physics_loss = physics_loss / valid_terms if valid_terms > 0 else torch.tensor(0.0, device=self.device)

                # Combine losses and perform optimization step
                total_loss = mse_loss + PHYSICS_WEIGHT * physics_loss
                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                optimizer.step()
                total_train_loss += total_loss.item()

            scheduler.step()
            avg_train_loss = total_train_loss / len(tr_loader)

            # Validation phase
            for net in self.base_nets.values():
                net.eval()
            self.meta.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb, mb in va_loader:
                    xb, yb, mb = xb.to(self.device), yb.to(self.device), mb.to(self.device)
                    base_out = torch.stack([self.base_nets[p](xb) for p in self.present_targets], dim=1)
                    pred = base_out + self.meta(base_out)
                    val_loss += ((pred - yb) ** 2 * mb).sum().item() / mb.sum().item()
            val_loss /= len(va_loader)

            print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # Early stopping and model saving
            if val_loss < best_val_loss - 1e-4:
                best_val_loss, patience = val_loss, 0
                for prop, net in self.base_nets.items():
                    torch.save(net.state_dict(), self.model_dir / f"base_{prop}_kan.pth")
                torch.save(self.meta.state_dict(), self.model_dir / "meta_kan.pth")
            else:
                patience += 1
                if patience >= PATIENCE_LIMIT:
                    print(" ⇢ Early stopping triggered")
                    break

        # Load the best models after training
        for prop, net in self.base_nets.items():
            net.load_state_dict(torch.load(self.model_dir / f"base_{prop}_kan.pth"))
        self.meta.load_state_dict(torch.load(self.model_dir / "meta_kan.pth"))

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
        """Predict properties from composition with proper model loading and feature handling"""
        # 1. Load pretrained models (sorted alphabetically)
        model_dir = Path("../data/trained_models")
        sorted_targets = sorted(self.present_targets)

        # Load base KANs
        for prop in sorted_targets:
            model_path = model_dir / f"base_{prop}_kan.pth"
            if model_path.exists():
                self.base_nets[prop].load_state_dict(torch.load(model_path))
            else:
                raise FileNotFoundError(f"Base KAN model for {prop} not found at {model_path}")

        # Load meta KAN
        meta_path = model_dir / "meta_kan.pth"
        if meta_path.exists():
            self.meta.load_state_dict(torch.load(meta_path))
        else:
            raise FileNotFoundError(f"Meta KAN model not found at {meta_path}")

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

        # 3. Create aligned input features
        # Get feature columns in original sorted order
        feat_columns = self.X_comp.columns.tolist()

        # Initialize feature vector with zeros
        feat_vector = np.zeros(len(feat_columns), dtype=np.float32)

        # Fill in available features
        for i, col in enumerate(feat_columns):
            feat_vector[i] = normalized.get(col, 0.0)

        # 4. Generate polynomial features and scale
        raw_df = pd.DataFrame([feat_vector], columns=feat_columns)
        raw_poly = self.poly.transform(raw_df)
        scaled_poly = self.scaler.transform(raw_poly)

        # Combine with original fractions
        final_feats = np.hstack([scaled_poly, feat_vector[None, :]]).astype(np.float32)
        if self.embedding_method != 'none':
            final_feats = self.embedder.transform(final_feats)

        # 5. Make prediction
        xb = torch.tensor(final_feats, device=device)

        with torch.no_grad():
            # Process base networks
            base_outputs = []
            for prop in sorted_targets:
                out = self.base_nets[prop](xb)
                base_outputs.append(out)

            # Stack outputs correctly (batch_size × num_properties)
            base_out = torch.stack(base_outputs, dim=1)  # Shape: (1, num_properties)

            # Apply meta network
            meta_out = self.meta(base_out)  # Should be (1, num_properties)
            pred = (base_out + meta_out).cpu().numpy()[0]

        # Return predictions with original target order and unstandardize
        return {prop: (pred[self.present_targets.index(prop)] * self.σ[self.present_targets.index(prop)] + self.μ[self.present_targets.index(prop)])
                for prop in self.present_targets}

    @staticmethod
    def parse_compound(c: str) -> Dict[str, int]:
        """Parse compound formula into constituent elements"""
        elements = {}
        for el, n in re.findall(r"([A-Z][a-z]*)(\d*)", c):
            elements[el] = elements.get(el, 0) + int(n or "1")
        return elements

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
            torch.save(net.state_dict(), path / f"base_{prop}_kan.pth")
        torch.save(self.meta.state_dict(), path / "meta_kan.pth")
        np.save(path / "μ_kan.npy", self.μ)
        np.save(path / "σ_kan.npy", self.σ)
        pd.to_pickle(self.poly, path / "poly_kan.pkl")
        pd.to_pickle(self.scaler, path / "scaler_kan.pkl")
        pd.to_pickle(self.X_comp.columns.tolist(), path / "elements_kan.pkl")

    def load(self, path: str):
        path = Path(path)
        for prop in self.present_targets:
            self.base_nets[prop].load_state_dict(torch.load(path / f"base_{prop}_kan.pth"))
        self.meta.load_state_dict(torch.load(path / "meta_kan.pth"))
        self.μ = np.load(path / "μ_kan.npy")
        self.σ = np.load(path / "σ_kan.npy")
        self.poly = pd.read_pickle(path / "poly_kan.pkl")
        self.scaler = pd.read_pickle(path / "scaler_kan.pkl")
        self.X_comp.columns = pd.read_pickle(path / "elements_kan.pkl")

# if __name__ == "__main__":
#     df = pd.read_csv("saltdblean_processed.csv").rename(columns=str.strip)
#     trainer = KANMetaTrainer(df, TARGETS, DERIVED_PROPS)
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
