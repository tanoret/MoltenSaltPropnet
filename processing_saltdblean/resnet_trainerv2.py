# processing_saltdblean/resnet_trainerv2.py

import re
import math
import random
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

from processing_saltdblean.embedding_preconditioner import EmbeddingPreconditioner

# ============================================================
# Globals
# ============================================================
SEED = 42
R = 8.314

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# Targets & physics
# ============================================================
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

ELEMENT_FEATURE_COLS = [
    "polarizability_element[10-24Cm3]",
    "atomic_mass_element",
    "electronegativity_element",
    "atomic_radius_element[Angstrom]",
    "ionic_radius_element[Angstrom]",
    "covalent_radius_element[Angstrom]",
    "first_ionization_energy[kJ_per_mol]",
]

# ============================================================
# Utils
# ============================================================
def rel_mse_pct(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    denom = np.mean(y_true ** 2) or 1e-12
    return 100.0 * mse / denom

# ============================================================
# Networks
# ============================================================
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

# ============================================================
# Trainer
# ============================================================
class ResNetMetaTrainer:

    def __init__(
        self,
        df: pd.DataFrame,
        target_columns,
        derived_props,
        degree_poly: int = 3,
        embedding_method: str = "none",
        n_components: int = 10,
    ):
        self.df = df.copy()
        self.device = device
        self.derived_props = derived_props

        # --------------------------------------------------
        # Global element property maps (used in predict)
        # --------------------------------------------------
        self.global_elem_maps = {}
        for col in ELEMENT_FEATURE_COLS:
            found = {}
            if col in self.df.columns:
                for x in self.df[col]:
                    if isinstance(x, dict) and x:
                        found = x
                        break
            self.global_elem_maps[col] = found

        # --------------------------------------------------
        # Targets
        # --------------------------------------------------
        self.present_targets = []
        for t in target_columns:
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
            raise RuntimeError("No valid targets found.")

        # --------------------------------------------------
        # Composition → element fractions
        # --------------------------------------------------
        self.df["Composition"] = self.df.apply(self.row_composition, axis=1)
        self.X_comp = pd.json_normalize(self.df["Composition"]).fillna(0.0)
        self.X_comp = self.X_comp.reindex(sorted(self.X_comp.columns), axis=1)

        # --------------------------------------------------
        # Polynomial features
        # --------------------------------------------------
        self.poly = PolynomialFeatures(degree_poly, include_bias=False)
        X_poly = self.poly.fit_transform(self.X_comp)
        self.scaler = StandardScaler()
        X_poly = self.scaler.fit_transform(X_poly).astype(np.float32)

        fractions = self.X_comp.to_numpy(np.float32)

        # --------------------------------------------------
        # Element aggregated features (mean, std)
        # --------------------------------------------------
        elem_feats = []
        for _, row in self.df.iterrows():
            comp = row["Composition"]
            row_feats = []
            for col in ELEMENT_FEATURE_COLS:
                prop_dict = row.get(col, {})
                m, s = self.aggregate_element_property(comp, prop_dict)
                row_feats.extend([m, s])
            elem_feats.append(row_feats)

        X_elem = np.asarray(elem_feats, dtype=np.float32)

        # --------------------------------------------------
        # Final feature matrix
        # --------------------------------------------------
        self.X = np.hstack([X_poly, fractions, X_elem])
        self.feat_dim = self.X.shape[1]

        # --------------------------------------------------
        # Feature names
        # --------------------------------------------------
        self.feature_names = (
            [f"poly_{i}" for i in range(X_poly.shape[1])]
            + list(self.X_comp.columns)
            + [f"{c}_mean" for c in ELEMENT_FEATURE_COLS]
            + [f"{c}_std" for c in ELEMENT_FEATURE_COLS]
        )

        # --------------------------------------------------
        # Targets, masks
        # --------------------------------------------------
        self.mask_all = np.isfinite(self.df[self.present_targets]).to_numpy(bool)
        self.df[self.present_targets] = self.df[self.present_targets].fillna(0.0)
        self.y_raw = self.df[self.present_targets].to_numpy(np.float32)

        # --------------------------------------------------
        # Train / val / test split
        # --------------------------------------------------
        idx = np.arange(len(self.X))
        self.tr_idx, self.te_idx = train_test_split(idx, test_size=0.20, random_state=SEED)
        self.tr_idx, self.va_idx = train_test_split(self.tr_idx, test_size=0.20, random_state=SEED)

        # --------------------------------------------------
        # Embedding
        # --------------------------------------------------
        self.embedding_method = embedding_method
        self.embedder = EmbeddingPreconditioner(embedding_method, n_components)
        self.embedder.fit(self.X[self.tr_idx])
        self.X_embedded = self.embedder.transform(self.X)
        if embedding_method != "none":
            self.feat_dim = n_components

        # --------------------------------------------------
        # Target standardization
        # --------------------------------------------------
        self.mu = self.y_raw[self.tr_idx].mean(0)
        self.sigma = self.y_raw[self.tr_idx].std(0)
        self.sigma[self.sigma == 0] = 1.0
        self.y_std = (self.y_raw - self.mu) / self.sigma

        # --------------------------------------------------
        # Models
        # --------------------------------------------------
        self.idx_map = {n: i for i, n in enumerate(self.present_targets)}
        self.base_nets = nn.ModuleDict({
            p: BaseNet(self.feat_dim).to(device)
            for p in self.present_targets
        })
        self.meta = MetaNet(len(self.present_targets)).to(device)

    # =====================================================
    # Helpers
    # =====================================================
    def row_composition(self, row) -> Dict[str, float]:
        comps = row["System"].split("-")
        fracs = (
            [1.0] * len(comps)
            if row["Mol Frac"].strip() == "Pure Salt"
            else list(map(float, row["Mol Frac"].split("-")))
        )
        total = {}
        for cmp, f in zip(comps, fracs):
            for el, cnt in re.findall(r"([A-Z][a-z]*)(\d*)", cmp):
                total[el] = total.get(el, 0) + int(cnt or "1") * f
        s = sum(total.values())
        return {el: c / s for el, c in total.items()}

    def aggregate_element_property(
        self,
        elem_fracs: Dict[str, float],
        prop_dict: Dict[str, float],
    ) -> Tuple[float, float]:
        vals, weights = [], []
        for el, frac in elem_fracs.items():
            if el in prop_dict:
                vals.append(float(prop_dict[el]))
                weights.append(float(frac))
        if not vals:
            return 0.0, 0.0
        v = np.asarray(vals, np.float32)
        w = np.asarray(weights, np.float32)
        mean = np.sum(v * w)
        std = np.sqrt(np.sum(w * (v - mean) ** 2))
        return mean, std

    # =====================================================
    # Training
    # =====================================================
    def train_base(self):
        for prop in self.present_targets:
            print(f" • Training base net for {prop}")
            net = self.base_nets[prop]
            j = self.idx_map[prop]

            mask = self.mask_all[:, j]
            mask_tr = mask & np.isin(np.arange(len(mask)), self.tr_idx)
            mask_va = mask & np.isin(np.arange(len(mask)), self.va_idx)

            x_tr, y_tr = self.X_embedded[mask_tr], self.y_std[mask_tr, j]
            x_va, y_va = self.X_embedded[mask_va], self.y_std[mask_va, j]

            trL = DataLoader(TensorDataset(torch.tensor(x_tr), torch.tensor(y_tr)),
                             batch_size=64, shuffle=True)
            vaL = DataLoader(TensorDataset(torch.tensor(x_va), torch.tensor(y_va)),
                             batch_size=256) if len(x_va) else None

            opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
            best, wait = 1e9, 0

            for _ in range(300):
                net.train()
                for xb, yb in trL:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()
                    nn.functional.mse_loss(net(xb), yb).backward()
                    opt.step()

                if vaL:
                    net.eval()
                    with torch.no_grad():
                        val = sum(
                            nn.functional.mse_loss(net(xb.to(device)), yb.to(device)).item()
                            for xb, yb in vaL
                        ) / len(vaL)

                    if val < best:
                        best, wait = val, 0
                    else:
                        wait += 1
                        if wait >= 25:
                            break

    def train_meta(self):
        for net in self.base_nets.values():
            for p in net.parameters():
                p.requires_grad_(False)

        trL = DataLoader(
            TensorDataset(
                torch.tensor(self.X_embedded[self.tr_idx]),
                torch.tensor(self.y_std[self.tr_idx]),
                torch.tensor(self.mask_all[self.tr_idx]),
            ),
            batch_size=64,
            shuffle=True,
        )

        opt = torch.optim.AdamW(self.meta.parameters(), lr=1e-3, weight_decay=1e-4)

        for _ in range(600):
            self.meta.train()
            for xb, yb, mb in trL:
                xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                with torch.no_grad():
                    base = torch.stack([self.base_nets[p](xb) for p in self.present_targets], 1)
                pred = base + self.meta(base)
                loss = ((pred - yb) ** 2 * mb).sum() / mb.sum()
                opt.zero_grad()
                loss.backward()
                opt.step()

    # =====================================================
    # Predict
    # =====================================================
    def predict(self, composition: Dict[str, float]) -> Dict[str, float]:
        elems = {}
        for k, v in composition.items():
            for el, n in self.parse_compound(k).items():
                elems[el] = elems.get(el, 0) + v * n
        s = sum(elems.values())
        elem_fracs = {el: v / s for el, v in elems.items()}

        frac = np.zeros(len(self.X_comp.columns), dtype=np.float32)
        for i, col in enumerate(self.X_comp.columns):
            frac[i] = elem_fracs.get(col, 0.0)

        X_poly = self.poly.transform(pd.DataFrame([frac], columns=self.X_comp.columns))
        X_poly = self.scaler.transform(X_poly)

        elem_feats = []
        for col in ELEMENT_FEATURE_COLS:
            prop_map = self.global_elem_maps.get(col, {})
            m, s = self.aggregate_element_property(elem_fracs, prop_map)
            elem_feats.extend([m, s])

        X = np.hstack([X_poly, frac[None, :], np.array(elem_feats)[None, :]])
        if self.embedding_method != "none":
            X = self.embedder.transform(X)

        with torch.no_grad():
            xb = torch.tensor(X, device=device)
            base = torch.stack([self.base_nets[p](xb) for p in self.present_targets], 1)
            pred = (base + self.meta(base)).cpu().numpy()[0]

        return {p: float(pred[i] * self.sigma[i] + self.mu[i])
                for i, p in enumerate(self.present_targets)}

    @staticmethod
    def parse_compound(c: str) -> Dict[str, int]:
        out = {}
        for el, n in re.findall(r"([A-Z][a-z]*)(\d*)", c):
            out[el] = out.get(el, 0) + int(n or "1")
        return out


    def save(self, path: str):
        path = Path(path)
        path.mkdir(exist_ok=True)
        for p, net in self.base_nets.items():
            torch.save(net.state_dict(), path / f"base_{p}_resnet.pth")
        torch.save(self.meta.state_dict(), path / "meta_resnet.pth")
        np.save(path / "mu.npy", self.mu)
        np.save(path / "sigma.npy", self.sigma)

    def load(self, path: str):
        path = Path(path)

        for p in self.present_targets:
            ckpt = path / f"base_{p}_resnet.pth"
            if not ckpt.exists():
                raise FileNotFoundError(ckpt)

            state = torch.load(ckpt, map_location=device)
            in_dim_ckpt = state["net.0.weight"].shape[1]
            in_dim_model = self.base_nets[p].net[0].in_features

            if in_dim_ckpt != in_dim_model:
                raise RuntimeError(
                    f"Feature dim mismatch for {p}: "
                    f"ckpt={in_dim_ckpt}, model={in_dim_model}"
                )

            self.base_nets[p].load_state_dict(state)

        self.meta.load_state_dict(
            torch.load(path / "meta_resnet.pth", map_location=device)
        )

        self.mu = np.load(path / "mu.npy")
        self.sigma = np.load(path / "sigma.npy")




#if __name__ == "__main__":
 #   df = pd.read_csv("/Users/meggie/Documents/MoltenSaltPropnet/data/new_data.csv").rename(columns=str.strip)
  #  trainer = ResNetMetaTrainer(df, TARGETS, DERIVED_PROPS)
  #  print(f"Using {len(trainer.present_targets)} properties:", ", ".join(trainer.present_targets))
   # trainer.train_base()
   # trainer.train_meta()
#    trainer.evaluate()
#     coeff = trainer.predict({'Na': 0.5, 'Cl': 0.5})
#     print("\nPredicted coefficients for 50-50 NaCl:")
#     for k, v in coeff.items(): print(f"{k:7s}: {v:11.4f}")
#     print("\nDerived properties @ 900K:")
#     deriv = trainer.derived(coeff, 900)
#     for k, v in deriv.items(): print(f"{k:4s}: {v:11.4f}")
