
"""
mlp_dae_mccv.py

DAE → MLP with Monte-Carlo Cross-Validation + physics regularisation,
designed to mirror your SNNMetaTrainer physics behaviour.

- Handles missing data via Denoising Autoencoder (DAE)
- Multi-output MLP on DAE latent space
- 30 Monte-Carlo CV splits (random 80/20 train/test)
- Physics loss for rho, muA, muB, k, cp (same formulas as SNN)
- Tracks:
    * train loss per epoch
    * val loss per epoch
    * per-target R² and relMSE (% of <y²>)
    * per-split avg R² and avg relMSE
    * overall CV summary (mean ± std of avg R² / relMSE)
- Saves best model:
    ../data/trained_models/best_dae_mccv.pth
    ../data/trained_models/best_mlp_mccv.pth
    ../data/trained_models/best_mu_mccv.npy
    ../data/trained_models/best_sigma_mccv.npy

Demo at bottom: trains, prints CV summary, then predicts 50-50 NaCl
and prints coefficients + derived properties at 900 K.
"""

import re
import math
import random
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# ────────────────────────────────────────────────────────────────
#  Global config
# ────────────────────────────────────────────────────────────────
SEED = 42
R = 8.314
device = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
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


def _rel_mse_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Relative MSE as % of ⟨y²⟩ — avoids unit issues."""
    mse = mean_squared_error(y_true, y_pred)
    denom = np.mean(y_true ** 2) or 1e-12
    return 100.0 * mse / denom


# ────────────────────────────────────────────────────────────────
#  Denoising Autoencoder
# ────────────────────────────────────────────────────────────────
class DAE(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int = 64,
                 hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_dim),
        )

    def forward(self, x: torch.Tensor, noise_std: float = 0.1):
        if self.training and noise_std > 0:
            noise = torch.randn_like(x) * noise_std
            x_noisy = x + noise
        else:
            x_noisy = x

        z = self.encoder(x_noisy)
        xr = self.decoder(z)
        return xr, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ────────────────────────────────────────────────────────────────
#  Multi-output MLP
# ────────────────────────────────────────────────────────────────
class MultiOutputMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int,
                 hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ────────────────────────────────────────────────────────────────
#  Main Trainer with Monte-Carlo CV
# ────────────────────────────────────────────────────────────────
class MLPDAEMCCVTrainer:
    """
    DAE + MLP trainer with Monte-Carlo CV and physics-based regularisation.

    After training:
        trainer.cv_results_  # list of per-split metrics
        trainer.cv_summary_  # aggregated summary
    """

    def __init__(self,
                 df: pd.DataFrame,
                 target_cols=TARGETS,
                 derived_props=DERIVED_PROPS,
                 dae_latent_dim: int = 64):

        self.df = df.copy()
        self.derived_props = derived_props
        self.dae_latent_dim = dae_latent_dim

        # ── determine present targets & clean ───────────────────
        self.present_targets: List[str] = []
        for t in target_cols:
            if t in self.df.columns:
                col = (self.df[t]
                       .replace(["----", ""], np.nan)
                       .replace(r"\*", "", regex=True)
                       .pipe(pd.to_numeric, errors="coerce"))
                if np.isfinite(col).any():
                    self.present_targets.append(t)
                    self.df[t] = col

        if not self.present_targets:
            raise RuntimeError("No valid target columns found in DataFrame.")

        self.idx_map = {t: i for i, t in enumerate(self.present_targets)}

        # mask of where targets are actually present
        mask_all = np.isfinite(self.df[self.present_targets]).to_numpy(bool)

        # fill missing with 0.0 for numeric stability (mask tracks real values)
        self.df[self.present_targets] = self.df[self.present_targets].fillna(0.0)
        y_raw_full = self.df[self.present_targets].to_numpy(np.float32)

        self.y_raw_full = y_raw_full
        self.mask_all_full = mask_all

        # ── composition → element fractions ─────────────────────
        self.df["Composition"] = self.df.apply(self._row_composition, axis=1)
        self.X_comp = pd.json_normalize(self.df["Composition"]).fillna(0.0)
        self.X_comp = self.X_comp.reindex(sorted(self.X_comp.columns), axis=1)

        # ── polynomial features + scaling ───────────────────────
        self.poly = PolynomialFeatures(3, include_bias=False)
        X_poly = self.poly.fit_transform(self.X_comp)
        self.scaler = StandardScaler()
        X_poly = self.scaler.fit_transform(X_poly).astype(np.float32)

        frac = self.X_comp.to_numpy(np.float32)
        self.X = np.hstack([X_poly, frac])  # (N, feat_dim)
        self.feat_dim = self.X.shape[1]

        # model dir
        self.model_dir = Path("../data/trained_models")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # best model stats for prediction
        self.best_mu = None
        self.best_sigma = None

        # CV results will be stored here
        self.cv_results_: List[Dict] = []
        self.cv_summary_: Dict = {}

    # ──────────────────────── composition parser ────────────────
    @staticmethod
    def _row_composition(row) -> Dict[str, float]:
        comps = str(row["System"]).split("-")
        fracs = ([1.0] * len(comps)
                 if str(row["Mol Frac"]).strip() == "Pure Salt"
                 else list(map(float, str(row["Mol Frac"]).split("-"))))

        total = {}
        for cmp, f in zip(comps, fracs):
            for el, cnt in re.findall(r"([A-Z][a-z]*)(\d*)", cmp):
                total[el] = total.get(el, 0) + int(cnt or "1") * f
        s = sum(total.values()) or 1.0
        return {el: cnt / s for el, cnt in total.items()}

    # ──────────────────────── physics loss ──────────────────────
    def _physics_loss(self,
                       pred_raw: torch.Tensor,
                       y_raw: torch.Tensor,
                       mask_b: torch.Tensor,
                       T: torch.Tensor) -> torch.Tensor:
        """
        Same physics structure as original SNN trainer.
        pred_raw, y_raw: (B, n_targets) in physical units
        mask_b:         (B, n_targets) bool / {0,1}
        T:              (B,) temperature
        """
        loss = 0.0
        terms = 0

        for dprop, coeffs in self.derived_props:
            idxs = [self.idx_map[c] for c in coeffs if c in self.idx_map]
            if len(idxs) != len(coeffs):
                continue

            mb = mask_b[:, idxs].bool()
            m = torch.all(mb, dim=1)
            if not m.any():
                continue

            y = y_raw[m][:, idxs]      # (M, k)
            p = pred_raw[m][:, idxs]   # (M, k)
            Tm = T[m]

            if dprop == 'rho':
                # rho = rho_a - rho_b * T
                loss_t = F.mse_loss(p[:, 0] - p[:, 1] * Tm,
                                    y[:, 0] - y[:, 1] * Tm)
            elif dprop == 'muA':
                # muA = mu1_a * exp(mu1_b / RT)
                loss_t = F.mse_loss(
                    torch.log(torch.clamp(p[:, 0], 1e-6) *
                              torch.exp(p[:, 1] / (R * Tm))),
                    torch.log(y[:, 0] * torch.exp(y[:, 1] / (R * Tm))))
            elif dprop == 'muB':
                # muB ~ 10^(a + b/T + c/T^2)
                loss_t = F.mse_loss(
                    p[:, 0] + p[:, 1] / Tm + p[:, 2] / Tm**2,
                    y[:, 0] + y[:, 1] / Tm + y[:, 2] / Tm**2)
            elif dprop == 'k':
                # k = k_a + k_b*T
                loss_t = F.mse_loss(p[:, 0] + p[:, 1] * Tm,
                                    y[:, 0] + y[:, 1] * Tm)
            elif dprop == 'cp':
                # cp = cp_a + cp_b*T + cp_c/T^2
                loss_t = F.mse_loss(
                    p[:, 0] + p[:, 1] * Tm + p[:, 2] / Tm**2,
                    y[:, 0] + y[:, 1] * Tm + y[:, 2] / Tm**2)
            else:
                continue

            loss += loss_t
            terms += 1

        if terms == 0:
            return torch.tensor(0.0, device=device)
        return loss / terms

    # ──────────────────────── MCCV training ─────────────────────
    def train_mccv(self,
                   n_splits: int = 30,
                   train_frac: float = 0.8,
                   dae_epochs: int = 200,
                   mlp_epochs: int = 300,
                   batch_size: int = 128,
                   physics_weight: float = 0.1,
                   temp_range: Tuple[float, float] = (500.0, 1200.0)):
        """
        Run Monte-Carlo CV: random train/test splits repeated n_splits times.
        For each split:
            - fit DAE on training X
            - compute Z (latent) for all X
            - fit MLP on Z[train] with physics regularization
            - evaluate on test
        Only the best model (highest avg R²) is saved.
        """

        N = len(self.X)
        X_full = self.X
        y_full = self.y_raw_full
        mask_full = self.mask_all_full

        self.cv_results_ = []
        best_avg_r2 = -1e9

        for split in range(n_splits):
            print(f"\n================ Monte-Carlo split {split+1}/{n_splits} ================")
            rng = np.random.RandomState(SEED + split)
            perm = rng.permutation(N)
            n_tr = int(train_frac * N)
            tr_idx = perm[:n_tr]
            te_idx = perm[n_tr:]

            # train-based normalisation of targets
            y_tr_raw = y_full[tr_idx]
            mu = y_tr_raw.mean(0)
            sigma = y_tr_raw.std(0)
            sigma[sigma == 0] = 1.0

            # standardised targets (for whole dataset, but μ,σ from train)
            y_std_full = (y_full - mu) / sigma

            # ── train DAE on X[tr_idx] ───────────────────────────
            dae = DAE(self.feat_dim,
                      latent_dim=self.dae_latent_dim).to(device)
            self._fit_dae_for_split(dae, X_full[tr_idx],
                                    epochs=dae_epochs,
                                    batch_size=batch_size)

            # get latent Z for all samples
            with torch.no_grad():
                X_t = torch.tensor(X_full, dtype=torch.float32, device=device)
                Z_full = dae.encode(X_t).cpu().numpy()

            # inner split train/val from training set for MLP early stopping
            tr_inner_idx, va_inner_idx = self._inner_split(tr_idx, rng)

            Z_tr = Z_full[tr_inner_idx]
            Z_va = Z_full[va_inner_idx]
            Z_te = Z_full[te_idx]

            y_tr_std = y_std_full[tr_inner_idx]
            y_va_std = y_std_full[va_inner_idx]

            mask_tr = mask_full[tr_inner_idx]
            mask_va = mask_full[va_inner_idx]
            mask_te = mask_full[te_idx]

            # convert to torch
            Z_tr_t = torch.tensor(Z_tr, dtype=torch.float32, device=device)
            Z_va_t = torch.tensor(Z_va, dtype=torch.float32, device=device)
            Z_te_t = torch.tensor(Z_te, dtype=torch.float32, device=device)

            y_tr_t = torch.tensor(y_tr_std, dtype=torch.float32, device=device)
            y_va_t = torch.tensor(y_va_std, dtype=torch.float32, device=device)

            mask_tr_t = torch.tensor(mask_tr.astype(np.float32),
                                     dtype=torch.float32, device=device)
            mask_va_t = torch.tensor(mask_va.astype(np.float32),
                                     dtype=torch.float32, device=device)

            # ── MLP for this split ───────────────────────────────
            mlp = MultiOutputMLP(self.dae_latent_dim,
                                 out_dim=len(self.present_targets)).to(device)

            mu_t = torch.tensor(mu, dtype=torch.float32, device=device)
            sigma_t = torch.tensor(sigma, dtype=torch.float32, device=device)

            train_hist, val_hist = self._fit_mlp_for_split(
                mlp, Z_tr_t, y_tr_t, mask_tr_t,
                Z_va_t, y_va_t, mask_va_t,
                mu_t, sigma_t,
                mlp_epochs=mlp_epochs,
                batch_size=batch_size,
                physics_weight=physics_weight,
                temp_range=temp_range,
            )

            # ── evaluation on held-out test ─────────────────────
            metrics, avg_r2 = self._evaluate_split(
                mlp, dae, Z_te_t, te_idx,
                mu, sigma, mask_te, y_full[te_idx]
            )

            self.cv_results_.append({
                "split": split,
                "train_loss": train_hist,
                "val_loss": val_hist,
                "final_metrics": metrics,
                "avg_r2": avg_r2,
            })

            print(f"[Split {split+1}/{n_splits}] "
                  f"avg R²={avg_r2:+.3f}  "
                  f"avg relMSE={metrics['avg_relmse_pct']:.2f}%")

            # ── update best model ───────────────────────────────
            if avg_r2 > best_avg_r2:
                best_avg_r2 = avg_r2
                self.best_mu = mu
                self.best_sigma = sigma

                torch.save(dae.state_dict(),
                           self.model_dir / "best_dae_mccv.pth")
                torch.save(mlp.state_dict(),
                           self.model_dir / "best_mlp_mccv.pth")
                np.save(self.model_dir / "best_mu_mccv.npy", mu)
                np.save(self.model_dir / "best_sigma_mccv.npy", sigma)

        # ── build summary stats over splits ─────────────────────
        self.cv_summary_ = self._aggregate_cv_results()
        print("\n================ Monte-Carlo CV summary ================")
        print(f"  mean avg R²     = {self.cv_summary_['avg_r2_mean']:+.3f} "
              f"± {self.cv_summary_['avg_r2_std']:.3f}")
        print(f"  mean avg relMSE = {self.cv_summary_['avg_relmse_mean']:.2f}% "
              f"± {self.cv_summary_['avg_relmse_std']:.2f}%")

    # ──────────────────────── helpers: DAE/MLP training ─────────
    @staticmethod
    def _inner_split(tr_idx: np.ndarray,
                     rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
        """Split training indices into inner train/val for early stopping."""
        perm = rng.permutation(len(tr_idx))
        n_tr_inner = int(0.8 * len(tr_idx))
        tr_inner = tr_idx[perm[:n_tr_inner]]
        va_inner = tr_idx[perm[n_tr_inner:]]
        return tr_inner, va_inner

    def _fit_dae_for_split(self, dae: DAE,
                           X_tr: np.ndarray,
                           epochs: int,
                           batch_size: int):
        X_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
        opt = torch.optim.Adam(dae.parameters(), lr=1e-3, weight_decay=1e-5)

        for ep in range(epochs):
            perm = torch.randperm(X_t.size(0), device=device)
            Xb = X_t[perm]
            losses = []

            for i in range(0, Xb.size(0), batch_size):
                xb = Xb[i:i+batch_size]
                xr, _ = dae(xb, noise_std=0.1)
                loss = F.mse_loss(xr, xb)

                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())

            if ep % 50 == 0 or ep == epochs - 1:
                print(f"  DAE ep {ep:3d} | recon {np.mean(losses):.6f}")

    def _fit_mlp_for_split(self, mlp: MultiOutputMLP,
                           Z_tr: torch.Tensor, y_tr: torch.Tensor,
                           m_tr: torch.Tensor,
                           Z_va: torch.Tensor, y_va: torch.Tensor,
                           m_va: torch.Tensor,
                           mu_t: torch.Tensor, sigma_t: torch.Tensor,
                           mlp_epochs: int, batch_size: int,
                           physics_weight: float,
                           temp_range: Tuple[float, float]):
        opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
        best_val = 1e9
        patience = 50
        wait = 0

        N_tr = Z_tr.size(0)
        Tmin, Tmax = temp_range

        train_loss_history = []
        val_loss_history = []

        for ep in range(mlp_epochs):
            mlp.train()
            perm = torch.randperm(N_tr, device=device)
            Zb = Z_tr[perm]
            yb = y_tr[perm]
            mb = m_tr[perm]
            losses = []

            for i in range(0, N_tr, batch_size):
                xb = Zb[i:i+batch_size]
                y_batch = yb[i:i+batch_size]
                m_batch = mb[i:i+batch_size]

                pred_std = mlp(xb)  # (B, n_targets)

                # coefficient MSE with missing mask
                num = ((pred_std - y_batch) ** 2 * m_batch).sum()
                denom = m_batch.sum() + 1e-8
                loss_coeff = num / denom

                # physics loss in raw units
                T = torch.rand(len(xb), device=device) * (Tmax - Tmin) + Tmin
                pred_raw = pred_std * sigma_t + mu_t
                y_raw = y_batch * sigma_t + mu_t
                loss_phys = self._physics_loss(pred_raw, y_raw,
                                               m_batch, T)

                loss = loss_coeff + physics_weight * loss_phys

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
                opt.step()
                losses.append(loss.item())

            # validation loss
            mlp.eval()
            with torch.no_grad():
                pred_va_std = mlp(Z_va)
                num_va = ((pred_va_std - y_va) ** 2 * m_va).sum()
                denom_va = m_va.sum() + 1e-8
                val_loss = num_va / denom_va

            train_loss_history.append(float(np.mean(losses)))
            val_loss_history.append(float(val_loss.item()))

            if ep % 20 == 0 or ep == mlp_epochs - 1:
                print(f"  MLP ep {ep:3d} | train {train_loss_history[-1]:.5f} "
                      f"| val {val_loss.item():.5f}")

            # early stopping
            if val_loss.item() < best_val - 1e-5:
                best_val = val_loss.item()
                best_state = mlp.state_dict()
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print("  ⇢ early stopping MLP")
                    break

        # load best weights
        mlp.load_state_dict(best_state)
        return train_loss_history, val_loss_history

    # ──────────────────────── evaluation per split ──────────────
    def _evaluate_split(self,
                        mlp: MultiOutputMLP,
                        dae: DAE,
                        Z_te_t: torch.Tensor,
                        te_idx: np.ndarray,
                        mu: np.ndarray,
                        sigma: np.ndarray,
                        mask_te: np.ndarray,
                        y_te_raw: np.ndarray):
        mlp.eval()
        dae.eval()

        with torch.no_grad():
            pred_std = mlp(Z_te_t).cpu().numpy()  # (N_te, n_targets)

        pred_raw = pred_std * sigma + mu         # de-standardised

        metrics = {"per_target": {}}
        r2s = []
        relmses = []

        for j, t in enumerate(self.present_targets):
            m = mask_te[:, j]
            if not np.any(m):
                continue
            y_true = y_te_raw[m, j]
            y_pred = pred_raw[m, j]

            r2 = r2_score(y_true, y_pred)
            relmse = _rel_mse_pct(y_true, y_pred)

            metrics["per_target"][t] = {
                "r2": float(r2),
                "relmse_pct": float(relmse),
            }
            r2s.append(r2)
            relmses.append(relmse)

        avg_r2 = float(np.mean(r2s)) if r2s else float("nan")
        avg_relmse = float(np.mean(relmses)) if relmses else float("nan")

        metrics["avg_r2"] = avg_r2
        metrics["avg_relmse_pct"] = avg_relmse
        return metrics, avg_r2

    # ──────────────────────── aggregate CV metrics ──────────────
    def _aggregate_cv_results(self) -> Dict[str, float]:
        if not self.cv_results_:
            return {}

        avg_r2_list = [res["final_metrics"]["avg_r2"] for res in self.cv_results_]
        avg_relmse_list = [res["final_metrics"]["avg_relmse_pct"]
                           for res in self.cv_results_]

        return {
            "avg_r2_mean": float(np.mean(avg_r2_list)),
            "avg_r2_std": float(np.std(avg_r2_list)),
            "avg_relmse_mean": float(np.mean(avg_relmse_list)),
            "avg_relmse_std": float(np.std(avg_relmse_list)),
        }

    # ──────────────────────── prediction API ────────────────────
    def predict(self, composition: Dict[str, float]) -> Dict[str, float]:
        """
        Predict coefficients for a given composition using best
        DAE+MLP model learned in Monte-Carlo CV.
        """
        # load μ, σ from disk if not already
        if self.best_mu is None or self.best_sigma is None:
            mu_path = self.model_dir / "best_mu_mccv.npy"
            sigma_path = self.model_dir / "best_sigma_mccv.npy"
            if not mu_path.exists() or not sigma_path.exists():
                raise RuntimeError("Best model statistics not found. "
                                   "Run train_mccv() first.")
            self.best_mu = np.load(mu_path)
            self.best_sigma = np.load(sigma_path)

        # build features for this composition
        feat = self._composition_to_features(composition)  # (1, feat_dim)

        # instantiate DAE & MLP and load best weights
        dae = DAE(self.feat_dim, latent_dim=self.dae_latent_dim).to(device)
        mlp = MultiOutputMLP(self.dae_latent_dim,
                             out_dim=len(self.present_targets)).to(device)

        dae_path = self.model_dir / "best_dae_mccv.pth"
        mlp_path = self.model_dir / "best_mlp_mccv.pth"
        if not dae_path.exists() or not mlp_path.exists():
            raise RuntimeError("Best model weights not found. "
                               "Run train_mccv() first.")

        dae.load_state_dict(torch.load(dae_path, map_location=device))
        mlp.load_state_dict(torch.load(mlp_path, map_location=device))
        dae.eval(); mlp.eval()

        x = torch.tensor(feat, dtype=torch.float32, device=device)
        with torch.no_grad():
            z = dae.encode(x)
            pred_std = mlp(z).cpu().numpy()[0]

        pred = pred_std * self.best_sigma + self.best_mu

        return {t: float(pred[self.idx_map[t]]) for t in self.present_targets}

    # ──────────────────────── helpers: composition → features ────
    def _composition_to_features(self, comp: Dict[str, float]) -> np.ndarray:
        """
        Convert user composition dict into the same feature pipeline
        used for training (poly + scaler + concatenated fractions).
        """
        elements = {}
        for key, v in comp.items():
            parsed = re.findall(r"([A-Z][a-z]*)(\d*)", key)
            if not parsed:  # treat as single element symbol
                elements[key] = elements.get(key, 0.0) + v
            else:
                for el, cnt in parsed:
                    elements[el] = elements.get(el, 0.0) + v * int(cnt or "1")

        total = sum(elements.values()) or 1.0
        elements = {k: v / total for k, v in elements.items()}

        feat_vec = np.zeros(len(self.X_comp.columns), dtype=np.float32)
        for i, col in enumerate(self.X_comp.columns):
            feat_vec[i] = elements.get(col, 0.0)

        poly = self.poly.transform([feat_vec])
        poly = self.scaler.transform(poly)

        return np.hstack([poly, feat_vec[None, :]]).astype(np.float32)


# ────────────────────────────────────────────────────────────────
#  Physics-derived properties from coefficients (for demo)
# ────────────────────────────────────────────────────────────────
def derived_properties(coeffs: Dict[str, float], T: float) -> Dict[str, float]:
    out = {}
    if {'rho_a', 'rho_b'}.issubset(coeffs):
        out['rho'] = coeffs['rho_a'] - coeffs['rho_b'] * T
    if {'mu1_a', 'mu1_b'}.issubset(coeffs):
        out['muA'] = coeffs['mu1_a'] * math.exp(coeffs['mu1_b'] / (R * T))
    if {'mu2_a', 'mu2_b', 'mu2_c'}.issubset(coeffs):
        out['muB'] = 10**(coeffs['mu2_a'] +
                          coeffs['mu2_b'] / T +
                          coeffs['mu2_c'] / T**2)
    if {'k_a', 'k_b'}.issubset(coeffs):
        out['k'] = coeffs['k_a'] + coeffs['k_b'] * T
    if {'cp_a', 'cp_b', 'cp_c'}.issubset(coeffs):
        out['cp'] = (coeffs['cp_a'] +
                     coeffs['cp_b'] * T +
                     coeffs['cp_c'] / T**2)
    return out


# ────────────────────────────────────────────────────────────────
#  Standalone script: train + predict
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Load your processed MSTDB CSV
    csv_path = "/Users/meggie/Documents/MoltenSaltPropnet/data/new_mstdb_janz.csv"   # adjust if needed
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path).rename(columns=str.strip)

    # 2. Create trainer
    trainer = MLPDAEMCCVTrainer(df, TARGETS, DERIVED_PROPS)

    # 3. Train with Monte-Carlo CV (30 splits)
    trainer.train_mccv(
        n_splits=30,
        train_frac=0.8,
        dae_epochs=200,
        mlp_epochs=300,
        batch_size=128,
        physics_weight=0.1,
        temp_range=(500.0, 1200.0),
    )

    # 4. Demo prediction: 50-50 NaCl
    example = {"Na": 0.5, "Cl": 0.5}
    print("\nPredicting coefficients for 50-50 NaCl using best MCCV model...")
    coeff = trainer.predict(example)

    for k in sorted(coeff.keys()):
        print(f"{k:7s}: {coeff[k]:11.4f}")

    T_demo = 900.0
    print(f"\nDerived thermo-physical properties at {T_demo:.0f} K:")
    deriv = derived_properties(coeff, T_demo)
    for k in sorted(deriv.keys()):
        print(f"{k:4s}: {deriv[k]:11.4f}")

    # 5. Optional: quick CV summary printout
    print("\nPer-split avg R² and avg relMSE (%):")
    for res in trainer.cv_results_:
        s = res["split"]
        m = res["final_metrics"]
        print(f" split {s:2d}:  avg R²={m['avg_r2']:+.3f}   "
              f"avg relMSE={m['avg_relmse_pct']:.2f}%")

