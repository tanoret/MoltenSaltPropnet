"""
Overview
more or less the architecture to remember


Model architecture - already adjusted for the smaller targets in the dataset
The system has two modelling layers. 
The first layer consists of individual base models predicting raw coefficients for specific physical relationships.
CatBoost, Ridge regression and Gaussian Process Regression are selected automatically depending on the number of available samples for each target. 
These models generate initial coefficient estimates.


The second layer is a neural meta network trained to correct those initial predictions. It receives
the standardized base model outputs and learns small residual adjustments. This network is composed
of a feedforward block with SiLU activation and several residual blocks. Residual blocks help the
network focus on learning corrections instead of replacing predictions completely. The network is
trained with AdamW, which helps stabilize training through decoupled weight decay.


Physics in the modell
The meta network is trained with respect physical
relationships between temperature and properties.
A physics loss term evaluates predicted physicalbehaviour such as thermal conductivity or viscosity over random temperature samples


Whats inside new or optimized from other failures :)
The training uses AdamW optimization to provide better weight decay handling. CosineAnnealingLR is
used as a scheduler to gradually reduce the learning rate. Normalization per coefficient type keeps
the learning stable across varying magnitudes. Composition parsing extracts elemental fractions
from chemical formulas and includes them as explicit features. Error metrics, cross validation, and
plotting tools support model evaluation and diagnostics.


What actually did not work? - i tried at some point data augmentation with the temperature as a coifficient to generate more samples
It did not work i guess, could be more reasons did really not understand how this works, and it seemed like not really to work
As well first there was just Catboost over all of the samples. But catboost was overfitting for the small samples a lot
and actually still predicting bad. this is why the architecture is now changed


The outcome - do have to say looked from time promessing on the plots, but still some where performing actually quite bad
"""

import os
import json
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel

import torch
import torch.nn as nn

sns.set(style="whitegrid")

# why: store configuration values in one place so the workflow stays consistent
CSV_PATH = "/Users/meggie/Documents/MoltenSaltPropnet/data/new_mstdb_janz.csv"
USE_TARGET_SCALING = False
MAX_N_SPLITS_CV = 3
MIN_SAMPLES = 8
R_GAS = 8.314
T_SWITCH = 850.0
REG_MU1B = 0.01
BASE_PLOT_DIR = "visualisation"
CATBOOST_PLOT_DIR = os.path.join(BASE_PLOT_DIR, "catboost_v2")
os.makedirs(CATBOOST_PLOT_DIR, exist_ok=True)


device = "cuda" if torch.cuda.is_available() else "cpu"


# why: helper for saving plots consistently
def _savefig(filename):
    plt.savefig(os.path.join(CATBOOST_PLOT_DIR, filename), dpi=200, bbox_inches="tight")
    print(f"Saved plot → {os.path.join(CATBOOST_PLOT_DIR, filename)}")

def rel_mse_pct(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mse = mean_squared_error(y_true, y_pred)
    denom = np.mean(y_true ** 2)
    if denom < 1e-12:
        denom = 1e-12
    return 100.0 * mse / denom

df = pd.read_csv(CSV_PATH)


def parse_composition_str(x):
    if not isinstance(x, str) or x.strip() == "":
        return None
    try:
        return json.loads(x.replace("'", '"'))
    except Exception:
        return None


def parse_formula(compound):
    if not isinstance(compound, str):
        return {}
    result = {}
    for el, num in re.findall(r"([A-Z][a-z]*)(\d*)", compound):
        count = int(num) if num else 1
        result[el] = result.get(el, 0) + count
    return result


def normalize_counts_to_fractions(d):
    # why: machine learning works better with normalized fractions than raw counts
    if not d:
        return {}
    total = float(sum(d.values()))
    if total <= 0:
        return d
    return {k: v / total for k, v in d.items()}


def extract_composition(row):
    comp = parse_composition_str(row.get("Composition", None))
    if comp is None:
        comp = parse_formula(row.get("System", ""))
    return normalize_counts_to_fractions(comp)


df["comp"] = df.apply(extract_composition, axis=1)
elements = sorted({el for d in df["comp"] for el in d.keys()})


for el in elements:
    df[f"elem_{el}"] = df["comp"].apply(lambda d: d.get(el, 0.0))

X = df.drop(columns=[c for c in ["System", "Mol Frac", "Composition", "comp"] if c in df.columns])
X = X.fillna(0.0)

TARGETS = [
    "Melt(K)", "Boil(K)",
    "rho_a", "rho_b",
    "mu1_a", "mu1_b",
    "mu2_a", "mu2_b", "mu2_c",
    "k_a", "k_b",
    "cp_a", "cp_b"
]

# why: group related coefficients so the meta network can apply physics laws
DERIVED_GROUPS = [
    ("rho", ["rho_a", "rho_b"]),
    ("muA", ["mu1_a", "mu1_b"]),
    ("muB", ["mu2_a", "mu2_b", "mu2_c"]),
    ("k",   ["k_a", "k_b"]),
    ("cp",  ["cp_a", "cp_b"])
]

models = {}
model_types = {}
scalers = {}
evals_results = {}

# why: choose model types based on available data so each target gets an appropriate regressor
def choose_model_type(n_samples, target_name):
    if n_samples >= 200:
        return "catboost"
    elif n_samples >= 50:
        return "ridge"
    else:
        return "gpr"


# why: train one independent model per coefficient so each pattern is captured separately
for t in TARGETS:
    if t not in df.columns:
        print(f"Skipping {t}: column not found.")
        continue
    mask = df[t].notna()
    n = int(mask.sum())
    if n < MIN_SAMPLES:
        print(f"Skipping {t}: not enough data ({n} rows).")
        continue

    model_type = choose_model_type(n, t)
    model_types[t] = model_type
    print(f"Training base model for {t} (n={n}, model={model_type})")

    X_t = X[mask]
    y_t = df.loc[mask, t]

    if USE_TARGET_SCALING:
        scaler = StandardScaler()
        y_scaled = scaler.fit_transform(y_t.values.reshape(-1, 1)).ravel()
        scalers[t] = scaler
    else:
        y_scaled = y_t.values

    if model_type == "catboost":
        X_train, X_val, y_train, y_val = train_test_split(X_t, y_scaled, test_size=0.2, random_state=42)
        model = CatBoostRegressor(
            loss_function="RMSE",
            depth=8,
            learning_rate=0.03,
            iterations=1200,
            eval_metric="RMSE",
            random_seed=42,
            verbose=False
        )
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        evals_results[t] = model.get_evals_result()

    elif model_type == "ridge":
        model = Ridge(alpha=1.0)
        model.fit(X_t, y_scaled)

    elif model_type == "gpr":
        kernel = (
            C(1.0, (1e-2, 1e3)) * Matern(length_scale=1.0, nu=1.5)
            + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-9, 1e1))
        )
        model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=42
        )
        model.fit(X_t, y_scaled)

    models[t] = model

print("\n============================================")
print("Finished training all available base target models (mixed types).")
print("============================================\n")


# why: gather raw true and predicted coefficient matrices for meta network training
present_targets = [t for t in TARGETS if t in models]
print("Targets with trained base models:", present_targets)

y_raw_mat = df[present_targets].to_numpy(float)
mask_all = np.isfinite(y_raw_mat)

n_samples = len(df)
n_props = len(present_targets)
idx_all = np.arange(n_samples)

base_pred_raw = np.full((n_samples, n_props), np.nan)
for j, t in enumerate(present_targets):
    mask = df[t].notna()
    if not mask.any():
        continue
    pred_s = models[t].predict(X[mask])
    if USE_TARGET_SCALING and t in scalers:
        pred = scalers[t].inverse_transform(pred_s.reshape(-1, 1)).ravel()
    else:
        pred = pred_s
    base_pred_raw[mask.values, j] = pred

# ============================================
# TRAIN / VAL SPLIT FOR META NETWORK
# ============================================

# why: split dataset for meta network training and validation
tr_idx, va_idx = train_test_split(idx_all, test_size=0.2, random_state=42)

# why: compute normalization constants so all coefficients share similar scale
μ_vec = np.zeros(n_props)
σ_vec = np.ones(n_props)
for j in range(n_props):
    vals = y_raw_mat[tr_idx, j]
    vals = vals[np.isfinite(vals)]
    if len(vals) > 0:
        μ_vec[j] = np.mean(vals)
        σ_vec[j] = max(np.std(vals), 1e-12)

print("\nPer-target normalization (μ, σ):")
for j, t in enumerate(present_targets):
    print(f"  {t}: mean={μ_vec[j]:.4g}, std={σ_vec[j]:.4g}")

# why: standardized inputs help stabilize meta network training
y_std_mat = (y_raw_mat - μ_vec) / σ_vec
base_pred_std = (base_pred_raw - μ_vec) / σ_vec

y_std_mat_nan0 = np.nan_to_num(y_std_mat, nan=0.0).astype(np.float32)
base_pred_std_nan0 = np.nan_to_num(base_pred_std, nan=0.0).astype(np.float32)
mask_all_float = mask_all.astype(np.float32)

# ============================================
# META NETWORK DEFINITION
# ============================================

# why: residual blocks let the model learn corrections rather than full replacements
class ResidualBlock(nn.Module):
    def __init__(self, dim, p_drop=0.1):
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


# why: meta net predicts small adjustments respecting physical relationships
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


meta_net = MetaNet(n_props, hidden=128, depth=2).to(device)
present_idx_map = {name: j for j, name in enumerate(present_targets)}


# PHYSICS LOSS


# why: physics loss encourages the meta network to output coefficients that produce physically valid temperature dependence
def physics_loss(pred_raw, y_raw, mask, T_tensor):
    loss = 0.0
    valid_terms = 0
    for dname, coeffs in DERIVED_GROUPS:
        idxs = [present_idx_map[c] for c in coeffs if c in present_idx_map]
        if len(idxs) != len(coeffs):
            continue
        # rows where all coefficients for that group are present
        m_group = mask[:, idxs].min(dim=1).values > 0.5
        if not m_group.any():
            continue

        T = T_tensor[m_group]
        y_c = y_raw[m_group][:, idxs]
        p_c = pred_raw[m_group][:, idxs]

        if dname == "rho":
            y_vals = y_c[:, 0] + y_c[:, 1] * T
            p_vals = p_c[:, 0] + p_c[:, 1] * T

        elif dname == "muA":
            # Arrhenius-like form, compare in log space
            p_mu1_a = torch.clamp(p_c[:, 0], min=1e-6)
            y_mu1_a = torch.clamp(y_c[:, 0], min=1e-6)
            p_vals = p_mu1_a * torch.exp(p_c[:, 1] / (R_GAS * T))
            y_vals = y_mu1_a * torch.exp(y_c[:, 1] / (R_GAS * T))
            p_vals = torch.log(p_vals + 1e-8)
            y_vals = torch.log(y_vals + 1e-8)

        elif dname == "muB":
            # simple polynomial in 1/T and 1/T^2
            T2 = T * T
            y_vals = y_c[:, 0] + y_c[:, 1] / T + y_c[:, 2] / T2
            p_vals = p_c[:, 0] + p_c[:, 1] / T + p_c[:, 2] / T2

        elif dname == "k":
            y_vals = y_c[:, 0] + y_c[:, 1] * T
            p_vals = p_c[:, 0] + p_c[:, 1] * T

        elif dname == "cp":
            y_vals = y_c[:, 0] + y_c[:, 1] * T
            p_vals = p_c[:, 0] + p_c[:, 1] * T

        term = nn.functional.mse_loss(p_vals, y_vals)
        loss += term
        valid_terms += 1

    if valid_terms == 0:
        return torch.tensor(0.0, device=pred_raw.device)
    return loss / valid_terms


# META NETWORK TRAINING


# why: train meta network to adjust base predictions toward physically consistent outputs
def train_meta_net(
    base_pred_std_np,
    y_std_np,
    mask_np,
    tr_idx,
    va_idx,
    μ_vec_np,
    σ_vec_np,
    epochs=400,
    batch_size=64,
    physics_weight=0.1,
    temp_range=(500.0, 1200.0),
):
    optimizer = torch.optim.AdamW(meta_net.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-4
    )

    μ_t = torch.tensor(μ_vec_np, device=device, dtype=torch.float32)
    σ_t = torch.tensor(σ_vec_np, device=device, dtype=torch.float32)

    base_tr = base_pred_std_np[tr_idx]
    y_tr = y_std_np[tr_idx]
    m_tr = mask_np[tr_idx]

    base_va = base_pred_std_np[va_idx]
    y_va = y_std_np[va_idx]
    m_va = mask_np[va_idx]

    def batches(idx):
        idx = np.array(idx)
        np.random.shuffle(idx)
        for start in range(0, len(idx), batch_size):
            yield idx[start : start + batch_size]

    best_val = float("inf")
    patience = 0
    PAT = 40

    print("\n=== Training Meta Network ===")
    for epoch in range(epochs):
        meta_net.train()
        train_loss = 0.0
        count = 0

        for idb in batches(tr_idx):
            xb = torch.tensor(base_pred_std_np[idb], device=device)
            yb = torch.tensor(y_std_np[idb], device=device)
            mb = torch.tensor(mask_np[idb], device=device)

            T_rand = torch.rand(len(idb), device=device) * (temp_range[1] - temp_range[0]) + temp_range[0]

            delta = meta_net(xb)
            pred_std = xb + delta

            diff = (pred_std - yb) * mb
            loss_coeff = (diff ** 2).sum() / mb.sum().clamp_min(1)

            pred_raw = pred_std * σ_t + μ_t
            y_raw = yb * σ_t + μ_t

            loss_phys = physics_loss(pred_raw, y_raw, mb, T_rand) * physics_weight

            if "mu1_b" in present_idx_map:
                mu1_b_idx = present_idx_map["mu1_b"]
                loss_mu1b = REG_MU1B * ((pred_std[:, mu1_b_idx] - xb[:, mu1_b_idx]) ** 2).mean()
            else:
                loss_mu1b = 0.0

            loss = loss_coeff + loss_phys + loss_mu1b

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(meta_net.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            count += 1

        scheduler.step()

        meta_net.eval()
        with torch.no_grad():
            xb_va = torch.tensor(base_va, device=device)
            yb_va = torch.tensor(y_va, device=device)
            mb_va = torch.tensor(m_va, device=device)
            delta_va = meta_net(xb_va)
            pred_va = xb_va + delta_va
            diff = (pred_va - yb_va) * mb_va
            val_loss = (diff ** 2).sum() / mb_va.sum().clamp_min(1)

        if count > 0:
            avg_train = train_loss / count
        else:
            avg_train = float("nan")

        print(
            f"Epoch {epoch+1:4d} | train loss = {avg_train:.6f} | val loss = {val_loss.item():.6f}"
        )

        if val_loss.item() < best_val - 1e-4:
            best_val = val_loss.item()
            patience = 0
            best_state = meta_net.state_dict()
        else:
            patience += 1
            if patience >= PAT:
                print("Early stopping meta net training.")
                break

    meta_net.load_state_dict(best_state)
    print("Meta net training finished. Best val loss =", best_val)


train_meta_net(
    base_pred_std_nan0,
    y_std_mat_nan0,
    mask_all_float,
    tr_idx,
    va_idx,
    μ_vec,
    σ_vec,
)


# APPLY META NET TO ALL SAMPLES


# why: apply meta corrections to all samples
meta_net.eval()
with torch.no_grad():
    xb_all = torch.tensor(base_pred_std_nan0, device=device)
    delta_all = meta_net(xb_all).cpu().numpy()
    pred_std_all = base_pred_std + delta_all
    pred_raw_all = pred_std_all * σ_vec + μ_vec


# EVALUATION: R², MSE, REL MSE


results = {}
print("\n=== MODEL PERFORMANCE AFTER META NET (R², MSE, rel MSE %) ===\n")
for j, t in enumerate(present_targets):
    mask = np.isfinite(y_raw_mat[:, j])
    if not mask.any():
        continue
    y_true = y_raw_mat[mask, j]
    y_pred = pred_raw_all[mask, j]
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    relm = rel_mse_pct(y_true, y_pred)
    results[t] = {
        "r2": r2,
        "mse": mse,
        "rel_mse_pct": relm,
        "y_true": y_true,
        "y_pred": y_pred,
    }
    print(f"{t}:")
    print(f"  R²          = {r2:.4f}")
    print(f"  MSE         = {mse:.6g}")
    print(f"  rel MSE (%) = {relm:.3f}")
    print(f"  samples     = {len(y_true)}\n")

# ============================================
# CROSS-VALIDATION FOR BASE MODELS
# ============================================

# helper for computing cross validation metrics for base models only
# since we have not that much data in some rows the limit 5 which i wanted to do first with montecarlo sampling
# is as well not possible since it needs many splitts to be stable. I found here 2/3 where good
# but paper are suggesting 2 is kind of bit not enough to find out if its a stable performance

def crossval_for_target(X_t, y_t, model_type, use_scaling=False, max_splits=MAX_N_SPLITS_CV):
    n_samples = len(X_t)
    n_splits = min(max_splits, n_samples)
    if n_splits < 2:
        return None, []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in kf.split(X_t):
        X_tr, X_va = X_t.iloc[train_idx], X_t.iloc[val_idx]
        y_tr, y_va = y_t.iloc[train_idx], y_t.iloc[val_idx]

        if use_scaling:
            scaler = StandardScaler()
            y_tr_s = scaler.fit_transform(y_tr.values.reshape(-1, 1)).ravel()
            y_va_s = scaler.transform(y_va.values.reshape(-1, 1)).ravel()
        else:
            y_tr_s = y_tr.values
            y_va_s = y_va.values

        if model_type == "catboost":
            m = CatBoostRegressor(
                loss_function="RMSE",
                depth=8,
                learning_rate=0.03,
                iterations=800,
                eval_metric="RMSE",
                random_seed=42,
                verbose=False
            )
            m.fit(X_tr, y_tr_s, eval_set=(X_va, y_va_s), verbose=False)
        elif model_type == "ridge":
            m = Ridge(alpha=1.0)
            m.fit(X_tr, y_tr_s)
        elif model_type == "gpr":
            kernel = (
                C(1.0, (1e-2, 1e3)) * Matern(length_scale=1.0, nu=1.5)
                + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-9, 1e1))
            )
            m = GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                n_restarts_optimizer=1,
                random_state=42
            )
            m.fit(X_tr, y_tr_s)
        else:
            continue

        y_pred = m.predict(X_va)
        if use_scaling:
            y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()

        scores.append(r2_score(y_va, y_pred))

    return float(np.mean(scores)), scores


print("\n=== BASE MODEL K-FOLD CROSS-VALIDATION (R²) ===\n")
cv_results = {}
for t in present_targets:
    model_type = model_types.get(t, None)
    if model_type is None:
        continue
    mask = df[t].notna()
    n = int(mask.sum())
    if n < MIN_SAMPLES:
        continue
    X_t = X[mask]
    y_t = df.loc[mask, t]
    mean_r2, folds_r2 = crossval_for_target(X_t, y_t, model_type, use_scaling=USE_TARGET_SCALING)
    if mean_r2 is None:
        continue
    cv_results[t] = {"mean_r2": mean_r2, "folds_r2": folds_r2}
    print(f"{t}: model={model_type}, mean CV R² = {mean_r2:.4f}, folds = {[f'{v:.3f}' for v in folds_r2]}")



# PREDICTED vs ACTUAL TABLE (FOR META OUTPUT)


rows = []
for j, t in enumerate(present_targets):
    mask = np.isfinite(y_raw_mat[:, j])
    idxs = np.where(mask)[0]
    y_true = y_raw_mat[mask, j]
    y_pred = pred_raw_all[mask, j]
    for k, (true_val, pred_val) in enumerate(zip(y_true, y_pred)):
        rows.append({
            "index": idxs[k],
            "target": t,
            "actual": true_val,
            "predicted": pred_val,
            "abs_error": abs(true_val - pred_val),
            "percent_error": 100.0 * abs(true_val - pred_val) / (abs(true_val) + 1e-12)
        })

pred_vs_actual = pd.DataFrame(rows)
pred_vs_actual.to_csv("predicted_vs_actual_meta.csv", index=False)
print("\nSaved predicted_vs_actual_meta.csv")


def plot_actual_vs_predicted(df_pa, target, save=True):
    d = df_pa[df_pa["target"] == target]
    if len(d) == 0:
        print(f"No data for {target}")
        return

    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=d, x="actual", y="predicted", alpha=0.6)
    lo = min(d["actual"].min(), d["predicted"].min())
    hi = max(d["actual"].max(), d["predicted"].max())
    plt.plot([lo, hi], [lo, hi], "r--")

    plt.title(f"Actual vs Predicted – {target}")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()

    if save:
        _savefig(f"actual_vs_pred_{target}.png")

    plt.show()


def plot_residuals(df_pa, target, save=True):
    d = df_pa[df_pa["target"] == target].copy()
    if len(d) == 0:
        print(f"No data for {target}")
        return

    d["residual"] = d["predicted"] - d["actual"]

    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=d, x="actual", y="residual", alpha=0.6)
    plt.axhline(0, color="red", linestyle="--")

    plt.title(f"Residuals – {target}")
    plt.xlabel("Actual")
    plt.ylabel("Residual")
    plt.tight_layout()

    if save:
        _savefig(f"residuals_{target}.png")

    plt.show()


def plot_error_distribution(df_pa, target, percent=False, save=True):
    d = df_pa[df_pa["target"] == target]
    if len(d) == 0:
        print(f"No data for {target}")
        return

    plt.figure(figsize=(6, 4))
    values = d["percent_error"] if percent else d["abs_error"]
    sns.histplot(values, bins=30, kde=True)

    plt.xlabel("Percent Error (%)" if percent else "Absolute Error")
    plt.title(f"Error Distribution – {target}")
    plt.tight_layout()

    if save:
        suffix = "pct" if percent else "abs"
        _savefig(f"error_dist_{target}_{suffix}.png")

    plt.show()


def plot_confusion_style_heatmap(df_pa, target, bins=40, save=True):
    d = df_pa[df_pa["target"] == target]
    if len(d) == 0:
        print(f"No data for {target}")
        return

    actual = d["actual"].values
    pred   = d["predicted"].values

    vmin = min(actual.min(), pred.min())
    vmax = max(actual.max(), pred.max())

    plt.figure(figsize=(6, 5))
    plt.hist2d(actual, pred, bins=bins,
               range=[[vmin, vmax], [vmin, vmax]],
               cmap="viridis")
    plt.plot([vmin, vmax], [vmin, vmax], "r--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Confusion-style Heatmap – {target}")
    plt.colorbar(label="Count")
    plt.tight_layout()

    if save:
        _savefig(f"confusion_heatmap_{target}.png")

    plt.show()


def plot_all_targets_grid(df_pa, targets, cols=4, save=True):
    n = len(targets)
    rows_grid = (n + cols - 1) // cols
    plt.figure(figsize=(cols * 4, rows_grid * 3.2))

    for i, t in enumerate(targets):
        d = df_pa[df_pa["target"] == t]
        if len(d) == 0:
            continue

        plt.subplot(rows_grid, cols, i + 1)
        sns.scatterplot(data=d, x="actual", y="predicted", alpha=0.6)
        lo = min(d["actual"].min(), d["predicted"].min())
        hi = max(d["actual"].max(), d["predicted"].max())
        plt.plot([lo, hi], [lo, hi], "r--")
        plt.title(t)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")

    plt.tight_layout()
    if save:
        _savefig("all_targets_grid.png")

    plt.show()


def build_error_bin_matrix(df_pa, targets, bins=(0, 5, 10, 20, 50, 100)):
    labels = []
    for i in range(len(bins) - 1):
        labels.append(f"{bins[i]}–{bins[i+1]}%")
    labels.append(f">{bins[-1]}%")

    mat = pd.DataFrame(0, index=targets, columns=labels)

    for t in targets:
        d = df_pa[df_pa["target"] == t]
        if len(d) == 0:
            continue
        errs = d["percent_error"].values
        idxs = np.digitize(errs, bins)

        for idx in idxs:
            if idx < len(bins):
                col = labels[idx - 1] if idx > 0 else labels[0]
            else:
                col = labels[-1]
            mat.loc[t, col] += 1

    return mat


def plot_error_bin_matrix(df_pa, targets, bins=(0, 5, 10, 20, 50, 100), save=True):
    mat = build_error_bin_matrix(df_pa, targets, bins=bins)

    plt.figure(figsize=(1.4 * len(mat.columns), 0.5 * len(mat.index) + 3))
    sns.heatmap(mat, annot=True, fmt="d", cmap="viridis", cbar_kws={"label": "Count"})
    plt.xlabel("Percent Error Bin")
    plt.ylabel("Target")
    plt.title("Error-bin Confusion Matrix")
    plt.tight_layout()

    if save:
        _savefig("error_bin_confusion_matrix.png")

    plt.show()


def plot_error_by_system(df_raw, df_pa, target, top_n_systems=10, percent=True, save=True):
    d = df_pa[df_pa["target"] == target]
    if len(d) == 0:
        print(f"No data for {target}")
        return

    if "System" not in df_raw.columns:
        print("System column not in df_raw; cannot plot error by system.")
        return

    merged = d.merge(df_raw[["System"]], left_on="index", right_index=True, how="left")

    err_col = "percent_error" if percent else "abs_error"
    ylabel  = "Percent Error (%)" if percent else "Absolute Error"

    top = merged["System"].value_counts().head(top_n_systems).index
    merged = merged[merged["System"].isin(top)]

    plt.figure(figsize=(max(8, 0.7 * len(top)), 4))
    sns.boxplot(data=merged, x="System", y=err_col)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Error by System – {target}")
    plt.ylabel(ylabel)
    plt.tight_layout()

    if save:
        _savefig(f"error_by_system_{target}.png")

    plt.show()


def plot_catboost_learning_curve(target, save=True):
    if target not in evals_results:
        print(f"No CatBoost evals found for {target}.")
        return

    ev = evals_results[target]
    train_loss = ev["learn"]["RMSE"]
    val_loss = ev["validation"]["RMSE"]

    plt.figure(figsize=(6, 4))
    plt.plot(train_loss, label="Train RMSE")
    plt.plot(val_loss, label="Validation RMSE")
    plt.xlabel("Iteration")
    plt.ylabel("RMSE")
    plt.title(f"CatBoost Learning Curve – {target}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save:
        _savefig(f"learning_curve_{target}.png")

    plt.show()


def generate_all_plots(df_raw, pred_df, targets, top_n_systems=10):
    """
    Generate all diagnostic plots for all targets and save them
    into visualisation/catboost.
    """
    print("\n=== Generating per-target plots ===")
    for t in targets:
        d = pred_df[pred_df["target"] == t]
        if len(d) == 0:
            print(f"[{t}] – no data, skipping.")
            continue

        print(f"[{t}] – plotting...")

        plot_actual_vs_predicted(pred_df, t, save=True)
        plot_residuals(pred_df, t, save=True)
        plot_error_distribution(pred_df, t, percent=False, save=True)
        plot_error_distribution(pred_df, t, percent=True,  save=True)
        plot_confusion_style_heatmap(pred_df, t, bins=40, save=True)
        plot_error_by_system(df_raw, pred_df, t,
                             top_n_systems=top_n_systems,
                             percent=True,
                             save=True)

        if model_types.get(t) == "catboost":
            plot_catboost_learning_curve(t, save=True)

    print("\n=== Generating global plots (all targets) ===")
    plot_all_targets_grid(pred_df, targets, cols=4, save=True)
    plot_error_bin_matrix(pred_df, targets, bins=(0, 5, 10, 20, 50, 100), save=True)
    print("\nAll plots generated and saved in 'visualisation/catboost'.")


if __name__ == "__main__":
    print("\n=== Running full pipeline (base models + meta net + diagnostics) ===\n")
    generate_all_plots(df, pred_vs_actual, present_targets, top_n_systems=10)
    print("\nDone.")
