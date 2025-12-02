# train_meta_network.py

import os
import json
import re
import math
import numpy as np
import pandas as pd
import joblib

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel

import torch
import torch.nn as nn



CSV_PATH = "/Users/meggie/Documents/MoltenSaltPropnet/data/new_mstdb_janz.csv"
USE_TARGET_SCALING = False
MAX_N_SPLITS_CV = 3
MIN_SAMPLES = 8
R_GAS = 8.314
T_SWITCH = 850.0
REG_MU1B = 0.01

device = "cuda" if torch.cuda.is_available() else "cpu"

TARGETS = [
    "Melt(K)", "Boil(K)",
    "rho_a", "rho_b",
    "mu1_a", "mu1_b",
    "mu2_a", "mu2_b", "mu2_c",
    "k_a", "k_b",
    "cp_a", "cp_b"
]

DERIVED_GROUPS = [
    ("rho", ["rho_a", "rho_b"]),
    ("muA", ["mu1_a", "mu1_b"]),
    ("muB", ["mu2_a", "mu2_b", "mu2_c"]),
    ("k",   ["k_a", "k_b"]),
    ("cp",  ["cp_a", "cp_b"])
]

# model saving dirs
MODEL_DIR = "models"
BASE_MODEL_DIR = os.path.join(MODEL_DIR, "base_models")
SCALER_DIR = os.path.join(MODEL_DIR, "scalers")
os.makedirs(BASE_MODEL_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)
META_MODEL_PATH = os.path.join(MODEL_DIR, "best_meta_network.pth")


# HELPERS


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



# FEATURE PREPARATION


df["comp"] = df.apply(extract_composition, axis=1)
elements = sorted({el for d in df["comp"] for el in d.keys()})

for el in elements:
    df[f"elem_{el}"] = df["comp"].apply(lambda d: d.get(el, 0.0))

X = df.drop(columns=[c for c in ["System", "Mol Frac", "Composition", "comp"] if c in df.columns])
X = X.fillna(0.0)

models = {}
model_types = {}
scalers = {}
evals_results = {}


def choose_model_type(n_samples, target_name):
    if n_samples >= 200:
        return "catboost"
    elif n_samples >= 50:
        return "ridge"
    else:
        return "gpr"



# TRAIN BASE MODELS


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

    
    # SAVE BASE MODEL TO DISK

    if model_type == "catboost":
        model_path = os.path.join(BASE_MODEL_DIR, f"{t}_catboost.cbm")
        model.save_model(model_path)
    else:
        model_path = os.path.join(BASE_MODEL_DIR, f"{t}_{model_type}.pkl")
        joblib.dump(model, model_path)

    print(f"Saved base model for {t} → {model_path}")

    # if target scaling is used, save the scaler as well
    if USE_TARGET_SCALING and t in scalers:
        scaler_path = os.path.join(SCALER_DIR, f"{t}_scaler.pkl")
        joblib.dump(scalers[t], scaler_path)
        print(f"Saved scaler for {t} → {scaler_path}")

    models[t] = model

print("\n============================================")
print("Finished training all available base target models (mixed types).")
print("============================================\n")


# BUILD MATRICES FOR META NET


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

# train/val split
tr_idx, va_idx = train_test_split(idx_all, test_size=0.2, random_state=42)

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

y_std_mat = (y_raw_mat - μ_vec) / σ_vec
base_pred_std = (base_pred_raw - μ_vec) / σ_vec

y_std_mat_nan0 = np.nan_to_num(y_std_mat, nan=0.0).astype(np.float32)
base_pred_std_nan0 = np.nan_to_num(base_pred_std, nan=0.0).astype(np.float32)
mask_all_float = mask_all.astype(np.float32)


# META NET


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


def physics_loss(pred_raw, y_raw, mask, T_tensor):
    loss = 0.0
    valid_terms = 0
    for dname, coeffs in DERIVED_GROUPS:
        idxs = [present_idx_map[c] for c in coeffs if c in present_idx_map]
        if len(idxs) != len(coeffs):
            continue

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
            p_mu1_a = torch.clamp(p_c[:, 0], min=1e-6)
            y_mu1_a = torch.clamp(y_c[:, 0], min=1e-6)
            p_vals = p_mu1_a * torch.exp(p_c[:, 1] / (R_GAS * T))
            y_vals = y_mu1_a * torch.exp(y_c[:, 1] / (R_GAS * T))
            p_vals = torch.log(p_vals + 1e-8)
            y_vals = torch.log(y_vals + 1e-8)

        elif dname == "muB":
            T2 = T * T
            y_vals = y_c[:, 0] + y_c[:, 1] / T + y_c[:, 2] / T2
            p_vals = p_c[:, 0] + p_c[:, 1] / T + p_c[:, 2] / T2

        elif dname in ("k", "cp"):
            y_vals = y_c[:, 0] + y_c[:, 1] * T
            p_vals = p_c[:, 0] + p_c[:, 1] * T

        term = nn.functional.mse_loss(p_vals, y_vals)
        loss += term
        valid_terms += 1

    if valid_terms == 0:
        return torch.tensor(0.0, device=pred_raw.device)
    return loss / valid_terms


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
    best_state = None

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

        avg_train = train_loss / max(count, 1)
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

    if best_state is not None:
        meta_net.load_state_dict(best_state)
    print("Meta net training finished. Best val loss =", best_val)
    return best_state, best_val



# TRAIN META NET


best_state, best_val = train_meta_net(
    base_pred_std_nan0,
    y_std_mat_nan0,
    mask_all_float,
    tr_idx,
    va_idx,
    μ_vec,
    σ_vec,
)

# save best meta net weights
if best_state is not None:
    torch.save(best_state, META_MODEL_PATH)
    print(f"Saved best meta network weights → {META_MODEL_PATH}")

# APPLY META NET
meta_net.eval()
with torch.no_grad():
    xb_all = torch.tensor(base_pred_std_nan0, device=device)
    delta_all = meta_net(xb_all).cpu().numpy()
    pred_std_all = base_pred_std + delta_all
    pred_raw_all = pred_std_all * σ_vec + μ_vec

# EVALUATION: metrics + predicted_vs_actual table
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
        "r2": float(r2),
        "mse": float(mse),
        "rel_mse_pct": float(relm),
        "n_samples": int(len(y_true)),
    }
    print(f"{t}:")
    print(f"  R²          = {r2:.4f}")
    print(f"  MSE         = {mse:.6g}")
    print(f"  rel MSE (%) = {relm:.3f}")
    print(f"  samples     = {len(y_true)}\n")


# CROSS-VALIDATION
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
    folds_str = [f"{v:.3f}" for v in folds_r2]
    print(f"{t}: model={model_type}, mean CV R² = {mean_r2:.4f}, folds = {folds_str}")


rows = []
for j, t in enumerate(present_targets):
    mask = np.isfinite(y_raw_mat[:, j])
    idxs = np.where(mask)[0]
    y_true = y_raw_mat[mask, j]
    y_pred = pred_raw_all[mask, j]
    for k, (true_val, pred_val) in enumerate(zip(y_true, y_pred)):
        rows.append({
            "index": int(idxs[k]),
            "target": t,
            "actual": float(true_val),
            "predicted": float(pred_val),
            "abs_error": float(abs(true_val - pred_val)),
            "percent_error": float(100.0 * abs(true_val - pred_val) / (abs(true_val) + 1e-12))
        })

pred_vs_actual = pd.DataFrame(rows)
pred_vs_actual.to_csv("predicted_vs_actual_meta.csv", index=False)

with open("catboost_evals_results.json", "w") as f:
    json.dump(evals_results, f)

with open("meta_performance_summary.json", "w") as f:
    json.dump(results, f, indent=2)

with open("cv_results_base_models.json", "w") as f:
    json.dump(cv_results, f, indent=2)

print("\nSaved predicted_vs_actual_meta.csv")
print("Saved catboost_evals_results.json")
print("Saved meta_performance_summary.json")
print("Saved cv_results_base_models.json")
print("\nDone training + exporting for plotting.")
