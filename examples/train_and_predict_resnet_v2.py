#!/usr/bin/env python
"""
Train ResNet+Meta model on MSTDB data, evaluate, visualize, and store outputs.

All outputs are saved under:
    evaluate_modelperformance/resnet_v2/
"""

import os
import sys
import json
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from processing_mstdb.processor import MSTDBProcessor
from processing_mstdb.resnet_v2 import ResNetMetaTrainerv2, cross_validate_resnet_clean


OUTDIR = os.path.join("evaluate_modelperformance", "resnet_v2")
os.makedirs(OUTDIR, exist_ok=True)

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


def _rel_mse_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mse = np.mean((y_true - y_pred) ** 2)
    denom = np.mean(y_true ** 2)
    if not np.isfinite(denom) or denom == 0.0:
        denom = 1e-12
    return 100.0 * mse / denom


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _p90_rel_err(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rel = np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-12)
    return float(np.percentile(rel, 90))


def compute_split_metrics(
    trainer: ResNetMetaTrainerv2,
    split: str = "val",
    min_n: int = 5,
) -> Dict:
    if split == "val":
        idx = trainer.va_idx
    elif split == "test":
        idx = trainer.te_idx
    else:
        raise ValueError("split must be 'val' or 'test'")

    trainer.meta.eval()
    for net in trainer.base_nets.values():
        net.eval()

    μ, σ = trainer.μ, trainer.σ
    Xs = trainer.X_embedded[idx]
    ys = trainer.y_raw[idx]
    ms = trainer.mask_all[idx]

    with torch.no_grad():
        xb = torch.tensor(Xs, dtype=torch.float32, device=trainer.device)
        base_out = torch.stack(
            [trainer.base_nets[p](xb).cpu() for p in trainer.present_targets],
            dim=1,
        ).numpy()
        pred_std = base_out + trainer.meta(
            torch.tensor(base_out, device=trainer.device, dtype=torch.float32)
        ).cpu().numpy()

    pred = pred_std * σ + μ

    per_target: Dict[str, Dict] = {}
    rel_mses, r2s, maes, p90s = [], [], [], []

    print(f"\n{split.capitalize()} results — relMSE (%), R², MAE, p90_rel")
    for j, prop in enumerate(trainer.present_targets):
        mask_j = ms[:, j]
        n_j = int(mask_j.sum())
        if n_j < min_n:
            print(f" • {prop:<8s}: [skipped: only {n_j} samples]")
            continue

        yt = ys[mask_j, j]
        yp = pred[mask_j, j]

        m_rel = _rel_mse_pct(yt, yp)
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        mae = _mae(yt, yp)
        p90 = _p90_rel_err(yt, yp)

        per_target[prop] = {
            "relMSE_pct": float(m_rel),
            "R2": float(r2),
            "MAE": float(mae),
            "p90_rel_err": float(p90),
            "n": n_j,
        }

        print(
            f" • {prop:<8s}: {m_rel:8.2f}%   R²={r2:+.3f}   "
            f"MAE={mae:9.3g}   p90={p90:6.3f}   (n={n_j})"
        )

        rel_mses.append(m_rel)
        r2s.append(r2)
        maes.append(mae)
        p90s.append(p90)

    if rel_mses:
        return {
            "avg_relMSE_pct": float(np.mean(rel_mses)),
            "avg_R2": float(np.mean(r2s)),
            "avg_MAE": float(np.mean(maes)),
            "avg_p90_rel_err": float(np.mean(p90s)),
            "per_target": per_target,
        }
    else:
        return {
            "avg_relMSE_pct": float("nan"),
            "avg_R2": float("nan"),
            "avg_MAE": float("nan"),
            "avg_p90_rel_err": float("nan"),
            "per_target": per_target,
        }


def save_plot(filename: str):
    plt.tight_layout()
    path = os.path.join(OUTDIR, filename)
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


# Plot: per-target R² (val vs test)

def plot_r2_bar(val_metrics: Dict, test_metrics: Dict):
    val_pt = val_metrics["per_target"]
    test_pt = test_metrics["per_target"]

    targets = sorted(set(val_pt.keys()) & set(test_pt.keys()))
    if not targets:
        return

    val_r2 = [val_pt[t]["R2"] for t in targets]
    test_r2 = [test_pt[t]["R2"] for t in targets]

    x = np.arange(len(targets))
    w = 0.35

    plt.figure(figsize=(12, 5))
    plt.bar(x - w/2, val_r2,  w, label="Val R²")
    plt.bar(x + w/2, test_r2, w, label="Test R²")
    plt.axhline(0, color="black")

    plt.xticks(x, targets, rotation=45, ha="right")
    plt.ylabel("R²")
    plt.title("Per-target R² (val vs test)")
    plt.legend()
    save_plot("r2_bar.png")


# Plot: per-target MAE (val vs test, log scale)

def plot_mae_bar(val_metrics: Dict, test_metrics: Dict):
    val_pt = val_metrics["per_target"]
    test_pt = test_metrics["per_target"]

    targets = sorted(set(val_pt.keys()) & set(test_pt.keys()))
    if not targets:
        return

    val_mae = np.array([val_pt[t]["MAE"] for t in targets], dtype=float)
    test_mae = np.array([test_pt[t]["MAE"] for t in targets], dtype=float)

    x = np.arange(len(targets))
    w = 0.35

    plt.figure(figsize=(12, 5))
    plt.bar(x - w/2, val_mae,  w, label="Val MAE")
    plt.bar(x + w/2, test_mae, w, label="Test MAE")

    plt.xticks(x, targets, rotation=45, ha="right")
    plt.ylabel("MAE (log scale)")
    plt.title("Per-target MAE")
    plt.yscale("log")
    plt.legend()
    save_plot("mae_bar.png")



# Combined true vs predicted for all targets

def plot_all_true_vs_pred(trainer: ResNetMetaTrainerv2, split: str = "test", min_n: int = 5):
    if split == "test":
        idx = trainer.te_idx
    elif split == "val":
        idx = trainer.va_idx
    else:
        raise ValueError("split must be 'val' or 'test'")

    trainer.meta.eval()
    for net in trainer.base_nets.values():
        net.eval()

    Xs = trainer.X_embedded[idx]
    ys_all = trainer.y_raw[idx]
    ms_all = trainer.mask_all[idx]
    μ, σ = trainer.μ, trainer.σ

    with torch.no_grad():
        xb = torch.tensor(Xs, dtype=torch.float32, device=trainer.device)
        base_out = torch.stack(
            [trainer.base_nets[p](xb) for p in trainer.present_targets],
            dim=1,
        )
        pred_std = base_out + trainer.meta(base_out)
        pred_all = pred_std.cpu().numpy() * σ + μ

    targets = trainer.present_targets
    n_targets = len(targets)
    n_cols = 3
    n_rows = int(np.ceil(n_targets / n_cols))

    plt.figure(figsize=(5 * n_cols, 4 * n_rows))
    plot_idx = 1

    for j, prop in enumerate(targets):
        mask_j = ms_all[:, j]
        n_j = int(mask_j.sum())
        if n_j < min_n:
            continue

        yt = ys_all[mask_j, j]
        yp = pred_all[mask_j, j]

        plt.subplot(n_rows, n_cols, plot_idx)
        plt.scatter(yt, yp, alpha=0.5, s=10)
        lo = min(yt.min(), yp.min())
        hi = max(yt.max(), yp.max())
        plt.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        plt.title(prop)
        plt.xlabel("True")
        plt.ylabel("Pred")
        plot_idx += 1

    plt.suptitle(f"True vs predicted ({split} set)", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_plot(f"true_vs_pred_all_{split}.png")



# CV R² boxplot

def plot_cv_r2_boxplot(cv_results: List[Dict]):
    r2_per_target: Dict[str, List[float]] = {}

    for fold_res in cv_results:
        test = fold_res.get("test", {})
        pt = test.get("per_target", {})
        for t, m in pt.items():
            r2_per_target.setdefault(t, []).append(m["R2"])

    if not r2_per_target:
        print("cv_results has no per-target R²; skipping CV boxplot.")
        return

    targets = sorted(r2_per_target.keys())
    data = [r2_per_target[t] for t in targets]

    plt.figure(figsize=(12, 5))
    plt.boxplot(data, labels=targets, showmeans=True)
    plt.axhline(0, color="black")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("R² (test across folds)")
    plt.title("Cross-validation R² stability per target")
    save_plot("cv_r2_boxplot.png")



# Optional learning curve (if you logged losses in train_meta)

def plot_learning_curve(trainer: ResNetMetaTrainerv2):
    if not hasattr(trainer, "meta_train_loss") or not hasattr(trainer, "meta_val_loss"):
        print("No meta_train_loss/meta_val_loss on trainer; skipping learning curve.")
        return

    train_losses = trainer.meta_train_loss
    val_losses = trainer.meta_val_loss
    epochs = np.arange(1, len(train_losses) + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Meta network learning curve")
    plt.legend()
    save_plot("learning_curve_meta.png")



# Main

def main():
    # 1. Load data
    csv_path = "/Users/meggie/Documents/MoltenSaltPropnet/data/new_mstdb_janz.csv"
    processor = MSTDBProcessor.from_csv(csv_path)
    processor.df.columns = processor.df.columns.str.strip()

    # 2. Train model
    trainer = ResNetMetaTrainerv2(processor.df, TARGETS, DERIVED_PROPS)
    trainer.train_base()
    trainer.train_meta()

    # (optional) learning curve
    plot_learning_curve(trainer)

    # 3. Metrics
    val_metrics = compute_split_metrics(trainer, split="val")
    test_metrics = compute_split_metrics(trainer, split="test")

    with open(os.path.join(OUTDIR, "val_test_metrics.json"), "w") as f:
        json.dump({"val": val_metrics, "test": test_metrics}, f, indent=4)

    # 4. Plots
    plot_r2_bar(val_metrics, test_metrics)
    plot_mae_bar(val_metrics, test_metrics)
    plot_all_true_vs_pred(trainer, split="test")

    # 5. Cross-validation
    print("\nRunning 5-fold cross-validation...")
    cv_results = cross_validate_resnet_clean(processor.df, TARGETS, DERIVED_PROPS, k=5)
    with open(os.path.join(OUTDIR, "cv_results.json"), "w") as f:
        json.dump(cv_results, f, indent=4)
    plot_cv_r2_boxplot(cv_results)

    print("\nAll results saved under:", OUTDIR)


if __name__ == "__main__":
    main()
