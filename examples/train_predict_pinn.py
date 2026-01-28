import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from processing_saltdblean.processor import SALTDBLEANProcessor
from processing_saltdblean.pinn_trainer import (
    PINNMetaTrainer,
    TARGETS,
    DERIVED_PROPS,
    ELEMENT_FEATURE_COLS,
)

DATA_PATH = "/Users/meggie/Documents/MoltenSaltPropnet/data/new_mstdb_janz_with_ionic_polarizability.csv"

BASE_OUTDIR = os.path.join("data", "pinn_compare")
PLOT_DIR_A = os.path.join(BASE_OUTDIR, "pinn_A_plots")
PLOT_DIR_B = os.path.join(BASE_OUTDIR, "pinn_B_plots")
PLOT_DIR_COMPARE = os.path.join(BASE_OUTDIR, "pinn_A_vs_B_plots")

MODEL_DIR_A = os.path.join(BASE_OUTDIR, "models_A_without_elements")
MODEL_DIR_B = os.path.join(BASE_OUTDIR, "models_B_with_elements")

TEMPERATURE = 900.0
MAX_TARGETS_PER_FIG = 8
ELEMENT_FILTERS = ["Cl", "F"]

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs(PLOT_DIR_A, exist_ok=True)
os.makedirs(PLOT_DIR_B, exist_ok=True)
os.makedirs(PLOT_DIR_COMPARE, exist_ok=True)
os.makedirs(MODEL_DIR_A, exist_ok=True)
os.makedirs(MODEL_DIR_B, exist_ok=True)


def _filter_indices_by_element(trainer: PINNMetaTrainer, idxs: np.ndarray, element: str) -> np.ndarray:
    return trainer._indices_with_element(idxs, element)

def _true_phys_matrix(trainer: PINNMetaTrainer, idxs: np.ndarray) -> np.ndarray:
    return trainer._true_phys_matrix(idxs)

def _predict_phys_matrix(trainer: PINNMetaTrainer, idxs: np.ndarray) -> np.ndarray:
    Xs = trainer.X_embedded[idxs]
    return trainer.predict_batch(Xs)

def _make_coeff_compare_plots(
    trainerA: PINNMetaTrainer,
    trainerB: PINNMetaTrainer,
    idxs: np.ndarray,
    element_label: str,
    split_label: str,
    out_dir: str,
    max_targets_per_fig: int = 8,
):
    """
    Creates multi-figure coefficient comparison:
      left column = A (no elem feats)
      right column = B (with elem feats)
    grouped by targets (0..7), (8..12), etc.
    """
    os.makedirs(out_dir, exist_ok=True)

    y_true = _true_phys_matrix(trainerA, idxs)
    M = trainerA.mask_all[idxs].astype(bool)

    y_pred_A = _predict_phys_matrix(trainerA, idxs)
    y_pred_B = _predict_phys_matrix(trainerB, idxs)

    targets = trainerA.present_targets
    nT = len(targets)

    start = 0
    fig_id = 0
    while start < nT:
        end = min(nT, start + max_targets_per_fig)
        chunk = list(range(start, end))
        fig_id += 1

        nrows = len(chunk)
        fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(10, 3.2 * nrows))
        if nrows == 1:
            axes = np.array([axes])

        for r, j in enumerate(chunk):
            tname = targets[j]
            m = M[:, j]
            axA = axes[r, 0]
            axB = axes[r, 1]

            if not np.any(m):
                axA.set_axis_off()
                axB.set_axis_off()
                continue

            yt = y_true[m, j].astype(float)

            ypA = y_pred_A[m, j].astype(float)
            ypB = y_pred_B[m, j].astype(float)

            # A
            axA.scatter(yt, ypA, alpha=0.7)
            mnA = float(min(np.min(yt), np.min(ypA)))
            mxA = float(max(np.max(yt), np.max(ypA)))
            axA.plot([mnA, mxA], [mnA, mxA], "k--", linewidth=1.0)
            axA.set_title(f"{tname} — A (WITHOUT elem-feats)")
            axA.set_xlabel("Actual")
            axA.set_ylabel("Predicted")
            axA.grid(True, alpha=0.3)

            # B
            axB.scatter(yt, ypB, alpha=0.7)
            mnB = float(min(np.min(yt), np.min(ypB)))
            mxB = float(max(np.max(yt), np.max(ypB)))
            axB.plot([mnB, mxB], [mnB, mxB], "k--", linewidth=1.0)
            axB.set_title(f"{tname} — B (WITH elem-feats)")
            axB.set_xlabel("Actual")
            axB.set_ylabel("Predicted")
            axB.grid(True, alpha=0.3)

        fig.suptitle(f"Element filter: {element_label} | split={split_label} | N={len(idxs)}", fontsize=12)
        plt.tight_layout(rect=[0, 0.0, 1, 0.98])

        fname = f"{element_label}_{split_label}_coeff_A_vs_B_{start}_{end-1}.png"
        fig.savefig(os.path.join(out_dir, fname), dpi=160)
        plt.close(fig)

        start = end


def _make_derived_compare_plot(
    trainerA: PINNMetaTrainer,
    trainerB: PINNMetaTrainer,
    idxs: np.ndarray,
    element_label: str,
    split_label: str,
    out_dir: str,
    temperature: float = 900.0,
    derived_list=None,
):
    """
    A single multi-row x 2-col figure like your screenshot:
      rows = derived properties
      col 0 = A
      col 1 = B
    """
    os.makedirs(out_dir, exist_ok=True)

    if derived_list is None:
        # show more if you want; but rho/k are the most stable
        derived_list = ["rho", "muA", "muB", "k", "cp"]

    y_true = _true_phys_matrix(trainerA, idxs)
    y_pred_A = _predict_phys_matrix(trainerA, idxs)
    y_pred_B = _predict_phys_matrix(trainerB, idxs)

    targets = trainerA.present_targets
    nrows = len(derived_list)

    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(10, 3.0 * nrows))
    if nrows == 1:
        axes = np.array([axes])

    # Build derived vectors
    def derived_pairs(trainer: PINNMetaTrainer, ymat: np.ndarray):
        out = {k: {"act": [], "pred": []} for k in derived_list}
        for i in range(len(idxs)):
            coeff_true = {t: float(y_true[i, j]) for j, t in enumerate(targets)}
            coeff_pred = {t: float(ymat[i, j]) for j, t in enumerate(targets)}

            a_props = trainer.derived(coeff_true, temperature)
            p_props = trainer.derived(coeff_pred, temperature)

            for k in derived_list:
                av = a_props.get(k, None)
                pv = p_props.get(k, None)
                if av is None or pv is None:
                    continue
                if not np.isfinite(av) or not np.isfinite(pv):
                    continue
                if abs(av) < 1e-12:
                    continue
                out[k]["act"].append(av)
                out[k]["pred"].append(pv)
        return out

    dA = derived_pairs(trainerA, y_pred_A)
    dB = derived_pairs(trainerB, y_pred_B)

    for r, pname in enumerate(derived_list):
        axA = axes[r, 0]
        axB = axes[r, 1]

        # A
        ytA = np.asarray(dA[pname]["act"], dtype=float)
        ypA = np.asarray(dA[pname]["pred"], dtype=float)
        if ytA.size >= 10:
            axA.scatter(ytA, ypA, alpha=0.7)
            mn = float(min(np.min(ytA), np.min(ypA)))
            mx = float(max(np.max(ytA), np.max(ypA)))
            axA.plot([mn, mx], [mn, mx], "k--", linewidth=1.0)
        axA.set_title(f"{pname} @ {temperature:.0f}K — A (WITHOUT elem-feats)")
        axA.set_xlabel("Actual")
        axA.set_ylabel("Predicted")
        axA.grid(True, alpha=0.3)

        # B
        ytB = np.asarray(dB[pname]["act"], dtype=float)
        ypB = np.asarray(dB[pname]["pred"], dtype=float)
        if ytB.size >= 10:
            axB.scatter(ytB, ypB, alpha=0.7)
            mn = float(min(np.min(ytB), np.min(ypB)))
            mx = float(max(np.max(ytB), np.max(ypB)))
            axB.plot([mn, mx], [mn, mx], "k--", linewidth=1.0)
        axB.set_title(f"{pname} @ {temperature:.0f}K — B (WITH elem-feats)")
        axB.set_xlabel("Actual")
        axB.set_ylabel("Predicted")
        axB.grid(True, alpha=0.3)

    fig.suptitle(f"Element filter: {element_label} | split={split_label} | N={len(idxs)}", fontsize=12)
    plt.tight_layout(rect=[0, 0.0, 1, 0.98])

    fname = f"{element_label}_derived_A_vs_B_T{int(temperature)}K.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=160)
    plt.close(fig)


def main():
    processor = SALTDBLEANProcessor.from_csv(DATA_PATH)
    df = processor.df
    df.columns = df.columns.str.strip()

    print("\n=== PINN A (no element features) ===")
    trainerA = PINNMetaTrainer(
        df=df,
        target_cols=TARGETS,
        derived_props=DERIVED_PROPS,
        element_feature_cols=ELEMENT_FEATURE_COLS,
        use_element_features=False,
        embedding_method="pca",
        n_components=32,
        model_dir=MODEL_DIR_A,
        seed=SEED,
    )
    trainerA.train(
        lr=3e-4,              # lower LR for stability
        num_epochs=500,
        patience_limit=80,
        kl_warmup_epochs=60,
        phys_warmup_epochs=60,
    )
    mA_test = trainerA.evaluate("test", return_dict=True)
    mA_Cl = trainerA.evaluate_by_element("test", "Cl")
    mA_F = trainerA.evaluate_by_element("test", "F")

    trainerA.make_plots(PLOT_DIR_A, temperature=TEMPERATURE, splits=("train", "test"), derived_to_compare=["rho", "k"])

    trainerA.save_metrics_text(
        out_path=os.path.join(PLOT_DIR_A, "metrics_test.txt"),
        metrics_main={**mA_test, "split": "test"},
        metrics_by_element=[
            {**mA_Cl, "split": "test", "element": "Cl"},
            {**mA_F, "split": "test", "element": "F"},
        ],
        header="PINN A metrics (no element features)",
    )

    print("\n=== PINN B (with element features) ===")
    splits = (trainerA.tr_idx, trainerA.va_idx, trainerA.te_idx)
    trainerB = PINNMetaTrainer(
        df=df,
        target_cols=TARGETS,
        derived_props=DERIVED_PROPS,
        element_feature_cols=ELEMENT_FEATURE_COLS,
        use_element_features=True,
        embedding_method="pca",
        n_components=32,
        splits=splits,
        model_dir=MODEL_DIR_B,
        seed=SEED,
    )
    trainerB.train(
        lr=3e-4,              # same LR for fair A/B
        num_epochs=500,
        patience_limit=80,
        kl_warmup_epochs=60,
        phys_warmup_epochs=60,
    )
    mB_test = trainerB.evaluate("test", return_dict=True)
    mB_Cl = trainerB.evaluate_by_element("test", "Cl")
    mB_F = trainerB.evaluate_by_element("test", "F")

    trainerB.make_plots(PLOT_DIR_B, temperature=TEMPERATURE, splits=("train", "test"), derived_to_compare=["rho", "k"])

    trainerB.save_metrics_text(
        out_path=os.path.join(PLOT_DIR_B, "metrics_test.txt"),
        metrics_main={**mB_test, "split": "test"},
        metrics_by_element=[
            {**mB_Cl, "split": "test", "element": "Cl"},
            {**mB_F, "split": "test", "element": "F"},
        ],
        header="PINN B metrics (with element features)",
    )

    # ============================================================
    # A vs B Comparison plots for Cl and F on TEST split
    # ============================================================
    test_idxs = trainerA.te_idx

    compare_metrics_lines = []
    compare_metrics_lines.append("PINN A vs B — element-filter comparison (TEST split)")
    compare_metrics_lines.append("=" * 60)
    compare_metrics_lines.append("")

    for el in ELEMENT_FILTERS:
        idx_el = _filter_indices_by_element(trainerA, test_idxs, el)
        if idx_el.size == 0:
            print(f"[COMPARE] element={el}: no test rows.")
            continue

        print(f"\n[COMPARE] Making A vs B plots for element={el} | N={len(idx_el)}")
        _make_coeff_compare_plots(
            trainerA, trainerB,
            idxs=idx_el,
            element_label=el,
            split_label="test",
            out_dir=PLOT_DIR_COMPARE,
            max_targets_per_fig=MAX_TARGETS_PER_FIG,
        )
        _make_derived_compare_plot(
            trainerA, trainerB,
            idxs=idx_el,
            element_label=el,
            split_label="test",
            out_dir=PLOT_DIR_COMPARE,
            temperature=TEMPERATURE,
            derived_list=["rho", "muA", "muB", "k", "cp"],
        )

        # Store metrics lines for compare folder
        A_el = trainerA.evaluate_by_element("test", el)
        B_el = trainerB.evaluate_by_element("test", el)

        compare_metrics_lines.append(f"Element: {el} | N={A_el['n_rows']}")
        compare_metrics_lines.append("-" * 60)

        # Overview
        compare_metrics_lines.append(f"A avg MSE%={A_el['avg_mse_pct']:.4f} | avg R2={A_el['avg_r2']:.4f}")
        compare_metrics_lines.append(f"B avg MSE%={B_el['avg_mse_pct']:.4f} | avg R2={B_el['avg_r2']:.4f}")
        compare_metrics_lines.append("")

        # Per-target table
        compare_metrics_lines.append("Per-target comparison:")
        compare_metrics_lines.append(f"{'Target':12s} | {'A_MSE%':>10s} | {'A_R2':>8s} | {'B_MSE%':>10s} | {'B_R2':>8s} | N")
        compare_metrics_lines.append("-" * 60)

        targets = trainerA.present_targets
        for t in targets:
            Ad = A_el["per_target"].get(t, None)
            Bd = B_el["per_target"].get(t, None)

            if Ad is None or Bd is None:
                continue

            compare_metrics_lines.append(
                f"{t:12s} | "
                f"{Ad['MSE_pct']:10.4f} | {Ad['R2']:8.4f} | "
                f"{Bd['MSE_pct']:10.4f} | {Bd['R2']:8.4f} | "
                f"{Ad.get('N',0)}"
            )

        compare_metrics_lines.append("\n")

    # Save compare metrics summary in the compare plot folder
    compare_metrics_path = os.path.join(PLOT_DIR_COMPARE, "metrics_compare_test.txt")
    with open(compare_metrics_path, "w", encoding="utf-8") as f:
        f.write("\n".join(compare_metrics_lines))

    print(f"\nAll compare plots saved in: {PLOT_DIR_COMPARE}")
    print(f"Compare metrics saved in: {compare_metrics_path}")


if __name__ == "__main__":
    main()
