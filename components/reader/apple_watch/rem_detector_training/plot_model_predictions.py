#!/usr/bin/env python3
"""
Plot model predictions vs ground truth on test data.
For each patient: x-axis = time (hours since sleep start), y-axis = sleep stage (0=non-REM, 1=REM).
Shows ground truth and predictions as step plots.
"""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).parent))
from create_data_split_variants import load_splits

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "visualizations"
DATA_DIR = SCRIPT_DIR.parent.parent / "data" / "tlr_data" / "walch-apple-watch"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_subject_ids() -> list[str]:
    """Load sorted subject IDs (same order as used in create_data_split_variants)."""
    all_ids = []
    for split_file in ["train_ids.txt", "val_ids.txt", "test_ids.txt"]:
        path = DATA_DIR / split_file
        if path.exists():
            with open(path) as f:
                ids = [line.strip() for line in f if line.strip() and line.strip() != "id"]
                all_ids.extend(ids)
    return sorted(set(all_ids))


def infer_variant_from_model_path(model_path: Path) -> str:
    """Infer variant from model path, e.g. models/full/rem_detector_*.pkl -> full"""
    parts = model_path.parts
    if "models" in parts:
        idx = parts.index("models")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "baseline"


def plot_predictions_per_patient(
    model_path: Path,
    output_path: Path | None = None,
    patients: list[str] | None = None,
) -> None:
    """
    Load model, get predictions on test data, and plot ground truth vs preds per patient.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    variant = infer_variant_from_model_path(model_path)
    print(f"Loading model: {model_path}")
    print(f"Inferred variant: {variant}")

    with open(model_path, "rb") as f:
        save_data = pickle.load(f)

    model = save_data["model"]
    scaler = save_data.get("scaler")
    feature_names = save_data.get("feature_names", [])

    _, X_test, _, y_test, split_info, _, test_subj_idx = load_splits(variant)
    test_subjects = split_info["test_subjects"]

    if scaler is not None:
        X_test_transformed = scaler.transform(X_test)
    else:
        X_test_transformed = X_test

    y_pred = model.predict(X_test_transformed)

    # Get time_hours for x-axis if available
    if "time_hours" in feature_names:
        time_col = feature_names.index("time_hours")
        time_values = X_test[:, time_col]
    else:
        # Fallback: use epoch index, convert to hours (30s epochs)
        time_values = np.arange(len(X_test)) * (30 / 3600)

    all_subject_ids = load_subject_ids()
    unique_subjects = np.unique(test_subj_idx)
    subject_names = [all_subject_ids[i] for i in unique_subjects]

    if patients:
        patient_set = set(patients)
        keep_mask = np.array([s in patient_set for s in subject_names])
        unique_subjects = unique_subjects[keep_mask]
        subject_names = [s for s in subject_names if s in patient_set]

    n_patients = len(unique_subjects)
    n_cols = 2
    n_rows = (n_patients + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    if n_patients == 1:
        axes = np.array([[axes]])
    axes = axes.flatten()

    model_name = model_path.stem.replace("rem_detector_", "").replace("_", " ").title()

    for ax_idx, (subj_idx, subj_name) in enumerate(zip(unique_subjects, subject_names)):
        ax = axes[ax_idx]
        mask = test_subj_idx == subj_idx

        t = time_values[mask]
        y_true = y_test[mask]
        y_pr = y_pred[mask]

        # Step plot: ground truth (solid) vs predictions (dashed)
        ax.step(t, y_true, where="mid", color="tab:blue", label="Ground truth", linewidth=1.5)
        ax.step(t, y_pr, where="mid", color="tab:orange", label="Predicted", linewidth=1.2, linestyle="--", alpha=0.9)

        ax.set_xlabel("Time (hours since sleep start)")
        ax.set_ylabel("Sleep stage")
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Non-REM", "REM"])
        ax.set_title(f"Patient {subj_name}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    for ax_idx in range(n_patients, len(axes)):
        axes[ax_idx].set_visible(False)

    fig.suptitle(f"{model_name} - Ground Truth vs Predictions (Test Set)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path is None:
        output_path = OUTPUT_DIR / f"{model_path.stem}_predictions_per_patient.png"
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot model predictions vs ground truth per patient"
    )
    parser.add_argument(
        "model_path",
        type=Path,
        help="Path to model pickle (e.g. models/full/rem_detector_gaussian_nb.pkl)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output path for plot",
    )
    parser.add_argument(
        "--patients",
        nargs="+",
        default=None,
        help="Optional: plot only these patient IDs",
    )
    args = parser.parse_args()

    plot_predictions_per_patient(
        model_path=args.model_path,
        output_path=args.output,
        patients=args.patients,
    )


if __name__ == "__main__":
    main()
