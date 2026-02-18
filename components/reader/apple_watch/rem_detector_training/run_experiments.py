#!/usr/bin/env python3
"""
Run All Experiments: Benchmark Feature Variants vs Baseline
1. Creates data splits for all feature variants
2. Trains models on each variant with GroupKFold CV
3. Generates comparison report
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd

# Paths
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_cmd(cmd, description):
    """Run a command and return success status."""
    project_root = SCRIPT_DIR.parent.parent.parent
    print(f"\n{'='*60}")
    print(f"▶ {description}")
    print(f"  $ {' '.join(cmd)}")
    print('='*60)
    result = subprocess.run(cmd, cwd=project_root)
    return result.returncode == 0


def collect_results():
    """Collect results from all variant training runs."""
    results = {}
    models_dir = SCRIPT_DIR / "models"

    for variant_dir in models_dir.iterdir():
        if not variant_dir.is_dir():
            continue
        if variant_dir.name.endswith('_phase2'):
            continue  # Skip phase2 dirs - different pickle format (no metrics)
        variant = variant_dir.name
        results[variant] = {}

        for pkl_file in variant_dir.glob("rem_detector_*.pkl"):
            import pickle
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            model_name = pkl_file.stem.replace('rem_detector_', '').replace('_', ' ').title()
            metrics = data.get('metrics', {})
            if not metrics:
                continue  # Skip pickles without metrics (e.g. ensemble format)
            results[variant][model_name] = metrics

    return results


def load_latest_results():
    """Load latest results from JSON files if pickle collection fails."""
    results = {}
    for f in RESULTS_DIR.glob("train_*_*.json"):
        # Format: train_{variant}_{YYYYMMDD}_{HHMMSS}.json
        parts = f.stem.split('_')
        if len(parts) >= 4:
            variant = '_'.join(parts[1:-2])  # e.g. "baseline" or "normalized_zscore"
            if variant not in results:
                results[variant] = {}
            with open(f) as fp:
                data = json.load(fp)
            for model_name, metrics in data.items():
                results[variant][model_name] = metrics
    return results


def generate_report(results):
    """Generate comparison report."""
    rows = []

    for variant, models in results.items():
        for model_name, metrics in models.items():
            rows.append({
                'variant': variant,
                'model': model_name,
                'test_roc_auc': metrics.get('roc_auc', np.nan),
                'test_recall': metrics.get('recall', np.nan),
                'test_precision': metrics.get('precision', np.nan),
                'cv_roc_auc': metrics.get('cv_roc_auc_mean', np.nan),
                'cv_test_gap': metrics.get('cv_roc_auc_mean', np.nan) - metrics.get('roc_auc', np.nan)
                if not np.isnan(metrics.get('cv_roc_auc_mean', np.nan)) else np.nan,
            })

    df = pd.DataFrame(rows)

    # Pivot for easier comparison
    pivot_roc = df.pivot_table(index='model', columns='variant', values='test_roc_auc')
    pivot_cv = df.pivot_table(index='model', columns='variant', values='cv_roc_auc')

    return df, pivot_roc, pivot_cv


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run all REM detection experiments')
    parser.add_argument('--skip-create', action='store_true',
                        help='Skip creating data splits (use existing)')
    parser.add_argument('--skip-train', action='store_true',
                        help='Skip training (only generate report from existing results)')
    parser.add_argument('--variants', nargs='+', default=None,
                        help='Specific variants to run (default: key variants)')
    args = parser.parse_args()

    # Key variants to test (includes Phase 2 circadian variants)
    default_variants = [
        'baseline',
        'normalized_zscore',
        'normalized_differential',
        'normalized_ratios',
        'full',
        'normalized_differential_circadian',
        'full_circadian',
    ]
    variants = args.variants or default_variants

    print("=" * 60)
    print("🧪 REM Detection - Full Experiment Pipeline")
    print("=" * 60)
    print(f"\nVariants: {variants}")

    # Step 1: Create data splits
    project_root = SCRIPT_DIR.parent.parent.parent  # project-morpheus
    script_path = SCRIPT_DIR.relative_to(project_root)  # components/reader/apple_watch

    if not args.skip_create:
        success = run_cmd(
            ["uv", "run", str(script_path / "create_data_split_variants.py"), "--variants"] + variants,
            "Creating data splits for all variants"
        )
        if not success:
            print("❌ Failed to create data splits")
            return 1
    else:
        print("\n⏭️  Skipping data split creation")

    # Step 2: Train each variant
    if not args.skip_train:
        for variant in variants:
            success = run_cmd(
                ["uv", "run", str(script_path / "train_with_group_cv.py"), "--variant", variant],
                f"Training models on variant: {variant}"
            )
            if not success:
                print(f"⚠️  Training failed for {variant}")

    # Step 3: Collect results and generate report
    print("\n" + "=" * 60)
    print("📊 Collecting Results & Generating Report")
    print("=" * 60)

    results = collect_results()
    if not results:
        results = load_latest_results()

    if not results:
        print("❌ No results found. Run training first.")
        return 1

    df, pivot_roc, pivot_cv = generate_report(results)

    print("\n📈 Test ROC AUC by Variant & Model:")
    print(pivot_roc.to_string())

    print("\n📈 GroupKFold CV ROC AUC by Variant & Model:")
    print(pivot_cv.to_string())

    # Best per variant (drop variants with all-NaN roc_auc to avoid idxmax error)
    print("\n🏆 Best Model per Variant:")
    df_valid = df.dropna(subset=['test_roc_auc'])
    best_per_variant = pd.DataFrame()
    if df_valid.empty:
        print("   (No valid results with roc_auc)")
    else:
        best_per_variant = df_valid.loc[df_valid.groupby('variant')['test_roc_auc'].idxmax()]
        for _, row in best_per_variant.iterrows():
            gap = row['cv_test_gap']
            gap_str = f" (CV-Test gap: {gap:.3f})" if not np.isnan(gap) else ""
            print(f"   {row['variant']}: {row['model']} ROC AUC={row['test_roc_auc']:.3f}{gap_str}")

    # Improvement over baseline
    if 'baseline' in results:
        baseline_best = df[df['variant'] == 'baseline']['test_roc_auc'].max()
        print(f"\n📊 Improvement over baseline (best baseline ROC AUC: {baseline_best:.3f}):")
        for variant in variants:
            if variant == 'baseline':
                continue
            variant_best = df[df['variant'] == variant]['test_roc_auc'].max()
            delta = variant_best - baseline_best
            print(f"   {variant}: {delta:+.3f} ({variant_best:.3f})")

    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = RESULTS_DIR / f"experiment_report_{timestamp}.csv"
    df.to_csv(report_path, index=False)
    print(f"\n💾 Saved report to {report_path}")

    report_json = RESULTS_DIR / f"experiment_report_{timestamp}.json"
    summary = best_per_variant.fillna('').to_dict('records') if not best_per_variant.empty else []
    with open(report_json, 'w') as f:
        json.dump({
            'summary': summary,
            'pivot_roc_auc': pivot_roc.fillna(0).to_dict(),
            'pivot_cv_auc': pivot_cv.fillna(0).to_dict(),
        }, f, indent=2)
    print(f"💾 Saved JSON report to {report_json}")

    print("\n✨ Experiment pipeline complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
