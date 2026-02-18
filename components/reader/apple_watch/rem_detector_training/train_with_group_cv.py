#!/usr/bin/env python3
"""
Train REM Detection Models with GroupKFold CV
- Subject-level cross-validation (honest estimates)
- Supports multiple feature variants
- Trains all models: RF, XGBoost, MLP, Logistic Regression, Gaussian NB
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import pickle
import json
from datetime import datetime
import sys
import warnings

# Suppress FutureWarning for n_jobs
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

sys.path.append(str(Path(__file__).parent))
from create_data_split_variants import load_splits
from config import RANDOM_SEED, set_seeds, XGBOOST_PARAMS, RANDOM_FOREST_PARAMS, MLP_PARAMS, LOGISTIC_REGRESSION_PARAMS

OUTPUT_DIR = Path("/Users/seangorman/code-projects/project-morpheus/components/reader/apple_watch/models")
RESULTS_DIR = Path("/Users/seangorman/code-projects/project-morpheus/components/reader/apple_watch/results")
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


def get_models():
    """Define all models - uses shared config for consistency"""
    models = {}

    models['Random Forest'] = RandomForestClassifier(**RANDOM_FOREST_PARAMS)

    if HAS_XGBOOST:
        models['XGBoost'] = xgb.XGBClassifier(**XGBOOST_PARAMS)

    models['MLP'] = MLPClassifier(**MLP_PARAMS)

    models['Logistic Regression'] = LogisticRegression(**LOGISTIC_REGRESSION_PARAMS)

    models['Gaussian NB'] = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GaussianNB())
    ])

    return models


def train_and_evaluate(
    model,
    X_train, y_train,
    X_test, y_test,
    train_subject_indices,
    model_name,
    feature_names,
    variant,
    use_scaling=False
):
    """Train and evaluate with GroupKFold CV"""
    print(f"\n🔧 Training {model_name}...")

    scaler = None
    if use_scaling:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    start_time = datetime.now()
    model.fit(X_train_scaled, y_train)
    train_time = (datetime.now() - start_time).total_seconds()

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Handle edge case: single class in predictions
    tn, fp, fn, tp = 0, 0, 0, 0
    try:
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    except ValueError:
        cm = confusion_matrix(y_test, y_pred)
        if cm.shape == (1, 1):
            tn, fp, fn, tp = cm[0, 0], 0, 0, 0 if y_test[0] == 0 else 0

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'roc_auc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.5,
        'train_time_seconds': train_time,
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test)
    }

    # GroupKFold CV - honest subject-level cross-validation
    if train_subject_indices is not None and len(np.unique(train_subject_indices)) >= 5:
        gkf = GroupKFold(n_splits=5)  # shuffle=False by default for reproducibility
        cv_scores = cross_val_score(
            model, X_train_scaled, y_train,
            groups=train_subject_indices,
            cv=gkf,
            scoring='roc_auc',
            n_jobs=-1
        )
        metrics['cv_roc_auc_mean'] = cv_scores.mean()
        metrics['cv_roc_auc_std'] = cv_scores.std()
    else:
        metrics['cv_roc_auc_mean'] = np.nan
        metrics['cv_roc_auc_std'] = np.nan

    print(f"   ✅ {model_name} trained in {train_time:.1f}s")
    print(f"   📊 Test ROC AUC: {metrics['roc_auc']:.3f}")
    print(f"   📊 Test Recall: {metrics['recall']:.3f}")
    if not np.isnan(metrics['cv_roc_auc_mean']):
        print(f"   📊 GroupKFold CV ROC AUC: {metrics['cv_roc_auc_mean']:.3f} ± {metrics['cv_roc_auc_std']:.3f}")

    model_slug = model_name.lower().replace(' ', '_')
    variant_dir = OUTPUT_DIR / variant
    variant_dir.mkdir(exist_ok=True)
    model_path = variant_dir / f"rem_detector_{model_slug}.pkl"

    save_data = {
        'model': model,
        'scaler': scaler,
        'metrics': metrics,
        'feature_names': feature_names,
        'variant': variant,
    }

    with open(model_path, 'wb') as f:
        pickle.dump(save_data, f)

    print(f"   💾 Saved to {model_path}")

    return metrics


def main():
    set_seeds()
    import argparse
    parser = argparse.ArgumentParser(description='Train models with GroupKFold CV')
    parser.add_argument('--variant', default='baseline',
                        help='Feature variant to train on')
    parser.add_argument('--scale-all', action='store_true',
                        help='Use StandardScaler for all models (not just MLP/LogReg)')
    args = parser.parse_args()

    variant = args.variant

    print("=" * 60)
    print("🚀 REM Detection - Training with GroupKFold CV")
    print("=" * 60)
    print(f"\n📋 Feature variant: {variant}")

    try:
        X_train, X_test, y_train, y_test, split_info, train_subj_idx, test_subj_idx = load_splits(variant)
    except FileNotFoundError as e:
        print(f"   ❌ {e}")
        print("   Run: uv run components/reader/apple_watch/create_data_split_variants.py")
        return

    print(f"\n📊 Dataset Summary:")
    print(f"   Training: {len(X_train)} samples ({y_train.sum()} REM, {100*y_train.mean():.1f}%)")
    print(f"   Test: {len(X_test)} samples ({y_test.sum()} REM, {100*y_test.mean():.1f}%)")
    print(f"   Features: {split_info['feature_names']}")

    models = get_models()
    all_metrics = {}

    for name, model in models.items():
        use_scaling = name in ['MLP', 'Logistic Regression'] or args.scale_all

        metrics = train_and_evaluate(
            model, X_train, y_train, X_test, y_test,
            train_subj_idx, name, split_info['feature_names'], variant,
            use_scaling=use_scaling
        )
        all_metrics[name] = metrics

    print("\n" + "=" * 60)
    print("📊 MODEL COMPARISON")
    print("=" * 60)

    df = pd.DataFrame(all_metrics).T
    df = df.sort_values('roc_auc', ascending=False)
    display_cols = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'specificity', 'cv_roc_auc_mean', 'train_time_seconds']
    display_cols = [c for c in display_cols if c in df.columns]
    print("\n" + df[display_cols].to_string())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = RESULTS_DIR / f"train_{variant}_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump({k: {kk: (float(vv) if isinstance(vv, (np.floating, float)) else vv)
                       for kk, vv in v.items()} for k, v in all_metrics.items()}, f, indent=2)
    print(f"\n💾 Saved results to {results_path}")

    best = max(all_metrics.items(), key=lambda x: x[1]['roc_auc'])
    print(f"\n🏆 Best Model: {best[0]} (ROC AUC: {best[1]['roc_auc']:.3f})")


if __name__ == '__main__':
    main()
