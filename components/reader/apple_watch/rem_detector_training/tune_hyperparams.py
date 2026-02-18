#!/usr/bin/env python3
"""
Hyperparameter Tuning for MLP, RF & Logistic Regression
Optimizes for RECALL (lucid dreaming: catch REM periods).
Uses Phase 1 setup: same splits, GroupKFold CV, evaluates on held-out test.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, ParameterGrid, cross_val_score
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, recall_score, precision_score
import pickle
import json
from datetime import datetime
import sys
import warnings
from tqdm.auto import tqdm

warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')

sys.path.append(str(Path(__file__).parent))
from create_data_split_variants import load_splits
from config import RANDOM_SEED, set_seeds

OUTPUT_DIR = Path(__file__).parent / "models"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


# Param grids - MLP: small (max 64), strong regularization for generalization
MLP_PARAM_GRID = {
    'mlp__hidden_layer_sizes': [(64, 32), (64, 16), (48, 24), (32, 16), (32, 8), (48, 16), (16, 8)],
    'mlp__alpha': [0.01, 0.1, 0.5, 1.0],  # Strong L2 reg (sklearn has no dropout, alpha acts similarly)
    'mlp__learning_rate_init': [0.0005, 0.001],
    'mlp__max_iter': [200, 300],
}

RF_PARAM_GRID = {
    'n_estimators': [100, 200],
    'max_depth': [8, 10, 12],
    'min_samples_leaf': [32, 48, 64],
    'max_features': ['sqrt', 'log2'],
}

# Logistic Regression - C, l1_ratio (0=L2, 1=L1; penalty deprecated in sklearn 1.8+), solver
LOGREG_PARAM_GRID = {
    'logreg__C': [0.01, 0.1, 0.5, 1.0, 10.0],
    'logreg__l1_ratio': [0.0, 1.0],  # 0=L2, 1=L1 (replaces deprecated penalty)
    'logreg__solver': ['lbfgs', 'saga'],
    'logreg__class_weight': ['balanced', None],
    'logreg__max_iter': [1000, 2000],
}


def _valid_logreg_params(params):
    """lbfgs only supports L2 (l1_ratio=0)"""
    return not (params['logreg__l1_ratio'] == 1.0 and params['logreg__solver'] == 'lbfgs')


def tune_mlp(X_train, y_train, train_subject_indices):
    """Grid search MLP with GroupKFold (MLP needs scaling)"""
    print("\n🔍 Tuning MLP...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            activation='relu',
            solver='adam',
            batch_size=256,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=RANDOM_SEED,
        ))
    ])

    gkf = GroupKFold(n_splits=5)
    best_score, best_params, best_estimator = -1, None, None

    for params in tqdm(ParameterGrid(MLP_PARAM_GRID), desc="MLP param combinations"):
        pipeline.set_params(**params)
        scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=gkf, groups=train_subject_indices,
            scoring='recall', n_jobs=-1
        )
        mean_score = scores.mean()
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    # Refit best on full train (fresh clone)
    best_estimator = clone(pipeline).set_params(**best_params)
    best_estimator.fit(X_train, y_train)
    return type('GridResult', (), {'best_params_': best_params, 'best_score_': best_score, 'best_estimator_': best_estimator})()


def tune_rf(X_train, y_train, train_subject_indices):
    """Grid search Random Forest with GroupKFold"""
    print("\n🔍 Tuning Random Forest...")
    model = RandomForestClassifier(
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight='balanced',
    )

    gkf = GroupKFold(n_splits=5)
    best_score, best_params, best_estimator = -1, None, None

    for params in tqdm(ParameterGrid(RF_PARAM_GRID), desc="RF param combinations"):
        model.set_params(**params)
        scores = cross_val_score(
            model, X_train, y_train,
            cv=gkf, groups=train_subject_indices,
            scoring='recall', n_jobs=-1
        )
        mean_score = scores.mean()
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    # Refit best on full train (fresh model with best params)
    best_model = RandomForestClassifier(
        random_state=RANDOM_SEED, n_jobs=-1, class_weight='balanced', **best_params
    )
    best_model.fit(X_train, y_train)
    return type('GridResult', (), {'best_params_': best_params, 'best_score_': best_score, 'best_estimator_': best_model})()


def tune_logreg(X_train, y_train, train_subject_indices):
    """Grid search Logistic Regression with GroupKFold (needs scaling)"""
    print("\n🔍 Tuning Logistic Regression...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(random_state=RANDOM_SEED))
    ])

    gkf = GroupKFold(n_splits=5)
    best_score, best_params = -1, None

    param_list = [p for p in ParameterGrid(LOGREG_PARAM_GRID) if _valid_logreg_params(p)]
    for params in tqdm(param_list, desc="LogReg param combinations"):
        pipeline.set_params(**params)
        scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=gkf, groups=train_subject_indices,
            scoring='recall', n_jobs=-1
        )
        mean_score = scores.mean()
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    best_estimator = clone(pipeline).set_params(**best_params)
    best_estimator.fit(X_train, y_train)
    return type('GridResult', (), {'best_params_': best_params, 'best_score_': best_score, 'best_estimator_': best_estimator})()


def evaluate_on_test(model, X_train, y_train, X_test, y_test):
    """Train on full train set, evaluate on held-out test (model handles scaling if Pipeline)"""
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    return {
        'roc_auc': roc_auc_score(y_test, y_proba),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'precision': precision_score(y_test, y_pred, zero_division=0),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', default='full',
                        help='Feature variant (default: full, best from Phase 1)')
    parser.add_argument('--model', choices=['mlp', 'rf', 'logreg', 'all'], default='all',
                        help='Which model(s) to tune (all = mlp, rf, logreg)')
    args = parser.parse_args()

    set_seeds()

    print("=" * 60)
    print("🔬 Hyperparameter Tuning - MLP, RF & Logistic Regression")
    print("=" * 60)
    print(f"\nVariant: {args.variant}")

    try:
        X_train, X_test, y_train, y_test, split_info, train_subj, test_subj = load_splits(args.variant)
    except FileNotFoundError:
        print(f"   ❌ No splits for {args.variant}. Run create_data_split_variants.py first.")
        return

    print(f"\n📊 Data: train={len(X_train)}, test={len(X_test)}")
    print(f"   Train subjects: {len(np.unique(train_subj))}")

    results = {'variant': args.variant, 'tuned_models': {}}
    models_to_run = ['mlp', 'rf', 'logreg'] if args.model == 'all' else [args.model]

    if 'mlp' in models_to_run:
        grid_mlp = tune_mlp(X_train, y_train, train_subj)
        print(f"\n   Best MLP params: {grid_mlp.best_params_}")
        print(f"   Best CV Recall: {grid_mlp.best_score_:.3f}")

        test_metrics = evaluate_on_test(
            grid_mlp.best_estimator_, X_train, y_train, X_test, y_test
        )
        print(f"   Test ROC AUC: {test_metrics['roc_auc']:.3f}")
        print(f"   Test Recall: {test_metrics['recall']:.3f}")

        best_params = {k: list(v) if isinstance(v, tuple) else v for k, v in grid_mlp.best_params_.items()}
        results['tuned_models']['MLP'] = {
            'best_params': best_params,
            'cv_recall': float(grid_mlp.best_score_),
            'test_roc_auc': test_metrics['roc_auc'],
            'test_recall': test_metrics['recall'],
            'test_precision': test_metrics['precision'],
        }

        # Save best MLP
        out_dir = OUTPUT_DIR / f"{args.variant}_tuned"
        out_dir.mkdir(exist_ok=True)
        with open(out_dir / "rem_detector_mlp_tuned.pkl", 'wb') as f:
            pickle.dump({
                'model': grid_mlp.best_estimator_,
                'metrics': results['tuned_models']['MLP'],
                'variant': args.variant,
                'feature_names': split_info['feature_names'],
            }, f)

    if 'rf' in models_to_run:
        grid_rf = tune_rf(X_train, y_train, train_subj)
        print(f"\n   Best RF params: {grid_rf.best_params_}")
        print(f"   Best CV Recall: {grid_rf.best_score_:.3f}")

        test_metrics = evaluate_on_test(
            grid_rf.best_estimator_, X_train, y_train, X_test, y_test
        )
        print(f"   Test ROC AUC: {test_metrics['roc_auc']:.3f}")
        print(f"   Test Recall: {test_metrics['recall']:.3f}")

        best_params = {k: list(v) if isinstance(v, tuple) else v for k, v in grid_rf.best_params_.items()}
        results['tuned_models']['Random Forest'] = {
            'best_params': best_params,
            'cv_recall': float(grid_rf.best_score_),
            'test_roc_auc': test_metrics['roc_auc'],
            'test_recall': test_metrics['recall'],
            'test_precision': test_metrics['precision'],
        }

        out_dir = OUTPUT_DIR / f"{args.variant}_tuned"
        out_dir.mkdir(exist_ok=True)
        with open(out_dir / "rem_detector_random_forest_tuned.pkl", 'wb') as f:
            pickle.dump({
                'model': grid_rf.best_estimator_,
                'metrics': results['tuned_models']['Random Forest'],
                'variant': args.variant,
                'feature_names': split_info['feature_names'],
            }, f)

    if 'logreg' in models_to_run:
        grid_logreg = tune_logreg(X_train, y_train, train_subj)
        print(f"\n   Best LogReg params: {grid_logreg.best_params_}")
        print(f"   Best CV Recall: {grid_logreg.best_score_:.3f}")

        test_metrics = evaluate_on_test(
            grid_logreg.best_estimator_, X_train, y_train, X_test, y_test
        )
        print(f"   Test ROC AUC: {test_metrics['roc_auc']:.3f}")
        print(f"   Test Recall: {test_metrics['recall']:.3f}")

        best_params = {k: list(v) if isinstance(v, tuple) else v for k, v in grid_logreg.best_params_.items()}
        results['tuned_models']['Logistic Regression'] = {
            'best_params': best_params,
            'cv_recall': float(grid_logreg.best_score_),
            'test_roc_auc': test_metrics['roc_auc'],
            'test_recall': test_metrics['recall'],
            'test_precision': test_metrics['precision'],
        }

        out_dir = OUTPUT_DIR / f"{args.variant}_tuned"
        out_dir.mkdir(exist_ok=True)
        with open(out_dir / "rem_detector_logistic_regression_tuned.pkl", 'wb') as f:
            pickle.dump({
                'model': grid_logreg.best_estimator_,
                'metrics': results['tuned_models']['Logistic Regression'],
                'variant': args.variant,
                'feature_names': split_info['feature_names'],
            }, f)

    # Summary
    if results['tuned_models']:
        best = max(results['tuned_models'].items(), key=lambda x: x[1]['test_recall'])
        print(f"\n🏆 Best recall: {best[0]} (Test Recall: {best[1]['test_recall']:.3f}, ROC AUC: {best[1]['test_roc_auc']:.3f})")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(RESULTS_DIR / f"tune_{args.variant}_{timestamp}.json", 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Saved results to {RESULTS_DIR}")

    print("\n✨ Tuning complete!")


if __name__ == '__main__':
    main()
