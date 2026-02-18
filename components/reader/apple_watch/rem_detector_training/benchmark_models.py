#!/usr/bin/env python3
"""
Benchmark a large suite of basic ML models on CPU.
Uses "full" feature split, GroupKFold CV, optimizes for recall.
All from sklearn + xgboost - no extra dependencies.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import roc_auc_score, recall_score, precision_score
import sys
import warnings
from tqdm.auto import tqdm

warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

sys.path.append(str(Path(__file__).parent))
from create_data_split_variants import load_splits
from config import RANDOM_SEED, set_seeds

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


def get_models():
    """Suite of basic ML classifiers - tree-based need no scaling, others get StandardScaler"""
    models = []

    # Tree-based (no scaling)
    models.append(("Random Forest", RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_leaf=48,
        class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1
    )))
    models.append(("Extra Trees", ExtraTreesClassifier(
        n_estimators=100, max_depth=10, min_samples_leaf=48,
        class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1
    )))
    models.append(("Gradient Boosting", GradientBoostingClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        random_state=RANDOM_SEED
    )))
    models.append(("AdaBoost", AdaBoostClassifier(
        n_estimators=50, learning_rate=0.5, random_state=RANDOM_SEED
    )))
    models.append(("Bagging (RF)", BaggingClassifier(
        estimator=RandomForestClassifier(n_estimators=50, max_depth=8, random_state=RANDOM_SEED),
        n_estimators=10, random_state=RANDOM_SEED, n_jobs=-1
    )))

    if HAS_XGBOOST:
        models.append(("XGBoost", xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            scale_pos_weight=4, random_state=RANDOM_SEED, n_jobs=-1
        )))

    # Need scaling
    models.append(("Logistic Regression", Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_SEED))
    ])))
    models.append(("SVC (linear)", Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='linear', class_weight='balanced', probability=True, random_state=RANDOM_SEED))
    ])))
    models.append(("SVC (rbf)", Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=RANDOM_SEED))
    ])))
    models.append(("K-Neighbors (k=5)", Pipeline([
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier(n_neighbors=5, n_jobs=-1))
    ])))
    models.append(("K-Neighbors (k=15)", Pipeline([
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier(n_neighbors=15, n_jobs=-1))
    ])))
    models.append(("MLP", Pipeline([
        ('scaler', StandardScaler()),
        ('clf', MLPClassifier(
            hidden_layer_sizes=(64, 32), alpha=0.01, max_iter=200,
            early_stopping=True, validation_fraction=0.1, random_state=RANDOM_SEED
        ))
    ])))

    # Naive Bayes (scale for GaussianNB, BernoulliNB can work without but often benefits)
    models.append(("Gaussian NB", Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GaussianNB())
    ])))
    models.append(("Bernoulli NB", Pipeline([
        ('scaler', StandardScaler()),
        ('clf', BernoulliNB())
    ])))

    return models


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', default='full', help='Feature variant')
    args = parser.parse_args()

    set_seeds()

    print("=" * 60)
    print("🔬 Model Benchmark - Recall on Held-out Test")
    print("=" * 60)
    print(f"\nVariant: {args.variant}")

    try:
        X_train, X_test, y_train, y_test, split_info, train_subj, _ = load_splits(args.variant)
    except FileNotFoundError:
        print(f"   ❌ No splits for {args.variant}. Run create_data_split_variants.py first.")
        return

    print(f"\n📊 Data: train={len(X_train)}, test={len(X_test)}")
    models = get_models()
    print(f"   Models: {len(models)}")

    gkf = GroupKFold(n_splits=5)
    results = []

    for name, model in tqdm(models, desc="Benchmarking"):
        try:
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=gkf, groups=train_subj,
                scoring='recall', n_jobs=-1
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred.astype(float)

            rec = recall_score(y_test, y_pred, zero_division=0)
            prec = precision_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.5

            results.append({
                'model': name,
                'cv_recall': cv_scores.mean(),
                'cv_recall_std': cv_scores.std(),
                'test_recall': rec,
                'test_precision': prec,
                'test_roc_auc': auc,
            })
        except Exception as e:
            results.append({'model': name, 'error': str(e)})
            print(f"\n   ⚠️  {name}: {e}")

    df = pd.DataFrame([r for r in results if 'error' not in r])
    df = df.sort_values('test_recall', ascending=False)

    print("\n" + "=" * 60)
    print("📊 RESULTS (sorted by test recall)")
    print("=" * 60)
    print(df.to_string(index=False))

    best = df.iloc[0]
    print(f"\n🏆 Best recall: {best['model']} (Test Recall: {best['test_recall']:.3f})")

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"benchmark_{args.variant}_{timestamp}.csv"
    df.to_csv(out_path, index=False)
    print(f"\n💾 Saved to {out_path}")

    print("\n✨ Benchmark complete!")


if __name__ == '__main__':
    main()
