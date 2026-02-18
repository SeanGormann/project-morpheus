#!/usr/bin/env python3
"""
Multi-Model Training Pipeline for REM Detection
Trains and compares: Random Forest, XGBoost, MLP, Logistic Regression
Uses standardized train/val/test splits
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle
import json
from datetime import datetime
import sys

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️  XGBoost not installed. Install with: uv add xgboost")

# Import data loading utility
sys.path.append(str(Path(__file__).parent))
from create_data_split import load_splits

# Paths
OUTPUT_DIR = Path("/Users/seangorman/code-projects/project-morpheus/components/reader/apple_watch/models")
RESULTS_DIR = Path("/Users/seangorman/code-projects/project-morpheus/components/reader/apple_watch/results")
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

def get_models():
    """Define all models to train"""
    models = {}
    
    # 1. Random Forest (baseline from paper)
    models['Random Forest'] = RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=48,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    # 2. XGBoost (if available)
    if HAS_XGBOOST:
        models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=4  # Handle imbalance (80/20 split)
        )
    
    # 3. MLP (small neural network)
    models['MLP'] = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=256,
        learning_rate_init=0.001,
        max_iter=100,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    # 4. Logistic Regression (simple baseline)
    models['Logistic Regression'] = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    
    return models

def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name, use_scaling=False):
    """Train a single model and evaluate"""
    print(f"\n🔧 Training {model_name}...")
    
    # Scale features for neural network
    scaler = None
    if use_scaling:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
    
    # Train
    start_time = datetime.now()
    model.fit(X_train_scaled, y_train)
    train_time = (datetime.now() - start_time).total_seconds()
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'roc_auc': roc_auc_score(y_test, y_proba),
        'train_time_seconds': train_time,
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test)
    }
    
    # Cross-validation score on training data
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    metrics['cv_roc_auc_mean'] = cv_scores.mean()
    metrics['cv_roc_auc_std'] = cv_scores.std()
    
    print(f"   ✅ {model_name} trained in {train_time:.1f}s")
    print(f"   📊 Test ROC AUC: {metrics['roc_auc']:.3f}")
    print(f"   📊 Test Recall: {metrics['recall']:.3f}")
    print(f"   📊 CV ROC AUC: {metrics['cv_roc_auc_mean']:.3f} ± {metrics['cv_roc_auc_std']:.3f}")
    
    # Save model
    model_slug = model_name.lower().replace(' ', '_')
    model_path = OUTPUT_DIR / f"rem_detector_{model_slug}.pkl"
    
    save_data = {
        'model': model,
        'scaler': scaler,
        'metrics': metrics,
        'feature_names': ['hr_feature', 'activity_feature', 'time_hours']
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"   💾 Saved to {model_path}")
    
    return metrics, y_pred, y_proba

def create_comparison_table(all_metrics):
    """Create a comparison table of all models"""
    df = pd.DataFrame(all_metrics).T
    
    # Sort by ROC AUC
    df = df.sort_values('roc_auc', ascending=False)
    
    # Format for display
    display_cols = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'specificity', 'train_time_seconds']
    df_display = df[display_cols].copy()
    
    # Round for readability
    for col in df_display.columns:
        if col != 'train_time_seconds':
            df_display[col] = df_display[col].apply(lambda x: f"{x:.3f}")
        else:
            df_display[col] = df_display[col].apply(lambda x: f"{x:.1f}s")
    
    return df_display

def main():
    """Main training pipeline"""
    print("="*60)
    print("🚀 REM Detection - Multi-Model Training Pipeline")
    print("="*60)
    
    # Load standardized splits
    print("\n📂 Loading standardized data splits...")
    try:
        X_train, X_test, y_train, y_test, split_info = load_splits()
        print(f"   ✅ Loaded splits (seed={split_info['random_seed']})")
    except FileNotFoundError:
        print("   ❌ No standardized splits found!")
        print("   Run: uv run components/reader/apple_watch/create_data_split.py")
        return
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Training: {len(X_train)} samples ({y_train.sum()} REM, {100*y_train.mean():.1f}%)")
    print(f"   Test: {len(X_test)} samples ({y_test.sum()} REM, {100*y_test.mean():.1f}%)")
    
    # Get all models
    models = get_models()
    print(f"\n🤖 Training {len(models)} models:")
    for name in models.keys():
        print(f"   - {name}")
    
    # Train all models
    all_metrics = {}
    all_predictions = {}
    
    for name, model in models.items():
        # Use scaling for MLP and Logistic Regression
        use_scaling = name in ['MLP', 'Logistic Regression']
        
        metrics, y_pred, y_proba = train_and_evaluate(
            model, X_train, y_train, X_test, y_test, 
            name, use_scaling=use_scaling
        )
        
        all_metrics[name] = metrics
        all_predictions[name] = {'y_pred': y_pred, 'y_proba': y_proba}
    
    # Create comparison table
    print("\n" + "="*60)
    print("📊 MODEL COMPARISON")
    print("="*60)
    
    comparison_df = create_comparison_table(all_metrics)
    print("\n" + comparison_df.to_string())
    
    # Save comparison
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed metrics
    metrics_path = RESULTS_DIR / f"model_comparison_{timestamp}.json"
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n💾 Saved detailed metrics to {metrics_path}")
    
    # Save comparison table
    table_path = RESULTS_DIR / f"model_comparison_{timestamp}.csv"
    comparison_df.to_csv(table_path)
    print(f"💾 Saved comparison table to {table_path}")
    
    # Identify best model
    best_model = max(all_metrics.items(), key=lambda x: x[1]['roc_auc'])
    print(f"\n🏆 Best Model: {best_model[0]}")
    print(f"   ROC AUC: {best_model[1]['roc_auc']:.3f}")
    print(f"   Recall: {best_model[1]['recall']:.3f}")
    
    print("\n✨ Training complete!")
    print(f"📁 Models saved in: {OUTPUT_DIR}")
    print(f"📁 Results saved in: {RESULTS_DIR}")

if __name__ == '__main__':
    main()