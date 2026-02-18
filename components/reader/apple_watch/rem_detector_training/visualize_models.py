#!/usr/bin/env python3
"""
Visualize REM Detection Model Performance
Uses standardized train/val/test splits
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, auc, precision_recall_curve,
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
)
import pickle
import json
import sys

# Import data loading utility
sys.path.append(str(Path(__file__).parent))
from create_data_split import load_splits

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

MODEL_DIR = Path("/Users/seangorman/code-projects/project-morpheus/components/reader/apple_watch/models")
OUTPUT_DIR = Path("/Users/seangorman/code-projects/project-morpheus/components/reader/apple_watch/visualizations")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def plot_confusion_matrix(y_true, y_pred, model_name="Random Forest"):
    """Plot confusion matrix"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Not REM', 'REM']
    )
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    
    # Add percentages
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            pct = 100 * cm[i, j] / total
            ax.text(j, i + 0.3, f'({pct:.1f}%)', 
                   ha='center', va='center', fontsize=9, color='gray')
    
    plt.tight_layout()
    return fig

def plot_roc_curve(y_true, y_proba, model_name="Random Forest"):
    """Plot ROC curve"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    # Mark the optimal threshold (closest to top-left)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=10,
            label=f'Optimal (thresh={optimal_threshold:.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
    ax.set_title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_precision_recall_curve(y_true, y_proba, model_name="Random Forest"):
    """Plot Precision-Recall curve"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    
    ax.plot(recall, precision, color='darkgreen', lw=2,
            label=f'PR curve (AUC = {pr_auc:.3f})')
    
    # Mark threshold = 0.24 (from TLR paper)
    idx_24 = np.argmin(np.abs(thresholds - 0.24))
    if idx_24 < len(precision) and idx_24 < len(recall):
        ax.plot(recall[idx_24], precision[idx_24], 'ro', markersize=10,
                label=f'TLR threshold (0.24)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'Precision-Recall Curve - {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_feature_importance(model, feature_names, model_name="Random Forest"):
    """Plot feature importance (for tree-based models)"""
    if not hasattr(model, 'feature_importances_'):
        return None
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_names)))
    
    ax.bar(range(len(feature_names)), importances[indices], color=colors)
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
    ax.set_ylabel('Importance', fontsize=12)
    ax.set_title(f'Feature Importance - {model_name}', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add percentage labels on bars
    for i, v in enumerate(importances[indices]):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig

def plot_threshold_analysis(y_true, y_proba, model_name="Random Forest"):
    """Plot how metrics change with threshold"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    thresholds = np.linspace(0, 1, 100)
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics['accuracy'].append(accuracy)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1'].append(f1)
    
    # Plot all metrics
    axes[0, 0].plot(thresholds, metrics['accuracy'], 'b-', lw=2)
    axes[0, 0].axvline(x=0.24, color='r', linestyle='--', label='TLR threshold (0.24)')
    axes[0, 0].set_xlabel('Threshold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Accuracy vs Threshold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    axes[0, 1].plot(thresholds, metrics['precision'], 'g-', lw=2)
    axes[0, 1].axvline(x=0.24, color='r', linestyle='--', label='TLR threshold (0.24)')
    axes[0, 1].set_xlabel('Threshold')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision vs Threshold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    axes[1, 0].plot(thresholds, metrics['recall'], 'orange', lw=2)
    axes[1, 0].axvline(x=0.24, color='r', linestyle='--', label='TLR threshold (0.24)')
    axes[1, 0].set_xlabel('Threshold')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].set_title('Recall vs Threshold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    axes[1, 1].plot(thresholds, metrics['f1'], 'purple', lw=2)
    axes[1, 1].axvline(x=0.24, color='r', linestyle='--', label='TLR threshold (0.24)')
    axes[1, 1].set_xlabel('Threshold')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_title('F1 Score vs Threshold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.suptitle(f'Threshold Analysis - {model_name}', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    return fig

def plot_class_distribution(y_true, model_name="Dataset"):
    """Plot class distribution"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    unique, counts = np.unique(y_true, return_counts=True)
    labels = ['Not REM', 'REM']
    colors = ['#3498db', '#e74c3c']
    
    bars = ax.bar(labels, counts, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title(f'Class Distribution - {model_name}', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add count and percentage labels
    total = counts.sum()
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        pct = 100 * count / total
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_summary_dashboard(model_name, metrics_dict):
    """Create a summary dashboard with key metrics"""
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
    
    # Title
    fig.suptitle(f'{model_name} - Performance Summary', fontsize=16, fontweight='bold')
    
    # Metric boxes
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 'Specificity']
    metric_values = [
        metrics_dict.get('accuracy', 0),
        metrics_dict.get('precision', 0),
        metrics_dict.get('recall', 0),
        metrics_dict.get('f1', 0),
        metrics_dict.get('roc_auc', 0),
        metrics_dict.get('specificity', 0)
    ]
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
    
    for i, (name, value, color) in enumerate(zip(metric_names, metric_values, colors)):
        row = i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])
        
        # Create circular metric display
        ax.text(0.5, 0.5, f'{value:.3f}', 
                ha='center', va='center', fontsize=36, fontweight='bold',
                color=color)
        ax.text(0.5, 0.15, name, 
                ha='center', va='center', fontsize=12)
        
        # Add circle border
        circle = plt.Circle((0.5, 0.5), 0.4, color=color, fill=False, linewidth=3)
        ax.add_patch(circle)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    return fig

def visualize_model(model_path, X_test, y_test, model_name="Model"):
    """Generate all visualizations for a trained model"""
    print(f"\n🎨 Creating visualizations for {model_name}...")
    
    # Load model
    with open(model_path, 'rb') as f:
        save_data = pickle.load(f)
    
    # Handle both old and new format
    if isinstance(save_data, dict):
        model = save_data['model']
        scaler = save_data.get('scaler', None)
        feature_names = save_data.get('feature_names', ['HR Feature', 'Activity Feature', 'Time (hours)'])
    else:
        model = save_data
        scaler = None
        feature_names = ['HR Feature', 'Activity Feature', 'Time (hours)']
    
    # Scale test data if needed
    if scaler is not None:
        X_test_transformed = scaler.transform(X_test)
    else:
        X_test_transformed = X_test
    
    # Make predictions
    y_pred = model.predict(X_test_transformed)
    y_proba = model.predict_proba(X_test_transformed)[:, 1]  # Probability of REM
    
    # Calculate metrics
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    metrics = {
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
        'roc_auc': auc(*roc_curve(y_test, y_proba)[:2])
    }
    
    # Create plots
    plots = [
        (plot_class_distribution(y_test, model_name), 'class_distribution'),
        (plot_confusion_matrix(y_test, y_pred, model_name), 'confusion_matrix'),
        (plot_roc_curve(y_test, y_proba, model_name), 'roc_curve'),
        (plot_precision_recall_curve(y_test, y_proba, model_name), 'pr_curve'),
        (plot_threshold_analysis(y_test, y_proba, model_name), 'threshold_analysis'),
        (create_summary_dashboard(model_name, metrics), 'summary_dashboard')
    ]
    
    # Add feature importance if available
    if hasattr(model, 'feature_importances_'):
        plots.append((
            plot_feature_importance(model, feature_names, model_name),
            'feature_importance'
        ))
    
    # Save all plots
    model_slug = model_name.lower().replace(' ', '_')
    for fig, name in plots:
        if fig is not None:
            filepath = OUTPUT_DIR / f"{model_slug}_{name}.png"
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"✅ Saved {name}")
            plt.close(fig)
    
    # Save metrics to JSON
    metrics_path = OUTPUT_DIR / f"{model_slug}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Saved metrics to {metrics_path}")
    
    print(f"\n📊 {model_name} Metrics:")
    for key, value in metrics.items():
        print(f"   {key.capitalize()}: {value:.3f}")

if __name__ == '__main__':
    print("="*60)
    print("🎨 REM Detection - Model Visualization")
    print("="*60)
    
    # Load test data
    print("\n📂 Loading test data...")
    print("\n📂 Loading test data from standardized splits...")
    try:
        _, X_test, _, y_test, split_info = load_splits()
        print(f"✅ Loaded {len(X_test)} test samples (seed={split_info['random_seed']})")
    except FileNotFoundError:
        print("❌ No standardized splits found!")
        print("   Run: uv run components/reader/apple_watch/create_data_split.py")
        exit(1)
    
    # Find all trained models
    model_files = list(MODEL_DIR.glob("rem_detector_*.pkl"))
    
    if not model_files:
        print(f"\n❌ No models found in {MODEL_DIR}")
        print("   Run train_all_models.py first!")
        exit(1)
    
    print(f"\n🤖 Found {len(model_files)} trained models:")
    for model_file in model_files:
        print(f"   - {model_file.stem}")
    
    # Visualize each model
    for model_file in model_files:
        model_name = model_file.stem.replace('rem_detector_', '').replace('_', ' ').title()
        
        print(f"\n{'='*60}")
        print(f"📊 Visualizing: {model_name}")
        print(f"{'='*60}")
        
        visualize_model(model_file, X_test, y_test, model_name)
    
    print(f"\n✨ All visualizations complete!")
    print(f"📁 Saved to: {OUTPUT_DIR}")