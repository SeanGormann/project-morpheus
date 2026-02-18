#!/usr/bin/env python3
"""
FFT-Based REM Detection Model
Uses frequency-domain features from heart rate and motion signals
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal
from scipy.fft import fft, fftfreq
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, auc
from sklearn.model_selection import StratifiedKFold, cross_val_score
import pickle
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Import data loading and split utilities
import sys
sys.path.append(str(Path(__file__).parent))
from train_rem_detector import load_subject_data
from create_data_split import load_splits

# Paths
DATA_DIR = Path("/Users/seangorman/code-projects/project-morpheus/components/data/tlr_data/walch-apple-watch")
MODEL_DIR = Path("/Users/seangorman/code-projects/project-morpheus/components/reader/apple_watch/models")
VIZ_DIR = Path("/Users/seangorman/code-projects/project-morpheus/components/reader/apple_watch/visualizations")
VIZ_DIR.mkdir(exist_ok=True, parents=True)

RANDOM_SEED = 42

def extract_fft_features_from_signal(signal_data, timestamps, window_size=300, overlap=150):
    """
    Extract frequency-domain features from a time series signal
    
    Args:
        signal_data: Signal values (e.g., heart rate)
        timestamps: Corresponding timestamps
        window_size: Window size in seconds
        overlap: Overlap between windows in seconds
    
    Returns:
        DataFrame with FFT features per window
    """
    features = []
    window_timestamps = []
    
    start_time = timestamps.min()
    end_time = timestamps.max()
    
    current_time = start_time
    step_size = window_size - overlap
    
    while current_time + window_size <= end_time:
        window_end = current_time + window_size
        
        # Get data in this window
        mask = (timestamps >= current_time) & (timestamps < window_end)
        window_signal = signal_data[mask]
        
        if len(window_signal) < 10:  # Need minimum data points
            current_time += step_size
            continue
        
        # Compute FFT
        n = len(window_signal)
        fft_vals = fft(window_signal)
        fft_freqs = fftfreq(n, d=1.0)  # Assuming 1 Hz sampling after interpolation
        
        # Only take positive frequencies
        positive_freqs = fft_freqs[:n//2]
        magnitude = np.abs(fft_vals[:n//2])
        
        # Extract frequency bands
        # Very Low Frequency: 0.0033-0.04 Hz (heart rate variability)
        # Low Frequency: 0.04-0.15 Hz (sympathetic/parasympathetic balance)
        # High Frequency: 0.15-0.4 Hz (respiratory sinus arrhythmia)
        
        vlf_mask = (positive_freqs >= 0.0033) & (positive_freqs < 0.04)
        lf_mask = (positive_freqs >= 0.04) & (positive_freqs < 0.15)
        hf_mask = (positive_freqs >= 0.15) & (positive_freqs < 0.4)
        
        vlf_power = np.sum(magnitude[vlf_mask]**2) if np.any(vlf_mask) else 0
        lf_power = np.sum(magnitude[lf_mask]**2) if np.any(lf_mask) else 0
        hf_power = np.sum(magnitude[hf_mask]**2) if np.any(hf_mask) else 0
        
        total_power = vlf_power + lf_power + hf_power
        
        # Compute features
        feature_dict = {
            'timestamp': current_time,
            'mean': np.mean(window_signal),
            'std': np.std(window_signal),
            'vlf_power': vlf_power,
            'lf_power': lf_power,
            'hf_power': hf_power,
            'total_power': total_power,
            'lf_hf_ratio': lf_power / hf_power if hf_power > 0 else 0,
            'vlf_ratio': vlf_power / total_power if total_power > 0 else 0,
            'lf_ratio': lf_power / total_power if total_power > 0 else 0,
            'hf_ratio': hf_power / total_power if total_power > 0 else 0,
            'dominant_freq': positive_freqs[np.argmax(magnitude)] if len(magnitude) > 0 else 0,
            'spectral_entropy': -np.sum((magnitude**2 / np.sum(magnitude**2)) * np.log(magnitude**2 / np.sum(magnitude**2) + 1e-10)) if len(magnitude) > 0 else 0
        }
        
        features.append(feature_dict)
        current_time += step_size
    
    return pd.DataFrame(features)

def extract_subject_fft_features(subject_id, epoch_seconds=30):
    """Extract FFT features for a subject"""
    hr_data, motion_data, label_data = load_subject_data(subject_id)
    
    # Interpolate heart rate to regular 1 Hz sampling
    start_time = hr_data['timestamp'].min()
    end_time = hr_data['timestamp'].max()
    regular_timestamps = np.arange(start_time, end_time, 1.0)  # 1 Hz
    
    hr_interp = np.interp(regular_timestamps, hr_data['timestamp'].values, hr_data['heart_rate'].values)
    
    # Extract FFT features from heart rate (5-minute windows with 2.5-min overlap)
    hr_fft_features = extract_fft_features_from_signal(hr_interp, regular_timestamps, window_size=300, overlap=150)
    
    # Extract motion magnitude
    motion_data['magnitude'] = np.sqrt(
        motion_data['x']**2 + motion_data['y']**2 + motion_data['z']**2
    )
    
    # Interpolate motion to 1 Hz
    motion_interp = np.interp(regular_timestamps, motion_data['timestamp'].values, motion_data['magnitude'].values)
    
    # Extract FFT features from motion
    motion_fft_features = extract_fft_features_from_signal(motion_interp, regular_timestamps, window_size=300, overlap=150)
    
    # Combine features
    combined_features = hr_fft_features.copy()
    for col in motion_fft_features.columns:
        if col != 'timestamp':
            combined_features[f'motion_{col}'] = motion_fft_features[col].values[:len(combined_features)]
    
    # Add time feature
    combined_features['time_hours'] = (combined_features['timestamp'] - start_time) / 3600
    
    # Align with labels (use majority label in 5-min window)
    labels = []
    for _, row in combined_features.iterrows():
        window_start = row['timestamp']
        window_end = window_start + 300  # 5 minutes
        
        label_mask = (label_data['timestamp'] >= window_start) & (label_data['timestamp'] < window_end)
        window_labels = label_data[label_mask]['stage'].values
        
        if len(window_labels) > 0:
            # Use majority vote
            majority_label = np.bincount(window_labels.astype(int)).argmax()
            labels.append(majority_label)
        else:
            labels.append(0)  # Default to wake
    
    combined_features['label'] = labels
    
    return combined_features

def load_fft_dataset(subject_ids):
    """Load FFT features for multiple subjects"""
    all_features = []
    
    for subject_id in subject_ids:
        try:
            features = extract_subject_fft_features(subject_id)
            all_features.append(features)
        except Exception as e:
            print(f"   ⚠️  Error processing {subject_id}: {e}")
            continue
    
    if len(all_features) == 0:
        raise ValueError("No subjects successfully processed")
    
    combined = pd.concat(all_features, ignore_index=True)
    
    # Separate features and labels
    feature_cols = [col for col in combined.columns if col not in ['timestamp', 'label']]
    X = combined[feature_cols].values
    y = (combined['label'].values == 5).astype(int)  # Binary: REM vs not REM
    
    return X, y, feature_cols

def train_fft_model():
    """Train FFT-based REM detector"""
    print("="*60)
    print("🌊 FFT-Based REM Detection Model")
    print("="*60)
    
    # Load standardized splits
    print("\n📂 Loading data splits...")
    try:
        _, _, _, _, _, _, split_info = load_splits()
        train_subjects = split_info['train_subjects']
        val_subjects = split_info['val_subjects']
        test_subjects = split_info['test_subjects']
        print(f"   ✅ Using standardized splits (seed={split_info['random_seed']})")
    except FileNotFoundError:
        print("   ⚠️  No standardized splits found. Run create_data_split.py first!")
        return
    
    # Extract FFT features
    print(f"\n🔬 Extracting FFT features...")
    print(f"   Training on {len(train_subjects)} subjects...")
    X_train, y_train, feature_names = load_fft_dataset(train_subjects)
    
    print(f"   Testing on {len(test_subjects)} subjects...")
    X_test, y_test, _ = load_fft_dataset(test_subjects)
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Features: {len(feature_names)}")
    print(f"   Train: {len(X_train)} samples ({y_train.sum()} REM, {100*y_train.mean():.1f}%)")
    print(f"   Test: {len(X_test)} samples ({y_test.sum()} REM, {100*y_test.mean():.1f}%)")
    
    # Scale features
    print(f"\n⚙️  Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest on FFT features
    print(f"\n🌲 Training Random Forest on FFT features...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=20,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    start_time = datetime.now()
    model.fit(X_train_scaled, y_train)
    train_time = (datetime.now() - start_time).total_seconds()
    
    print(f"   ✅ Trained in {train_time:.1f}s")
    
    # Evaluate
    print(f"\n📊 Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    metrics = {
        'model_type': 'FFT-based Random Forest',
        'n_features': len(feature_names),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'roc_auc': roc_auc_score(y_test, y_proba),
        'train_time_seconds': train_time,
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'feature_names': feature_names
    }
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    metrics['cv_roc_auc_mean'] = cv_scores.mean()
    metrics['cv_roc_auc_std'] = cv_scores.std()
    
    print(f"\n   📈 Test Metrics:")
    print(f"      Accuracy: {metrics['accuracy']:.3f}")
    print(f"      Precision: {metrics['precision']:.3f}")
    print(f"      Recall: {metrics['recall']:.3f}")
    print(f"      F1 Score: {metrics['f1']:.3f}")
    print(f"      ROC AUC: {metrics['roc_auc']:.3f}")
    print(f"      Specificity: {metrics['specificity']:.3f}")
    print(f"\n   🔄 Cross-Validation:")
    print(f"      CV ROC AUC: {metrics['cv_roc_auc_mean']:.3f} ± {metrics['cv_roc_auc_std']:.3f}")
    
    # Save model
    model_data = {
        'model': model,
        'scaler': scaler,
        'metrics': metrics,
        'feature_names': feature_names
    }
    
    model_path = MODEL_DIR / "rem_detector_fft.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"\n💾 Model saved to {model_path}")
    
    # Create visualizations
    create_fft_visualizations(model, scaler, X_test_scaled, y_test, y_pred, y_proba, metrics, feature_names)
    
    return model, metrics

def create_fft_visualizations(model, scaler, X_test, y_test, y_pred, y_proba, metrics, feature_names):
    """Create comprehensive visualizations for FFT model"""
    print(f"\n🎨 Creating visualizations...")
    
    sns.set_style("whitegrid")
    
    # 1. Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Not REM', 'REM'],
                yticklabels=['Not REM', 'REM'])
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix - FFT Model', fontweight='bold', fontsize=14)
    plt.tight_layout()
    fig.savefig(VIZ_DIR / 'fft_confusion_matrix.png', dpi=150)
    plt.close(fig)
    print("   ✅ Confusion matrix")
    
    # 2. ROC Curve
    fig, ax = plt.subplots(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve - FFT Model', fontweight='bold', fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(VIZ_DIR / 'fft_roc_curve.png', dpi=150)
    plt.close(fig)
    print("   ✅ ROC curve")
    
    # 3. Feature Importance (top 20)
    fig, ax = plt.subplots(figsize=(10, 8))
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]  # Top 20
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, 20))
    ax.barh(range(20), importances[indices], color=colors)
    ax.set_yticks(range(20))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=9)
    ax.set_xlabel('Importance')
    ax.set_title('Top 20 Feature Importances - FFT Model', fontweight='bold', fontsize=14)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    fig.savefig(VIZ_DIR / 'fft_feature_importance.png', dpi=150)
    plt.close(fig)
    print("   ✅ Feature importance")
    
    # 4. Metrics Summary
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle('FFT Model - Performance Summary', fontsize=16, fontweight='bold')
    
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 'Specificity']
    metric_values = [
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1'],
        metrics['roc_auc'],
        metrics['specificity']
    ]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
    
    for idx, (ax, name, value, color) in enumerate(zip(axes.flat, metric_names, metric_values, colors)):
        ax.text(0.5, 0.5, f'{value:.3f}', 
                ha='center', va='center', fontsize=32, fontweight='bold', color=color)
        ax.text(0.5, 0.15, name, ha='center', va='center', fontsize=11)
        circle = plt.Circle((0.5, 0.5), 0.4, color=color, fill=False, linewidth=3)
        ax.add_patch(circle)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    plt.tight_layout()
    fig.savefig(VIZ_DIR / 'fft_summary_dashboard.png', dpi=150)
    plt.close(fig)
    print("   ✅ Summary dashboard")
    
    # Save metrics JSON
    metrics_path = VIZ_DIR / 'fft_metrics.json'
    with open(metrics_path, 'w') as f:
        # Convert numpy types to native Python types
        metrics_json = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                       for k, v in metrics.items() if k != 'feature_names'}
        json.dump(metrics_json, f, indent=2)
    print(f"   ✅ Metrics JSON")
    
    print(f"\n✨ All visualizations saved to: {VIZ_DIR}")

if __name__ == '__main__':
    model, metrics = train_fft_model()
