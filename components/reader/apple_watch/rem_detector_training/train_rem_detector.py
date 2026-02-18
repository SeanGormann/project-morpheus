#!/usr/bin/env python3
"""
Train Random Forest REM Detection Model
Based on Walch et al. 2019 + TLR implementation
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
import pickle
import json

# Paths
DATA_DIR = Path("/Users/seangorman/code-projects/project-morpheus/components/data/tlr_data/walch-apple-watch")
OUTPUT_DIR = Path("/Users/seangorman/code-projects/project-morpheus/components/reader/apple_watch/models")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def load_subject_data(subject_id):
    """Load heart rate, motion, and labels for a subject"""
    hr_file = DATA_DIR / "heart_rate" / f"{subject_id}_heartrate.csv"
    motion_file = DATA_DIR / "motion" / f"{subject_id}_acceleration.csv"
    label_file = DATA_DIR / "labels" / f"{subject_id}_labeled_sleep.csv"
    
    # Load heart rate (timestamp, HR in bpm) - skip header
    hr_data = pd.read_csv(hr_file, skiprows=1, names=['timestamp', 'heart_rate'])
    
    # Load motion (timestamp, x, y, z in g) - skip header
    motion_data = pd.read_csv(motion_file, skiprows=1, names=['timestamp', 'x', 'y', 'z'])
    
    # Load labels (timestamp, stage: 0=wake, 1=N1, 2=N2, 3=N3, 5=REM) - skip header
    label_data = pd.read_csv(label_file, skiprows=1, names=['timestamp', 'stage'])
    
    return hr_data, motion_data, label_data

def calculate_activity_counts(motion_data, window_seconds=30):
    """
    Convert raw acceleration to activity counts
    Based on Neishabouri et al. 2022 (ActiGraph method)
    """
    # Calculate magnitude of acceleration vector
    motion_data['magnitude'] = np.sqrt(
        motion_data['x']**2 + 
        motion_data['y']**2 + 
        motion_data['z']**2
    )
    
    # Simple activity count: sum of squared magnitudes in window
    # This is a simplified version - the full method is more complex
    activity_counts = []
    timestamps = []
    
    start_time = motion_data['timestamp'].min()
    end_time = motion_data['timestamp'].max()
    
    current_time = start_time
    while current_time < end_time:
        window_end = current_time + window_seconds
        window_data = motion_data[
            (motion_data['timestamp'] >= current_time) & 
            (motion_data['timestamp'] < window_end)
        ]
        
        if len(window_data) > 0:
            # Sum of squared magnitudes
            activity_count = (window_data['magnitude']**2).sum()
            activity_counts.append(activity_count)
            timestamps.append(current_time)
        
        current_time = window_end
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'activity_count': activity_counts
    })

def calculate_ema(series, alpha=0.3):
    """Calculate Exponential Moving Average"""
    ema = np.zeros(len(series))
    ema[0] = series.iloc[0]
    
    for i in range(1, len(series)):
        ema[i] = alpha * series.iloc[i] + (1 - alpha) * ema[i-1]
    
    return ema

def extract_features(hr_data, motion_data, epoch_seconds=30):
    """
    Extract features for each 30-second epoch
    Following TLR paper methodology
    """
    # Calculate activity counts from motion
    activity_df = calculate_activity_counts(motion_data, epoch_seconds)
    
    # Align heart rate to epochs
    start_time = min(hr_data['timestamp'].min(), activity_df['timestamp'].min())
    end_time = max(hr_data['timestamp'].max(), activity_df['timestamp'].max())
    
    features = []
    timestamps = []
    
    current_time = start_time
    while current_time < end_time:
        epoch_end = current_time + epoch_seconds
        
        # Get heart rate in this epoch
        hr_window = hr_data[
            (hr_data['timestamp'] >= current_time) & 
            (hr_data['timestamp'] < epoch_end)
        ]
        
        # Get activity count for this epoch
        activity_window = activity_df[
            (activity_df['timestamp'] >= current_time) & 
            (activity_df['timestamp'] < epoch_end)
        ]
        
        if len(hr_window) > 0 and len(activity_window) > 0:
            avg_hr = hr_window['heart_rate'].mean()
            activity = activity_window['activity_count'].mean()
            time_hours = (current_time - start_time) / 3600  # Hours since start
            
            features.append({
                'heart_rate': avg_hr,
                'activity_count': activity,
                'time_hours': time_hours
            })
            timestamps.append(current_time)
        
        current_time = epoch_end
    
    feature_df = pd.DataFrame(features)
    feature_df['timestamp'] = timestamps
    
    # Apply EMA smoothing to heart rate (as per TLR paper)
    if len(feature_df) > 0:
        feature_df['hr_ema'] = calculate_ema(feature_df['heart_rate'])
        # Cube and scale HR feature
        feature_df['hr_feature'] = (feature_df['hr_ema'] ** 3) / 1000
        
        # Apply EMA to activity and normalize
        feature_df['activity_ema'] = calculate_ema(feature_df['activity_count'])
        # Square and normalize activity
        max_activity = feature_df['activity_ema'].max()
        if max_activity > 0:
            feature_df['activity_feature'] = (feature_df['activity_ema'] ** 2) / max_activity
        else:
            feature_df['activity_feature'] = 0
    
    return feature_df

def align_labels(feature_df, label_data, epoch_seconds=30):
    """Align sleep stage labels with features"""
    labels = []
    
    for _, row in feature_df.iterrows():
        epoch_time = row['timestamp']
        # Find label for this epoch
        label_match = label_data[
            (label_data['timestamp'] >= epoch_time) & 
            (label_data['timestamp'] < epoch_time + epoch_seconds)
        ]
        
        if len(label_match) > 0:
            # Use most common label in this epoch
            stage = label_match['stage'].mode()[0] if len(label_match) > 0 else 0
            labels.append(stage)
        else:
            labels.append(0)  # Default to wake if no label
    
    return labels

def train_model():
    """Train Random Forest REM detection model"""
    print("Loading training data...")
    
    # Load train/val/test splits
    with open(DATA_DIR / "train_ids.txt") as f:
        train_ids = [line.strip() for line in f if line.strip() and line.strip() != 'id']
    
    with open(DATA_DIR / "val_ids.txt") as f:
        val_ids = [line.strip() for line in f if line.strip() and line.strip() != 'id']
    
    # Combine train and val for final model
    all_train_ids = train_ids + val_ids
    
    print(f"Training on {len(all_train_ids)} subjects")
    
    # Collect all training data
    X_train = []
    y_train = []
    
    for subject_id in all_train_ids:
        print(f"Processing subject {subject_id}...")
        try:
            hr_data, motion_data, label_data = load_subject_data(subject_id)
            feature_df = extract_features(hr_data, motion_data)
            labels = align_labels(feature_df, label_data)
            
            # Extract feature columns
            feature_cols = ['hr_feature', 'activity_feature', 'time_hours']
            X = feature_df[feature_cols].values
            y = np.array(labels)
            
            # Binary classification: REM (5) vs not REM
            y_binary = (y == 5).astype(int)
            
            X_train.append(X)
            y_train.append(y_binary)
            
        except Exception as e:
            print(f"Error processing {subject_id}: {e}")
            continue
    
    # Combine all data
    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)
    
    print(f"\nTotal training samples: {len(X_train)}")
    print(f"REM samples: {y_train.sum()} ({100*y_train.mean():.1f}%)")
    print(f"Non-REM samples: {len(y_train) - y_train.sum()} ({100*(1-y_train.mean()):.1f}%)")
    
    # Train Random Forest
    print("\nTraining Random Forest...")
    clf = RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=48,  # As per TLR paper
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'  # Handle class imbalance
    )
    
    clf.fit(X_train, y_train)
    
    # Evaluate on training data
    y_pred = clf.predict(X_train)
    print("\nTraining performance:")
    print(f"Accuracy: {accuracy_score(y_train, y_pred):.3f}")
    print(f"Precision: {precision_score(y_train, y_pred):.3f}")
    print(f"Recall: {recall_score(y_train, y_pred):.3f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_train, y_pred))
    
    # Save model
    model_path = OUTPUT_DIR / "rem_detector_rf.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    print(f"\n✅ Model saved to {model_path}")
    
    # Save feature names for reference
    feature_info = {
        'feature_names': ['hr_feature', 'activity_feature', 'time_hours'],
        'description': {
            'hr_feature': 'Heart rate EMA cubed and scaled by 1000',
            'activity_feature': 'Activity count squared and normalized by max',
            'time_hours': 'Hours since sleep start'
        },
        'rem_threshold': 0.24,  # As per TLR paper
        'epoch_seconds': 30
    }
    
    with open(OUTPUT_DIR / "model_info.json", 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    return clf

if __name__ == '__main__':
    model = train_model()