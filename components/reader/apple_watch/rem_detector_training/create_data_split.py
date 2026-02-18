#!/usr/bin/env python3
"""
Create Standardized Train/Val/Test Split
Ensures consistent data splits across all experiments with proper seeding
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import pickle
import json
from datetime import datetime

# Import data loading
import sys
sys.path.append(str(Path(__file__).parent))
from train_rem_detector import load_subject_data, extract_features, align_labels
from config import RANDOM_SEED, TEST_SIZE

# Paths
DATA_DIR = Path("/Users/seangorman/code-projects/project-morpheus/components/data/tlr_data/walch-apple-watch")
OUTPUT_DIR = Path("/Users/seangorman/code-projects/project-morpheus/components/reader/apple_watch/data_splits")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def load_all_subjects():
    """Load all subjects from the dataset"""
    print("📂 Loading all subjects...")
    
    # Get all subject IDs from the original splits
    all_ids = []
    for split_file in ['train_ids.txt', 'val_ids.txt', 'test_ids.txt']:
        with open(DATA_DIR / split_file) as f:
            ids = [line.strip() for line in f if line.strip() and line.strip() != 'id']
            all_ids.extend(ids)
    
    # Remove duplicates, sort for deterministic order (reproducible splits)
    all_ids = sorted(set(all_ids))
    print(f"   Found {len(all_ids)} unique subjects")
    
    X_list = []
    y_list = []
    subject_indices = []  # Track which subject each sample belongs to
    subject_names = []
    
    for idx, subject_id in enumerate(all_ids):
        print(f"   Processing {idx+1}/{len(all_ids)}: {subject_id}...", end='\r')
        try:
            hr_data, motion_data, label_data = load_subject_data(subject_id)
            feature_df = extract_features(hr_data, motion_data)
            labels = align_labels(feature_df, label_data)
            
            # Extract features
            feature_cols = ['hr_feature', 'activity_feature', 'time_hours']
            X = feature_df[feature_cols].values
            y = np.array(labels)
            
            # Binary: REM vs not REM
            y_binary = (y == 5).astype(int)
            
            X_list.append(X)
            y_list.append(y_binary)
            subject_indices.extend([idx] * len(X))
            subject_names.append(subject_id)
            
        except Exception as e:
            print(f"\n   ⚠️  Skipped {subject_id}: {e}")
            continue
    
    print()  # New line after progress
    
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    subject_indices = np.array(subject_indices)
    
    print(f"   ✅ Loaded {len(X)} samples from {len(subject_names)} subjects")
    print(f"   📊 REM: {y.sum()} ({100*y.mean():.1f}%)")
    print(f"   📊 Non-REM: {len(y)-y.sum()} ({100*(1-y.mean()):.1f}%)")
    
    return X, y, subject_indices, subject_names

def create_subject_level_split(X, y, subject_indices, subject_names,
                               test_size=TEST_SIZE, random_seed=RANDOM_SEED):
    """
    Create train/test split at the SUBJECT level (80/20)
    This ensures no data leakage (same subject doesn't appear in multiple splits)
    """
    print(f"\n🔀 Creating subject-level splits (seed={random_seed})...")
    
    # Get unique subjects
    unique_subjects = np.unique(subject_indices)
    n_subjects = len(unique_subjects)
    
    print(f"   Total subjects: {n_subjects}")
    print(f"   Target splits: Train={1-test_size:.0%}, Test={test_size:.0%}")
    
    # Split: train and test subjects
    train_subjects, test_subjects = train_test_split(
        unique_subjects,
        test_size=test_size,
        random_state=random_seed,
        shuffle=True
    )
    
    print(f"\n   Subject counts:")
    print(f"   - Train: {len(train_subjects)} subjects")
    print(f"   - Test: {len(test_subjects)} subjects")
    
    # Create boolean masks for samples
    train_mask = np.isin(subject_indices, train_subjects)
    test_mask = np.isin(subject_indices, test_subjects)
    
    # Split data
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"\n   Sample counts:")
    print(f"   - Train: {len(X_train)} samples ({y_train.sum()} REM, {100*y_train.mean():.1f}%)")
    print(f"   - Test: {len(X_test)} samples ({y_test.sum()} REM, {100*y_test.mean():.1f}%)")
    
    # Create split info
    split_info = {
        'random_seed': random_seed,
        'created_at': datetime.now().isoformat(),
        'n_subjects_total': n_subjects,
        'n_subjects_train': len(train_subjects),
        'n_subjects_test': len(test_subjects),
        'n_samples_train': len(X_train),
        'n_samples_test': len(X_test),
        'rem_ratio_train': float(y_train.mean()),
        'rem_ratio_test': float(y_test.mean()),
        'train_subjects': [subject_names[i] for i in train_subjects],
        'test_subjects': [subject_names[i] for i in test_subjects],
        'feature_names': ['hr_feature', 'activity_feature', 'time_hours']
    }
    
    return X_train, X_test, y_train, y_test, split_info

def save_splits(X_train, X_test, y_train, y_test, split_info):
    """Save data splits to disk"""
    print(f"\n💾 Saving splits...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as pickle (fast, preserves numpy dtypes)
    split_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'split_info': split_info
    }
    
    pickle_path = OUTPUT_DIR / "data_split.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(split_data, f)
    print(f"   ✅ Saved pickle: {pickle_path}")
    
    # Save split info as JSON
    json_path = OUTPUT_DIR / "split_info.json"
    with open(json_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    print(f"   ✅ Saved info: {json_path}")
    
    # Also save as numpy arrays for easy loading
    np.savez_compressed(
        OUTPUT_DIR / "data_split.npz",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )
    print(f"   ✅ Saved numpy: {OUTPUT_DIR / 'data_split.npz'}")
    
    return pickle_path

def load_splits():
    """Load previously saved splits"""
    pickle_path = OUTPUT_DIR / "data_split.pkl"
    
    if not pickle_path.exists():
        raise FileNotFoundError(f"No saved splits found at {pickle_path}. Run create_splits() first.")
    
    with open(pickle_path, 'rb') as f:
        split_data = pickle.load(f)
    
    return (
        split_data['X_train'],
        split_data['X_test'],
        split_data['y_train'],
        split_data['y_test'],
        split_data['split_info']
    )

def print_split_summary():
    """Print summary of saved splits"""
    try:
        X_train, X_test, y_train, y_test, split_info = load_splits()
        
        print("="*60)
        print("📊 SAVED DATA SPLIT SUMMARY")
        print("="*60)
        print(f"\n🌱 Random Seed: {split_info['random_seed']}")
        print(f"📅 Created: {split_info['created_at']}")
        
        print(f"\n👥 Subjects:")
        print(f"   Train: {split_info['n_subjects_train']}")
        print(f"   Test:  {split_info['n_subjects_test']}")
        print(f"   Total: {split_info['n_subjects_total']}")
        
        print(f"\n📈 Samples:")
        print(f"   Train: {split_info['n_samples_train']:,} ({100*split_info['rem_ratio_train']:.1f}% REM)")
        print(f"   Test:  {split_info['n_samples_test']:,} ({100*split_info['rem_ratio_test']:.1f}% REM)")
        
        print(f"\n🎯 Features:")
        for i, feat in enumerate(split_info['feature_names'], 1):
            print(f"   {i}. {feat}")
        
        print("\n" + "="*60)
        
    except FileNotFoundError as e:
        print(f"❌ {e}")

def main():
    """Create and save standardized splits"""
    from config import set_seeds
    set_seeds()
    print("="*60)
    print("🎲 Creating Standardized Data Splits (80/20)")
    print("="*60)

    # Load all data
    X, y, subject_indices, subject_names = load_all_subjects()

    # Create splits (subject-level to prevent data leakage)
    X_train, X_test, y_train, y_test, split_info = create_subject_level_split(
        X, y, subject_indices, subject_names,
        test_size=TEST_SIZE,
        random_seed=RANDOM_SEED
    )
    
    # Save splits
    save_splits(X_train, X_test, y_train, y_test, split_info)
    
    print("\n✨ Splits created successfully!")
    print(f"📁 Saved to: {OUTPUT_DIR}")
    
    # Print summary
    print()
    print_split_summary()

if __name__ == '__main__':
    main()
