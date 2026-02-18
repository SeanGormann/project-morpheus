#!/usr/bin/env python3
"""
Create Data Splits for Multiple Feature Variants
Generates train/test splits for each feature set (baseline, normalized, etc.)
Saves subject_indices for GroupKFold CV
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import pickle
import json
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent))
from train_rem_detector import load_subject_data, align_labels
from feature_extraction import (
    extract_raw_features,
    FEATURE_VARIANTS,
    get_feature_names,
)
from config import RANDOM_SEED, TEST_SIZE, set_seeds

# Paths
DATA_DIR = Path("/Users/seangorman/code-projects/project-morpheus/components/data/tlr_data/walch-apple-watch")
OUTPUT_DIR = Path("/Users/seangorman/code-projects/project-morpheus/components/reader/apple_watch/data_splits")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def load_all_subjects_with_features(variants=None):
    """Load all subjects and extract features for each variant."""
    variants = variants or list(FEATURE_VARIANTS.keys())

    print("📂 Loading all subjects...")

    all_ids = []
    for split_file in ['train_ids.txt', 'val_ids.txt', 'test_ids.txt']:
        with open(DATA_DIR / split_file) as f:
            ids = [line.strip() for line in f if line.strip() and line.strip() != 'id']
            all_ids.extend(ids)
    all_ids = sorted(set(all_ids))  # Deterministic order for reproducible splits
    print(f"   Found {len(all_ids)} unique subjects")

    # Per-variant: X_list, y_list, subject_indices
    data_by_variant = {v: {'X_list': [], 'y_list': [], 'subject_indices': []} for v in variants}

    for idx, subject_id in enumerate(all_ids):
        print(f"   Processing {idx+1}/{len(all_ids)}: {subject_id}...", end='\r')
        try:
            hr_data, motion_data, label_data = load_subject_data(subject_id)
            raw_df = extract_raw_features(hr_data, motion_data)
            labels = align_labels(raw_df, label_data)
            y_binary = (np.array(labels) == 5).astype(int)

            for variant in variants:
                extractor = FEATURE_VARIANTS[variant]['extractor']
                feature_df = extractor(raw_df)
                X = feature_df.values
                data_by_variant[variant]['X_list'].append(X)
                data_by_variant[variant]['y_list'].append(y_binary)
                data_by_variant[variant]['subject_indices'].extend([idx] * len(X))

        except Exception as e:
            print(f"\n   ⚠️  Skipped {subject_id}: {e}")
            continue

    print()

    subject_names = all_ids
    results = {}

    for variant in variants:
        X = np.vstack(data_by_variant[variant]['X_list'])
        y = np.concatenate(data_by_variant[variant]['y_list'])
        subject_indices = np.array(data_by_variant[variant]['subject_indices'])
        results[variant] = {
            'X': X,
            'y': y,
            'subject_indices': subject_indices,
            'subject_names': subject_names,
        }

    return results


def create_subject_level_split(X, y, subject_indices, subject_names, test_size=TEST_SIZE, random_seed=RANDOM_SEED):
    """Create train/test split at subject level."""
    unique_subjects = np.unique(subject_indices)
    train_subjects, test_subjects = train_test_split(
        unique_subjects,
        test_size=test_size,
        random_state=random_seed,
        shuffle=True
    )

    train_mask = np.isin(subject_indices, train_subjects)
    test_mask = np.isin(subject_indices, test_subjects)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'train_subject_indices': subject_indices[train_mask],
        'test_subject_indices': subject_indices[test_mask],
        'train_subjects': [subject_names[i] for i in train_subjects],
        'test_subjects': [subject_names[i] for i in test_subjects],
        'n_train_subjects': len(train_subjects),
        'n_test_subjects': len(test_subjects),
        'n_samples_train': len(X_train),
        'n_samples_test': len(X_test),
    }


def save_splits(variant, split_data, feature_names):
    """Save splits for a variant."""
    variant_dir = OUTPUT_DIR / variant
    variant_dir.mkdir(exist_ok=True)

    split_info = {
        'random_seed': RANDOM_SEED,
        'created_at': datetime.now().isoformat(),
        'variant': variant,
        'n_subjects_train': split_data['n_train_subjects'],
        'n_subjects_test': split_data['n_test_subjects'],
        'n_samples_train': len(split_data['X_train']),
        'n_samples_test': len(split_data['X_test']),
        'rem_ratio_train': float(split_data['y_train'].mean()),
        'rem_ratio_test': float(split_data['y_test'].mean()),
        'train_subjects': split_data['train_subjects'],
        'test_subjects': split_data['test_subjects'],
        'feature_names': feature_names,
    }

    pickle_data = {
        'X_train': split_data['X_train'],
        'X_test': split_data['X_test'],
        'y_train': split_data['y_train'],
        'y_test': split_data['y_test'],
        'train_subject_indices': split_data['train_subject_indices'],
        'test_subject_indices': split_data['test_subject_indices'],
        'split_info': split_info,
    }

    with open(variant_dir / "data_split.pkl", 'wb') as f:
        pickle.dump(pickle_data, f)

    with open(variant_dir / "split_info.json", 'w') as f:
        json.dump(split_info, f, indent=2)

    np.savez_compressed(
        variant_dir / "data_split.npz",
        X_train=split_data['X_train'],
        X_test=split_data['X_test'],
        y_train=split_data['y_train'],
        y_test=split_data['y_test'],
        train_subject_indices=split_data['train_subject_indices'],
        test_subject_indices=split_data['test_subject_indices'],
    )

    return variant_dir


def load_splits(variant='baseline'):
    """Load splits for a variant."""
    variant_dir = OUTPUT_DIR / variant
    pickle_path = variant_dir / "data_split.pkl"

    if not pickle_path.exists():
        raise FileNotFoundError(
            f"No splits for variant '{variant}'. Run create_data_split_variants.py first."
        )

    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    return (
        data['X_train'],
        data['X_test'],
        data['y_train'],
        data['y_test'],
        data['split_info'],
        data.get('train_subject_indices'),
        data.get('test_subject_indices'),
    )


def main():
    set_seeds()
    import argparse
    parser = argparse.ArgumentParser(description='Create data splits for all feature variants')
    parser.add_argument('--variants', nargs='+', default=None,
                        help='Specific variants to create (default: all)')
    args = parser.parse_args()

    variants = args.variants or list(FEATURE_VARIANTS.keys())

    print("=" * 60)
    print("🎲 Creating Data Splits for Multiple Feature Variants")
    print("=" * 60)

    results = load_all_subjects_with_features(variants)

    for variant in variants:
        data = results[variant]
        print(f"\n📊 Variant: {variant}")
        print(f"   Samples: {len(data['X'])} ({data['y'].sum()} REM, {100*data['y'].mean():.1f}%)")

    print("\n🔀 Creating subject-level splits (80/20)...")

    for variant in variants:
        data = results[variant]
        split_data = create_subject_level_split(
            data['X'], data['y'],
            data['subject_indices'],
            data['subject_names'],
        )
        feature_names = get_feature_names(variant)
        save_splits(variant, split_data, feature_names)
        print(f"   ✅ {variant}: train={split_data['n_samples_train']}, test={split_data['n_samples_test']}")

    print("\n✨ All splits created!")
    print(f"📁 Saved to: {OUTPUT_DIR}")
    print("\nVariants available:")
    for v in variants:
        print(f"   - {v}: {FEATURE_VARIANTS[v]['description']}")


if __name__ == '__main__':
    main()
