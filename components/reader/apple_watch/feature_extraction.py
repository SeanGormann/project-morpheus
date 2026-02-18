#!/usr/bin/env python3
"""
Feature Extraction for REM Detection
Multiple variants: baseline, subject-normalized, differential, etc.
Based on model_improvement_plan.md
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import percentileofscore

# Import core extraction from train_rem_detector
import sys
sys.path.append(str(Path(__file__).parent))
from train_rem_detector import (
    calculate_activity_counts,
    calculate_ema,
    load_subject_data,
    align_labels,
)


def extract_raw_features(hr_data, motion_data, epoch_seconds=30):
    """
    Extract raw epoch-level features (before any normalization).
    Returns DataFrame with: heart_rate, activity_count, time_hours, hr_ema, activity_ema
    """
    activity_df = calculate_activity_counts(motion_data, epoch_seconds)

    start_time = min(hr_data['timestamp'].min(), activity_df['timestamp'].min())
    end_time = max(hr_data['timestamp'].max(), activity_df['timestamp'].max())

    features = []
    timestamps = []

    current_time = start_time
    while current_time < end_time:
        epoch_end = current_time + epoch_seconds

        hr_window = hr_data[
            (hr_data['timestamp'] >= current_time) & 
            (hr_data['timestamp'] < epoch_end)
        ]
        activity_window = activity_df[
            (activity_df['timestamp'] >= current_time) & 
            (activity_df['timestamp'] < epoch_end)
        ]

        if len(hr_window) > 0 and len(activity_window) > 0:
            avg_hr = hr_window['heart_rate'].mean()
            activity = activity_window['activity_count'].mean()
            time_hours = (current_time - start_time) / 3600

            features.append({
                'heart_rate': avg_hr,
                'activity_count': activity,
                'time_hours': time_hours
            })
            timestamps.append(current_time)

        current_time = epoch_end

    feature_df = pd.DataFrame(features)
    feature_df['timestamp'] = timestamps

    # Circadian features (Walch et al.: REM probability increases through the night)
    # Use time_hours (0-8 for typical night) - REM peaks in middle-to-late sleep (~4-6h in)
    if len(feature_df) > 0:
        # Circadian phase: cosine peaking at ~5h into sleep (REM-rich period)
        feature_df['circadian_phase'] = np.cos(2 * np.pi * (feature_df['time_hours'] - 5) / 8)
        # Interaction: sleep progression * circadian alignment
        feature_df['sleep_circadian_interaction'] = feature_df['time_hours'] * feature_df['circadian_phase']

    if len(feature_df) > 0:
        feature_df['hr_ema'] = calculate_ema(feature_df['heart_rate'])
        feature_df['activity_ema'] = calculate_ema(feature_df['activity_count'])

    return feature_df


# =============================================================================
# FEATURE VARIANT: BASELINE (current absolute features)
# =============================================================================

def extract_baseline_features(feature_df):
    """Current TLR paper features: absolute HR, activity, time"""
    df = feature_df.copy()
    df['hr_feature'] = (df['hr_ema'] ** 3) / 1000
    max_activity = df['activity_ema'].max()
    if max_activity > 0:
        df['activity_feature'] = (df['activity_ema'] ** 2) / max_activity
    else:
        df['activity_feature'] = 0
    return df[['hr_feature', 'activity_feature', 'time_hours']]


# =============================================================================
# FEATURE VARIANT: SUBJECT-NORMALIZED (z-score)
# =============================================================================

def extract_normalized_zscore(feature_df):
    """
    Z-score normalization per subject.
    hr_feature = (hr_ema - subject_mean) / subject_std
    activity_feature = (activity_ema - subject_mean) / subject_std
    """
    df = feature_df.copy()
    hr_mean, hr_std = df['hr_ema'].mean(), df['hr_ema'].std()
    act_mean, act_std = df['activity_ema'].mean(), df['activity_ema'].std()

    if hr_std > 0:
        df['hr_feature'] = (df['hr_ema'] - hr_mean) / hr_std
    else:
        df['hr_feature'] = 0

    if act_std > 0:
        df['activity_feature'] = (df['activity_ema'] - act_mean) / act_std
    else:
        df['activity_feature'] = 0

    # Time stays as-is (already relative to night start)
    return df[['hr_feature', 'activity_feature', 'time_hours']]


# =============================================================================
# FEATURE VARIANT: SUBJECT-NORMALIZED (percentile 0-1)
# =============================================================================

def extract_normalized_percentile(feature_df):
    """
    Percentile within subject (0-1 scale).
    More robust to outliers than z-score.
    """
    df = feature_df.copy()
    hr_vals = df['hr_ema'].values
    act_vals = df['activity_ema'].values

    df['hr_feature'] = np.array([percentileofscore(hr_vals, x, kind='rank') / 100 for x in hr_vals])
    df['activity_feature'] = np.array([percentileofscore(act_vals, x, kind='rank') / 100 for x in act_vals])

    return df[['hr_feature', 'activity_feature', 'time_hours']]


# =============================================================================
# FEATURE VARIANT: MIN-MAX NORMALIZATION (0-1 per subject)
# =============================================================================

def extract_normalized_minmax(feature_df):
    """Min-max scale to 0-1 within subject"""
    df = feature_df.copy()
    hr_min, hr_max = df['hr_ema'].min(), df['hr_ema'].max()
    act_min, act_max = df['activity_ema'].min(), df['activity_ema'].max()

    if hr_max > hr_min:
        df['hr_feature'] = (df['hr_ema'] - hr_min) / (hr_max - hr_min)
    else:
        df['hr_feature'] = 0.5

    if act_max > act_min:
        df['activity_feature'] = (df['activity_ema'] - act_min) / (act_max - act_min)
    else:
        df['activity_feature'] = 0.5

    return df[['hr_feature', 'activity_feature', 'time_hours']]


# =============================================================================
# FEATURE VARIANT: NORMALIZED + DIFFERENTIAL (deltas, trends)
# =============================================================================

def extract_normalized_differential(feature_df, n_epochs_30s=2):
    """
    Z-score normalized + differential features.
    Deltas capture REM transition patterns.
    n_epochs_30s=2 means 1 min lookback for delta
    """
    df = feature_df.copy()

    # First normalize
    hr_mean, hr_std = df['hr_ema'].mean(), df['hr_ema'].std()
    act_mean, act_std = df['activity_ema'].mean(), df['activity_ema'].std()

    if hr_std > 0:
        df['hr_norm'] = (df['hr_ema'] - hr_mean) / hr_std
    else:
        df['hr_norm'] = 0
    if act_std > 0:
        df['activity_norm'] = (df['activity_ema'] - act_mean) / act_std
    else:
        df['activity_norm'] = 0

    # Deltas (change over last n epochs = n*30 seconds)
    df['hr_delta'] = df['hr_norm'].diff().fillna(0)
    df['activity_delta'] = df['activity_norm'].diff().fillna(0)

    # Rolling mean/std over ~2 min (4 epochs) for trend
    n_roll = min(4, len(df) // 2) if len(df) > 4 else 2
    df['hr_roll_mean'] = df['hr_norm'].rolling(n_roll, min_periods=1).mean()
    df['hr_roll_std'] = df['hr_norm'].rolling(n_roll, min_periods=1).std().fillna(0)
    df['activity_roll_mean'] = df['activity_norm'].rolling(n_roll, min_periods=1).mean()
    df['activity_roll_std'] = df['activity_norm'].rolling(n_roll, min_periods=1).std().fillna(0)

    cols = [
        'hr_norm', 'activity_norm', 'time_hours',
        'hr_delta', 'activity_delta',
        'hr_roll_mean', 'hr_roll_std', 'activity_roll_mean', 'activity_roll_std'
    ]
    return df[cols].rename(columns={
        'hr_norm': 'hr_feature',
        'activity_norm': 'activity_feature'
    })


# =============================================================================
# FEATURE VARIANT: NORMALIZED + DIFFERENTIAL + CIRCADIAN
# =============================================================================

def extract_normalized_differential_circadian(feature_df):
    """
    Z-score + differential + circadian features (Walch et al.).
    circadian_phase: REM peaks in middle-to-late sleep
    sleep_circadian_interaction: time_hours * circadian_phase
    """
    df = extract_normalized_differential(feature_df)
    # Add circadian (already in raw feature_df from extract_raw_features)
    if 'circadian_phase' in feature_df.columns:
        df['circadian_phase'] = feature_df['circadian_phase'].values[:len(df)]
        df['sleep_circadian_interaction'] = feature_df['sleep_circadian_interaction'].values[:len(df)]
    return df


# =============================================================================
# FEATURE VARIANT: NORMALIZED + RATIOS (scale-invariant)
# =============================================================================

def extract_normalized_ratios(feature_df):
    """
    Z-score normalized + physiological ratios.
    activity_hr_ratio = activity / hr (scale-invariant)
    """
    df = feature_df.copy()

    hr_mean, hr_std = df['hr_ema'].mean(), df['hr_ema'].std()
    act_mean, act_std = df['activity_ema'].mean(), df['activity_ema'].std()

    if hr_std > 0:
        df['hr_feature'] = (df['hr_ema'] - hr_mean) / hr_std
    else:
        df['hr_feature'] = 0
    if act_std > 0:
        df['activity_feature'] = (df['activity_ema'] - act_mean) / act_std
    else:
        df['activity_feature'] = 0

    # Activity/HR ratio (avoid div by zero)
    df['activity_hr_ratio'] = df['activity_ema'] / (df['hr_ema'].clip(lower=40) + 1e-6)
    # Normalize ratio within subject
    ratio_mean, ratio_std = df['activity_hr_ratio'].mean(), df['activity_hr_ratio'].std()
    if ratio_std > 0:
        df['activity_hr_ratio'] = (df['activity_hr_ratio'] - ratio_mean) / ratio_std

    return df[['hr_feature', 'activity_feature', 'time_hours', 'activity_hr_ratio']]


# =============================================================================
# FEATURE VARIANT: ALL COMBINED (normalized + differential + ratios)
# =============================================================================

def extract_full_features(feature_df):
    """Combines normalized + differential + ratios for maximum information"""
    df_norm_diff = extract_normalized_differential(feature_df)
    df_ratios = extract_normalized_ratios(feature_df)

    df = df_norm_diff.copy()
    df['activity_hr_ratio'] = df_ratios['activity_hr_ratio']

    return df


def extract_full_circadian_features(feature_df):
    """Full features + circadian (best of all worlds)"""
    df = extract_full_features(feature_df)
    if 'circadian_phase' in feature_df.columns:
        df['circadian_phase'] = feature_df['circadian_phase'].values[:len(df)]
        df['sleep_circadian_interaction'] = feature_df['sleep_circadian_interaction'].values[:len(df)]
    return df


# =============================================================================
# FEATURE VARIANT REGISTRY (for easy iteration)
# =============================================================================

FEATURE_VARIANTS = {
    'baseline': {
        'extractor': extract_baseline_features,
        'description': 'Current absolute features (hr³/1000, activity²/max, time)',
    },
    'normalized_zscore': {
        'extractor': extract_normalized_zscore,
        'description': 'Z-score per subject (hr, activity, time)',
    },
    'normalized_percentile': {
        'extractor': extract_normalized_percentile,
        'description': 'Percentile 0-1 per subject',
    },
    'normalized_minmax': {
        'extractor': extract_normalized_minmax,
        'description': 'Min-max 0-1 per subject',
    },
    'normalized_differential': {
        'extractor': extract_normalized_differential,
        'description': 'Z-score + HR/activity deltas + rolling stats',
    },
    'normalized_ratios': {
        'extractor': extract_normalized_ratios,
        'description': 'Z-score + activity/HR ratio',
    },
    'full': {
        'extractor': extract_full_features,
        'description': 'Normalized + differential + ratios (all features)',
    },
    'normalized_differential_circadian': {
        'extractor': extract_normalized_differential_circadian,
        'description': 'Z-score + differential + circadian (Walch)',
    },
    'full_circadian': {
        'extractor': extract_full_circadian_features,
        'description': 'Full + circadian phase & interaction',
    },
}


def get_feature_matrix(feature_df, variant='baseline'):
    """Get feature matrix for a given variant."""
    extractor = FEATURE_VARIANTS[variant]['extractor']
    return extractor(feature_df).values


FEATURE_NAMES = {
    'baseline': ['hr_feature', 'activity_feature', 'time_hours'],
    'normalized_zscore': ['hr_feature', 'activity_feature', 'time_hours'],
    'normalized_percentile': ['hr_feature', 'activity_feature', 'time_hours'],
    'normalized_minmax': ['hr_feature', 'activity_feature', 'time_hours'],
    'normalized_differential': [
        'hr_feature', 'activity_feature', 'time_hours',
        'hr_delta', 'activity_delta',
        'hr_roll_mean', 'hr_roll_std', 'activity_roll_mean', 'activity_roll_std'
    ],
    'normalized_ratios': ['hr_feature', 'activity_feature', 'time_hours', 'activity_hr_ratio'],
    'full': [
        'hr_feature', 'activity_feature', 'time_hours',
        'hr_delta', 'activity_delta',
        'hr_roll_mean', 'hr_roll_std', 'activity_roll_mean', 'activity_roll_std',
        'activity_hr_ratio'
    ],
    'normalized_differential_circadian': [
        'hr_feature', 'activity_feature', 'time_hours',
        'hr_delta', 'activity_delta',
        'hr_roll_mean', 'hr_roll_std', 'activity_roll_mean', 'activity_roll_std',
        'circadian_phase', 'sleep_circadian_interaction'
    ],
    'full_circadian': [
        'hr_feature', 'activity_feature', 'time_hours',
        'hr_delta', 'activity_delta',
        'hr_roll_mean', 'hr_roll_std', 'activity_roll_mean', 'activity_roll_std',
        'activity_hr_ratio',
        'circadian_phase', 'sleep_circadian_interaction'
    ],
}


def get_feature_names(variant='baseline'):
    """Get feature names for a given variant."""
    return FEATURE_NAMES.get(variant, FEATURE_NAMES['baseline'])
