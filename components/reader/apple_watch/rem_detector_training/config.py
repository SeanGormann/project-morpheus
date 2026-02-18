"""
Shared configuration for REM detection experiments.
Ensures consistent splits, seeds, and hyperparameters across all scripts.
"""

import numpy as np

# Seeds - use everywhere for reproducibility
RANDOM_SEED = 42


def set_seeds(seed=RANDOM_SEED):
    """Set numpy and random seeds for reproducibility."""
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except ImportError:
        pass


# Model hyperparameters (canonical - used by train_with_group_cv, train_phase2)
XGBOOST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'scale_pos_weight': 4,
    'random_state': RANDOM_SEED,
    'n_jobs': -1,
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'min_samples_leaf': 48,
    'max_depth': 10,
    'max_features': 'sqrt',
    'random_state': RANDOM_SEED,
    'n_jobs': -1,
    'class_weight': 'balanced',
}

MLP_PARAMS = {
    'hidden_layer_sizes': (64, 32),
    'activation': 'relu',
    'solver': 'adam',
    'alpha': 0.01,
    'batch_size': 256,
    'learning_rate_init': 0.001,
    'max_iter': 200,
    'random_state': RANDOM_SEED,
    'early_stopping': True,
    'validation_fraction': 0.1,
}

LOGISTIC_REGRESSION_PARAMS = {
    'max_iter': 1000,
    'random_state': RANDOM_SEED,
    'class_weight': 'balanced',
    'C': 0.5,
}

# Data split
TEST_SIZE = 0.20
