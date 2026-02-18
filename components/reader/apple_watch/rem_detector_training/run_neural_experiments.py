#!/usr/bin/env python3
"""
Run Neural Network Experiments: MLP, RNN (LSTM), CNN
- Creates rolling-window sequences from existing data splits
- Tests MLP (baseline), LSTM, CNN with varying window sizes and kernel sizes
- Uses GroupKFold CV for subject-level validation

Requires: pip install torch (or uv add torch, or uv pip install -e ".[neural]")
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)
import pickle
import json
from datetime import datetime
import sys
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='torch')

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

sys.path.append(str(Path(__file__).parent))
from create_data_split_variants import load_splits
from config import RANDOM_SEED, set_seeds

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "models"
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# -----------------------------------------------------------------------------
# Windowed data creation
# -----------------------------------------------------------------------------

def create_windowed_data(X, y, subject_indices, window_size):
    """
    Create rolling windows from sequential data, grouped by subject.
    Returns X_windows (n_samples, window_size, n_features), y_labels, subject_indices.
    Label is the last epoch in each window.
    """
    X_windows = []
    y_labels = []
    subj_out = []

    for subj_id in np.unique(subject_indices):
        mask = subject_indices == subj_id
        X_subj = X[mask]
        y_subj = y[mask]

        for i in range(window_size, len(X_subj) + 1):
            X_windows.append(X_subj[i - window_size:i])
            y_labels.append(y_subj[i - 1])  # predict last epoch in window
            subj_out.append(subj_id)

    return np.array(X_windows, dtype=np.float32), np.array(y_labels), np.array(subj_out)


def prepare_flat_mlp_data(X_train, X_test, y_train, y_test, scaler=None):
    """For MLP baseline: use flat features as-is (no window)."""
    if scaler is None:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    return (
        torch.FloatTensor(X_train), torch.FloatTensor(X_test),
        torch.LongTensor(y_train), torch.LongTensor(y_test),
        scaler,
    )


# -----------------------------------------------------------------------------
# PyTorch models (only defined when torch is available)
# -----------------------------------------------------------------------------

if HAS_TORCH:
    import torch.nn as nn

    class MLP(nn.Module):
        """Simple MLP for flat or flattened window input."""

        def __init__(self, input_dim, hidden_dims=(64, 32), dropout=0.2):
            super().__init__()
            layers = []
            prev = input_dim
            for h in hidden_dims:
                layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
                prev = h
            layers.append(nn.Linear(prev, 2))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    class LSTMClassifier(nn.Module):
        """LSTM for sequential input (batch, seq_len, n_features)."""

        def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size, hidden_size, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0,
            )
            self.fc = nn.Linear(hidden_size, 2)

        def forward(self, x):
            out, (h_n, _) = self.lstm(x)
            return self.fc(h_n[-1])

    class CNN1DClassifier(nn.Module):
        """1D CNN for (batch, n_channels, seq_len). Channels = n_features."""

        def __init__(self, n_channels, seq_len, kernel_sizes=(3, 5), n_filters=32):
            super().__init__()
            convs = []
            for k in kernel_sizes:
                convs.append(nn.Conv1d(n_channels, n_filters, k, padding=k // 2))
            self.convs = nn.ModuleList(convs)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(n_filters * len(kernel_sizes), 2)

        def forward(self, x):
            outs = []
            for conv in self.convs:
                h = torch.relu(conv(x))
                h = self.pool(h)
                outs.append(h.squeeze(-1))
            out = torch.cat(outs, dim=1)
            return self.fc(out)


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, X, y, device, batch_size=512):
    model.eval()
    loader = DataLoader(
        TensorDataset(X, y),
        batch_size=batch_size,
        shuffle=False,
    )
    probs = []
    with torch.no_grad():
        for Xb, _ in loader:
            Xb = Xb.to(device)
            logits = model(Xb)
            probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    probs = np.concatenate(probs)
    preds = (probs >= 0.5).astype(int)
    return probs, preds


def run_group_kfold_cv(model_fn, X, y, groups, device, n_splits=5, epochs=50, batch_size=256):
    """Run GroupKFold CV, returning mean ± std ROC AUC."""
    gkf = GroupKFold(n_splits=n_splits)
    scores = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        if X_tr.ndim == 3:
            n, seq, feat = X_tr.shape
            X_tr = scaler.fit_transform(X_tr.reshape(-1, feat)).reshape(n, seq, feat)
            nv, seqv, _ = X_val.shape
            X_val = scaler.transform(X_val.reshape(-1, feat)).reshape(nv, seqv, feat)
        else:
            X_tr = scaler.fit_transform(X_tr)
            X_val = scaler.transform(X_val)

        X_tr = torch.FloatTensor(X_tr)
        X_val = torch.FloatTensor(X_val)
        y_tr = torch.LongTensor(y_tr)
        y_val_np = y_val

        loader = DataLoader(
            TensorDataset(X_tr, y_tr),
            batch_size=batch_size,
            shuffle=True,
        )
        model = model_fn().to(device)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 4.0], dtype=torch.float32).to(device))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for _ in range(epochs):
            train_epoch(model, loader, criterion, optimizer, device)

        probs, _ = evaluate(model, X_val, y_val_np, device, batch_size=512)
        auc = roc_auc_score(y_val_np, probs) if len(np.unique(y_val_np)) > 1 else 0.5
        scores.append(auc)
    return np.mean(scores), np.std(scores)


def train_and_evaluate_neural(
    model_name,
    model_fn,
    X_train, X_test, y_train, y_test,
    train_subj, test_subj,
    device,
    epochs=80,
    batch_size=256,
):
    """Train on full train set, evaluate on test set."""
    # Scale
    scaler = StandardScaler()
    if X_train.ndim == 3:
        n, seq, feat = X_train.shape
        X_train = scaler.fit_transform(X_train.reshape(-1, feat)).reshape(n, seq, feat)
        nt, seqt, _ = X_test.shape
        X_test = scaler.transform(X_test.reshape(-1, feat)).reshape(nt, seqt, feat)
    else:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    X_tr = torch.FloatTensor(X_train)
    X_te = torch.FloatTensor(X_test)
    y_tr = torch.LongTensor(y_train)

    loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=batch_size,
        shuffle=True,
    )
    model = model_fn().to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 4.0], dtype=torch.float32).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    start = datetime.now()
    for _ in range(epochs):
        train_epoch(model, loader, criterion, optimizer, device)
    train_time = (datetime.now() - start).total_seconds()

    probs, preds = evaluate(model, X_te, y_test, device, batch_size=512)

    metrics = {
        'accuracy': accuracy_score(y_test, preds),
        'precision': precision_score(y_test, preds, zero_division=0),
        'recall': recall_score(y_test, preds, zero_division=0),
        'f1': f1_score(y_test, preds, zero_division=0),
        'roc_auc': roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else 0.5,
        'train_time_seconds': train_time,
    }

    # GroupKFold CV on train
    if train_subj is not None and len(np.unique(train_subj)) >= 5:
        cv_mean, cv_std = run_group_kfold_cv(
            model_fn, X_train, y_train, train_subj, device,
            n_splits=5, epochs=epochs, batch_size=batch_size,
        )
        metrics['cv_roc_auc_mean'] = cv_mean
        metrics['cv_roc_auc_std'] = cv_std
    else:
        metrics['cv_roc_auc_mean'] = np.nan
        metrics['cv_roc_auc_std'] = np.nan

    return model, scaler, metrics


# -----------------------------------------------------------------------------
# Experiment configs
# -----------------------------------------------------------------------------

def get_experiment_configs():
    """Return list of (name, window_size, model_type, kwargs)."""
    configs = []

    # MLP baseline (no window) - flat features
    configs.append(('MLP_flat', None, 'mlp', {}))

    # MLP with different window sizes
    for w in [5, 10, 20]:
        configs.append((f'MLP_window{w}', w, 'mlp', {}))

    # LSTM with different window sizes
    for w in [5, 10, 20]:
        configs.append((f'LSTM_window{w}', w, 'lstm', {'hidden_size': 64, 'num_layers': 2}))

    for w in [5, 10, 20]:
        configs.append((f'LSTM_window{w}_big', w, 'lstm', {'hidden_size': 128, 'num_layers': 2}))

    # CNN with different window sizes and kernel sizes
    for w in [5, 10, 20]:
        for k in [2, 3, 5]:
            if k <= w:
                configs.append((f'CNN_window{w}_k{k}', w, 'cnn', {'kernel_sizes': (k,)}))

    return configs


def build_model_fn(config, n_features, device):
    """Build a callable that returns a fresh model for the given config."""
    name, window_size, model_type, kwargs = config

    if model_type == 'mlp':
        if window_size is None:
            input_dim = n_features
        else:
            input_dim = window_size * n_features

        def fn():
            return MLP(input_dim, hidden_dims=(64, 32), **{k: v for k, v in kwargs.items() if k in ('dropout',)})

        return fn

    if model_type == 'lstm':
        def fn():
            return LSTMClassifier(
                input_size=n_features,
                hidden_size=kwargs.get('hidden_size', 64),
                num_layers=kwargs.get('num_layers', 2),
            )

        return fn

    if model_type == 'cnn':
        kernel_sizes = kwargs.get('kernel_sizes', (3, 5))

        def fn():
            return CNN1DClassifier(
                n_channels=n_features,
                seq_len=window_size,
                kernel_sizes=kernel_sizes,
            )

        return fn

    raise ValueError(f"Unknown model_type: {model_type}")


def prepare_data_for_config(config, X_train, X_test, y_train, y_test, train_subj, test_subj):
    """Prepare (X_train, X_test, y_train, y_test, train_subj, test_subj) for the config."""
    name, window_size, model_type, _ = config

    if window_size is None:
        # MLP flat: use as-is
        return X_train, X_test, y_train, y_test, train_subj, test_subj

    X_tr_w, y_tr_w, subj_tr_w = create_windowed_data(X_train, y_train, train_subj, window_size)
    X_te_w, y_te_w, subj_te_w = create_windowed_data(X_test, y_test, test_subj, window_size)

    if model_type == 'mlp':
        # Flatten for MLP
        n, seq, feat = X_tr_w.shape
        X_tr_w = X_tr_w.reshape(n, seq * feat)
        nt, _, _ = X_te_w.shape
        X_te_w = X_te_w.reshape(nt, seq * feat)

    return X_tr_w, X_te_w, y_tr_w, y_te_w, subj_tr_w, subj_te_w


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    if not HAS_TORCH:
        print("❌ PyTorch not installed. Run: pip install torch (or uv add torch)")
        return 1

    set_seeds()
    torch.manual_seed(RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    import argparse
    parser = argparse.ArgumentParser(description='Run neural network experiments')
    parser.add_argument('--variant', default='baseline', help='Feature variant')
    parser.add_argument('--epochs', type=int, default=80, help='Training epochs')
    parser.add_argument('--quick', action='store_true', help='Fewer configs, fewer epochs')
    args = parser.parse_args()

    print("=" * 60)
    print("🧠 Neural Network Experiments: MLP, RNN, CNN")
    print("=" * 60)
    print(f"\nVariant: {args.variant}")

    try:
        X_train, X_test, y_train, y_test, split_info, train_subj, test_subj = load_splits(args.variant)
    except FileNotFoundError as e:
        print(f"   ❌ {e}")
        return 1

    n_features = X_train.shape[1]
    print(f"\n📊 Train: {len(X_train)}, Test: {len(X_test)}, Features: {n_features}")

    configs = get_experiment_configs()
    if args.quick:
        configs = [
            c for c in configs
            if c[0] in ('MLP_flat', 'MLP_window10', 'LSTM_window10', 'CNN_window10_k3')
        ]

    all_metrics = {}
    variant_dir = OUTPUT_DIR / args.variant
    variant_dir.mkdir(exist_ok=True)

    for i, config in enumerate(configs):
        name, window_size, model_type, kwargs = config
        print(f"\n[{i+1}/{len(configs)}] 🔧 {name} (window={window_size}, type={model_type})")

        X_tr, X_te, y_tr, y_te, subj_tr, subj_te = prepare_data_for_config(
            config, X_train, X_test, y_train, y_test, train_subj, test_subj
        )

        model_fn = build_model_fn(config, n_features, device)
        model, scaler, metrics = train_and_evaluate_neural(
            name, model_fn, X_tr, X_te, y_tr, y_te, subj_tr, subj_te,
            device, epochs=args.epochs if not args.quick else 30,
        )

        all_metrics[name] = metrics
        print(f"   ✅ Test ROC AUC: {metrics['roc_auc']:.3f}")
        if not np.isnan(metrics.get('cv_roc_auc_mean', np.nan)):
            print(f"   📊 CV ROC AUC: {metrics['cv_roc_auc_mean']:.3f} ± {metrics['cv_roc_auc_std']:.3f}")

        model_path = variant_dir / f"rem_detector_{name.lower().replace('-', '_')}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model_state': model.state_dict(),
                'model_class': type(model).__name__,
                'config': config,
                'scaler': scaler,
                'metrics': metrics,
                'feature_names': split_info['feature_names'],
                'variant': args.variant,
            }, f)
        print(f"   💾 Saved to {model_path}")

    print("\n" + "=" * 60)
    print("📊 RESULTS")
    print("=" * 60)

    df = pd.DataFrame(all_metrics).T
    df = df.sort_values('roc_auc', ascending=False)
    print(df[['roc_auc', 'recall', 'precision', 'cv_roc_auc_mean', 'train_time_seconds']].to_string())

    best = df['roc_auc'].idxmax()
    print(f"\n🏆 Best: {best} (ROC AUC: {df.loc[best, 'roc_auc']:.3f})")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = RESULTS_DIR / f"neural_{args.variant}_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump({
            k: {kk: (float(vv) if isinstance(vv, (np.floating, float)) else vv)
                for kk, vv in v.items()}
            for k, v in all_metrics.items()
        }, f, indent=2)
    print(f"\n💾 Results saved to {results_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
