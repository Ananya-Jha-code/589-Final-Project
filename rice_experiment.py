"""
Rice Grains Dataset – Experiments
==================================
Algorithms evaluated
  1. k-Nearest Neighbours  (KNN)
  2. Neural Network         (NN)

Both algorithms suit all-numerical data well.

KNN: parameter-free once k is fixed; normalised Euclidean distance is
well-defined for the 7 morphological measurements (area, perimeter, etc.).
Tuning k trades off bias vs. variance directly.

Neural Network: can capture non-linear decision boundaries between
Cammeo and Osmancik species. 3810 instances give enough data to train
a small NN reliably without excessive overfitting.

Evaluation: stratified 10-fold cross-validation, accuracy + F1-score.
At least 6 hyperparameter configurations per algorithm.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'algo'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from knn import KNN
from neural_network import NeuralNetwork

# ── shared helpers ────────────────────────────────────────────────────────────

def normalize(X_train, X_test):
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0)
    std[std == 0] = 1.0
    return (X_train - mean) / std, (X_test - mean) / std


def stratified_kfold(y, k=10, seed=42):
    """Return list of (train_idx, test_idx) pairs for stratified k-fold CV."""
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    fold_test = [[] for _ in range(k)]
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        for i, part in enumerate(np.array_split(idx, k)):
            fold_test[i].extend(part.tolist())
    all_idx = np.arange(len(y))
    folds = []
    for i in range(k):
        test_set = set(fold_test[i])
        folds.append((
            np.array([j for j in all_idx if j not in test_set]),
            np.array(fold_test[i], dtype=int),
        ))
    return folds


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def f1_binary(y_true, y_pred, pos=1):
    tp = int(np.sum((y_true == pos) & (y_pred == pos)))
    fp = int(np.sum((y_true != pos) & (y_pred == pos)))
    fn = int(np.sum((y_true == pos) & (y_pred != pos)))
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2.0 * p * r / (p + r) if (p + r) > 0 else 0.0


# ── data loading ──────────────────────────────────────────────────────────────

def load_rice(data_dir='data'):
    df = pd.read_csv(os.path.join(data_dir, 'rice.csv'))
    feat_cols = [c for c in df.columns if c != 'label']
    X = df[feat_cols].values.astype(float)
    # Cammeo = 0, Osmancik = 1
    y = (df['label'] == 'Osmancik').astype(int).values
    return X, y


# ── KNN experiments ───────────────────────────────────────────────────────────

K_VALUES = [1, 3, 5, 7, 9, 11, 15, 21]   # 8 settings (≥ 6 required)


def knn_cv(X, y, k_val, folds):
    accs, f1s = [], []
    for train_idx, test_idx in folds:
        X_tr, X_te = normalize(X[train_idx], X[test_idx])
        y_tr, y_te = y[train_idx], y[test_idx]
        knn = KNN(k=k_val)
        knn.fit(X_tr, y_tr)
        y_pred = knn.predict(X_te)
        accs.append(accuracy(y_te, y_pred))
        f1s.append(f1_binary(y_te, y_pred))
    return np.mean(accs), np.mean(f1s)


def run_knn(X, y, folds):
    print('\n── KNN on Rice ─────────────────────────────────────────────')
    print(f'  {"k":>4}  {"Accuracy":>10}  {"F1-Score":>10}')
    print('  ' + '-' * 30)
    results = []
    for k in K_VALUES:
        acc, f1 = knn_cv(X, y, k, folds)
        print(f'  {k:>4}  {acc:>10.4f}  {f1:>10.4f}')
        results.append((k, acc, f1))
    return results


def plot_knn(results, out_dir):
    ks  = [r[0] for r in results]
    acc = [r[1] for r in results]
    f1  = [r[2] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, acc, marker='o', linewidth=2, label='Accuracy')
    ax.plot(ks, f1,  marker='s', linewidth=2, label='F1-Score', linestyle='--')
    ax.set_xlabel('k (number of neighbours)', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(
        'KNN on Rice Dataset\nAccuracy and F1-Score vs k  (10-fold stratified CV)',
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ks)
    plt.tight_layout()
    path = os.path.join(out_dir, 'rice_knn_vs_k.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved: {path}')


# ── Neural Network experiments ────────────────────────────────────────────────
#
# Configurations vary: hidden-layer architecture, learning rate (lr),
# L2 regularisation strength (lam), and number of gradient-descent iterations.
# 8 configurations (> 6 required).

NN_CONFIGS = [
    # (layer_sizes,         lr,    lam,    n_iters, label)
    ([7, 16,      1],     0.10,  0.000,  1000,  'arch=[7,16,1]    lr=0.10 λ=0.000'),
    ([7, 32,      1],     0.10,  0.000,  1000,  'arch=[7,32,1]    lr=0.10 λ=0.000'),
    ([7, 64,      1],     0.05,  0.000,  1000,  'arch=[7,64,1]    lr=0.05 λ=0.000'),
    ([7, 16,  8,  1],     0.10,  0.000,  1000,  'arch=[7,16,8,1]  lr=0.10 λ=0.000'),
    ([7, 32, 16,  1],     0.10,  0.000,  1000,  'arch=[7,32,16,1] lr=0.10 λ=0.000'),
    ([7, 32,      1],     0.10,  0.010,  1000,  'arch=[7,32,1]    lr=0.10 λ=0.010'),
    ([7, 32,      1],     0.10,  0.001,  2000,  'arch=[7,32,1]    lr=0.10 λ=0.001 2k'),
    ([7, 16,      1],     0.10,  0.001,  2000,  'arch=[7,16,1]    lr=0.10 λ=0.001 2k'),
]


def nn_cv(X, y, arch, lr, lam, n_iters, folds):
    accs, f1s = [], []
    for train_idx, test_idx in folds:
        X_tr, X_te = normalize(X[train_idx], X[test_idx])
        y_tr, y_te = y[train_idx], y[test_idx]
        np.random.seed(0)   # reproducible weight initialisation per fold
        nn = NeuralNetwork(arch, lam=lam)
        nn.train(X_tr, y_tr, lr=lr, n_iters=n_iters)
        y_pred = nn.predict(X_te)
        accs.append(accuracy(y_te, y_pred))
        f1s.append(f1_binary(y_te, y_pred))
    return np.mean(accs), np.mean(f1s)


def run_nn(X, y, folds):
    print('\n── Neural Network on Rice ──────────────────────────────────')
    print(f'  {"Config":<42}  {"Accuracy":>10}  {"F1-Score":>10}')
    print('  ' + '-' * 68)
    results = []
    for arch, lr, lam, n_iters, label in NN_CONFIGS:
        acc, f1 = nn_cv(X, y, arch, lr, lam, n_iters, folds)
        print(f'  {label:<42}  {acc:>10.4f}  {f1:>10.4f}')
        results.append((arch, lr, lam, n_iters, label, acc, f1))
    return results


def plot_nn_learning_curve(X, y, best_arch, best_lr, best_lam, best_n_iters,
                           out_dir, seed=99):
    """
    Cost J on a held-out test set as a function of training-set size.
    Uses one stratified 80/20 split (fold 0 of 5-fold CV) for speed.
    """
    folds = stratified_kfold(y, k=5, seed=seed)
    train_idx, test_idx = folds[0]

    X_tr_full = X[train_idx]
    y_tr_full = y[train_idx]
    X_te      = X[test_idx]
    y_te      = y[test_idx]

    rng   = np.random.RandomState(7)
    n_max = len(X_tr_full)

    # Build a list of training sizes: 20, 40, ..., n_max
    sizes = list(range(20, n_max, 40)) + [n_max]
    sizes = sorted(set(sizes))

    Js = []
    print(f'\n  Computing NN learning curve ({len(sizes)} points) …')
    for sz in sizes:
        # Stratified subsample of sz examples from the training fold
        classes   = np.unique(y_tr_full)
        counts    = np.array([np.sum(y_tr_full == c) for c in classes])
        allocs    = np.round(sz * counts / counts.sum()).astype(int)
        allocs[0] += sz - allocs.sum()   # fix rounding
        allocs    = np.clip(allocs, 0, counts)
        sub_idx   = []
        for c, n in zip(classes, allocs):
            c_idx = np.where(y_tr_full == c)[0]
            perm  = rng.permutation(len(c_idx))
            sub_idx.append(c_idx[perm[:n]])
        sub_idx = np.concatenate(sub_idx)

        X_sub, y_sub     = X_tr_full[sub_idx], y_tr_full[sub_idx]
        X_sub_n, X_te_n  = normalize(X_sub, X_te)

        np.random.seed(0)
        nn = NeuralNetwork(best_arch, lam=best_lam)
        nn.train(X_sub_n, y_sub, lr=best_lr, n_iters=best_n_iters)
        J = nn.compute_cost(X_te_n, y_te.reshape(-1, 1))
        Js.append(J)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, Js, marker='o', linewidth=2, color='steelblue')
    ax.set_xlabel('Number of training examples', fontsize=12)
    ax.set_ylabel('Cost J  (test set)', fontsize=12)
    arch_str = str(best_arch)
    ax.set_title(
        f'Neural Network Learning Curve – Rice Dataset\n'
        f'arch={arch_str}, lr={best_lr}, λ={best_lam}',
        fontsize=13,
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, 'rice_nn_learning_curve.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved: {path}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(42)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
    os.makedirs(out_dir, exist_ok=True)

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    X, y = load_rice(data_dir)
    print(f'Rice dataset: {X.shape[0]} instances, {X.shape[1]} features')
    print(f'Class distribution: Cammeo={np.sum(y == 0)}  Osmancik={np.sum(y == 1)}')

    folds = stratified_kfold(y, k=10, seed=42)

    # ── KNN ──────────────────────────────────────────────────────────────────
    knn_results = run_knn(X, y, folds)
    plot_knn(knn_results, out_dir)
    best_knn = max(knn_results, key=lambda r: r[1])
    print(f'\n  Best KNN: k={best_knn[0]}  '
          f'Accuracy={best_knn[1]:.4f}  F1={best_knn[2]:.4f}')

    # ── Neural Network ────────────────────────────────────────────────────────
    nn_results = run_nn(X, y, folds)
    best_nn = max(nn_results, key=lambda r: r[5])
    print(f'\n  Best NN: {best_nn[4]}\n'
          f'           Accuracy={best_nn[5]:.4f}  F1={best_nn[6]:.4f}')

    plot_nn_learning_curve(
        X, y,
        best_arch=best_nn[0], best_lr=best_nn[1],
        best_lam=best_nn[2],  best_n_iters=best_nn[3],
        out_dir=out_dir,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n══ Final Summary – Rice Dataset ══════════════════════════════')
    print(f'  KNN  (k={best_knn[0]:>2}):  '
          f'Accuracy = {best_knn[1]:.4f}   F1 = {best_knn[2]:.4f}')
    print(f'  NN   ({best_nn[4]}):')
    print(f'          Accuracy = {best_nn[5]:.4f}   F1 = {best_nn[6]:.4f}')


if __name__ == '__main__':
    main()
