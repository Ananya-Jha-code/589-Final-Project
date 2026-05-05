"""
Credit Approval Dataset – Experiments
=======================================
Algorithms evaluated
  1. Decision Tree  (DT)
  2. k-Nearest Neighbours  (KNN, with one-hot encoding)

Algorithm justification
  Decision Tree: The credit dataset has 9 categorical features and 6
  numerical features.  The existing DecisionTree implementation splits on
  discrete feature values using information gain, making it a natural fit
  for categorical data.  Numerical features are binned into quartile-based
  categories before training, so the tree sees only a small, fixed set of
  values per column.  The early-stop threshold is the main hyperparameter
  and controls tree depth / overfitting.

  KNN: After one-hot encoding all categorical features and z-score
  normalising the numerical features, every instance is a real-valued
  vector on which Euclidean distance is well-defined.  KNN is a strong
  non-parametric baseline for mixed-type data once the encoding is done.
  With only 653 instances a neural network would be at greater risk of
  overfitting, while KNN generalises smoothly as k increases.

Evaluation: stratified 10-fold CV, accuracy + F1-score.
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

from decision_tree import DecisionTree
from knn import KNN

# ── shared helpers ────────────────────────────────────────────────────────────

def stratified_kfold(y, k=10, seed=42):
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


# ── column definitions ────────────────────────────────────────────────────────

CAT_COLS = [
    'attr1_cat', 'attr4_cat', 'attr5_cat', 'attr6_cat', 'attr7_cat',
    'attr9_cat', 'attr10_cat', 'attr11_cat', 'attr12_cat', 'attr13_cat',
]
NUM_COLS = ['attr2_num', 'attr3_num', 'attr8_num', 'attr14_num', 'attr15_num']


# ── data loading ──────────────────────────────────────────────────────────────

def load_credit(data_dir='data'):
    df = pd.read_csv(os.path.join(data_dir, 'credit_approval.csv'))
    X_df = df.drop(columns=['label'])
    y    = df['label'].values.astype(int)
    return X_df, y


# ── preprocessing ─────────────────────────────────────────────────────────────

def bin_numericals(X_train_df, X_test_df, n_bins=4):
    """
    Bin each numerical column into quartile-based discrete categories using
    only training-set statistics (no data leakage).  Resulting values are
    strings so the DecisionTree treats them as categories.
    """
    X_tr = X_train_df.copy()
    X_te = X_test_df.copy()
    for col in NUM_COLS:
        vals_tr = X_tr[col].astype(float).values
        # n_bins-1 interior quantile boundaries → n_bins bins
        boundaries = np.unique(
            np.percentile(vals_tr, np.linspace(0, 100, n_bins + 1)[1:-1])
        )
        X_tr[col] = np.searchsorted(boundaries, vals_tr).astype(str)
        X_te[col] = np.searchsorted(
            boundaries, X_te[col].astype(float).values
        ).astype(str)
    return X_tr, X_te


def one_hot_and_normalize(X_train_df, X_test_df):
    """
    One-hot encode categorical columns (categories determined from training
    set only) and z-score normalise all columns.  Returns numpy arrays.
    """
    cat_mapping = {
        col: sorted(X_train_df[col].astype(str).unique())
        for col in CAT_COLS
    }

    def encode(df):
        parts = []
        for col in CAT_COLS:
            for cat in cat_mapping[col]:
                parts.append((df[col].astype(str) == cat).astype(float).values)
        for col in NUM_COLS:
            parts.append(df[col].astype(float).values)
        return np.column_stack(parts)

    X_tr = encode(X_train_df)
    X_te = encode(X_test_df)

    mean = X_tr.mean(axis=0)
    std  = X_tr.std(axis=0)
    std[std == 0] = 1.0
    return (X_tr - mean) / std, (X_te - mean) / std


# ── Decision Tree experiments ──────────────────────────────────────────────────
#
# The early-stop threshold stops splitting when a node's dominant class
# already covers that fraction of examples.  Lower values (looser stopping)
# allow a deeper, potentially overfit tree; higher values prune earlier.
# None = grow until pure or no attributes remain.
# 7 settings (≥ 6 required).

DT_THRESHOLDS = [None, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]


def dt_cv(X_df, y, threshold, folds, n_bins=4):
    accs, f1s = [], []
    for train_idx, test_idx in folds:
        X_tr_df = X_df.iloc[train_idx].reset_index(drop=True)
        X_te_df = X_df.iloc[test_idx].reset_index(drop=True)
        y_tr    = y[train_idx]
        y_te    = y[test_idx]

        # Bin numericals using only training-fold statistics
        X_tr_bin, X_te_bin = bin_numericals(X_tr_df, X_te_df, n_bins=n_bins)

        dt = DecisionTree(
            criterion='information_gain',
            early_stop_threshold=threshold,
        )
        dt.fit(X_tr_bin, pd.Series(y_tr))
        y_pred = dt.predict(X_te_bin)

        accs.append(accuracy(y_te, y_pred))
        f1s.append(f1_binary(y_te, y_pred))
    return np.mean(accs), np.mean(f1s)


def run_dt(X_df, y, folds):
    print('\n── Decision Tree on Credit Approval ────────────────────────')
    print(f'  {"Threshold":<12}  {"Accuracy":>10}  {"F1-Score":>10}')
    print('  ' + '-' * 38)
    results = []
    for thresh in DT_THRESHOLDS:
        acc, f1 = dt_cv(X_df, y, thresh, folds)
        label = 'None' if thresh is None else f'{thresh:.2f}'
        print(f'  {label:<12}  {acc:>10.4f}  {f1:>10.4f}')
        results.append((thresh, acc, f1))
    return results


def plot_dt(results, out_dir):
    labels = ['None' if r[0] is None else str(r[0]) for r in results]
    acc    = [r[1] for r in results]
    f1     = [r[2] for r in results]
    x      = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, acc, marker='o', linewidth=2, label='Accuracy')
    ax.plot(x, f1,  marker='s', linewidth=2, label='F1-Score', linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Early-Stop Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(
        'Decision Tree on Credit Approval\n'
        'Accuracy and F1-Score vs Early-Stop Threshold  (10-fold stratified CV)',
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, 'credit_dt_threshold.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved: {path}')


# ── KNN experiments ───────────────────────────────────────────────────────────
#
# 8 values of k (≥ 6 required).  Odd values avoid ties.

KNN_K_VALUES = [1, 3, 5, 7, 9, 11, 15, 21]


def knn_cv(X_df, y, k_val, folds):
    accs, f1s = [], []
    for train_idx, test_idx in folds:
        X_tr_df = X_df.iloc[train_idx].reset_index(drop=True)
        X_te_df = X_df.iloc[test_idx].reset_index(drop=True)
        y_tr    = y[train_idx]
        y_te    = y[test_idx]

        # One-hot encode categoricals + normalise numericals
        X_tr, X_te = one_hot_and_normalize(X_tr_df, X_te_df)

        knn = KNN(k=k_val)
        knn.fit(X_tr, y_tr)
        y_pred = knn.predict(X_te)

        accs.append(accuracy(y_te, y_pred))
        f1s.append(f1_binary(y_te, y_pred))
    return np.mean(accs), np.mean(f1s)


def run_knn(X_df, y, folds):
    print('\n── KNN on Credit Approval ──────────────────────────────────')
    print(f'  {"k":>4}  {"Accuracy":>10}  {"F1-Score":>10}')
    print('  ' + '-' * 30)
    results = []
    for k in KNN_K_VALUES:
        acc, f1 = knn_cv(X_df, y, k, folds)
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
        'KNN on Credit Approval\n'
        'Accuracy and F1-Score vs k  (10-fold stratified CV, one-hot + normalised)',
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ks)
    plt.tight_layout()
    path = os.path.join(out_dir, 'credit_knn_vs_k.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved: {path}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(42)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir  = os.path.join(base_dir, 'figures')
    data_dir = os.path.join(base_dir, 'data')
    os.makedirs(out_dir, exist_ok=True)

    X_df, y = load_credit(data_dir)
    print(f'Credit Approval dataset: {len(y)} instances, '
          f'{X_df.shape[1]} features  '
          f'({len(CAT_COLS)} categorical, {len(NUM_COLS)} numerical)')
    print(f'Class distribution: rejected (0)={np.sum(y == 0)}  '
          f'approved (1)={np.sum(y == 1)}')

    folds = stratified_kfold(y, k=10, seed=42)

    # ── Decision Tree ─────────────────────────────────────────────────────────
    dt_results = run_dt(X_df, y, folds)
    plot_dt(dt_results, out_dir)
    best_dt = max(dt_results, key=lambda r: r[1])
    dt_label = 'None' if best_dt[0] is None else str(best_dt[0])
    print(f'\n  Best DT: threshold={dt_label}  '
          f'Accuracy={best_dt[1]:.4f}  F1={best_dt[2]:.4f}')

    # ── KNN ───────────────────────────────────────────────────────────────────
    knn_results = run_knn(X_df, y, folds)
    plot_knn(knn_results, out_dir)
    best_knn = max(knn_results, key=lambda r: r[1])
    print(f'\n  Best KNN: k={best_knn[0]}  '
          f'Accuracy={best_knn[1]:.4f}  F1={best_knn[2]:.4f}')

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n══ Final Summary – Credit Approval Dataset ═══════════════════')
    print(f'  Decision Tree (threshold={dt_label:<4}): '
          f'Accuracy = {best_dt[1]:.4f}   F1 = {best_dt[2]:.4f}')
    print(f'  KNN           (k={best_knn[0]:<2}):         '
          f'Accuracy = {best_knn[1]:.4f}   F1 = {best_knn[2]:.4f}')


if __name__ == '__main__':
    main()
