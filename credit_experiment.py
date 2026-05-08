"""
Credit Approval Dataset – Experiments + Extra Credit
====================================================
Base algorithms evaluated
  1. Decision Tree (DT)
  2. k-Nearest Neighbours (KNN)
  3. Gaussian Naive Bayes (GNB)   [EC1 additional algorithm]

Ensemble evaluated (EC3)
  - Heterogeneous bootstrap ensemble with majority voting:
    DT + KNN + GNB + Random Forest

Evaluation
  - Stratified 10-fold cross-validation
  - Metrics: accuracy and F1-score
  - Plots + table + write-up exported under figures/
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'algo'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

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


def confusion_binary(y_true, y_pred, pos=1):
    tp = int(np.sum((y_true == pos) & (y_pred == pos)))
    fp = int(np.sum((y_true != pos) & (y_pred == pos)))
    fn = int(np.sum((y_true == pos) & (y_pred != pos)))
    tn = int(np.sum((y_true != pos) & (y_pred != pos)))
    return tp, fp, fn, tn


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
    print('\n-- Decision Tree on Credit Approval ------------------------')
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
    print('\n-- KNN on Credit Approval ----------------------------------')
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


# ── Gaussian Naive Bayes experiments (EC1 additional algorithm) ──────────────
#
# 7 variance smoothing settings (>= 6 required).

GNB_SMOOTH_VALUES = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6]


def gnb_cv(X_df, y, var_smoothing, folds):
    accs, f1s = [], []
    for train_idx, test_idx in folds:
        X_tr_df = X_df.iloc[train_idx].reset_index(drop=True)
        X_te_df = X_df.iloc[test_idx].reset_index(drop=True)
        y_tr = y[train_idx]
        y_te = y[test_idx]

        X_tr, X_te = one_hot_and_normalize(X_tr_df, X_te_df)
        gnb = GaussianNB(var_smoothing=var_smoothing)
        gnb.fit(X_tr, y_tr)
        y_pred = gnb.predict(X_te)

        accs.append(accuracy(y_te, y_pred))
        f1s.append(f1_binary(y_te, y_pred))
    return np.mean(accs), np.mean(f1s)


def run_gnb(X_df, y, folds):
    print('\n-- Gaussian Naive Bayes on Credit Approval ------------------')
    print(f'  {"Var smoothing":<16}  {"Accuracy":>10}  {"F1-Score":>10}')
    print('  ' + '-' * 42)
    results = []
    for smooth in GNB_SMOOTH_VALUES:
        acc, f1 = gnb_cv(X_df, y, smooth, folds)
        print(f'  {smooth:<16.1e}  {acc:>10.4f}  {f1:>10.4f}')
        results.append((smooth, acc, f1))
    return results


def plot_gnb(results, out_dir):
    smooth_vals = [r[0] for r in results]
    acc = [r[1] for r in results]
    f1 = [r[2] for r in results]
    x = np.arange(len(smooth_vals))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, acc, marker='o', linewidth=2, label='Accuracy')
    ax.plot(x, f1, marker='s', linewidth=2, label='F1-Score', linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{v:.0e}' for v in smooth_vals], rotation=20)
    ax.set_xlabel('GaussianNB var_smoothing', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(
        'Gaussian Naive Bayes on Credit Approval\n'
        'Accuracy and F1-Score vs var_smoothing (10-fold stratified CV)',
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, 'credit_gnb_vs_smoothing.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved: {path}')


# ── Bootstrap heterogeneous ensemble (EC3) ───────────────────────────────────

def bootstrap_indices(n_samples, rng):
    return rng.randint(0, n_samples, size=n_samples)


def majority_vote_matrix(pred_matrix):
    # pred_matrix shape: (n_models, n_samples)
    n_samples = pred_matrix.shape[1]
    out = np.zeros(n_samples, dtype=int)
    for j in range(n_samples):
        counts = Counter(pred_matrix[:, j])
        max_count = max(counts.values())
        tied = [lbl for lbl, c in counts.items() if c == max_count]
        out[j] = min(tied)  # deterministic tie-break
    return out


def ensemble_predict_fold(X_tr_df, y_tr, X_te_df, rng):
    n = len(y_tr)

    # Shared preprocessings
    X_tr_bin_base, X_te_bin = bin_numericals(X_tr_df, X_te_df, n_bins=4)
    X_tr_num_base, X_te_num = one_hot_and_normalize(X_tr_df, X_te_df)

    model_preds = []

    # 1) Decision Tree (bootstrap on binned representation)
    idx_dt = bootstrap_indices(n, rng)
    dt = DecisionTree(criterion='information_gain', early_stop_threshold=0.70)
    dt.fit(X_tr_bin_base.iloc[idx_dt].reset_index(drop=True), pd.Series(y_tr[idx_dt]))
    model_preds.append(dt.predict(X_te_bin))

    # 2) KNN (bootstrap on encoded + normalized representation)
    idx_knn = bootstrap_indices(n, rng)
    knn = KNN(k=11)
    knn.fit(X_tr_num_base[idx_knn], y_tr[idx_knn])
    model_preds.append(knn.predict(X_te_num))

    # 3) Gaussian Naive Bayes (bootstrap on encoded + normalized representation)
    idx_gnb = bootstrap_indices(n, rng)
    gnb = GaussianNB(var_smoothing=1e-9)
    gnb.fit(X_tr_num_base[idx_gnb], y_tr[idx_gnb])
    model_preds.append(gnb.predict(X_te_num))

    # 4) Random Forest (bootstrap internal + explicit external bootstrap)
    idx_rf = bootstrap_indices(n, rng)
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=int(rng.randint(0, 10_000_000)),
    )
    rf.fit(X_tr_num_base[idx_rf], y_tr[idx_rf])
    model_preds.append(rf.predict(X_te_num))

    pred_matrix = np.vstack(model_preds)
    return majority_vote_matrix(pred_matrix)


def ensemble_cv(X_df, y, folds, seed=42):
    rng = np.random.RandomState(seed)
    accs, f1s = [], []
    oof_pred = np.zeros_like(y)
    for train_idx, test_idx in folds:
        X_tr_df = X_df.iloc[train_idx].reset_index(drop=True)
        X_te_df = X_df.iloc[test_idx].reset_index(drop=True)
        y_tr = y[train_idx]
        y_te = y[test_idx]

        pred = ensemble_predict_fold(X_tr_df, y_tr, X_te_df, rng)
        oof_pred[test_idx] = pred
        accs.append(accuracy(y_te, pred))
        f1s.append(f1_binary(y_te, pred))

    return {
        'accuracy': float(np.mean(accs)),
        'f1': float(np.mean(f1s)),
        'oof_pred': oof_pred,
    }


def plot_model_comparison(rows, out_dir):
    labels = [r['model'] for r in rows]
    acc = [r['accuracy'] for r in rows]
    f1 = [r['f1_score'] for r in rows]
    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, acc, width=width, label='Accuracy')
    ax.bar(x + width / 2, f1, width=width, label='F1-Score')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Credit Approval: Best model scores + ensemble', fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, 'credit_model_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved: {path}')


def plot_ensemble_confusion(y_true, y_pred, out_dir):
    tp, fp, fn, tn = confusion_binary(y_true, y_pred, pos=1)
    matrix = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap='Blues')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Pred 0', 'Pred 1'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['True 0', 'True 1'])
    ax.set_title('Ensemble Confusion Matrix (OOF predictions)')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center', color='black')
    plt.tight_layout()
    path = os.path.join(out_dir, 'credit_ensemble_confusion_matrix.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved: {path}')


def export_results_artifacts(rows, out_dir):
    table_df = pd.DataFrame(rows)
    table_path = os.path.join(out_dir, 'credit_evaluation_table.csv')
    table_df.to_csv(table_path, index=False)
    print(f'  Saved: {table_path}')

    md_header = '| Model | Accuracy | F1-score |'
    md_sep = '|---|---:|---:|'
    md_rows = [
        f'| {r["model"]} | {r["accuracy"]:.4f} | {r["f1_score"]:.4f} |'
        for r in rows
    ]

    report_lines = [
        '# Credit Approval Extra Credit Write-up',
        '',
        '## EC1: Additional algorithm',
        'Added Gaussian Naive Bayes as a third standalone model in addition to',
        'Decision Tree and KNN. All algorithms were evaluated with stratified',
        '10-fold CV using accuracy and F1-score.',
        '',
        '## EC3: Ensemble',
        'Built a heterogeneous ensemble composed of:',
        '- Decision Tree',
        '- KNN',
        '- Gaussian Naive Bayes',
        '- Random Forest',
        '',
        'For each CV fold, every model is trained on its own bootstrap sample',
        'drawn from the training fold. Final predictions are produced by',
        'majority voting.',
        '',
        '## Results table',
        md_header,
        md_sep,
        *md_rows,
        '',
        '## Brief interpretation',
        '- Ensemble voting improves robustness by averaging model-specific errors.',
        '- If ensemble accuracy/F1 exceeds individual baselines, it indicates',
        '  complementary decision boundaries across model families.',
        '- If one standalone model remains best, this suggests low disagreement',
        '  among models or insufficient diversity in the base learners.',
    ]

    report_path = os.path.join(out_dir, 'credit_extra_credit_writeup.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines) + '\n')
    print(f'  Saved: {report_path}')


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

    # ── Gaussian Naive Bayes (EC1) ───────────────────────────────────────────
    gnb_results = run_gnb(X_df, y, folds)
    plot_gnb(gnb_results, out_dir)
    best_gnb = max(gnb_results, key=lambda r: r[1])
    print(f'\n  Best GNB: var_smoothing={best_gnb[0]:.1e}  '
          f'Accuracy={best_gnb[1]:.4f}  F1={best_gnb[2]:.4f}')

    # ── Ensemble (EC3) ───────────────────────────────────────────────────────
    print('\n-- Bootstrap Heterogeneous Ensemble --------------------------')
    ensemble_res = ensemble_cv(X_df, y, folds, seed=42)
    print(f'  Ensemble Accuracy={ensemble_res["accuracy"]:.4f}  '
          f'F1={ensemble_res["f1"]:.4f}')
    plot_ensemble_confusion(y, ensemble_res['oof_pred'], out_dir)

    # ── Final table + chart + write-up ───────────────────────────────────────
    rows = [
        {
            'model': f'Decision Tree (thr={dt_label})',
            'accuracy': float(best_dt[1]),
            'f1_score': float(best_dt[2]),
        },
        {
            'model': f'KNN (k={best_knn[0]})',
            'accuracy': float(best_knn[1]),
            'f1_score': float(best_knn[2]),
        },
        {
            'model': f'GaussianNB (vs={best_gnb[0]:.0e})',
            'accuracy': float(best_gnb[1]),
            'f1_score': float(best_gnb[2]),
        },
        {
            'model': 'Ensemble (DT+KNN+GNB+RF)',
            'accuracy': float(ensemble_res['accuracy']),
            'f1_score': float(ensemble_res['f1']),
        },
    ]
    plot_model_comparison(rows, out_dir)
    export_results_artifacts(rows, out_dir)

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n== Final Summary - Credit Approval Dataset ===================')
    print(f'  Decision Tree (threshold={dt_label:<4}): '
          f'Accuracy = {best_dt[1]:.4f}   F1 = {best_dt[2]:.4f}')
    print(f'  KNN           (k={best_knn[0]:<2}):         '
          f'Accuracy = {best_knn[1]:.4f}   F1 = {best_knn[2]:.4f}')
    print(f'  GaussianNB    (var_smoothing={best_gnb[0]:.0e}): '
          f'Accuracy = {best_gnb[1]:.4f}   F1 = {best_gnb[2]:.4f}')
    print(f'  Ensemble      (DT+KNN+GNB+RF): '
          f'Accuracy = {ensemble_res["accuracy"]:.4f}   F1 = {ensemble_res["f1"]:.4f}')


if __name__ == '__main__':
    main()
