import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from gaussian_naive_bayes import GaussianNaiveBayes
from knn import KNN


def load_abalone():
    data = fetch_openml('abalone', version=1, as_frame=True, parser='auto')
    df = data.data.copy()
    rings = data.target.astype(int).values

    sex_dummies = pd.get_dummies(df['Sex'], prefix='Sex')
    num_cols = [c for c in df.columns if c != 'Sex']
    X = pd.concat([sex_dummies, df[num_cols].astype(float)], axis=1).values.astype(float)

    y = np.digitize(rings, bins=[9, 14])
    return X, y


def normalize(X_train, X_test):
    mu = X_train.mean(axis=0)
    sig = X_train.std(axis=0)
    sig[sig == 0] = 1.0
    return (X_train - mu) / sig, (X_test - mu) / sig


def stratified_kfold(X, y, k=10, seed=42):
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    folds = [[] for _ in range(k)]
    for c in classes:
        idx = np.where(y == c)[0].copy()
        rng.shuffle(idx)
        for i, s in enumerate(idx):
            folds[i % k].append(s)
    splits = []
    for i in range(k):
        test = np.array(folds[i])
        train = np.concatenate([np.array(folds[j]) for j in range(k) if j != i])
        splits.append((train.astype(int), test.astype(int)))
    return splits


def f1_macro(y_true, y_pred):
    f1s = []
    for c in np.unique(y_true):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0)
    return float(np.mean(f1s))


def run_cv(model_fn, X, y, k=10, use_norm=False):
    splits = stratified_kfold(X, y, k=k)
    accs, f1s = [], []
    for train_idx, test_idx in splits:
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        if use_norm:
            X_tr, X_te = normalize(X_tr, X_te)
        m = model_fn()
        m.fit(X_tr, y_tr)
        preds = m.predict(X_te)
        accs.append(np.mean(preds == y_te))
        f1s.append(f1_macro(y_te, preds))
    return np.mean(accs), np.std(accs), np.mean(f1s), np.std(f1s)



if __name__ == '__main__':
    X, y = load_abalone()
    print(f"abalone: {len(X)} samples, {X.shape[1]} features, classes={np.unique(y)}")
    vals, counts = np.unique(y, return_counts=True)
    print(f"class counts: { {int(v): int(c) for v, c in zip(vals, counts)} }")

    print("\n\nkNN:")
    k_values = [1, 3, 5, 7, 11, 15]
    knn_results = []
    
    for k in k_values:
        acc, acc_std, f1, f1_std = run_cv(lambda k=k: KNN(k=k), X, y, use_norm=True)
        print(f"  k={k:2d}  acc={acc:.4f}  f1={f1:.4f}")
        knn_results.append((k, acc, acc_std, f1, f1_std))

    print("\n\nGaussian Naive Bayes:")
    smoothing_vals = [1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 1.0]
    gnb_results = []

    for sv in smoothing_vals:
        acc, acc_std, f1, f1_std = run_cv(lambda sv=sv: GaussianNaiveBayes(var_smoothing=sv), X, y)
        print(f"  var_smoothing={sv:.0e}  acc={acc:.4f}  f1={f1:.4f}")
        gnb_results.append((sv, acc, acc_std, f1, f1_std))

    plt.figure()
    plt.plot(k_values, [r[1] for r in knn_results], marker='o')
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.title('kNN on Abalone')
    plt.savefig('ec2_knn_abalone.png', dpi=300)
    plt.close()

    sv_labels = [f'{v:.0e}' for v in smoothing_vals]
    plt.figure()
    plt.plot(range(len(sv_labels)), [r[1] for r in gnb_results], marker='o')
    plt.xticks(range(len(sv_labels)), sv_labels)
    plt.xlabel('var_smoothing')
    plt.ylabel('accuracy')
    plt.title('Gaussian NB on Abalone')
    plt.savefig('ec2_gnb_abalone.png', dpi=300)
    plt.close()

    print("\nplots saved")

    bk = max(knn_results, key=lambda r: r[1])
    bg = max(gnb_results, key=lambda r: r[1])
    print(f"\nbest kNN: k={bk[0]}  acc={bk[1]:.4f}  f1={bk[3]:.4f}")
    print(f"best GNB: smoothing={bg[0]:.0e}  acc={bg[1]:.4f}  f1={bg[3]:.4f}")
