import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

from adaboost import AdaBoost


def load_digits():
    X, y = datasets.load_digits(return_X_y=True)
    return X.astype(float), y


def load_parkinsons():
    df = pd.read_csv('../data/parkinsons.csv')
    y = df['Diagnosis'].values
    X = df.drop(columns=['Diagnosis']).values.astype(float)

    return X, y


def load_rice():
    df = pd.read_csv('../data/rice.csv')
    y = (df['label'] == 'Cammeo').astype(int).values
    X = df.drop(columns=['label']).values.astype(float)

    return X, y


def load_credit():
    df = pd.read_csv('../data/credit_approval.csv')
    y = df['label'].values.astype(int)
    df = df.drop(columns=['label'])

    cat_cols = [c for c in df.columns if '_cat' in c]
    num_cols = [c for c in df.columns if '_num' in c]

    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].mean())

    df = pd.get_dummies(df, columns=cat_cols)

    return df.values.astype(float), y


def stratified_kfold(X, y, k=10, seed=42):
    rng = np.random.RandomState(seed)
    folds = [[] for _ in range(k)]

    for c in np.unique(y):
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
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)

    return float(np.mean(f1s))


def run_cv(n_estimators, X, y, k=10):
    splits = stratified_kfold(X, y, k=k)
    accs, f1s = [], []

    for train_idx, test_idx in splits:
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        m = AdaBoost(n_estimators=n_estimators)
        m.fit(X_tr, y_tr)
        preds = m.predict(X_te)
        accs.append(np.mean(preds == y_te))
        f1s.append(f1_macro(y_te, preds))

    return np.mean(accs), np.std(accs), np.mean(f1s), np.std(f1s)


def plot_learning_curve(X, y, dataset_name):
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(y))
    split = int(0.9 * len(y))
    X_tr, X_te = X[idx[:split]], X[idx[split:]]
    y_tr, y_te = y[idx[:split]], y[idx[split:]]

    steps = list(range(10, 151, 10))
    train_accs, test_accs = [], []
    
    for n in steps:
        m = AdaBoost(n_estimators=n)
        m.fit(X_tr, y_tr)
        train_accs.append(np.mean(m.predict(X_tr) == y_tr))
        test_accs.append(np.mean(m.predict(X_te) == y_te))

    plt.figure()
    plt.plot(steps, train_accs, label='train')
    plt.plot(steps, test_accs, label='test')
    plt.xlabel('n_estimators')
    plt.ylabel('accuracy')
    plt.title(f'AdaBoost learning curve - {dataset_name}')
    plt.legend()
    plt.savefig(f'ec4_adaboost_{dataset_name.lower()}.png', dpi=300)
    plt.close()
    


if __name__ == '__main__':
    datasets_map = {
        'Digits': load_digits,
        'Parkinsons': load_parkinsons,
        'Rice': load_rice,
        'Credit': load_credit,
    }

    n_settings = [10, 25, 50, 75, 100, 150]

    for name, loader in datasets_map.items():
        X, y = loader()
        print(f"\n{name} ({len(X)} samples, {X.shape[1]} features)")

        results = []
        for n in n_settings:
            acc, acc_std, f1, f1_std = run_cv(n, X, y)
            print(f"  n={n}  acc={acc:.4f}  f1={f1:.4f}")
            results.append((n, acc, acc_std, f1, f1_std))

        best = max(results, key=lambda r: r[1])
        print(f"  best: n={best[0]}  acc={best[1]:.4f}  f1={best[3]:.4f}")

        plot_learning_curve(X, y, name)
