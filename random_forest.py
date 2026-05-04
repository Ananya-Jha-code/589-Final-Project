import numpy as np
import pandas as pd
from collections import Counter
from decision_tree import DecisionTree


class RandomForest:
    def __init__(self, ntree, min_size_for_split=2, max_depth=None):
        self.ntree = ntree
        self.min_size_for_split = min_size_for_split
        self.max_depth = max_depth
        self.trees = []

    def _bootstrap(self, X, y):
        n = len(y)
        idx = np.random.randint(0, n, n)
        return X[idx], y[idx]

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.col_names = list(X.columns)
            X = X.values
        y = np.array(y)
        n_features = max(1, int(np.sqrt(X.shape[1])))
        self.trees = []
        for _ in range(self.ntree):
            bX, by = self._bootstrap(X, y)
            tree = DecisionTree(min_size_for_split=self.min_size_for_split,
                                max_depth=self.max_depth,
                                n_features=n_features)
            tree.attribute_names = self.col_names
            tree.fit(pd.DataFrame(bX, columns=self.col_names), by)
            self.trees.append(tree)

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        # shape: (ntree, n_samples)
        all_preds = np.array([t.predict(X) for t in self.trees])
        result = []
        for j in range(all_preds.shape[1]):
            result.append(Counter(all_preds[:, j]).most_common(1)[0][0])
        return np.array(result)
