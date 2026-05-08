import numpy as np


class DecisionStump:
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left_pred = None
        self.right_pred = None

    def fit(self, X, y, weights, classes):
        n, n_features = X.shape
        K = len(classes)

        class_to_idx = {c: i for i, c in enumerate(classes)}
        yi = np.array([class_to_idx[v] for v in y])

        total_w = np.array([weights[yi == k].sum() for k in range(K)])
        best_err = float('inf')

        for feat in range(n_features):
            order = np.argsort(X[:, feat])
            Xf = X[order, feat]
            yf = yi[order]
            wf = weights[order]
            left_w = np.zeros(K)

            for i in range(n - 1):
                left_w[yf[i]] += wf[i]
                if Xf[i] == Xf[i + 1]:
                    continue
                thr = (Xf[i] + Xf[i + 1]) / 2.0
                right_w = total_w - left_w
                lm = int(np.argmax(left_w))
                rm = int(np.argmax(right_w))
                err = (left_w.sum() - left_w[lm]) + (right_w.sum() - right_w[rm])
                if err < best_err:
                    best_err = err
                    self.feature = feat
                    self.threshold = thr
                    self.left_pred = classes[lm]
                    self.right_pred = classes[rm]

        if self.feature is None:
            self.feature = 0
            self.threshold = float('inf')
            self.left_pred = classes[int(np.argmax(total_w))]
            self.right_pred = self.left_pred
            best_err = 1.0 - np.max(total_w)

        return best_err

    def predict(self, X):
        mask = X[:, self.feature] < self.threshold
        return np.where(mask, self.left_pred, self.right_pred)


class AdaBoost:
    def __init__(self, n_estimators=50, learning_rate=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y)
        n = len(y)

        self.classes_ = np.unique(y)
        K = len(self.classes_)
        weights = np.ones(n) / n

        self.stumps_ = []
        self.alphas_ = []

        for _ in range(self.n_estimators):
            stump = DecisionStump()
            err = stump.fit(X, y, weights, self.classes_)
            err = np.clip(err, 1e-10, 1 - 1e-10)

            if K == 2:
                alpha = self.learning_rate * 0.5 * np.log((1 - err) / err)
            else:
                alpha = self.learning_rate * (np.log((1 - err) / err) + np.log(K - 1))

            pred = stump.predict(X)
            weights *= np.exp(alpha * (pred != y).astype(float))
            weights /= weights.sum()

            self.stumps_.append(stump)
            self.alphas_.append(alpha)

    def predict(self, X):
        X = np.array(X, dtype=float)
        n = len(X)

        vote_scores = {c: np.zeros(n) for c in self.classes_}

        for alpha, stump in zip(self.alphas_, self.stumps_):
            pred = stump.predict(X)
            for i, p in enumerate(pred):
                vote_scores[p][i] += alpha

        score_mat = np.stack([vote_scores[c] for c in self.classes_])
        
        return self.classes_[np.argmax(score_mat, axis=0)]
