import numpy as np


class GaussianNaiveBayes:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y)
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_feats = X.shape[1]

        self.priors = np.zeros(n_classes)
        self.means = np.zeros((n_classes, n_feats))
        self.vars = np.zeros((n_classes, n_feats))

        for i, c in enumerate(self.classes):
            mask = y == c
            self.priors[i] = mask.sum() / len(y)
            self.means[i] = X[mask].mean(axis=0)
            self.vars[i] = X[mask].var(axis=0) + self.var_smoothing

    def predict(self, X):
        X = np.array(X, dtype=float)
        preds = []
        for x in X:
            scores = []
            for i in range(len(self.classes)):
                log_p = np.log(self.priors[i])
                log_p += -0.5 * np.sum(np.log(2 * np.pi * self.vars[i]) + (x - self.means[i])**2 / self.vars[i])
                scores.append(log_p)
            preds.append(self.classes[np.argmax(scores)])
        return np.array(preds)
