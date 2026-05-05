import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class NeuralNetwork:
    def __init__(self, layer_sizes, lam=0.0):
        self.layer_sizes = layer_sizes
        self.lam = lam
        self.weights = []
        for i in range(len(layer_sizes) - 1):
            n_in = layer_sizes[i] + 1
            n_out = layer_sizes[i + 1]
            W = np.random.uniform(-1, 1, (n_out, n_in))
            self.weights.append(W)

    def forward_single(self, x):
        a = np.concatenate([[1.0], x])
        activations = [a]
        zs = []
        for l, W in enumerate(self.weights):
            z = W @ a
            zs.append(z)
            a_new = sigmoid(z)
            if l < len(self.weights) - 1:
                a = np.concatenate([[1.0], a_new])
            else:
                a = a_new
            activations.append(a)
        return activations, zs

    def forward_batch(self, X):
        n = len(X)
        a = np.hstack([np.ones((n, 1)), X])
        activations = [a]
        for l, W in enumerate(self.weights):
            z = a @ W.T
            a_new = sigmoid(z)
            if l < len(self.weights) - 1:
                a = np.hstack([np.ones((n, 1)), a_new])
            else:
                a = a_new
            activations.append(a)
        return activations

    def compute_cost(self, X, y):
        n = len(X)
        activations = self.forward_batch(X)
        out = activations[-1]
        y_arr = y.reshape(n, -1)
        J = np.sum(-y_arr * np.log(out + 1e-15) - (1 - y_arr) * np.log(1 - out + 1e-15)) / n
        reg = sum(np.sum(W[:, 1:] ** 2) for W in self.weights)
        return J + (self.lam / (2 * n)) * reg

    def backprop_batch(self, X, y):
        n = len(X)
        y_arr = y.reshape(n, -1)
        activations = self.forward_batch(X)
        out = activations[-1]

        deltas = [None] * len(self.weights)
        deltas[-1] = out - y_arr

        for l in range(len(self.weights) - 2, -1, -1):
            a_hidden = activations[l + 1][:, 1:]
            sig_d = a_hidden * (1.0 - a_hidden)
            W_no_bias = self.weights[l + 1][:, 1:]
            deltas[l] = (deltas[l + 1] @ W_no_bias) * sig_d

        grads = []
        for l in range(len(self.weights)):
            grad = deltas[l].T @ activations[l] / n
            reg = (self.lam / n) * self.weights[l].copy()
            reg[:, 0] = 0.0
            grads.append(grad + reg)

        return grads

    def train(self, X, y, lr=0.1, n_iters=500):
        for _ in range(n_iters):
            grads = self.backprop_batch(X, y)
            for l in range(len(self.weights)):
                self.weights[l] -= lr * grads[l]

    def predict(self, X):
        activations = self.forward_batch(X)
        out = activations[-1]
        if out.shape[1] == 1:
            return (out[:, 0] >= 0.5).astype(int)
        return np.argmax(out, axis=1)
