import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt


class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []
        for x in X:
            # euclidean dist
            dist = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            nearest_indices = np.argsort(dist)[:self.k]
            nearest_labels = self.y_train[nearest_indices]
            labels, counts = np.unique(nearest_labels, return_counts=True)
            # majority vote
            predictions.append(labels[np.argmax(counts)])
        return np.array(predictions)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


def normalize(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    # no div by zero
    std[std == 0] = 1
    return (X_train - mean) / std, (X_test - mean) / std


def run_experiment(X, y, k_values, n_runs=20, use_normalization=True):
    train_means, test_means = [], []
    train_stds, test_stds = [], []
    
    for k in k_values:
        train_scores, test_scores = [], []
        
        for run in range(n_runs):
            X_shuffled, y_shuffled = shuffle(X, y, random_state=run)
            X_train, X_test, y_train, y_test = train_test_split(
                X_shuffled, y_shuffled, test_size=0.2, random_state=run
            )
            
            if use_normalization:
                X_train, X_test = normalize(X_train, X_test)
            
            knn = KNN(k)
            knn.fit(X_train, y_train)
            train_scores.append(knn.score(X_train, y_train))
            test_scores.append(knn.score(X_test, y_test))
        
        train_means.append(np.mean(train_scores))
        test_means.append(np.mean(test_scores))
        train_stds.append(np.std(train_scores))
        test_stds.append(np.std(test_scores))
    
    return train_means, test_means, train_stds, test_stds


def plot_results(k_values, accuracies, stds, ylabel, title, filename):
    plt.figure(figsize=(10, 6))
    plt.errorbar(k_values, accuracies, yerr=stds, marker='o', capsize=5, linewidth=2)
    plt.xlabel('Value of k')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename + '.pdf', dpi=300)
    plt.savefig(filename + '.png', dpi=300)
    plt.close()


def main():
    # Load data
    data = pd.read_csv('HW1_CMPSCI_589_Spring2026_Supporting_Files/datasets/wdbc.csv', header=None)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    k_values = list(range(1, 52, 2))  # odd numbers to avoid ties
    
    train_acc, test_acc, train_std, test_std = run_experiment(X, y, k_values, use_normalization=True)
    
    plot_results(k_values, train_acc, train_std,
                 'Accuracy over training data',
                 'k-NN Training Accuracy (normalized)',
                 'q1_1_training_accuracy')
    
    plot_results(k_values, test_acc, test_std,
                 'Accuracy over testing data',
                 'k-NN Testing Accuracy (normalized)',
                 'q1_2_testing_accuracy')
    
    _, test_acc_no_norm, _, test_std_no_norm = run_experiment(X, y, k_values, use_normalization=False)
    
    plot_results(k_values, test_acc_no_norm, test_std_no_norm,
                 'Accuracy over testing data',
                 'k-NN Testing Accuracy (no normalization)',
                 'q1_6_testing_accuracy_no_norm')
    
    best_k = k_values[np.argmax(test_acc)]
    best_k_no_norm = k_values[np.argmax(test_acc_no_norm)]
    print(f"Best k (with normalization): {best_k} (accuracy: {max(test_acc):.4f})")
    print(f"Best k (without normalization): {best_k_no_norm} (accuracy: {max(test_acc_no_norm):.4f})")
    print("Plots saved.")


if __name__ == "__main__":
    main()
