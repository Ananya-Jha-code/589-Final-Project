import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter


class TreeNode:
    def __init__(self, attribute=None, branches=None, label=None, majority=None):
        self.attribute = attribute
        self.label = label
        self.majority = majority  # fallback for unseen values during prediction
        
        if branches is None:
            self.branches = {}
        else:
            self.branches = branches

    def is_leaf(self):
        return self.label is not None


class DecisionTree:
    def __init__(self, criterion='information_gain', early_stop_threshold=None):
        self.root = None
        self.criterion = criterion
        self.early_stop_threshold = early_stop_threshold
        self.attribute_names = None

    def entropy(self, y):
        if len(y) == 0:
            return 0
        
        counts = Counter(y)
        total = len(y)
        probs = np.array(list(counts.values())) / total
        
        probs = probs[probs > 0]  # no log(0)
        log_probs = np.log2(probs)
        entropy_value = -np.sum(probs * log_probs)
        return entropy_value

    def gini(self, y):
        if len(y) == 0:
            return 0
        
        counts = Counter(y)
        total = len(y)
        probs = np.array(list(counts.values())) / total
        
        gini_value = 1 - np.sum(probs ** 2)
        return gini_value

    def impurity(self, y):
        if self.criterion == 'gini':
            return self.gini(y)
        else:
            return self.entropy(y)

    # Calculating information gain
    def information_gain(self, X, y, attribute):
        parent_impurity = self.impurity(y)

        child_impurity = 0
        values = X[attribute].unique()
        
        for value in values:
            mask = X[attribute] == value
            subset_y = y[mask]
            weight = len(subset_y) / len(y)
            child_impurity += weight * self.impurity(subset_y)
        gain = parent_impurity - child_impurity
        return gain

    def majority_class(self, y):
        counts = Counter(y)
        most_common = counts.most_common(1)
        return most_common[0][0]

    # Recursively builds decision tree by finding the best attribute to split on and splitting the data.
    def build_tree(self, X, y, attributes):
        unique_labels = np.unique(y)
        
        if len(unique_labels) == 1:
            if isinstance(y, pd.Series):
                label = y.iloc[0]
            else:
                label = y[0]
            return TreeNode(label=label)

        if len(attributes) == 0:
            return TreeNode(label=self.majority_class(y))
        # stop if most samples are the same class
        if self.early_stop_threshold is not None:
            counts = Counter(y)
            max_count = max(counts.values())
            fraction = max_count / len(y)
            
            if fraction >= self.early_stop_threshold:
                return TreeNode(label=self.majority_class(y))

        if len(y) == 0:
            return TreeNode(label=None)

        # find best attribute to split on
        gains = []
        for attr in attributes:
            gains.append(self.information_gain(X, y, attr))
        
        best_attr = attributes[np.argmax(gains)]

        node = TreeNode(attribute=best_attr, majority=self.majority_class(y))
        
        remaining_attrs = [a for a in attributes if a != best_attr]

        values = X[best_attr].unique()
        # split on each unique value
        for value in values:
            mask = X[best_attr] == value
            subset_X = X[mask]
            subset_y = y[mask]

            if len(subset_y) == 0:
                node.branches[value] = TreeNode(label=self.majority_class(y))
            else:
                child_node = self.build_tree(subset_X, subset_y, remaining_attrs)
                node.branches[value] = child_node

        return node

    def fit(self, X, y):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        self.attribute_names = list(X.columns)
        self.root = self.build_tree(X, y, self.attribute_names)

    def predict_single(self, x, node):
        if node.is_leaf():
            return node.label

        attr_value = x[node.attribute]
        
        if attr_value in node.branches:
            next_node = node.branches[attr_value]
            return self.predict_single(x, next_node)
        
        return node.majority

    
    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.attribute_names)

        predictions = []
        for index, row in X.iterrows():
            pred = self.predict_single(row, self.root)
            predictions.append(pred)
    
        return np.array(predictions)
    
    
    def score(self, X, y):
        predictions = self.predict(X)
        
        if isinstance(y, pd.Series):
            y = y.values
        
        correct = predictions == y
        accuracy = np.mean(correct)
        return accuracy



# Runs the experiment n times where each time the data is being shuffled and is being split into a train and test set.
def run_experiment(X, y, n_runs=100, criterion='information_gain', early_stop_threshold=None):
    train_accs = []
    test_accs = []

    for run in range(n_runs):
        X_shuffled, y_shuffled = shuffle(X, y, random_state=run)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_shuffled, y_shuffled, test_size=0.2, random_state=run
        )
        # reset indices after split
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        
        dt = DecisionTree(criterion=criterion, early_stop_threshold=early_stop_threshold)
        dt.fit(X_train, y_train)

        train_acc = dt.score(X_train, y_train)
        test_acc = dt.score(X_test, y_test)
        
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        if (run + 1) % 20 == 0:
            print(f"  {run + 1}/{n_runs} runs done")

    return train_accs, test_accs


def plot_histogram(accuracies, title, filename):
    mean = np.mean(accuracies)
    std = np.std(accuracies)

    plt.figure(figsize=(10, 6))
    plt.hist(accuracies, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    plt.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean:.4f}')
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.title(f'{title}\nMean: {mean:.4f}, Std: {std:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(filename + '.pdf', dpi=300)
    plt.savefig(filename + '.png', dpi=300)
    plt.close()


def print_results(name, train_accs, test_accs):
    train_mean = np.mean(train_accs)
    train_std = np.std(train_accs)
    test_mean = np.mean(test_accs)
    test_std = np.std(test_accs)
    
    print(f"\n{name}:")
    print(f"  Train: {train_mean:.4f} +/- {train_std:.4f}")
    print(f"  Test:  {test_mean:.4f} +/- {test_std:.4f}")


def main():
    data = pd.read_csv('HW1_CMPSCI_589_Spring2026_Supporting_Files/datasets/car.csv')
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    print(f"Loaded {len(X)} instances, {X.shape[1]} features")

    print("\nRunning with Information Gain...")
    train_accs, test_accs = run_experiment(X, y, n_runs=100, criterion='information_gain')
    print_results("Information Gain", train_accs, test_accs)
    plot_histogram(train_accs, 'Decision Tree Training Accuracy', 'q2_1_training_histogram')
    plot_histogram(test_accs, 'Decision Tree Testing Accuracy', 'q2_2_testing_histogram')

    print("\nRunning with Gini criterion...")
    train_gini, test_gini = run_experiment(X, y, n_runs=100, criterion='gini')
    print_results("Gini", train_gini, test_gini)
    plot_histogram(train_gini, 'Decision Tree Training Accuracy (Gini)', 'qe1_training_histogram')
    plot_histogram(test_gini, 'Decision Tree Testing Accuracy (Gini)', 'qe1_testing_histogram')

    print("\nRunning with early stopping (85%)...")
    train_early, test_early = run_experiment(X, y, n_runs=100, criterion='information_gain', early_stop_threshold=0.85)
    print_results("Early Stop", train_early, test_early)
    plot_histogram(train_early, 'Decision Tree Training Accuracy (Early Stop)', 'qe2_training_histogram')
    plot_histogram(test_early, 'Decision Tree Testing Accuracy (Early Stop)', 'qe2_testing_histogram')

    print("\nDone! Plots saved.")


if __name__ == "__main__":
    main()
