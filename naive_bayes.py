import math
from collections import defaultdict


class MultinomialNaiveBayes:
    def __init__(self, alpha=0, use_log=False):
        self.alpha = alpha
        self.use_log = use_log
        self.vocab = set()
        self.priors = {}
        self.word_counts = {}
        self.class_totals = {}

    def train(self, pos_instances, neg_instances, vocab):
        self.vocab = vocab

        total_docs = len(pos_instances) + len(neg_instances)
        self.priors["pos"] = len(pos_instances) / total_docs
        self.priors["neg"] = len(neg_instances) / total_docs

        # if the word isn't there then it creates it with 0 automatically
        # afterwhich increments to 1.
        self.word_counts["pos"] = defaultdict(int)
        self.word_counts["neg"] = defaultdict(int)

        for doc in pos_instances:
            for word in doc:
                self.word_counts["pos"][word] += 1

        for doc in neg_instances:
            for word in doc:
                self.word_counts["neg"][word] += 1

        self.class_totals["pos"] = sum(self.word_counts["pos"].values())
        self.class_totals["neg"] = sum(self.word_counts["neg"].values())

    def get_word_prob(self, word, class_label):
        count = self.word_counts[class_label].get(word, 0)
        total = self.class_totals[class_label]
        vocab_size = len(self.vocab)

        prob = (count + self.alpha) / (total + self.alpha * vocab_size)
        return prob

    def classify(self, document):
        if self.use_log:
            return self.classify_log(document)
        else:
            return self.classify_regular(document)

    def classify_regular(self, document):
        scores = {}

        for class_label in ["pos", "neg"]:
            score = self.priors[class_label]
            for word in document:
                if word in self.vocab:
                    score *= self.get_word_prob(word, class_label)
            scores[class_label] = score

        return "pos" if scores["pos"] > scores["neg"] else "neg"

    def classify_log(self, document):
        scores = {}

        for class_label in ["pos", "neg"]:
            score = math.log(self.priors[class_label])
            for word in document:
                if word in self.vocab:
                    prob = self.get_word_prob(word, class_label)
                    score += math.log(prob)
            scores[class_label] = score

        return "pos" if scores["pos"] > scores["neg"] else "neg"


def evaluate(classifier, pos_test, neg_test):
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for doc in pos_test:
        prediction = classifier.classify(doc)
        if prediction == "pos":
            tp += 1
        else:
            fn += 1

    for doc in neg_test:
        prediction = classifier.classify(doc)
        if prediction == "neg":
            tn += 1
        else:
            fp += 1

    # Accuracy = (True Positives + True Negatives) / Total Predictions
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    # Precision = True Positives / (True Positives + False Positives)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # Recall = True Positives / (True Positives + False Negatives)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    confusion_matrix = [[tp, fn], [fp, tn]]

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": confusion_matrix,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
    }
