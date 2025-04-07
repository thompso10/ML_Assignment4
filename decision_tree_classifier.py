import numpy as np

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        X = np.array(X)  # Ensure compatibility with NumPy arrays
        y = np.array(y)
        self.tree = self._build_tree(X, y, depth=0)

    def predict(self, X):
        X = np.array(X)
        predictions = [self._predict_row(row, self.tree) for row in X]
        return np.array(predictions)

    def _build_tree(self, X, y, depth):
        if depth == self.max_depth or len(np.unique(y)) == 1:
            return self._create_leaf(y)

        feature_index, threshold = self._best_split(X, y)
        if feature_index is None:
            return self._create_leaf(y)

        left_indices = X[:, feature_index] <= threshold
        right_indices = X[:, feature_index] > threshold

        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return {
            "feature_index": feature_index,
            "threshold": threshold,
            "left": left_subtree,
            "right": right_subtree,
        }

    def _best_split(self, X, y):
        best_gini = float("inf")
        best_feature, best_threshold = None, None

        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                gini = self._gini_index(X, y, feature_index, threshold)
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    def _gini_index(self, X, y, feature_index, threshold):
        left_indices = X[:, feature_index] <= threshold
        right_indices = X[:, feature_index] > threshold

        left_gini = self._gini(y[left_indices])
        right_gini = self._gini(y[right_indices])

        left_weight = len(y[left_indices]) / len(y)
        right_weight = len(y[right_indices]) / len(y)

        return left_weight * left_gini + right_weight * right_gini

    def _gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        prob = counts / counts.sum()
        return 1 - np.sum(prob ** 2)

    def _create_leaf(self, y):
        classes, counts = np.unique(y, return_counts=True)
        return {"class": classes[np.argmax(counts)]}

    def _predict_row(self, row, tree):
        if "class" in tree:
            return tree["class"]
        if row[tree["feature_index"]] <= tree["threshold"]:
            return self._predict_row(row, tree["left"])
        else:
            return self._predict_row(row, tree["right"])

