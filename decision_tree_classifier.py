import numpy as np

class DecisionTreeClassifier:
    def __init__(self, max_depth=10):
        """
        Initialize the Decision Tree Classifier.

        Parameters:
        - max_depth (int or None): The maximum depth of the tree. None for unlimited depth.
        """
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        """
        Train the Decision Tree on the provided data.

        Parameters:
        - X (numpy array): Feature matrix (shape: [n_samples, n_features]).
        - y (numpy array): Target labels (shape: [n_samples]).
        """
        X = np.array(X)  # Ensure compatibility with NumPy arrays
        y = np.array(y)
        self.tree = self._build_tree(X, y, depth=0)

    def predict(self, X):
        """
        Predict using the trained Decision Tree.

        Parameters:
        - X (numpy array): Feature matrix to predict (shape: [n_samples, n_features]).

        Returns:
        - numpy array: Predicted labels (shape: [n_samples]).
        """
        X = np.array(X)
        predictions = [self._predict_row(row, self.tree) for row in X]
        return np.array(predictions)

    def _build_tree(self, X, y, depth):
        """
        Recursively build the Decision Tree.

        Parameters:
        - X (numpy array): Feature matrix.
        - y (numpy array): Target labels.
        - depth (int): Current depth of the tree.

        Returns:
        - dict: A representation of the Decision Tree.
        """
        # Stop if max depth is reached, node is pure, or no samples are left
        if depth == self.max_depth or len(np.unique(y)) == 1 or len(y) == 0:
            return self._create_leaf(y)

        # Find the best feature and threshold for splitting
        feature_index, threshold = self._best_split(X, y)
        if feature_index is None:
            return self._create_leaf(y)

        # Split data based on the best feature and threshold
        left_indices = X[:, feature_index] <= threshold
        right_indices = X[:, feature_index] > threshold

        # Stop recursion if either subset is empty
        if not np.any(left_indices) or not np.any(right_indices):
            return self._create_leaf(y)

        # Recursively build left and right subtrees
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        # Return the tree node
        return {
            "feature_index": feature_index,
            "threshold": threshold,
            "left": left_subtree,
            "right": right_subtree,
        }

    def _best_split(self, X, y):
        """
        Find the best feature and threshold for splitting the data.

        Parameters:
        - X (numpy array): Feature matrix.
        - y (numpy array): Target labels.

        Returns:
        - (int, float): Index of the best feature and the best threshold.
        """
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
        """
        Calculate the Gini Index for a given split.

        Parameters:
        - X (numpy array): Feature matrix.
        - y (numpy array): Target labels.
        - feature_index (int): Index of the feature to split on.
        - threshold (float): Threshold value for the split.

        Returns:
        - float: Gini Index for the split.
        """
        left_indices = X[:, feature_index] <= threshold
        right_indices = X[:, feature_index] > threshold

        left_gini = self._gini(y[left_indices])
        right_gini = self._gini(y[right_indices])

        left_weight = len(y[left_indices]) / len(y)
        right_weight = len(y[right_indices]) / len(y)

        return left_weight * left_gini + right_weight * right_gini

    def _gini(self, y):
        """
        Calculate the Gini Impurity for a given node.

        Parameters:
        - y (numpy array): Target labels for the node.

        Returns:
        - float: Gini Impurity.
        """
        # Handle the edge case where the subset is empty
        if len(y) == 0:
            return 0
        classes, counts = np.unique(y, return_counts=True)
        prob = counts / counts.sum()
        return 1 - np.sum(prob ** 2)

    def _create_leaf(self, y):
        """
        Create a leaf node.

        Parameters:
        - y (numpy array): Target labels for the leaf.

        Returns:
        - dict: A leaf node containing the majority class.
        """
        # Handle the edge case where the subset is empty
        if len(y) == 0:
            return {"class": None}  # Return None if no samples are left
        classes, counts = np.unique(y, return_counts=True)
        return {"class": classes[np.argmax(counts)]}

    def _predict_row(self, row, tree):
        """
        Predict the class for a single row of data.

        Parameters:
        - row (numpy array): Feature values for the row.
        - tree (dict): The Decision Tree.

        Returns:
        - int: Predicted class for the row.
        """
        if "class" in tree:
            return tree["class"]
        if row[tree["feature_index"]] <= tree["threshold"]:
            return self._predict_row(row, tree["left"])
        else:
            return self._predict_row(row, tree["right"])

