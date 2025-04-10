import numpy as np
from decision_tree_classifier import DecisionTreeClassifier 
class RandomForestClassifier:
    def __init__(self, n_trees=20, data_frac=1.0, feature_subcount=None):
        """
        Initialize the Random Forest Classifier.

        Parameters:
        - n_trees (int): Number of trees in the forest.
        - data_frac (float): Fraction of data to sample for each tree (bootstrap sampling).
        - feature_subcount (int or None): Number of features to use for each tree. If None, use all features.
        """
        self.n_trees = n_trees
        self.data_frac = data_frac
        self.feature_subcount = feature_subcount
        self.trees = []  # Stores tree, features pairs

    def fit(self, X, y):
        """
        Train the Random Forest Classifier.

        Parameters:
        - X (numpy array): Feature matrix (shape: [n_samples, n_features]).
        - y (numpy array): Target labels (shape: [n_samples]).
        """
        n_samples, n_features = X.shape
        self.feature_subcount = self.feature_subcount or n_features  # Default to all features if not specified

        for _ in range(self.n_trees):
            # Bootstrap sampling (random sampling with replacement)
            indices = np.random.choice(n_samples, size=int(self.data_frac * n_samples), replace=True)
            X_subset = X[indices]
            y_subset = y[indices]

            # Random feature selection
            features = np.random.choice(n_features, size=self.feature_subcount, replace=False)
            X_subset = X_subset[:, features]

            # Train a Decision Tree
            tree = DecisionTreeClassifier(max_depth=None)
            tree.fit(X_subset, y_subset)
            self.trees.append((tree, features))  # Save the tree and the selected features

    def predict(self, X):
        """
        Predict using the trained Random Forest Classifier.

        Parameters:
        - X (numpy array): Feature matrix to predict (shape: [n_samples, n_features]).

        Returns:
        - numpy array: Predicted labels (shape: [n_samples]).
        """
        # Collect predictions from all trees
        predictions = np.zeros((X.shape[0], len(self.trees)))  # Matrix to store predictions from each tree

        for i, (tree, features) in enumerate(self.trees):
            X_subset = X[:, features]  # Select the same features used during training for this tree
            predictions[:, i] = tree.predict(X_subset)

        # Voting mechanism: use majority vote across trees
        final_predictions = [np.bincount(row.astype(int)).argmax() for row in predictions.astype(int)]
        return np.array(final_predictions)

