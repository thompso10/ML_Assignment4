import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        X = X.to_numpy()
        y = y.to_numpy()
        self.tree = self._build_tree(X, y, depth=0)

    def predict(self, X):
        X = X.to_numpy()
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


# Load dataset
data_path = "A4_Materials/list_of_devices.txt"  # Ensure this is the correct path

try:
    with open(data_path, "r") as file:
        lines = file.readlines()
    print(f"Dataset loaded successfully from {data_path}.")
except FileNotFoundError:
    print(f"Error: The dataset file was not found at the specified path: {data_path}. Please check the file path and try again.")
    exit()

data = []
for line in lines[1:]:
    parts = line.rsplit(maxsplit=2)
    if len(parts) == 3:
        device, mac, connection = parts
        data.append([device, connection])

df = pd.DataFrame(data, columns=["Device", "Connection_Type"])

if df.empty:
    print(f"Error: The dataset loaded from {data_path} is empty or improperly formatted. Please check the file contents.")
    exit()

print(f"Loaded {len(df)} device entries.")

# Encode features
label_encoder = LabelEncoder()
df['Device'] = label_encoder.fit_transform(df['Device'])

X = df[["Device"]]
y = df["Connection_Type"]

# Cross-Validation Hyperparameter Tuning
def tune_hyperparameters():
    max_depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    results = []

    # 5-Fold Cross-Validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    for depth in max_depths:
        accuracies = []

        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Train Decision Tree Classifier
            clf = DecisionTreeClassifier(max_depth=depth)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            # Calculate accuracy for this fold
            acc = accuracy_score(y_test, y_pred)
            accuracies.append(acc)

        # Average accuracy across folds
        avg_accuracy = np.mean(accuracies)
        results.append({"max_depth": depth, "accuracy": avg_accuracy})
        print(f"max_depth={depth}, cross-validated accuracy={avg_accuracy:.2f}")

    # Plotting results
    plt.figure(figsize=(10, 6))
    depths = [str(d) for d in max_depths]
    accuracies = [r["accuracy"] for r in results]
    plt.plot(depths, accuracies, marker='o', label="Accuracy")
    plt.title("Cross-Validated Accuracy vs. Max Depth")
    plt.xlabel("Max Depth")
    plt.ylabel("Average Accuracy")
    plt.grid()
    plt.legend()
    plt.show()

    return results


# Perform Hyperparameter Tuning with Cross-Validation
results = tune_hyperparameters()

