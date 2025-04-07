import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from decision_tree_classifier import DecisionTreeClassifier  # Import your decision tree classifier


class RandomForestClassifier:
    def __init__(self, n_trees=10, data_frac=0.8, feature_subcount=None):
        self.n_trees = n_trees
        self.data_frac = data_frac
        self.feature_subcount = feature_subcount
        self.trees = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Ensure feature_subcount is valid
        if self.feature_subcount > n_features:
            raise ValueError(f"feature_subcount={self.feature_subcount} exceeds number of features={n_features}. Set feature_subcount <= n_features.")

        self.feature_subcount = self.feature_subcount or n_features  # Default to all features if not specified

        for _ in range(self.n_trees):
            # Bootstrap sampling (random sampling with replacement)
            indices = np.random.choice(range(n_samples), size=int(self.data_frac * n_samples), replace=True)
            X_subset = X.iloc[indices]
            y_subset = y.iloc[indices]

            # Random feature selection
            features = np.random.choice(range(n_features), size=self.feature_subcount, replace=False)
            X_subset = X_subset.iloc[:, features]

            # Build the tree using DecisionTreeClassifier
            tree = DecisionTreeClassifier()
            tree.fit(X_subset, y_subset)
            self.trees.append((tree, features))

    def predict(self, X):
        predictions = []
        for tree, features in self.trees:
            X_subset = X.iloc[:, features]
            predictions.append(tree.predict(X_subset))

        # Gather all predictions and transpose
        predictions = np.array(predictions).T  # Transpose to gather votes per sample

        # Voting mechanism using np.bincount
        final_predictions = [np.bincount(row.astype(int)).argmax() for row in predictions]
        return np.array(final_predictions)


# Load dataset
data_path = "A4_Materials/list_of_devices.txt"

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

# Encode the 'Device' column (features) and 'Connection_Type' column (target)
device_encoder = LabelEncoder()
df["Device"] = device_encoder.fit_transform(df["Device"])

connection_encoder = LabelEncoder()
df["Connection_Type"] = connection_encoder.fit_transform(df["Connection_Type"])

X = df[["Device"]]
y = df["Connection_Type"]

# Hyperparameter Tuning with Cross-Validation
def tune_hyperparameters():
    # Define ranges for hyperparameters
    n_trees_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    data_frac_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    feature_subcount_values = [1]  # Limit feature_subcount to the number of features

    results = []

    # 5-Fold Cross-Validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Test all combinations of hyperparameters
    for n_trees in n_trees_values:
        for data_frac in data_frac_values:
            for feature_subcount in feature_subcount_values:
                accuracies = []

                for train_idx, test_idx in cv.split(X):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                    # Train Random Forest Classifier
                    rf = RandomForestClassifier(
                        n_trees=n_trees,
                        data_frac=data_frac,
                        feature_subcount=feature_subcount,
                    )
                    rf.fit(X_train, y_train)
                    y_pred = rf.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    accuracies.append(acc)

                # Average accuracy across folds
                avg_accuracy = np.mean(accuracies)
                results.append({
                    "n_trees": n_trees,
                    "data_frac": data_frac,
                    "feature_subcount": feature_subcount,
                    "accuracy": avg_accuracy,
                })
                print(f"n_trees={n_trees}, data_frac={data_frac}, cross-validated accuracy={avg_accuracy:.2f}")

    return results


# Plot Results
def plot_accuracy_by_parameter(results, parameter):
    plt.figure(figsize=(10, 6))

    # Group results by the selected parameter
    unique_values = sorted(set(r[parameter] for r in results))
    avg_accuracies = [
        np.mean([r["accuracy"] for r in results if r[parameter] == val]) for val in unique_values
    ]

    # Plot
    plt.plot(unique_values, avg_accuracies, marker='o')
    plt.title(f"Accuracy vs {parameter}")
    plt.xlabel(parameter)
    plt.ylabel("Average Accuracy")
    plt.grid(True)
    plt.show()


# Perform Tuning
results = tune_hyperparameters()

# Visualize Results
plot_accuracy_by_parameter(results, "n_trees")
plot_accuracy_by_parameter(results, "data_frac")
