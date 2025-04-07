import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        # Convert to numpy arrays for compatibility
        X = X.to_numpy()
        y = y.to_numpy()
        self.tree = self._build_tree(X, y, depth=0)

    def predict(self, X):
        X = X.to_numpy()
        predictions = [self._predict_row(row, self.tree) for row in X]
        return np.array(predictions)

    def _build_tree(self, X, y, depth):
        # Stop if max depth is reached or if the node is pure
        if depth == self.max_depth or len(np.unique(y)) == 1:
            return self._create_leaf(y)

        # Find the best split
        feature_index, threshold = self._best_split(X, y)
        if feature_index is None:
            return self._create_leaf(y)

        # Split data
        left_indices = X[:, feature_index] <= threshold
        right_indices = X[:, feature_index] > threshold

        # Recursively build left and right branches
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return {
            "feature_index": feature_index,
            "threshold": threshold,
            "left": left_subtree,
            "right": right_subtree,
        }

    def _best_split(self, X, y):
        # Find the best feature and threshold to split the data
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
        # Calculate Gini impurity for a split
        left_indices = X[:, feature_index] <= threshold
        right_indices = X[:, feature_index] > threshold

        left_gini = self._gini(y[left_indices])
        right_gini = self._gini(y[right_indices])

        left_weight = len(y[left_indices]) / len(y)
        right_weight = len(y[right_indices]) / len(y)

        return left_weight * left_gini + right_weight * right_gini

    def _gini(self, y):
        # Calculate Gini impurity for a node
        classes, counts = np.unique(y, return_counts=True)
        prob = counts / counts.sum()
        return 1 - np.sum(prob ** 2)

    def _create_leaf(self, y):
        # Create a leaf node
        classes, counts = np.unique(y, return_counts=True)
        return {"class": classes[np.argmax(counts)]}

    def _predict_row(self, row, tree):
        # Traverse the tree to make a prediction
        if "class" in tree:
            return tree["class"]
        if row[tree["feature_index"]] <= tree["threshold"]:
            return self._predict_row(row, tree["left"])
        else:
            return self._predict_row(row, tree["right"])


# Load dataset from text file
data_path = "A4_Materials/list_of_devices.txt"  # Ensure this is the correct path

try:
    with open(data_path, "r") as file:
        lines = file.readlines()
    
    print(f"Dataset loaded successfully from {data_path}.")
except FileNotFoundError:
    print(f"Error: The dataset file was not found at the specified path: {data_path}. Please check the file path and try again.")
    exit()

# Process dataset
data = []
for line in lines[1:]:  # Skip header row
    parts = line.rsplit(maxsplit=2)  # Split from the right to extract Connection Type reliably
    if len(parts) == 3:
        device, mac, connection = parts
        data.append([device, connection])

# Convert to DataFrame
df = pd.DataFrame(data, columns=["Device", "Connection_Type"])

# Ensure dataset is not empty
if df.empty:
    print(f"Error: The dataset loaded from {data_path} is empty or improperly formatted. Please check the file contents.")
    exit()

print(f"Loaded {len(df)} device entries.")

# Encode the 'Device' column
label_encoder = LabelEncoder()
df['Device'] = label_encoder.fit_transform(df['Device'])

# Split dataset
X = df[["Device"]]  # Now 'Device' is numeric
y = df["Connection_Type"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")

# Train Decision Tree classifier
clf = DecisionTreeClassifier(max_depth=2)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree Model Accuracy: {accuracy:.2f}")

