import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from decision_tree_classifier import DecisionTreeClassifier  # Importing your existing Decision Tree code

class RandomForestClassifier:
    def __init__(self, n_trees=10, data_frac=0.8, feature_subcount=None):
        self.n_trees = n_trees
        self.data_frac = data_frac
        self.feature_subcount = feature_subcount
        self.trees = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.feature_subcount = self.feature_subcount or n_features  # Default to all features if not specified

        for _ in range(self.n_trees):
            # Bootstrap sampling (random sampling with replacement)
            indices = np.random.choice(range(n_samples), size=int(self.data_frac * n_samples), replace=True)
            X_subset = X.iloc[indices]
            y_subset = y.iloc[indices]

            # Random feature selection
            features = np.random.choice(range(n_features), size=self.feature_subcount, replace=False)
            X_subset = X_subset.iloc[:, features]

            # Build the tree using DecisionTreeClassifier from decision_tree_classifier.py
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

# Encode the 'Device' column (features) and 'Connection_Type' column (target)
device_encoder = LabelEncoder()
df["Device"] = device_encoder.fit_transform(df["Device"])

connection_encoder = LabelEncoder()
df["Connection_Type"] = connection_encoder.fit_transform(df["Connection_Type"])

# Split dataset
X = df[["Device"]]  # Feature column (encoded as numeric)
y = df["Connection_Type"]  # Target column (encoded as numeric)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")

# Train Random Forest classifier
rf = RandomForestClassifier(n_trees=10, data_frac=0.8, feature_subcount=1)  # Adjust parameters as needed
rf.fit(X_train, y_train)

# Predict and decode predictions back to original labels
y_pred = rf.predict(X_test)
y_pred_decoded = connection_encoder.inverse_transform(y_pred)

# Decode y_test for evaluation
y_test_decoded = connection_encoder.inverse_transform(y_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test_decoded, y_pred_decoded)
print(f"Random Forest Model Accuracy: {accuracy:.2f}")

