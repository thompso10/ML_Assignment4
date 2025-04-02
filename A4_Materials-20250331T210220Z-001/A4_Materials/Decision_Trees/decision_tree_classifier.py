import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load dataset from text file
data_path = "A4_Materials-20250331T210220Z-001\A4_Materials\list_of_devices.txt"  # Ensure this is the correct path

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
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
