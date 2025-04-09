import json
import os
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from decision_tree_classifier import DecisionTreeClassifier
from random_forest_classifier import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import time  # Added for timing measurements

SEED = 0

# Hardcoded paths - CHANGE THESE TO MATCH YOUR SYSTEM
ROOT_PATH = r"A4_Materials/iot_data/iot_data"  # Raw string for Windows paths
SPLIT_RATIO = 0.7  # Training/test split ratio


def plot_confusion_matrix(y_true, y_pred, classes, normalize=True, title='Confusion Matrix'):
    """
    Plot a normalized confusion matrix with wider squares.
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: List of class names
        normalize: Whether to normalize the matrix
        title: Title for the plot
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Calculate dynamic figure size based on number of classes
    n_classes = len(classes)
    fig_width = max(10, n_classes * .8)  # Minimum 10 inches, 1.2 inches per class
    fig_height = max(8, n_classes * .5)  # Minimum 8 inches, 1.0 inches per class

    plt.figure(figsize=(fig_width, fig_height))

    # Create heatmap with square cells
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        annot_kws={'size': 12},
        cbar_kws={'shrink': 0.75},
        linewidths=0.5,
        linecolor='lightgray',
        square=True  # Force square-shaped cells
    )

    # Adjust label appearance
    ax.set_xticklabels(ax.get_xticklabels(),
                       rotation=45,
                       ha='right',
                       fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(),
                       rotation=0,
                       fontsize=12)

    plt.title(title, fontsize=14, pad=20)
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)

    # Adjust layout
    plt.tight_layout()
    plt.show()


def load_data(root, min_samples=20, max_samples=1000):
    """
    Loads and processes JSON feature files from the given root directory.
    """
    X, X_p, X_d, X_c, Y = [], [], [], [], []
    port_dict, domain_set, cipher_set = {}, set(), set()

    # Create paths and instance count filtering
    fpaths, fcounts = [], {}
    for rt, dirs, files in os.walk(root):
        for fname in files:
            path = os.path.join(rt, fname)
            label = os.path.basename(os.path.dirname(path))
            if fname.startswith("features") and fname.endswith(".json"):
                fpaths.append((path, label))
                fcounts[label] = fcounts.get(label, 0) + 1

    for label in os.listdir(root):
        label_path = os.path.join(root, label)
        if not os.path.isdir(label_path):
            continue
        fpaths.extend([(os.path.join(label_path, f), label) for f in os.listdir(label_path)
                       if f.startswith("features") and f.endswith(".json")])
        fcounts[label] = len([f for f in os.listdir(label_path)
                              if f.startswith("features") and f.endswith(".json")])

    processed_counts = {label: 0 for label in fcounts}
    for fpath in tqdm(fpaths, desc="Loading feature files"):
        path, label = fpath
        if fcounts[label] < min_samples or processed_counts[label] >= max_samples:
            continue

        try:
            with open(path, "r") as fp:
                features = json.load(fp)
                processed_counts[label] += 1
                instance = [features["flow_volume"], features["flow_duration"],
                            features["flow_rate"], features["sleep_time"],
                            features["dns_interval"], features["ntp_interval"]]
                X.append(instance)
                X_p.append(list(features["ports"]))
                X_d.append(list(features["domains"]))
                X_c.append(list(features["ciphers"]))
                Y.append(label)
                domain_set.update(features["domains"])
                cipher_set.update(features["ciphers"])
                for port in features["ports"]:
                    port_dict[port] = port_dict.get(port, 0) + 1
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error processing file {path}: {str(e)}")
            continue

    # Prune rarely seen ports
    port_set = {port for port in port_dict if port_dict[port] > 10}

    # Map to wordbag
    print("Generating wordbags...")
    for i in tqdm(range(len(Y)), desc="Processing wordbags"):
        X_p[i] = [X_p[i].count(port) for port in port_set]
        X_d[i] = [X_d[i].count(domain) for domain in domain_set]
        X_c[i] = [X_c[i].count(cipher) for cipher in cipher_set]

    return (np.array(X).astype(float), np.array(X_p),
            np.array(X_d), np.array(X_c), np.array(Y))


def classify_bayes(X_tr, Y_tr, X_ts, Y_ts):
    """
    Use a multinomial naive bayes classifier for 'bag of words'.
    """
    classifier = MultinomialNB()
    classifier.fit(X_tr, Y_tr)
    C_tr = [(np.argmax(instance), np.max(instance))
            for instance in classifier.predict_proba(X_tr)]
    C_ts = [(np.argmax(instance), np.max(instance))
            for instance in classifier.predict_proba(X_ts)]
    return np.array(C_tr), np.array(C_ts)


def do_stage_0(Xp_tr, Xp_ts, Xd_tr, Xd_ts, Xc_tr, Xc_ts, Y_tr, Y_ts):
    """
    Perform stage 0 classification with naive bayes.
    """
    resp_tr, resp_ts = classify_bayes(Xp_tr, Y_tr, Xp_ts, Y_ts)
    resd_tr, resd_ts = classify_bayes(Xd_tr, Y_tr, Xd_ts, Y_ts)
    resc_tr, resc_ts = classify_bayes(Xc_tr, Y_tr, Xc_ts, Y_ts)
    return resp_tr, resp_ts, resd_tr, resd_ts, resc_tr, resc_ts


def evaluate_model(model, X_tr, X_ts, Y_tr, Y_ts, model_name, le):
    """
    Evaluate a given model and print results with timing information.
    """
    print(f"\nTraining {model_name}...")

    # Measure training time
    start_time = time.time()
    model.fit(X_tr, Y_tr)
    training_time = time.time() - start_time

    # Measure prediction time
    start_time = time.time()
    train_pred = model.predict(X_tr)
    train_time = time.time() - start_time

    start_time = time.time()
    test_pred = model.predict(X_ts)
    test_time = time.time() - start_time

    # Training accuracy
    train_acc = np.mean(train_pred == Y_tr)

    # Test accuracy
    test_acc = np.mean(test_pred == Y_ts)

    # Print results with timing information
    print(f"{model_name} Training Time: {training_time:.4f} seconds")
    print(f"{model_name} Training Prediction Time: {train_time:.4f} seconds")
    print(f"{model_name} Test Prediction Time: {test_time:.4f} seconds")
    print(f"{model_name} Training Accuracy: {train_acc:.4f}")
    print(f"{model_name} Test Accuracy: {test_acc:.4f}")

    # Classification report with zero_division parameter
    print(f"\n{model_name} Classification Report:")
    print(classification_report(Y_ts, test_pred, target_names=le.classes_, zero_division=0))

    return test_pred, training_time


def main():
    print("Loading dataset")
    X, X_p, X_d, X_c, Y = load_data(ROOT_PATH)

    print("Encoding labels")
    le = LabelEncoder()
    Y = le.fit_transform(Y)

    # Shuffle and split
    print("Shuffling and splitting dataset")
    s = np.arange(len(Y))
    np.random.seed(SEED)
    np.random.shuffle(s)
    X, X_p, X_d, X_c, Y = X[s], X_p[s], X_d[s], X_c[s], Y[s]
    cut = int(len(Y) * SPLIT_RATIO)
    X_tr, Xp_tr, Xd_tr, Xc_tr, Y_tr = X[:cut], X_p[:cut], X_d[:cut], X_c[:cut], Y[:cut]
    X_ts, Xp_ts, Xd_ts, Xc_ts, Y_ts = X[cut:], X_p[cut:], X_d[cut:], X_c[cut:], Y[cut:]

    # Stage 0
    print("Performing Stage 0")
    p_tr, p_ts, d_tr, d_ts, c_tr, c_ts = do_stage_0(Xp_tr, Xp_ts, Xd_tr, Xd_ts, Xc_tr, Xc_ts, Y_tr, Y_ts)

    # Combine for Stage 1
    X_tr_full = np.hstack((X_tr, p_tr, d_tr, c_tr))
    X_ts_full = np.hstack((X_ts, p_ts, d_ts, c_ts))

    # Decision Tree Classifier
    dt = DecisionTreeClassifier(max_depth=10)
    dt_pred, dt_time = evaluate_model(dt, X_tr_full, X_ts_full, Y_tr, Y_ts, "Decision Tree", le)
    plot_confusion_matrix(Y_ts, dt_pred, classes=le.classes_,
                          title='Decision Tree - Normalized Confusion Matrix')

    # Random Forest Classifier
    rf = RandomForestClassifier(n_trees=20, data_frac=1.0, feature_subcount=None)
    rf_pred, rf_time = evaluate_model(rf, X_tr_full, X_ts_full, Y_tr, Y_ts, "Random Forest", le)
    plot_confusion_matrix(Y_ts, rf_pred, classes=le.classes_,
                          title='Random Forest - Normalized Confusion Matrix')



if __name__ == "__main__":
    main()