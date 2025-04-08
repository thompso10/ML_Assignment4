#!/usr/bin/env python3

import json
import argparse
import os
import numpy as np
import tqdm
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from random_forest_classifier import RandomForestClassifier  
from sklearn.naive_bayes import MultinomialNB

SEED = 0


def parse_args():
    """
    Parse arguments.
    """
    parser = argparse.ArgumentParser()

    def check_path(parser, x):
        if not os.path.exists(x):
            parser.error(f"That directory {x} does not exist!")
        else:
            return x
    parser.add_argument('-r', '--root', type=lambda x: check_path(parser, x),
                        help='The path to the root directory containing feature files.')
    parser.add_argument('-s', '--split', type=float, default=0.7,
                        help='The percentage of samples to use for training.')

    return parser.parse_args()


def load_data(root, min_samples=20, max_samples=1000):
    """
    Loads and processes JSON feature files from the given root directory.

    Filters labels by sample count, extracts numerical features, and builds
    wordbag-style vectors for ports, domains, and ciphers.

    Args:
        root (str): Path to the root dataset directory.
        min_samples (int): Minimum samples required per label.
        max_samples (int): Maximum samples to load per label.

    Returns:
        tuple: (X, X_p, X_d, X_c, Y) as NumPy arrays.
    """
    X, X_p, X_d, X_c, Y = [], [], [], [], []
    port_dict, domain_set, cipher_set = {}, set(), set()

    # Create paths and instance count filtering
    fpaths, fcounts = [], {}
    
    try:
        for label in os.listdir(root):
            label_path = os.path.join(root, label)
            if not os.path.isdir(label_path):
                continue
            fpaths.extend([(os.path.join(label_path, f), label) for f in os.listdir(label_path)])
            fcounts[label] = len(os.listdir(label_path))
    except TypeError:
        print(("TypeError: expected str, bytes or os.PathLike object, not NoneType. \n(HINT: use the flag --root: <path to root dir>)"))
        quit()
    processed_counts = {label: 0 for label in fcounts}
    for fpath in tqdm.tqdm(fpaths):
        path, label = fpath
        if fcounts[label] < min_samples or processed_counts[label] >= max_samples:
            continue
        processed_counts[label] += 1
        with open(path, "r") as fp:
            features = json.load(fp)
            instance = [features["flow_volume"], features["flow_duration"],
                        features["flow_rate"], features["sleep_time"],
                        features["dns_interval"], features["ntp_interval"]]
            X.append(instance)
            # Ensure X_p remains a list of ports
            X_p.append(list(features["ports"]))
            X_d.append(list(features["domains"]))
            X_c.append(list(features["ciphers"]))
            Y.append(label)
            domain_set.update(features["domains"])
            cipher_set.update(features["ciphers"])
            for port in features["ports"]:  # This stays as a list of ports
                port_dict[port] = port_dict.get(port, 0) + 1

    # Prune rarely seen ports
    port_set = {port for port in port_dict if port_dict[port] > 10}

    # Map to wordbag
    print("Generating wordbags ...")
    for i in tqdm.tqdm(range(len(Y))):
        # Ensure X_p[i] remains a list and is processed correctly
        X_p[i] = [X_p[i].count(port) for port in port_set]
        X_d[i] = [X_d[i].count(domain) for domain in domain_set]
        X_c[i] = [X_c[i].count(cipher) for cipher in cipher_set]

    return (np.array(X).astype(float), np.array(X_p),
            np.array(X_d), np.array(X_c), np.array(Y))


def classify_bayes(X_tr, Y_tr, X_ts, Y_ts):
    """
    Use a multinomial naive bayes classifier for 'bag of words'.

    Returns
    -------
    C_tr : numpy array
    C_ts : numpy array
    """
    classifier = MultinomialNB()
    classifier.fit(X_tr, Y_tr)
    C_tr = [(np.argmax(instance), max(instance))
            for instance in classifier.predict_proba(X_tr)]
    C_ts = [(np.argmax(instance), max(instance))
            for instance in classifier.predict_proba(X_ts)]
    return C_tr, C_ts



def do_stage_0(Xp_tr, Xp_ts, Xd_tr, Xd_ts, Xc_tr, Xc_ts, Y_tr, Y_ts):
    """
    Perform stage 0 classification with naive bayes.
    """
    timings = {}

    # Classification for ports
    start_time = time.time()
    resp_tr, resp_ts = classify_bayes(Xp_tr, Y_tr, Xp_ts, Y_ts)
    timings['ports'] = time.time() - start_time

    # Classification for domains
    start_time = time.time()
    resd_tr, resd_ts = classify_bayes(Xd_tr, Y_tr, Xd_ts, Y_ts)
    timings['domains'] = time.time() - start_time

    # Classification for ciphers
    start_time = time.time()
    resc_tr, resc_ts = classify_bayes(Xc_tr, Y_tr, Xc_ts, Y_ts)
    timings['ciphers'] = time.time() - start_time

    return resp_tr, resp_ts, resd_tr, resd_ts, resc_tr, resc_ts, timings


def do_stage_1(X_tr, X_ts, Y_tr, Y_ts):
    """
    Perform stage 1 classification using custom Random Forest.
    """
    start_time = time.time()
    model = RandomForestClassifier(n_trees=9, data_frac=0.8, feature_subcount=None)
    model.fit(X_tr, Y_tr)
    pred = model.predict(X_ts)
    score = classification_report(Y_ts, pred, output_dict=False)
    print(f"Stage 1 Classification Report:\n{score}")
    stage_1_time = time.time() - start_time
    return pred, stage_1_time


def main(args):
    print("Loading dataset")
    X, X_p, X_d, X_c, Y = load_data(args.root)

    print("Encoding labels")
    le = LabelEncoder()
    Y = le.fit_transform(Y)

    # Shuffle and split
    print("Shuffling and splitting dataset")
    s = np.arange(len(Y))
    np.random.seed(SEED)
    np.random.shuffle(s)
    X, X_p, X_d, X_c, Y = X[s], X_p[s], X_d[s], X_c[s], Y[s]
    cut = int(len(Y) * args.split)
    X_tr, Xp_tr, Xd_tr, Xc_tr, Y_tr = X[:cut], X_p[:cut], X_d[:cut], X_c[:cut], Y[:cut]
    X_ts, Xp_ts, Xd_ts, Xc_ts, Y_ts = X[cut:], X_p[cut:], X_d[cut:], X_c[cut:], Y[cut:]

    # Stage 0
    print("Performing Stage 0")
    p_tr, p_ts, d_tr, d_ts, c_tr, c_ts, stage_0_timings = do_stage_0(
        Xp_tr, Xp_ts, Xd_tr, Xd_ts, Xc_tr, Xc_ts, Y_tr, Y_ts
    )

    # Combine for Stage 1
    X_tr_full = np.hstack((X_tr, p_tr, d_tr, c_tr))
    X_ts_full = np.hstack((X_ts, p_ts, d_ts, c_ts))

    # Stage 1
    print("Performing Stage 1")
    pred, stage_1_time = do_stage_1(X_tr_full, X_ts_full, Y_tr, Y_ts)

    # Report timings
    print("\nTiming Metrics:")
    print(f"Stage 0 - Ports Classification Time: {stage_0_timings['ports']:.2f} seconds")
    print(f"Stage 0 - Domains Classification Time: {stage_0_timings['domains']:.2f} seconds")
    print(f"Stage 0 - Ciphers Classification Time: {stage_0_timings['ciphers']:.2f} seconds")
    print(f"Stage 1 - Random Forest Training Time: {stage_1_time:.2f} seconds")

    print(classification_report(Y_ts, pred, target_names=le.classes_))


if __name__ == "__main__":
    args = parse_args()
    main(args)