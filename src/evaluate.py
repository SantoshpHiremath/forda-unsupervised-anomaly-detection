"""
Honest evaluation harness for the unsupervised autoencoder detector and
the semi-supervised comparison.

The discipline this module enforces (checked by tests, not just
described): the anomaly threshold is chosen using ONLY normal-class
data (a held-out validation split of normal signals never seen in
training), never by peeking at the test set's true labels. True labels
are used strictly for scoring after predictions are already fixed --
the same "don't let the answer key leak into the decision" pattern used
in this portfolio's other honestly-evaluated projects.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score


def choose_threshold_from_normal_validation(val_normal_scores, percentile=95):
    """
    Pick an anomaly threshold using ONLY reconstruction errors from a
    held-out slice of NORMAL data (never abnormal data, never test
    labels). percentile=95 means: 95% of held-out normal signals should
    score below the threshold, i.e. we accept a ~5% false-positive rate
    on normal data by construction, before ever looking at how well this
    catches real anomalies.
    """
    return float(np.percentile(val_normal_scores, percentile))


def evaluate_predictions(y_true, y_pred, scores):
    """
    Score already-fixed predictions against true labels. This function
    is the ONLY place true labels are consulted, and only after
    predictions and threshold were already decided from normal-only
    data -- never used to tune the threshold itself.
    """
    y_true_binary = (y_true == 1).astype(int)
    y_pred_binary = (y_pred == 1).astype(int)

    auc = roc_auc_score(y_true_binary, scores)
    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    base_rate = y_true_binary.mean()

    return {
        "roc_auc": auc,
        "precision": precision,
        "recall": recall,
        "base_anomaly_rate": base_rate,
    }


def semi_supervised_baseline_auc(X_train, y_train, X_test, y_test, label_fraction, seed=0):
    """
    Semi-supervised comparison: train a supervised classifier (logistic
    regression, same honest-evaluation family as the rest of this
    portfolio) using only `label_fraction` of the available training
    labels, simulating the real-world constraint semi-supervised methods
    exist to address -- labels are expensive, so most of the data is
    unlabeled. This is a *baseline* to compare the unsupervised
    autoencoder against, showing how much a supervised approach degrades
    as labeled data shrinks, and where the unsupervised approach (which
    needs zero labels at training time) becomes competitive.
    """
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(seed)
    n = X_train.shape[0]
    n_labeled = max(2, int(round(n * label_fraction)))
    idx = rng.choice(n, size=n_labeled, replace=False)

    X_labeled = X_train[idx]
    y_labeled = y_train[idx]

    if len(np.unique(y_labeled)) < 2:
        return None  # degenerate: only one class in the tiny label budget

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_labeled, y_labeled)
    scores = clf.decision_function(X_test)
    y_test_binary = (y_test == 1).astype(int)
    return roc_auc_score(y_test_binary, scores)
