"""
End-to-end pipeline: generate data, split, train the unsupervised
autoencoder on normal-only data, choose a threshold from a normal-only
validation split, evaluate against the true test labels, run the
semi-supervised label-budget comparison, and print a summary.
"""

import numpy as np

from src.generate_signals import generate_dataset
from src.autoencoder import train_autoencoder, reconstruction_error, anomaly_scores_to_predictions
from src.evaluate import choose_threshold_from_normal_validation, evaluate_predictions, semi_supervised_baseline_auc


def split_dataset(X, y, seed=0, test_fraction=0.3, val_fraction_of_normal=0.15):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    perm = rng.permutation(n)
    X, y = X[perm], y[perm]

    n_test = int(round(n * test_fraction))
    X_test, y_test = X[:n_test], y[:n_test]
    X_trainval, y_trainval = X[n_test:], y[n_test:]

    normal_mask = y_trainval == -1
    X_normal = X_trainval[normal_mask]

    n_val = int(round(len(X_normal) * val_fraction_of_normal))
    X_val_normal = X_normal[:n_val]
    X_train_normal = X_normal[n_val:]

    return {
        "X_train_normal": X_train_normal,
        "X_val_normal": X_val_normal,
        "X_test": X_test,
        "y_test": y_test,
        "X_trainval": X_trainval,
        "y_trainval": y_trainval,
    }


def run_unsupervised_pipeline(seed=0, percentile=95, epochs=60):
    X, y, _fault_labels = generate_dataset(seed=42)
    split = split_dataset(X, y, seed=seed)

    model, mean, std = train_autoencoder(
        split["X_train_normal"], epochs=epochs, seed=seed
    )

    val_scores = reconstruction_error(model, split["X_val_normal"], mean, std)
    threshold = choose_threshold_from_normal_validation(val_scores, percentile=percentile)

    test_scores = reconstruction_error(model, split["X_test"], mean, std)
    test_preds = anomaly_scores_to_predictions(test_scores, threshold)

    metrics = evaluate_predictions(split["y_test"], test_preds, test_scores)
    metrics["threshold"] = threshold
    return metrics


def run_semi_supervised_comparison(seed=0):
    X, y, _ = generate_dataset(seed=42)
    split = split_dataset(X, y, seed=seed)
    results = {}
    for frac in (1.0, 0.5, 0.2, 0.1, 0.05, 0.02):
        auc = semi_supervised_baseline_auc(
            split["X_trainval"], split["y_trainval"],
            split["X_test"], split["y_test"],
            label_fraction=frac, seed=seed,
        )
        results[frac] = auc
    return results


if __name__ == "__main__":
    print("=== Unsupervised autoencoder anomaly detection (FordA-shaped synthetic data) ===")
    metrics = run_unsupervised_pipeline()
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print()
    print("=== Semi-supervised comparison: supervised AUC vs. fraction of labels used ===")
    comparison = run_semi_supervised_comparison()
    for frac, auc in sorted(comparison.items(), reverse=True):
        auc_str = f"{auc:.4f}" if auc is not None else "N/A (degenerate label sample)"
        print(f"  {int(frac*100):3d}% of labels -> supervised AUC = {auc_str}")
