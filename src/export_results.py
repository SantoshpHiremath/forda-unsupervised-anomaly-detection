"""
export_results.py
------------------

Runs this project's real pipeline (unsupervised anomaly detection across
several seeds, one run's raw reconstruction-error scores, and the
semi-supervised label-budget comparison) and writes the results to a
JSON file -- added specifically so an external project
(results-storytelling-dashboard) can visualize these real results
without duplicating or reimplementing any of this project's training/
evaluation logic. This module calls this project's own existing,
already-tested functions; it does not add any new modeling logic itself.

Usage: python3 -m src.export_results [output_path]
(defaults to results.json in the current directory)
"""

from __future__ import annotations

import json
import sys

import numpy as np

from src.autoencoder import train_autoencoder, reconstruction_error
from src.evaluate import choose_threshold_from_normal_validation
from src.generate_signals import generate_dataset
from src.pipeline import split_dataset, run_unsupervised_pipeline, run_semi_supervised_comparison


def collect_auc_across_seeds(seeds=(0, 1, 2, 3, 4)) -> list[dict]:
    results = []
    for seed in seeds:
        metrics = run_unsupervised_pipeline(seed=seed)
        results.append({"seed": seed, **{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                          for k, v in metrics.items()}})
    return results


def collect_reconstruction_error_distribution(seed=0) -> dict:
    X, y, _ = generate_dataset(seed=42)
    split = split_dataset(X, y, seed=seed)

    model, mean, std = train_autoencoder(split["X_train_normal"], epochs=60, seed=seed)

    val_scores = reconstruction_error(model, split["X_val_normal"], mean, std)
    threshold = choose_threshold_from_normal_validation(val_scores, percentile=95)

    test_scores = reconstruction_error(model, split["X_test"], mean, std)
    y_test = split["y_test"]

    return {
        "seed": seed,
        "threshold": float(threshold),
        "normal_scores": test_scores[y_test == -1].tolist(),
        "anomaly_scores": test_scores[y_test == 1].tolist(),
    }


def collect_semi_supervised_curve(seed=0) -> dict:
    comparison = run_semi_supervised_comparison(seed=seed)
    return {str(frac): (None if auc is None else float(auc)) for frac, auc in comparison.items()}


def collect_all(seeds=(0, 1, 2, 3, 4)) -> dict:
    return {
        "auc_across_seeds": collect_auc_across_seeds(seeds=seeds),
        "reconstruction_error_distribution": collect_reconstruction_error_distribution(seed=0),
        "semi_supervised_curve": collect_semi_supervised_curve(seed=0),
    }


def main():
    output_path = sys.argv[1] if len(sys.argv) > 1 else "results.json"
    print("Re-running the real pipeline across 5 seeds, collecting raw results...", file=sys.stderr)
    results = collect_all()
    with open(output_path, "w") as f:
        json.dump(results, f)
    print(f"Wrote {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
