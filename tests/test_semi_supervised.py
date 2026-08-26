import numpy as np
import pytest

from src.generate_signals import generate_dataset
from src.pipeline import split_dataset
from src.evaluate import semi_supervised_baseline_auc


def test_semi_supervised_auc_degrades_as_labels_shrink():
    """
    The core semi-supervised finding: AUC should trend downward as the
    label budget shrinks from 100% to 2%, demonstrating the real-world
    motivation for semi-supervised methods (labels are the expensive
    resource). We check the trend directionally (100% >= 2% by a real
    margin) rather than requiring strict monotonicity at every step,
    since a few points can jitter with a fixed random label sample.
    """
    X, y, _ = generate_dataset(seed=42)
    split = split_dataset(X, y, seed=0)

    auc_full = semi_supervised_baseline_auc(
        split["X_trainval"], split["y_trainval"], split["X_test"], split["y_test"],
        label_fraction=1.0, seed=0,
    )
    auc_scarce = semi_supervised_baseline_auc(
        split["X_trainval"], split["y_trainval"], split["X_test"], split["y_test"],
        label_fraction=0.02, seed=0,
    )
    assert auc_full is not None and auc_scarce is not None
    assert auc_full - auc_scarce > 0.02, (
        f"expected a real degradation from full labels ({auc_full:.4f}) "
        f"to a 2% label budget ({auc_scarce:.4f}), but the gap is too small "
        "to demonstrate the semi-supervised motivation honestly"
    )


def test_semi_supervised_with_degenerate_single_class_sample_returns_none():
    """
    If the tiny label budget happens to sample only one class, fitting a
    binary classifier is impossible -- the function must report this
    honestly (return None) rather than crash or silently fabricate a
    result.
    """
    X = np.random.default_rng(0).normal(size=(20, 5))
    y = np.array([-1] * 20)  # only one class present
    result = semi_supervised_baseline_auc(X, y, X, y, label_fraction=0.1, seed=0)
    assert result is None


def test_semi_supervised_deterministic_given_seed():
    X, y, _ = generate_dataset(seed=42, n_samples=500)
    split = split_dataset(X, y, seed=0)
    auc1 = semi_supervised_baseline_auc(
        split["X_trainval"], split["y_trainval"], split["X_test"], split["y_test"],
        label_fraction=0.2, seed=3,
    )
    auc2 = semi_supervised_baseline_auc(
        split["X_trainval"], split["y_trainval"], split["X_test"], split["y_test"],
        label_fraction=0.2, seed=3,
    )
    assert auc1 == auc2
