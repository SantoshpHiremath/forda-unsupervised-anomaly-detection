import numpy as np
import pytest

from src.generate_signals import generate_dataset, N_TIMESTEPS


def test_shape_matches_forda_dimensions():
    X, y, _ = generate_dataset(n_samples=3601, seed=42)
    assert X.shape == (3601, N_TIMESTEPS)
    assert y.shape == (3601,)


def test_labels_are_only_plus_minus_one():
    _, y, _ = generate_dataset(seed=42)
    assert set(np.unique(y).tolist()) <= {-1, 1}


def test_class_balance_is_reasonable_not_degenerate():
    _, y, _ = generate_dataset(seed=42, anomaly_fraction=0.42)
    frac = (y == 1).mean()
    assert 0.35 < frac < 0.50


def test_deterministic_with_fixed_seed():
    X1, y1, f1 = generate_dataset(seed=7)
    X2, y2, f2 = generate_dataset(seed=7)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)
    assert f1 == f2


def test_different_seeds_produce_different_data():
    X1, _, _ = generate_dataset(seed=1)
    X2, _, _ = generate_dataset(seed=2)
    assert not np.allclose(X1, X2)


def test_fault_labels_are_none_for_normal_and_set_for_abnormal():
    _, y, fault_labels = generate_dataset(seed=42)
    for label, fault in zip(y, fault_labels):
        if label == -1:
            assert fault is None
        else:
            assert fault in ("bearing_knock", "sensor_drift", "broadband_noise")


def test_classes_are_not_trivially_separable_by_energy():
    """
    Regression test for a real bug caught during development: an earlier
    version of the fault generator used large fault magnitudes, which
    made abnormal samples have ~28% higher total signal energy than
    normal samples on average -- meaning a trivial "sum of squares"
    threshold could nearly solve the anomaly-detection task without any
    model at all (the unsupervised autoencoder scored a suspicious 0.987
    AUC as a result). This test locks in that the fix (smaller fault
    magnitudes) keeps the energy gap small, so the dataset actually
    requires the autoencoder to learn signal *shape*, not just
    amplitude/energy.
    """
    X, y, _ = generate_dataset(seed=42)
    normal_energy = (X[y == -1] ** 2).sum(axis=1)
    abnormal_energy = (X[y == 1] ** 2).sum(axis=1)
    pct_gap = abs(abnormal_energy.mean() / normal_energy.mean() - 1) * 100
    assert pct_gap < 10, (
        f"energy gap between classes is {pct_gap:.1f}%, which is large enough "
        "that a trivial energy threshold could solve the task -- the dataset "
        "should require learning signal shape, not just amplitude"
    )


def test_all_three_fault_types_appear_in_abnormal_class():
    _, y, fault_labels = generate_dataset(seed=42, n_samples=1000)
    abnormal_faults = {f for f, label in zip(fault_labels, y) if label == 1}
    assert abnormal_faults == {"bearing_knock", "sensor_drift", "broadband_noise"}
