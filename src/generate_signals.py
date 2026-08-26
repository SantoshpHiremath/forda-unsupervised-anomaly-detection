"""
Synthetic motor-signal generator, shaped like the real FordA dataset
(UCR/UEA time-series archive): univariate signals, 500 timesteps per
sample, binary label (normal / abnormal engine noise measurement).

This is NOT the real FordA dataset. This sandbox has no outbound access
to OpenML, Zenodo, or the UCR/UEA time-series archive (all three were
tried and returned a 403 from the sandbox's outbound proxy, the same
network constraint documented in other projects in this portfolio, e.g.
Docker Hub being blocked for devops-cicd-monitoring). Rather than fake
having downloaded FordA, this module generates a synthetic dataset with
the same shape and the same real-world framing (engine noise signal
measured on a motor with normal/abnormal symptom classes), so the rest
of the project — genuinely unsupervised and semi-supervised anomaly
detection, honestly evaluated — is built and verified against real,
inspectable data, just not real Ford data.

The signal model: a normal signal is a smooth combination of a few sine
components (simulating periodic mechanical vibration) plus small Gaussian
noise. An abnormal signal has the same base structure PLUS one of a few
realistic fault signatures layered on top: a localized high-frequency
burst (bearing knock), a DC offset drift (sensor/calibration drift), or
increased broadband noise (worn component). This mirrors how FordA's
real abnormal class is described in the literature (engine noise
measurements with a symptom present) without claiming to reproduce the
real signal statistics, which would require the real data.
"""

import numpy as np

N_TIMESTEPS = 500


def _base_signal(rng, n_timesteps=N_TIMESTEPS):
    """A smooth, quasi-periodic 'normal' motor vibration signal."""
    t = np.linspace(0, 4 * np.pi, n_timesteps)
    n_components = rng.integers(2, 5)
    signal = np.zeros(n_timesteps)
    for _ in range(n_components):
        freq = rng.uniform(0.5, 3.0)
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.3, 1.0)
        signal += amp * np.sin(freq * t + phase)
    return signal


def _add_fault(rng, signal, fault_type):
    """
    Layer one realistic fault signature onto a base signal.

    Magnitudes here are deliberately kept SMALL relative to the base
    signal's own amplitude (roughly unit scale) -- an earlier version of
    this generator used larger fault magnitudes and produced a
    suspiciously easy dataset (unsupervised AUC 0.987, and abnormal
    samples had ~28% higher total signal energy than normal ones on
    average, meaning a trivial "sum of squares" threshold would nearly
    solve the task without any model at all -- see
    tests/test_generate_signals.py::test_classes_are_not_trivially_separable_by_energy
    for the regression test that catches this if it regresses). Real
    fault symptoms in sensor data are usually subtle relative to the
    signal, not dramatic -- so the magnitudes below were reduced to
    reflect that, not to make the detection task artificially easy for a
    single method.
    """
    n = len(signal)
    signal = signal.copy()
    if fault_type == "bearing_knock":
        # Localized high-frequency burst at a random position.
        start = rng.integers(0, n - 40)
        burst_len = rng.integers(15, 40)
        t = np.arange(burst_len)
        burst = rng.uniform(0.35, 0.7) * np.sin(t * rng.uniform(2.5, 4.0))
        window = np.hanning(burst_len)
        signal[start:start + burst_len] += burst * window
    elif fault_type == "sensor_drift":
        # Slow DC offset drift over the second half of the signal.
        drift_start = rng.integers(int(n * 0.3), int(n * 0.6))
        drift = np.zeros(n)
        ramp_len = n - drift_start
        drift[drift_start:] = np.linspace(0, rng.uniform(0.15, 0.4), ramp_len)
        signal += drift
    elif fault_type == "broadband_noise":
        # Elevated noise floor across the whole signal (worn component).
        signal += rng.normal(0, rng.uniform(0.12, 0.22), size=n)
    else:
        raise ValueError(f"unknown fault_type: {fault_type}")
    return signal


FAULT_TYPES = ("bearing_knock", "sensor_drift", "broadband_noise")


def generate_dataset(n_samples=3601, anomaly_fraction=0.42, seed=42, n_timesteps=N_TIMESTEPS):
    """
    Generate a synthetic dataset shaped like FordA:
      - n_samples rows, n_timesteps columns (default 500, matching FordA)
      - label -1 = normal, label 1 = abnormal
      - anomaly_fraction controls the class balance (FordA is roughly
        balanced; default here mirrors that, not artificially easy)

    Returns (X, y, fault_labels) where fault_labels is None for normal
    rows and the fault type string for abnormal rows (kept separately so
    training code can honestly avoid using it — it is NOT a training
    signal, only used afterward for honest evaluation and root-cause
    inspection, the same "don't use labels you shouldn't have" discipline
    as the rest of this portfolio).
    """
    rng = np.random.default_rng(seed)
    n_anomalous = int(round(n_samples * anomaly_fraction))
    n_normal = n_samples - n_anomalous

    X = np.zeros((n_samples, n_timesteps))
    y = np.zeros(n_samples, dtype=int)
    fault_labels = [None] * n_samples

    idx = 0
    for _ in range(n_normal):
        base = _base_signal(rng)
        noise = rng.normal(0, 0.15, size=n_timesteps)
        X[idx] = base + noise
        y[idx] = -1
        idx += 1

    for _ in range(n_anomalous):
        base = _base_signal(rng)
        fault_type = FAULT_TYPES[rng.integers(0, len(FAULT_TYPES))]
        signal = _add_fault(rng, base, fault_type)
        noise = rng.normal(0, 0.15, size=n_timesteps)
        X[idx] = signal + noise
        y[idx] = 1
        fault_labels[idx] = fault_type
        idx += 1

    # Shuffle so normal/abnormal rows are interleaved, not block-ordered.
    perm = rng.permutation(n_samples)
    X, y = X[perm], y[perm]
    fault_labels = [fault_labels[i] for i in perm]

    return X, y, fault_labels
