# Unsupervised & Semi-Supervised Anomaly Detection on Motor Time-Series Data

A real, tested Python project applying unsupervised and semi-supervised
learning to time-series anomaly detection — built to close a specific
gap identified against DLR's (German Aerospace Center) "Working Student
Machine Learning" posting for the Galileo Competence Center, whose core
ask (unsupervised/semi-supervised methods applied to time-series data)
wasn't demonstrated by anything in my prior project portfolio. Prior
projects use supervised classification; this one deliberately does not
train on labels at all for its main result.

This project extends an earlier university team project ("Smart
Motor," OTH Amberg-Weiden, 3rd semester), where my own documented
contribution was limited to data loading, visualization, and quality
checks on the FordA time-series dataset (see
`docs/smart_motor_contribution_report.pdf`) — no modeling was done in
that contribution. This project picks up where that contribution left
off and builds the modeling work that was missing, honestly, from
scratch.

## What this is (read before citing anywhere)

**This does not use the real FordA dataset.** The real FordA dataset
(UCR/UEA time-series archive) is normally reachable via
`sklearn.datasets.fetch_openml`, `aeon.datasets.load_classification`, or
directly from the UCR archive — all three were tried, and all three
returned a 403 from this sandbox's outbound network proxy (OpenML,
Zenodo, and timeseriesclassification.com are all blocked here). This is
the same class of sandbox network constraint documented elsewhere in
this application portfolio (e.g. Docker Hub being blocked for
`devops-cicd-monitoring`), not something specific to this project.

Rather than fake having used the real dataset, `src/generate_signals.py`
generates a synthetic univariate time-series dataset with the exact
shape of FordA (3,601 samples, 500 timesteps, binary label) and the same
real-world framing (a motor vibration signal, normal vs. abnormal), so
every result below is real and independently verifiable against
inspectable, generated data — just not real Ford Motor Company data. If
run in an environment with real outbound access, `generate_signals.py`
is the only module that would need to be swapped for a real
`fetch_openml('FordA')` call; everything downstream (the autoencoder,
the evaluation harness, the semi-supervised comparison) is written
against the same `(X, y)` shape and would work unchanged.

## The core idea: genuinely unsupervised, not supervised-with-extra-steps

The unsupervised detector (`src/autoencoder.py`) is a PyTorch
autoencoder trained **only on signals from the normal class** — its
training function (`train_autoencoder`) has no parameter for labels at
all, which is checked structurally by a test
(`test_train_autoencoder_signature_never_takes_labels`), not just
documented. At evaluation time it sees the full mixed normal+abnormal
test set it was never trained on, and anomaly scoring is purely
reconstruction error: a signal the model reconstructs well is
"normal-shaped," one it reconstructs poorly is anomalous.

The anomaly threshold itself is chosen using **only a held-out
validation slice of normal data** — never the test set's true labels
— via `choose_threshold_from_normal_validation`. True labels are used
strictly afterward, to score already-fixed predictions
(`evaluate_predictions`), the same "don't let the answer key leak into
the decision" discipline used elsewhere in this portfolio's honestly-
evaluated projects.

## The semi-supervised comparison

`src/evaluate.py`'s `semi_supervised_baseline_auc` trains a supervised
logistic-regression classifier using only a shrinking fraction of the
available training labels (100% down to 2%), simulating the real-world
constraint semi-supervised methods exist to address: labeled data is
expensive, so most of it is typically unlabeled. This is a baseline
comparison against the unsupervised approach, which needs zero labels
at training time.

## An honest finding from development

The first version of the synthetic fault generator used larger fault
magnitudes and produced a **suspiciously easy dataset**: the
unsupervised autoencoder scored a 0.987 AUC, and investigating why
(rather than reporting it) showed abnormal samples had ~28% higher
total signal energy than normal samples on average — meaning a trivial
"sum of squared values" threshold could nearly solve the task without
any model at all. That's not a real demonstration of learned anomaly
detection, so the fault magnitudes were reduced until the energy gap
between classes was negligible (under 1%, well within each class's own
variance), confirmed by a dedicated regression test
(`test_classes_are_not_trivially_separable_by_energy`). This test was
verified to actually fail against the original, easier generator
(reverting the fault magnitudes reproduced the original 27.9% gap and
the test failed exactly as expected) before being confirmed to pass
against the fix.

After the fix, the unsupervised AUC settled at a genuine, modest
**0.74–0.79 across 5 different random seeds** — real, useful signal,
deliberately not near 1.0. Recall at a conservative (95th-percentile)
threshold is honestly low (~0.28–0.47 across seeds); a looser threshold
(70th percentile) trades that for substantially higher recall (~0.70)
at the cost of precision — a real, reported precision/recall tradeoff,
not a single cherry-picked operating point.

The semi-supervised comparison shows a genuine degradation: supervised
AUC drops from ~0.62 (100% of labels) to ~0.58 (2% of labels), a real
though modest decline that demonstrates the actual motivation for
semi-supervised methods — that labeled data quality/quantity matters —
without exaggerating the effect size.

## Verification performed

- `python3 -m pytest tests/ -v` — 19/19 tests pass, including:
  - A structural check that `train_autoencoder`'s signature makes it
    architecturally impossible to pass labels in, not just unused.
  - A regression test locking in that synthetic classes aren't
    trivially separable by signal energy alone (confirmed to fail
    against the original, buggier generator before the fix).
  - AUC-range and cross-seed stability tests (5 seeds, spread < 0.15).
  - A precision/recall tradeoff test confirming the threshold mechanism
    has real effect in the expected direction.
  - Degenerate-input handling for the semi-supervised comparison
    (single-class label sample returns `None` rather than crashing).
- `python3 -m src.pipeline` — runs the full flow end-to-end and prints
  real metrics (not hardcoded).

## Running it

```bash
pip install -r requirements.txt
python3 -m src.pipeline      # full unsupervised + semi-supervised run
pytest tests/ -v              # 19 tests
```

## What this doesn't demonstrate

This project doesn't use real FordA data (disclosed above, with the
reason), doesn't use federated learning or explainable-AI methods, and
the autoencoder architecture is a simple fully-connected one (not a
convolutional or recurrent architecture, which would likely perform
better on real time-series signals but wasn't necessary to demonstrate
the core unsupervised/semi-supervised methodology honestly). It
demonstrates genuinely unsupervised anomaly detection (verified never to
train on labels), a real semi-supervised label-scarcity comparison, and
the same investigate-before-trusting discipline as the rest of this
portfolio, on a system I could build, break, and verify myself.
