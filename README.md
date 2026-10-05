# Unsupervised & Semi-Supervised Anomaly Detection on Motor Time-Series Data

## What it does

A tested Python project applying unsupervised and semi-supervised learning to time-series anomaly detection. My earlier projects use supervised classification; here the main result is produced without training on labels at all.

It builds on my contribution to the "Smart Motor" university team project (OTH Amberg-Weiden, 3rd semester), where I handled data loading, visualization, and quality checks on the FordA time-series dataset (see `docs/smart_motor_contribution_report.pdf`). This project adds the modeling work, written from scratch.

### The core idea: genuinely unsupervised

The unsupervised detector (`src/autoencoder.py`) is a PyTorch autoencoder trained **only on signals from the normal class**. Its training function (`train_autoencoder`) has no parameter for labels at all, which is checked structurally by a test (`test_train_autoencoder_signature_never_takes_labels`) as well as documented. At evaluation time it sees the full mixed normal+abnormal test set it was never trained on, and anomaly scoring is purely reconstruction error: a signal the model reconstructs well is "normal-shaped," one it reconstructs poorly is anomalous.

The anomaly threshold is chosen using **only a held-out validation slice of normal data**, never the test set's true labels, via `choose_threshold_from_normal_validation`. True labels are used strictly afterward, to score already-fixed predictions (`evaluate_predictions`), so the answer key never leaks into the decision.

### The semi-supervised comparison

`src/evaluate.py`'s `semi_supervised_baseline_auc` trains a supervised logistic-regression classifier using only a shrinking fraction of the available training labels (100% down to 2%), simulating the real-world constraint semi-supervised methods exist to address: labeled data is expensive, so most of it is typically unlabeled. This is a baseline comparison against the unsupervised approach, which needs zero labels at training time.

## Data

The data is synthetic. `src/generate_signals.py` generates a univariate time-series dataset with the exact shape of FordA (3,601 samples, 500 timesteps, binary label) and the same framing (a motor vibration signal, normal vs. abnormal). The real FordA dataset (UCR/UEA time-series archive) was not reachable from the environment where I built this, so every result below is verifiable against inspectable, generated data. `generate_signals.py` is the only module that would need to be swapped for a real `fetch_openml('FordA')` call; everything downstream (the autoencoder, the evaluation harness, the semi-supervised comparison) is written against the same `(X, y)` shape and works unchanged.

## Results

I first built the synthetic fault generator with larger fault magnitudes, which produced a very easy dataset: the unsupervised autoencoder scored a 0.987 AUC. Investigating showed abnormal samples had ~28% higher total signal energy than normal samples on average, so a trivial "sum of squared values" threshold could nearly solve the task without any model. I reduced the fault magnitudes until the energy gap between classes was negligible (under 1%, well within each class's own variance), and locked this in with a regression test (`test_classes_are_not_trivially_separable_by_energy`). I verified the test fails against the original generator (reverting the fault magnitudes reproduced the 27.9% gap) and passes against the fix.

After the fix, the unsupervised AUC settled at **0.74–0.79 across 5 different random seeds**, a modest and real signal. Recall at a conservative (95th-percentile) threshold is ~0.28–0.47 across seeds; a looser threshold (70th percentile) trades that for substantially higher recall (~0.70) at the cost of precision, so I report the precision/recall tradeoff rather than a single operating point.

The semi-supervised comparison shows a modest degradation: supervised AUC drops from ~0.62 (100% of labels) to ~0.58 (2% of labels), which illustrates the motivation for semi-supervised methods, that labeled data quantity matters.

## Tests

`python3 -m pytest tests/ -v` — 19/19 tests pass, including:

- A structural check that `train_autoencoder`'s signature makes it impossible to pass labels in, not just unused.
- A regression test locking in that synthetic classes aren't trivially separable by signal energy alone.
- AUC-range and cross-seed stability tests (5 seeds, spread < 0.15).
- A precision/recall tradeoff test confirming the threshold mechanism has real effect in the expected direction.
- Degenerate-input handling for the semi-supervised comparison (a single-class label sample returns `None` rather than crashing).

`python3 -m src.pipeline` runs the full flow end to end and prints metrics computed from the run (not hardcoded).

## Running it

```bash
pip install -r requirements.txt
python3 -m src.pipeline      # full unsupervised + semi-supervised run
pytest tests/ -v              # 19 tests
```

## Notes

The autoencoder is a simple fully-connected architecture, which is enough to demonstrate the unsupervised and semi-supervised methodology end to end.

## Possible extensions

- Run on the real FordA data by swapping `src/generate_signals.py`.
- Try convolutional or recurrent autoencoders, which would likely suit real time-series signals better.
- Add federated-learning or explainable-AI methods.
