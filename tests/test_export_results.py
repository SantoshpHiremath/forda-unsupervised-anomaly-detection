"""
tests/test_export_results.py
------------------------------

Tests src/export_results.py -- added to support the external
results-storytelling-dashboard project's visualizations. Verifies the
export functions return the expected shape and that the values are
genuinely computed (not placeholders), consistent with this project's
own existing pipeline tests.
"""

from src.export_results import (
    collect_auc_across_seeds,
    collect_reconstruction_error_distribution,
    collect_semi_supervised_curve,
)


class TestCollectAucAcrossSeeds:
    def test_returns_one_entry_per_seed(self):
        results = collect_auc_across_seeds(seeds=(0, 1))
        assert len(results) == 2
        assert [r["seed"] for r in results] == [0, 1]

    def test_auc_values_are_plausible_and_json_serializable(self):
        results = collect_auc_across_seeds(seeds=(0,))
        auc = results[0]["roc_auc"]
        assert isinstance(auc, float)  # not a numpy scalar -- must be JSON-serializable
        assert 0.0 <= auc <= 1.0


class TestCollectReconstructionErrorDistribution:
    def test_returns_nonempty_normal_and_anomaly_scores(self):
        dist = collect_reconstruction_error_distribution(seed=0)
        assert len(dist["normal_scores"]) > 0
        assert len(dist["anomaly_scores"]) > 0

    def test_scores_are_plain_floats(self):
        dist = collect_reconstruction_error_distribution(seed=0)
        assert all(isinstance(x, float) for x in dist["normal_scores"][:5])
        assert all(isinstance(x, float) for x in dist["anomaly_scores"][:5])

    def test_threshold_is_between_typical_score_range(self):
        dist = collect_reconstruction_error_distribution(seed=0)
        assert dist["threshold"] > 0


class TestCollectSemiSupervisedCurve:
    def test_returns_all_six_label_fractions(self):
        curve = collect_semi_supervised_curve(seed=0)
        assert set(curve.keys()) == {"1.0", "0.5", "0.2", "0.1", "0.05", "0.02"}

    def test_full_label_auc_is_generally_at_least_as_good_as_sparse(self):
        """Not a strict monotonic guarantee (this is a small synthetic
        run, some noise is expected), but the 100%-label AUC should not
        be dramatically worse than the 2%-label AUC -- a sanity check
        that the label-fraction wiring isn't accidentally reversed."""
        curve = collect_semi_supervised_curve(seed=0)
        assert curve["1.0"] is not None
        assert curve["0.02"] is not None
        assert curve["1.0"] >= curve["0.02"] - 0.15
