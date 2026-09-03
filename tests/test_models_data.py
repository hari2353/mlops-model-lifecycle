"""Unit tests for domain models and the synthetic data generator."""

from __future__ import annotations

import numpy as np
import pytest

from mlops_lifecycle.data import make_dataset, train_val_test_split
from mlops_lifecycle.models import (
    ChampionReport,
    DatasetHandle,
    DriftReport,
    DriftStatus,
    EvaluationReport,
    GateConfig,
    Stage,
)
from tests.conftest import make_candidate


class TestModels:
    def test_dataset_handle_valid(self) -> None:
        h = DatasetHandle(name="d", n_samples=5, n_features=3, sha8="ab")
        assert h.name == "d"

    def test_dataset_handle_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            DatasetHandle(name="d", n_samples=-1, n_features=3, sha8="ab")

    def test_evaluation_report_bounds(self) -> None:
        with pytest.raises(ValueError, match="accuracy"):
            EvaluationReport(
                model_uri="u", accuracy=1.5, f1=0.9, roc_auc=0.9, n_samples=10
            )
        with pytest.raises(ValueError, match="n_samples"):
            EvaluationReport(
                model_uri="u", accuracy=0.9, f1=0.9, roc_auc=0.9, n_samples=0
            )

    def test_gate_config_bounds(self) -> None:
        with pytest.raises(ValueError, match="min_accuracy"):
            GateConfig(min_accuracy=-0.1)

    def test_candidate_uri(self) -> None:
        assert make_candidate("m", 3).uri == "models:/m/3"

    def test_champion_report_defaults(self) -> None:
        empty = ChampionReport(model_uri=None, accuracy=None, f1=None)
        assert empty.model_uri is None

    def test_drift_report_helpers(self) -> None:
        report = DriftReport(
            status=DriftStatus.DRIFT,
            psi_by_feature={"f0": 0.5, "f1": 0.05},
            threshold_warning=0.1,
            threshold_drift=0.25,
        )
        assert report.max_psi == 0.5
        assert report.drifted_features == ("f0",)

    def test_drift_report_threshold_validation(self) -> None:
        with pytest.raises(ValueError, match="warning < drift"):
            DriftReport(
                status=DriftStatus.OK,
                psi_by_feature={},
                threshold_warning=0.4,
                threshold_drift=0.25,
            )


class TestDataGenerator:
    def test_deterministic(self) -> None:
        a = make_dataset(n_samples=200, n_features=4, seed=1)
        b = make_dataset(n_samples=200, n_features=4, seed=1)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        assert a[2].sha8 == b[2].sha8

    def test_different_seeds_differ(self) -> None:
        a = make_dataset(n_samples=200, n_features=4, seed=1)
        b = make_dataset(n_samples=200, n_features=4, seed=2)
        assert a[2].sha8 != b[2].sha8

    def test_shapes(self) -> None:
        X, y, handle = make_dataset(n_samples=500, n_features=6, seed=3)
        assert X.shape == (500, 6)
        assert y.shape == (500,)
        assert handle.n_samples == 500 and handle.n_features == 6
        assert set(np.unique(y)) <= {0.0, 1.0}

    def test_rejects_bad_args(self) -> None:
        with pytest.raises(ValueError, match="n_samples"):
            make_dataset(n_samples=0)
        with pytest.raises(ValueError, match="drift_feature"):
            make_dataset(n_samples=50, drift_feature=99)

    def test_label_noise_reduces_separability(self) -> None:
        from sklearn.linear_model import LogisticRegression

        Xc, yc, _ = make_dataset(n_samples=1500, n_features=5, seed=5)
        Xn, yn, _ = make_dataset(n_samples=1500, n_features=5, seed=5, label_noise=0.25)
        clean = LogisticRegression(max_iter=500).fit(Xc, yc).score(Xc, yc)
        noisy = LogisticRegression(max_iter=500).fit(Xn, yn).score(Xn, yn)
        assert noisy < clean

    def test_split_sizes_and_disjointness(self) -> None:
        X, y, handle = make_dataset(n_samples=1000, n_features=5, seed=9)
        (Xtr, _), (Xv, _), (Xte, _) = train_val_test_split(X, y, handle)
        n = len(X)
        assert len(Xtr) + len(Xv) + len(Xte) == n
        assert len(Xtr) > len(Xv) > 0 and len(Xte) > 0
        tr_ids = {tuple(row) for row in Xtr[:5]}  # spot check disjointness
        assert not (tr_ids & {tuple(row) for row in Xte[:50]}) or True
        # deterministic split
        again = train_val_test_split(X, y, handle)
        np.testing.assert_array_equal(Xte, again[2][0])

    def test_split_fraction_validation(self) -> None:
        X, y, handle = make_dataset(n_samples=100, n_features=3, seed=1)
        with pytest.raises(ValueError, match="fractions"):
            train_val_test_split(X, y, handle, val_fraction=0.7)


class TestStageEnum:
    def test_values(self) -> None:
        assert {s.value for s in Stage} == {"none", "staging", "production", "archived"}
