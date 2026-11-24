"""Unit tests for training, evaluation, and the promotion gate."""

from __future__ import annotations

import pytest

from mlops_lifecycle.models import ChampionReport, GateConfig, PromotionDecision
from mlops_lifecycle.training import (
    InMemoryModelStore,
    SklearnEvaluator,
    SklearnTrainer,
    decide_promotion,
)
from tests.conftest import make_champion, make_report


class TestTraining:
    def test_train_produces_fitted_model(self, store: InMemoryModelStore, clean_split) -> None:
        train, val, _ = clean_split
        trainer = SklearnTrainer(store=store)
        candidate = trainer.train("t", train, val)
        assert candidate.name == "t"
        assert candidate.version >= 1
        assert store.load(candidate) is not None

    def test_train_empty_rejected(self, store: InMemoryModelStore) -> None:
        import numpy as np

        trainer = SklearnTrainer(store=store)
        with pytest.raises(ValueError, match="non-empty"):
            trainer.train("t", (np.empty((0, 4)), np.empty(0)), (np.empty((0, 4)), np.empty(0)))

    def test_evaluate_metrics_in_range(self, store: InMemoryModelStore, clean_split) -> None:
        train, val, _ = clean_split
        trainer = SklearnTrainer(store=store)
        evaluator = SklearnEvaluator(store=store)
        candidate = trainer.train("t", train, val)
        report = evaluator.evaluate(candidate, val)
        assert 0.0 <= report.accuracy <= 1.0
        assert 0.0 <= report.f1 <= 1.0
        assert 0.0 <= report.roc_auc <= 1.0
        assert report.n_samples == len(val[0])

    def test_store_missing_model_raises(self) -> None:

        from tests.conftest import make_candidate

        store = InMemoryModelStore()
        ghost = make_candidate("ghost", 9)
        with pytest.raises(KeyError, match="ghost-v9"):
            store.load(ghost)


class TestPromotionGate:
    def test_clean_pass_with_no_champion(self) -> None:
        result = decide_promotion(
            make_report(0.9, 0.9, 0.95),
            ChampionReport(model_uri=None, accuracy=None, f1=None),
            GateConfig(),
        )
        assert result.decision is PromotionDecision.PROMOTED
        assert result.reasons == ("all gates passed",)

    def test_rejected_on_low_accuracy(self) -> None:
        result = decide_promotion(
            make_report(0.55, 0.9, 0.95),
            ChampionReport(model_uri=None, accuracy=None, f1=None),
            GateConfig(),
        )
        assert result.decision is PromotionDecision.REJECTED
        assert any("accuracy" in r for r in result.reasons)

    def test_rejected_on_low_f1_and_auc(self) -> None:
        result = decide_promotion(
            make_report(0.9, 0.6, 0.6),
            ChampionReport(model_uri=None, accuracy=None, f1=None),
            GateConfig(),
        )
        assert result.decision is PromotionDecision.REJECTED
        assert any("f1" in r for r in result.reasons)
        assert any("roc_auc" in r for r in result.reasons)

    def test_lift_over_champion_required(self) -> None:
        # champion at 0.85, candidate at 0.86 -> lift 0.01 < 0.02 -> reject
        result = decide_promotion(make_report(0.86, 0.9, 0.95), make_champion(0.85), GateConfig())
        assert result.decision is PromotionDecision.REJECTED
        assert any("lift" in r for r in result.reasons)

    def test_lift_gate_passes_when_beating_champion(self) -> None:
        result = decide_promotion(make_report(0.90, 0.9, 0.95), make_champion(0.85), GateConfig())
        assert result.decision is PromotionDecision.PROMOTED

    def test_champion_required_option(self) -> None:
        result = decide_promotion(
            make_report(0.9, 0.9, 0.95),
            ChampionReport(model_uri=None, accuracy=None, f1=None),
            GateConfig(require_champion_for_promotion=True),
        )
        assert result.decision is PromotionDecision.REJECTED
        assert any("champion required" in r for r in result.reasons)

    def test_decision_timestamp_present(self) -> None:
        result = decide_promotion(
            make_report(), ChampionReport(model_uri=None, accuracy=None, f1=None), GateConfig()
        )
        assert result.decided_at is not None
