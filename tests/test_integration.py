"""Integration tests: full lifecycle runs on deterministic data."""

from __future__ import annotations

import numpy as np
import pytest

from mlops_lifecycle.data import make_dataset, train_val_test_split
from mlops_lifecycle.lifecycle import Collector, LifecycleRunner
from mlops_lifecycle.models import GateConfig, PromotionDecision, Stage
from mlops_lifecycle.registry import InMemoryRegistry
from mlops_lifecycle.training import (
    InMemoryModelStore,
    SklearnEvaluator,
    SklearnTrainer,
)

pytestmark = pytest.mark.integration


def build_runner(store: InMemoryModelStore, registry: InMemoryRegistry, **kwargs):
    reporter = Collector(messages=[])
    runner = LifecycleRunner(
        trainer=SklearnTrainer(store=store),
        evaluator=SklearnEvaluator(store=store),
        registry=registry,
        store=store,
        reporter=reporter,
        **kwargs,
    )
    return runner, reporter


class TestFullLifecycle:
    def test_first_model_promotes_without_champion(self) -> None:
        store = InMemoryModelStore()
        registry = InMemoryRegistry()
        runner, reporter = build_runner(store, registry)
        X, y, h = make_dataset(n_samples=1500, n_features=6, seed=42)
        train, val, _ = train_val_test_split(X, y, h)
        result = runner.run("churn", train, val)
        assert result.promotion.decision is PromotionDecision.PROMOTED
        assert registry.current("churn", Stage.PRODUCTION) is not None
        assert any("promoted to PRODUCTION" in m for m in reporter.messages)

    def test_retraining_without_lift_gets_rejected(self) -> None:
        """A retrain on statistically identical data must NOT auto-promote."""
        store = InMemoryModelStore()
        registry = InMemoryRegistry()
        runner, _ = build_runner(store, registry)
        X, y, h = make_dataset(n_samples=1500, n_features=6, seed=42)
        train, val, _ = train_val_test_split(X, y, h)
        first = runner.run("churn", train, val)

        # record champion metrics as registry would in a real flow
        from mlops_lifecycle.models import ChampionReport

        registry.record_metrics(
            "churn",
            first.candidate.version,
            ChampionReport(
                model_uri=first.candidate.uri,
                accuracy=first.report.accuracy,
                f1=first.report.f1,
            ),
        )
        # retrain on the same data: high quality but no 2pp lift
        second = runner.run("churn", train, val)
        assert second.promotion.decision is PromotionDecision.REJECTED
        assert any("lift" in r for r in second.promotion.reasons)
        # production still serves the champion
        prod = registry.current("churn", Stage.PRODUCTION)
        assert prod is not None and prod.version == first.candidate.version

    def test_clearly_better_model_promotes(self) -> None:
        """Less label noise in fresh training data -> real lift -> promotion."""
        store = InMemoryModelStore()
        registry = InMemoryRegistry()
        runner, _ = build_runner(store, registry)

        # v1: noisy data (weaker champion)
        Xn, yn, hn = make_dataset(n_samples=1500, n_features=6, seed=42, label_noise=0.18)
        train_n, val_n, _ = train_val_test_split(Xn, yn, hn)
        first = runner.run("churn", train_n, val_n)
        assert first.promotion.decision is PromotionDecision.PROMOTED

        from mlops_lifecycle.models import ChampionReport

        registry.record_metrics(
            "churn",
            first.candidate.version,
            ChampionReport(
                model_uri=first.candidate.uri,
                accuracy=first.report.accuracy,
                f1=first.report.f1,
            ),
        )

        # v2: clean data (much stronger challenger)
        Xc, yc, hc = make_dataset(n_samples=1500, n_features=6, seed=1000)
        train_c, val_c, _ = train_val_test_split(Xc, yc, hc)
        second = runner.run("churn", train_c, val_c)
        lift = second.report.accuracy - first.report.accuracy
        expected = (
            PromotionDecision.PROMOTED if lift >= 0.02 else PromotionDecision.REJECTED
        )
        assert second.promotion.decision is expected
        if expected is PromotionDecision.PROMOTED:
            assert second.report.accuracy > first.report.accuracy

    def test_monitoring_flags_drift_and_decay(self) -> None:
        store = InMemoryModelStore()
        registry = InMemoryRegistry()
        runner, reporter = build_runner(store, registry)
        X, y, h = make_dataset(n_samples=1500, n_features=6, seed=42)
        train, val, _ = train_val_test_split(X, y, h)
        result = runner.run(
            "churn", train, val,
            monitor=make_dataset(n_samples=600, n_features=6, seed=99,
                                 drift_feature=0, drift_scale=1.2, label_noise=0.2)[:2],
            reference_X=X,
        )
        assert result.monitoring is not None
        assert result.monitoring.drift.status.value == "drift"
        assert result.monitoring.drift.drifted_features == ("f0",)
        assert result.monitoring.notes, "expected decay/drift alerts"
        assert any("drift" in m for m in reporter.messages)

    def test_monitoring_clean_batch_passes(self) -> None:
        store = InMemoryModelStore()
        registry = InMemoryRegistry()
        runner, _ = build_runner(store, registry)
        X, y, h = make_dataset(n_samples=1500, n_features=6, seed=42)
        train, val, _ = train_val_test_split(X, y, h)
        X_live, y_live, _ = make_dataset(n_samples=600, n_features=6, seed=77)
        result = runner.run("churn", train, val, monitor=(X_live, y_live), reference_X=X)
        assert result.monitoring is not None
        assert result.monitoring.drift.status.value == "ok"
        assert result.monitoring.notes == ()

    def test_gate_config_can_block_everything(self) -> None:
        store = InMemoryModelStore()
        registry = InMemoryRegistry()
        runner, _ = build_runner(
            store, registry,
            gate=GateConfig(min_accuracy=0.99, min_f1=0.99, min_roc_auc=0.99),
        )
        X, y, h = make_dataset(n_samples=800, n_features=4, seed=3)
        train, val, _ = train_val_test_split(X, y, h)
        result = runner.run("churn", train, val)
        assert result.promotion.decision is PromotionDecision.REJECTED
        assert registry.current("churn", Stage.PRODUCTION) is None


class TestDeterminism:
    def test_same_seed_same_metrics(self) -> None:
        results = []
        for _ in range(2):
            store = InMemoryModelStore()
            registry = InMemoryRegistry()
            runner, _ = build_runner(store, registry)
            X, y, h = make_dataset(n_samples=1000, n_features=5, seed=7)
            train, val, _ = train_val_test_split(X, y, h)
            results.append(runner.run("churn", train, val))
        r1, r2 = results
        assert r1.report.accuracy == r2.report.accuracy
        assert r1.report.f1 == r2.report.f1
        np.testing.assert_allclose(r1.report.roc_auc, r2.report.roc_auc)
