"""Market-steal additions (2026-11-18): one-call rollback, generated model
cards, alert-to-action policy.

Compared products treat these as baseline: SageMaker/Vertex give operators a
one-call revert to the archived previous champion; EU-AI-Act-era governance
expects a generated, bounded model card per promoted model; monitoring stacks
(Evidently, SageMaker Model Monitor) close alerts into an action decision.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

import pytest

from mlops_lifecycle.data import make_dataset, train_val_test_split
from mlops_lifecycle.lifecycle import LifecycleRunner
from mlops_lifecycle.models import (
    ChampionReport,
    DriftReport,
    DriftStatus,
    MonitoringEvent,
    Stage,
)
from mlops_lifecycle.policy import Action, ActionPolicy, decide_action
from mlops_lifecycle.registry import InMemoryRegistry, RegistryError, model_card
from mlops_lifecycle.training import (
    InMemoryModelStore,
    SklearnEvaluator,
    SklearnTrainer,
)
from tests.conftest import make_candidate

RegistryFactory = Callable[[], InMemoryRegistry]


@pytest.fixture(params=["memory"])
def registry_factory(request) -> RegistryFactory:
    return InMemoryRegistry


def promote(registry: InMemoryRegistry, name: str, version: int) -> None:
    registry.register(make_candidate(name, version))
    registry.transition(name, version, Stage.STAGING)
    registry.transition(name, version, Stage.PRODUCTION)


def event(accuracy: float, status: DriftStatus) -> MonitoringEvent:
    """Minimal MonitoringEvent for pure policy tests."""
    return MonitoringEvent(
        model_uri="models:/m/1",
        drift=DriftReport(
            status=status,
            psi_by_feature={"f0": 0.5 if status is DriftStatus.DRIFT else 0.0},
            threshold_warning=0.10,
            threshold_drift=0.25,
        ),
        performance_accuracy=accuracy,
        checked_at=datetime.now(tz=timezone.utc),
    )


# ── rollback ─────────────────────────────────────────────────────────────────
class TestRollbackContract:
    def test_rollback_repromotes_highest_archived(self, registry_factory) -> None:
        reg = registry_factory()
        promote(reg, "m", 1)
        promote(reg, "m", 2)  # auto-archives v1
        assert reg.current("m", Stage.PRODUCTION).version == 2

        entry = reg.rollback("m")

        assert entry.version == 1 and entry.to_stage is Stage.PRODUCTION
        assert reg.current("m", Stage.PRODUCTION).version == 1
        # v2 auto-archived by the rollback (promote-to-prod semantics)
        hist = reg.history("m")
        assert any(h.version == 2 and h.to_stage is Stage.ARCHIVED for h in hist)

    def test_rollback_default_picks_highest_archived(self, registry_factory) -> None:
        reg = registry_factory()
        for v in (1, 2, 3):
            promote(reg, "m", v)
        reg.transition("m", 1, Stage.NONE)  # drop v1 out of the running
        entry = reg.rollback("m")
        assert entry.version == 2  # highest remaining archived

    def test_rollback_explicit_target(self, registry_factory) -> None:
        reg = registry_factory()
        for v in (1, 2, 3):
            promote(reg, "m", v)
        entry = reg.rollback("m", to_version=1)
        assert entry.version == 1
        assert reg.current("m", Stage.PRODUCTION).version == 1
        assert reg.current("m", Stage.ARCHIVED).version == 3

    def test_rollback_errors_when_nothing_archived(self, registry_factory) -> None:
        reg = registry_factory()
        reg.register(make_candidate("m", 1))
        with pytest.raises(RegistryError):
            reg.rollback("m")

    def test_rollback_explicit_non_archived_errors(self, registry_factory) -> None:
        reg = registry_factory()
        promote(reg, "m", 1)
        with pytest.raises(RegistryError):
            reg.rollback("m", to_version=1)  # v1 is live, not archived

    def test_rollback_unknown_model_errors(self, registry_factory) -> None:
        reg = registry_factory()
        with pytest.raises(RegistryError):
            reg.rollback("ghost")


# ── model card ───────────────────────────────────────────────────────────────
class TestModelCard:
    def test_card_renders_identity_metrics_history(self, registry_factory) -> None:
        reg = registry_factory()
        promote(reg, "m", 1)
        reg.record_metrics(
            "m", 1, ChampionReport(model_uri="models:/m/1", accuracy=0.978, f1=0.979)
        )
        reg.register(make_candidate("m", 2))
        reg.transition("m", 2, Stage.STAGING)

        card = model_card(reg, "m")

        assert "# Model card: m" in card
        assert "v1" in card and "production" in card
        assert "0.978" in card and "0.979" in card
        assert "none -> staging" in card and "staging -> production" in card
        assert "generated" in card.lower()
        assert len(card) <= 4000

    def test_card_explicit_version_shows_its_stage(self, registry_factory) -> None:
        reg = registry_factory()
        promote(reg, "m", 1)
        reg.register(make_candidate("m", 2))
        reg.transition("m", 2, Stage.STAGING)
        card = model_card(reg, "m", version=2)
        assert "v2" in card and "staging" in card

    def test_card_never_invents_metrics(self, registry_factory) -> None:
        reg = registry_factory()
        promote(reg, "m", 1)  # no record_metrics call
        card = model_card(reg, "m")
        assert "not recorded" in card.lower()

    def test_card_bounded_under_long_history(self, registry_factory) -> None:
        reg = registry_factory()
        for v in range(1, 61):
            promote(reg, "m", v)
        card = model_card(reg, "m")
        assert len(card) <= 4000
        assert "truncated" in card.lower()

    def test_card_unknown_model_errors(self, registry_factory) -> None:
        reg = registry_factory()
        with pytest.raises(RegistryError):
            model_card(reg, "ghost")


# ── alert -> action policy ───────────────────────────────────────────────────
class TestDecideAction:
    def test_clean_batch_means_none(self) -> None:
        assert decide_action(event(0.95, DriftStatus.OK), ActionPolicy()) is Action.NONE

    def test_drift_alone_means_retrain(self) -> None:
        assert decide_action(event(0.95, DriftStatus.DRIFT), ActionPolicy()) is Action.RETRAIN

    def test_accuracy_floor_breach_means_rollback(self) -> None:
        assert decide_action(event(0.5, DriftStatus.OK), ActionPolicy()) is Action.ROLLBACK

    def test_both_signals_means_page(self) -> None:
        assert decide_action(event(0.5, DriftStatus.DRIFT), ActionPolicy()) is Action.PAGE

    def test_policy_floor_is_configurable(self) -> None:
        policy = ActionPolicy(min_accuracy=0.99)
        assert decide_action(event(0.97, DriftStatus.OK), policy) is Action.ROLLBACK


# ── runner wiring ────────────────────────────────────────────────────────────
def _runner() -> LifecycleRunner:
    store = InMemoryModelStore()
    return LifecycleRunner(
        store=store,
        registry=InMemoryRegistry(),
        trainer=SklearnTrainer(store=store),
        evaluator=SklearnEvaluator(store=store),
    )


class TestRunnerWiring:
    def test_recommended_action_set_when_monitoring_ran(self) -> None:
        runner = _runner()
        X, y, h = make_dataset(n_samples=1200, n_features=6, seed=42)
        train, val, _ = train_val_test_split(X, y, h)
        live = make_dataset(
            n_samples=600, n_features=6, seed=99,
            drift_feature=0, drift_scale=1.2, label_noise=0.2,
        )[:2]

        result = runner.run("m", train, val, monitor=live, reference_X=X)
        assert result.recommended_action is not None
        assert isinstance(result.recommended_action, Action)

    def test_recommended_action_none_without_monitoring(self) -> None:
        runner = _runner()
        X, y, h = make_dataset(n_samples=1200, n_features=6, seed=42)
        train, val, _ = train_val_test_split(X, y, h)
        result = runner.run("m", train, val)
        assert result.recommended_action is None
