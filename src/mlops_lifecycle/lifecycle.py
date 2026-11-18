"""Lifecycle orchestrator: wires train -> gate -> register -> promote -> monitor.

`LifecycleRunner` depends only on the protocols (Trainer, Evaluator,
ModelRegistry, ModelStore) - concrete implementations are injected at the
composition root. This is the class `scripts/run_lifecycle.py` drives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from mlops_lifecycle.models import (
    CandidateModel,
    ChampionReport,
    EvaluationReport,
    GateConfig,
    MonitoringEvent,
    PromotionDecision,
    PromotionResult,
    Stage,
)
from mlops_lifecycle.monitoring import MonitoringConfig, monitor_batch
from mlops_lifecycle.policy import Action, ActionPolicy, decide_action
from mlops_lifecycle.registry import ModelRegistry
from mlops_lifecycle.training import (
    Evaluator,
    ModelStore,
    Trainer,
    decide_promotion,
)


class ProgressReporter(Protocol):
    """Observability seam: the CLI prints, tests collect, CI logs."""

    def step(self, message: str) -> None: ...


@dataclass
class Collector:
    """Test-friendly reporter that records every step message."""

    messages: list[str]

    def step(self, message: str) -> None:
        self.messages.append(message)


class NullReporter:
    """Silent reporter."""

    def step(self, message: str) -> None:  # pragma: no cover - trivial
        pass


@dataclass
class LifecycleRunResult:
    """Everything one full lifecycle run produced, for CLI and tests."""

    candidate: CandidateModel
    report: EvaluationReport
    promotion: PromotionResult
    monitoring: MonitoringEvent | None = None
    recommended_action: Action | None = None


class LifecycleRunner:
    """End-to-end model lifecycle with automated promotion and monitoring."""

    def __init__(
        self,
        trainer: Trainer,
        evaluator: Evaluator,
        registry: ModelRegistry,
        store: ModelStore,
        gate: GateConfig | None = None,
        monitoring_config: MonitoringConfig | None = None,
        reporter: ProgressReporter | None = None,
    ) -> None:
        self._trainer = trainer
        self._evaluator = evaluator
        self._registry = registry
        self._store = store
        self._gate = gate or GateConfig()
        self._monitoring_config = monitoring_config or MonitoringConfig()
        self._reporter = reporter or NullReporter()

    def run(
        self,
        model_name: str,
        train: tuple,
        val: tuple,
        *,
        champion_source: ChampionReport | None = None,
        monitor: tuple | None = None,
        reference_X=None,
    ) -> LifecycleRunResult:
        """Train, evaluate, gate, register, promote (optionally monitor)."""
        self._reporter.step(f"training candidate '{model_name}'")
        candidate = self._trainer.train(model_name, train, val)

        self._reporter.step("evaluating on validation split")
        report = self._evaluator.evaluate(candidate, val)

        champion = champion_source or self._registry.champion_report(model_name)
        self._reporter.step(
            f"promotion gate: champion="
            f"{champion.model_uri or 'none'}"
        )
        promotion = decide_promotion(report, champion, self._gate)

        version = self._registry.register(candidate)
        candidate = _with_version(candidate, version)

        if promotion.decision is PromotionDecision.PROMOTED:
            self._reporter.step(f"v{version}: staging -> PRODUCTION")
            self._registry.transition(model_name, version, Stage.STAGING)
            entry = self._registry.transition(model_name, version, Stage.PRODUCTION)
            self._reporter.step(
                f"v{version} promoted to PRODUCTION at {entry.at.isoformat()}"
            )
        else:
            self._reporter.step(f"v{version} rejected: {'; '.join(promotion.reasons)}")

        monitoring = None
        recommended_action = None
        if monitor is not None and reference_X is not None:
            estimator = self._store.load(candidate)
            monitoring = monitor_batch(
                candidate,
                estimator,
                reference_X,
                monitor,
                self._monitoring_config,
            )
            recommended_action = decide_action(
                monitoring,
                ActionPolicy(min_accuracy=self._monitoring_config.min_accuracy),
            )
            self._reporter.step(
                f"monitoring: drift={monitoring.drift.status.value} "
                f"max_psi={monitoring.drift.max_psi:.3f} "
                f"live_acc={monitoring.performance_accuracy:.3f} "
                f"action={recommended_action.value}"
                + (f" notes={list(monitoring.notes)}" if monitoring.notes else "")
            )

        return LifecycleRunResult(
            candidate=candidate,
            report=report,
            promotion=promotion,
            monitoring=monitoring,
            recommended_action=recommended_action,
        )


def _with_version(candidate: CandidateModel, version: int) -> CandidateModel:
    from dataclasses import replace

    return replace(candidate, version=version)
