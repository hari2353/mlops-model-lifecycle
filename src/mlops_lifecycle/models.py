"""Domain models for the MLOps lifecycle.

Typed, frozen dataclasses are the stable core every stage depends on:
dataset handles, model candidates, evaluation reports, promotion decisions,
and monitoring verdicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Stage(str, Enum):
    """Registry stage of a model version (mirrors MLflow semantics)."""

    NONE = "none"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class PromotionDecision(str, Enum):
    """Outcome of the automated promotion gate."""

    PROMOTED = "promoted"
    REJECTED = "rejected"


class DriftStatus(str, Enum):
    """Verdict of a drift check on incoming data."""

    OK = "ok"
    WARNING = "warning"
    DRIFT = "drift"


@dataclass(frozen=True, slots=True)
class DatasetHandle:
    """Reference to a deterministic dataset split with its fingerprint."""

    name: str
    n_samples: int
    n_features: int
    sha8: str  # first 8 hex chars of the row-content hash

    def __post_init__(self) -> None:
        if self.n_samples < 0 or self.n_features < 0:
            msg = "dataset dimensions must be non-negative"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class CandidateModel:
    """A trained model plus the data it was trained on."""

    name: str
    version: int
    params: dict[str, float | int | str]
    train_data: DatasetHandle
    val_data: DatasetHandle

    @property
    def uri(self) -> str:
        return f"models:/{self.name}/{self.version}"


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Metrics for a candidate on held-out validation data."""

    model_uri: str
    accuracy: float
    f1: float
    roc_auc: float
    n_samples: int

    def __post_init__(self) -> None:
        for label, value in (
            ("accuracy", self.accuracy),
            ("f1", self.f1),
            ("roc_auc", self.roc_auc),
        ):
            if not 0.0 <= value <= 1.0:
                msg = f"{label} must be in [0, 1], got {value}"
                raise ValueError(msg)
        if self.n_samples <= 0:
            msg = "n_samples must be positive"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ChampionReport:
    """Metrics for the current production champion (may be None/empty)."""

    model_uri: str | None
    accuracy: float | None
    f1: float | None


@dataclass(frozen=True, slots=True)
class GateConfig:
    """Thresholds for automated promotion. All must hold to promote."""

    min_accuracy: float = 0.70
    min_f1: float = 0.70
    min_roc_auc: float = 0.75
    min_lift_over_champion: float = 0.02  # candidate must beat champion by 2pp
    require_champion_for_promotion: bool = False

    def __post_init__(self) -> None:
        for label, value in (
            ("min_accuracy", self.min_accuracy),
            ("min_f1", self.min_f1),
            ("min_roc_auc", self.min_roc_auc),
            ("min_lift_over_champion", self.min_lift_over_champion),
        ):
            if not 0.0 <= value <= 1.0:
                msg = f"{label} must be in [0, 1], got {value}"
                raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class PromotionResult:
    """Outcome of a promotion-gate evaluation."""

    decision: PromotionDecision
    reasons: tuple[str, ...]
    report: EvaluationReport
    champion: ChampionReport
    decided_at: datetime


@dataclass(frozen=True, slots=True)
class DriftReport:
    """Result of comparing live feature distributions to training ones."""

    status: DriftStatus
    psi_by_feature: dict[str, float]
    threshold_warning: float
    threshold_drift: float

    def __post_init__(self) -> None:
        if not 0 < self.threshold_warning < self.threshold_drift:
            msg = "thresholds must satisfy 0 < warning < drift"
            raise ValueError(msg)

    @property
    def max_psi(self) -> float:
        return max(self.psi_by_feature.values(), default=0.0)

    @property
    def drifted_features(self) -> tuple[str, ...]:
        return tuple(
            name for name, psi in self.psi_by_feature.items() if psi >= self.threshold_drift
        )


@dataclass(frozen=True, slots=True)
class MonitoringEvent:
    """One monitoring outcome for one batch of live data."""

    model_uri: str
    drift: DriftReport
    performance_accuracy: float
    checked_at: datetime
    notes: tuple[str, ...] = field(default_factory=tuple)
