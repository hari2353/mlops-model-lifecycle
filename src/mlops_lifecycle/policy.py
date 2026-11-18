"""Alert-to-action policy: closes a monitoring event into an operator action.

Precedence (documented): PAGE > ROLLBACK > RETRAIN > NONE.
- live accuracy below floor -> ROLLBACK (a bad model is serving traffic)
- drift verdict alone      -> RETRAIN (data moved, model still fine)
- both signals             -> PAGE (severe incident: bad model AND moved data)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from mlops_lifecycle.models import DriftStatus, MonitoringEvent


class Action(str, Enum):
    """Operator action a monitoring event maps to."""

    NONE = "none"
    RETRAIN = "retrain"
    ROLLBACK = "rollback"
    PAGE = "page"


@dataclass(frozen=True, slots=True)
class ActionPolicy:
    """Thresholds the policy judges a monitoring event against."""

    min_accuracy: float = 0.70

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_accuracy <= 1.0:
            msg = "min_accuracy must be in [0, 1]"
            raise ValueError(msg)


def decide_action(monitoring: MonitoringEvent, policy: ActionPolicy) -> Action:
    """Pure mapping: one monitoring event -> one operator action."""
    drifted = monitoring.drift.status is DriftStatus.DRIFT
    degraded = monitoring.performance_accuracy < policy.min_accuracy
    if degraded and drifted:
        return Action.PAGE
    if degraded:
        return Action.ROLLBACK
    if drifted:
        return Action.RETRAIN
    return Action.NONE
