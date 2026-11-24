"""Drift and performance monitoring.

Computes PSI (Population Stability Index) per feature against a reference
(training) sample, classifies drift, and watches live accuracy. PSI is the
industry-standard drift measure for tabular features; thresholds follow the
common 0.1 / 0.25 rule of thumb.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score

from mlops_lifecycle.models import (
    CandidateModel,
    DriftReport,
    DriftStatus,
    MonitoringEvent,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class MonitoringConfig:
    """PSI thresholds and decay alerting limits."""

    psi_warning: float = 0.10
    psi_drift: float = 0.25
    min_accuracy: float = 0.70

    def __post_init__(self) -> None:
        if not 0 < self.psi_warning < self.psi_drift:
            msg = "must satisfy 0 < psi_warning < psi_drift"
            raise ValueError(msg)
        if not 0.0 <= self.min_accuracy <= 1.0:
            msg = "min_accuracy must be in [0, 1]"
            raise ValueError(msg)


def compute_psi(
    reference: FloatArray, live: FloatArray, *, bins: int = 10, eps: float = 1e-6
) -> float:
    """Population Stability Index between two 1-D samples.

    Both samples are binned on shared quantile edges of the reference, so
    the result is symmetric in bin choice and stable for CI.
    """
    if reference.ndim != 1 or live.ndim != 1:
        msg = "psi inputs must be 1-D"
        raise ValueError(msg)
    if len(reference) < bins or len(live) < bins:
        msg = f"need at least {bins} samples per side for {bins} bins"
        raise ValueError(msg)

    edges = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 2:
        return 0.0  # degenerate constant feature: no drift signal

    ref_counts = np.histogram(reference, bins=edges)[0].astype(np.float64)
    live_counts = np.histogram(live, bins=edges)[0].astype(np.float64)

    ref_frac = ref_counts / ref_counts.sum()
    live_frac = live_counts / live_counts.sum()
    ref_frac = np.clip(ref_frac, eps, None)
    live_frac = np.clip(live_frac, eps, None)

    return float(np.sum((live_frac - ref_frac) * np.log(live_frac / ref_frac)))


def drift_report(
    reference_X: FloatArray,
    live_X: FloatArray,
    config: MonitoringConfig,
    feature_names: list[str] | None = None,
) -> DriftReport:
    """PSI per feature + a status verdict (ok / warning / drift)."""
    if reference_X.shape[1] != live_X.shape[1]:
        msg = "reference and live must share the feature dimension"
        raise ValueError(msg)
    names = feature_names or [f"f{i}" for i in range(reference_X.shape[1])]

    psi_by_feature: dict[str, float] = {}
    for i, name in enumerate(names):
        psi_by_feature[name] = round(compute_psi(reference_X[:, i], live_X[:, i]), 4)

    if any(v >= config.psi_drift for v in psi_by_feature.values()):
        status = DriftStatus.DRIFT
    elif any(v >= config.psi_warning for v in psi_by_feature.values()):
        status = DriftStatus.WARNING
    else:
        status = DriftStatus.OK

    return DriftReport(
        status=status,
        psi_by_feature=psi_by_feature,
        threshold_warning=config.psi_warning,
        threshold_drift=config.psi_drift,
    )


def monitor_batch(
    model: CandidateModel,
    estimator,
    reference_X: FloatArray,
    live: tuple[FloatArray, FloatArray],
    config: MonitoringConfig,
    *,
    now: datetime | None = None,
) -> MonitoringEvent:
    """One monitoring pass: drift check + live accuracy vs. floor."""
    X_live, y_live = live
    proba = estimator.predict_proba(X_live)[:, 1]
    preds = (proba >= 0.5).astype(np.float64)
    accuracy = float(accuracy_score(y_live, preds))

    drift = drift_report(reference_X, X_live, config)

    notes: list[str] = []
    if accuracy < config.min_accuracy:
        notes.append(f"live accuracy {accuracy:.3f} below floor {config.min_accuracy:.2f}")
    if drift.status is DriftStatus.DRIFT:
        notes.append(f"covariate drift on {', '.join(drift.drifted_features)}")

    return MonitoringEvent(
        model_uri=model.uri,
        drift=drift,
        performance_accuracy=accuracy,
        checked_at=now or datetime.now(tz=timezone.utc),
        notes=tuple(notes),
    )
