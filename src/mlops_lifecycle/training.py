"""Training and evaluation stages.

`Trainer` and `Evaluator` are protocols; the default implementations use
scikit-learn pipelines. The promotion gate is a pure function of an
`EvaluationReport` + `ChampionReport` + `GateConfig` - trivially testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlops_lifecycle.models import (
    CandidateModel,
    ChampionReport,
    DatasetHandle,
    EvaluationReport,
    GateConfig,
    PromotionDecision,
    PromotionResult,
)

FloatArray = NDArray[np.float64]


class Trainer(Protocol):
    """Anything that can train a candidate model on (X, y)."""

    def train(
        self,
        name: str,
        train: tuple[FloatArray, FloatArray],
        val: tuple[FloatArray, FloatArray],
    ) -> CandidateModel: ...


class Evaluator(Protocol):
    """Anything that can evaluate a fitted model on held-out data."""

    def evaluate(
        self, model: CandidateModel, data: tuple[FloatArray, FloatArray]
    ) -> EvaluationReport: ...


class ModelStore(Protocol):
    """Minimal persistence seam for fitted estimators."""

    def save(self, model: CandidateModel, estimator: Any) -> str: ...

    def load(self, model: CandidateModel) -> Any: ...


class InMemoryModelStore:
    """Default store: keeps fitted pipelines in a dict (tests/demo)."""

    def __init__(self) -> None:
        self._models: dict[str, Any] = {}

    def save(self, model: CandidateModel, estimator: Any) -> str:
        key = f"{model.name}-v{model.version}"
        self._models[key] = estimator
        return f"memory://{key}"

    def load(self, model: CandidateModel) -> Any:
        key = f"{model.name}-v{model.version}"
        if key not in self._models:
            msg = f"model not found: {key}"
            raise KeyError(msg)
        return self._models[key]


@dataclass(frozen=True, slots=True)
class SklearnTrainer:
    """Trains a standardized logistic regression and records its params."""

    store: ModelStore
    next_version: int = 1
    C: float = 1.0

    def train(
        self,
        name: str,
        train: tuple[FloatArray, FloatArray],
        val: tuple[FloatArray, FloatArray],
    ) -> CandidateModel:
        X_train, y_train = train
        if len(X_train) == 0:
            msg = "training data must be non-empty"
            raise ValueError(msg)

        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=self.C, max_iter=1000)),
            ]
        )
        pipeline.fit(X_train, y_train)

        train_handle = DatasetHandle(
            name=f"{name}-train",
            n_samples=len(X_train),
            n_features=X_train.shape[1],
            sha8="-",
        )
        val_handle = DatasetHandle(
            name=f"{name}-val",
            n_samples=len(val[0]),
            n_features=val[0].shape[1],
            sha8="-",
        )
        candidate = CandidateModel(
            name=name,
            version=self.next_version,
            params={"C": self.C, "max_iter": 1000},
            train_data=train_handle,
            val_data=val_handle,
        )
        self.store.save(candidate, pipeline)
        return candidate


@dataclass(frozen=True, slots=True)
class SklearnEvaluator:
    """Computes accuracy/F1/ROC-AUC on held-out data via the store."""

    store: ModelStore

    def evaluate(
        self, model: CandidateModel, data: tuple[FloatArray, FloatArray]
    ) -> EvaluationReport:
        X, y = data
        estimator = self.store.load(model)
        proba = estimator.predict_proba(X)[:, 1]
        preds = (proba >= 0.5).astype(np.float64)
        return EvaluationReport(
            model_uri=model.uri,
            accuracy=float(accuracy_score(y, preds)),
            f1=float(f1_score(y, preds)),
            roc_auc=float(roc_auc_score(y, proba)),
            n_samples=len(X),
        )


def decide_promotion(
    report: EvaluationReport,
    champion: ChampionReport,
    config: GateConfig,
    *,
    now=None,
) -> PromotionResult:
    """Apply the promotion gate: absolute thresholds + lift over champion."""
    from datetime import datetime, timezone

    reasons: list[str] = []

    if report.accuracy < config.min_accuracy:
        reasons.append(f"accuracy {report.accuracy:.3f} < min {config.min_accuracy:.2f}")
    if report.f1 < config.min_f1:
        reasons.append(f"f1 {report.f1:.3f} < min {config.min_f1:.2f}")
    if report.roc_auc < config.min_roc_auc:
        reasons.append(f"roc_auc {report.roc_auc:.3f} < min {config.min_roc_auc:.2f}")

    if champion.model_uri is not None and champion.accuracy is not None:
        lift = report.accuracy - champion.accuracy
        if lift < config.min_lift_over_champion:
            reasons.append(
                f"lift over champion {lift:+.3f} < required {config.min_lift_over_champion:+.2f}"
            )
    elif config.require_champion_for_promotion:
        reasons.append("no champion present but champion required for promotion")

    decision = PromotionDecision.PROMOTED if not reasons else PromotionDecision.REJECTED
    return PromotionResult(
        decision=decision,
        reasons=tuple(reasons) if reasons else ("all gates passed",),
        report=report,
        champion=champion,
        decided_at=now or datetime.now(tz=timezone.utc),
    )
