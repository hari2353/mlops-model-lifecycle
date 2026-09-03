"""Shared fixtures for the MLOps lifecycle tests."""

from __future__ import annotations

import pytest

from mlops_lifecycle.data import make_dataset, train_val_test_split
from mlops_lifecycle.models import (
    CandidateModel,
    ChampionReport,
    DatasetHandle,
    EvaluationReport,
)
from mlops_lifecycle.registry import InMemoryRegistry
from mlops_lifecycle.training import InMemoryModelStore, SklearnEvaluator, SklearnTrainer


@pytest.fixture
def clean_split():
    X, y, handle = make_dataset(n_samples=800, n_features=6, seed=42)
    return train_val_test_split(X, y, handle)


@pytest.fixture
def store() -> InMemoryModelStore:
    return InMemoryModelStore()


@pytest.fixture
def trained_candidate(store: InMemoryModelStore, clean_split):
    train, val, _ = clean_split
    trainer = SklearnTrainer(store=store)
    evaluator = SklearnEvaluator(store=store)
    candidate = trainer.train("test-model", train, val)
    report = evaluator.evaluate(candidate, val)
    return candidate, report


def make_candidate(name: str = "m", version: int = 1) -> CandidateModel:
    return CandidateModel(
        name=name,
        version=version,
        params={"C": 1.0},
        train_data=DatasetHandle(name="d-train", n_samples=10, n_features=4, sha8="abc"),
        val_data=DatasetHandle(name="d-val", n_samples=10, n_features=4, sha8="abd"),
    )


def make_report(accuracy: float = 0.9, f1: float = 0.9, roc_auc: float = 0.93) -> EvaluationReport:
    return EvaluationReport(
        model_uri="models:/m/1", accuracy=accuracy, f1=f1, roc_auc=roc_auc, n_samples=100
    )


def make_champion(accuracy: float = 0.8) -> ChampionReport:
    return ChampionReport(model_uri="models:/m/0", accuracy=accuracy, f1=accuracy - 0.05)


@pytest.fixture
def registry() -> InMemoryRegistry:
    return InMemoryRegistry()
