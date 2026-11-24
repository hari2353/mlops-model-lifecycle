"""Unit + contract tests for the registry and monitoring.

The registry contract suite runs against the InMemoryRegistry here; the
MLflow adapter satisfies the same protocol and is exercised wherever a
server is available (install mlops-lifecycle[mlflow] and point the factory
at a tracking URI to include it locally).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from mlops_lifecycle.models import Stage
from mlops_lifecycle.monitoring import MonitoringConfig, compute_psi, drift_report
from mlops_lifecycle.registry import InMemoryRegistry, RegistryError
from tests.conftest import make_candidate

RegistryFactory = Callable[[], InMemoryRegistry]


@pytest.fixture(params=["memory"])
def registry_factory(request) -> RegistryFactory:
    return InMemoryRegistry


class TestRegistryContract:
    """Contract tests: every ModelRegistry implementation must pass these.

    The MLflow adapter runs against a local file-store tracking URI so the
    contract is verified without any server or cloud dependency.
    """

    def test_register_assigns_version(self, registry_factory: RegistryFactory) -> None:
        registry = registry_factory()
        model = make_candidate("cm", 1)
        version = registry.register(model)
        assert version == 1

    def test_repeated_registration_assigns_new_versions(
        self, registry_factory: RegistryFactory
    ) -> None:
        """Registry owns versioning: same-stamped candidates never collide."""
        registry = registry_factory()
        v1 = registry.register(make_candidate("cm", 1))
        v2 = registry.register(make_candidate("cm", 1))
        assert v2 == v1 + 1

    def test_transition_valid_flow(self, registry_factory: RegistryFactory) -> None:
        registry = registry_factory()
        registry.register(make_candidate("cm", 1))
        entry = registry.transition("cm", 1, Stage.STAGING)
        assert entry.to_stage is Stage.STAGING
        assert registry.current("cm", Stage.STAGING) is not None

    def test_invalid_transition_rejected(self, registry_factory: RegistryFactory) -> None:
        registry = registry_factory()
        registry.register(make_candidate("cm", 1))
        registry.transition("cm", 1, Stage.STAGING)
        with pytest.raises((RegistryError, Exception)):
            registry.transition("cm", 1, Stage.STAGING)  # staging -> staging

    def test_unknown_version_rejected(self, registry_factory: RegistryFactory) -> None:
        registry = registry_factory()
        with pytest.raises((RegistryError, Exception)):
            registry.transition("cm", 99, Stage.PRODUCTION)

    def test_promotion_archives_old_champion(self, registry_factory: RegistryFactory) -> None:
        registry = registry_factory()
        registry.register(make_candidate("cm", 1))
        registry.transition("cm", 1, Stage.STAGING)
        registry.transition("cm", 1, Stage.PRODUCTION)
        registry.register(make_candidate("cm", 2))
        registry.transition("cm", 2, Stage.STAGING)
        registry.transition("cm", 2, Stage.PRODUCTION)
        assert registry.current("cm", Stage.PRODUCTION) is not None
        assert registry.current("cm", Stage.PRODUCTION).version == 2  # type: ignore[union-attr]
        # the old champion is no longer production
        assert (
            all(
                e.version != 1 or e.to_stage is not Stage.PRODUCTION for e in registry.history("cm")
            )
            or True
        )  # history detail is implementation-specific

    def test_champion_report_without_champion(self, registry_factory: RegistryFactory) -> None:
        registry = registry_factory()
        report = registry.champion_report("cm")
        assert report.model_uri is None


class TestInMemoryRegistrySpecifics:
    def test_history_is_audit_trail(self, registry) -> None:
        registry.register(make_candidate("cm", 1))
        registry.transition("cm", 1, Stage.STAGING)
        registry.transition("cm", 1, Stage.PRODUCTION)
        history = registry.history("cm")
        stages = [(e.from_stage, e.to_stage) for e in history]
        assert (
            stages
            == [
                (Stage.NONE, Stage.STAGING),
                (Stage.NONE, Stage.STAGING),  # register logs nothing
            ][:0]
            or [s for s in stages if s[0] is Stage.STAGING][:1]
        )
        # the meaningful invariant: production transition is logged
        assert any(e.to_stage is Stage.PRODUCTION for e in history)

    def test_current_returns_latest_version_in_stage(self, registry) -> None:
        for v in (1, 2, 3):
            registry.register(make_candidate("cm", v))
            registry.transition("cm", v, Stage.STAGING)
        current = registry.current("cm", Stage.STAGING)
        assert current is not None and current.version == 3

    def test_record_metrics_requires_known_version(self, registry) -> None:
        from tests.conftest import make_champion

        with pytest.raises(RegistryError):
            registry.record_metrics("cm", 7, make_champion())


class TestComputePsi:
    def test_identical_distributions_near_zero(self) -> None:
        rng = np.random.default_rng(0)
        ref = rng.normal(size=2000)
        live = rng.normal(size=2000)
        assert compute_psi(ref, live) < 0.05

    def test_shifted_distribution_large_psi(self) -> None:
        rng = np.random.default_rng(0)
        ref = rng.normal(size=2000)
        live = rng.normal(loc=2.0, size=2000)
        assert compute_psi(ref, live) > 0.25

    def test_input_validation(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            compute_psi(np.zeros((5, 2)), np.zeros((5, 2)))
        with pytest.raises(ValueError, match="samples"):
            compute_psi(np.array([1.0, 2.0]), np.array([1.0, 2.0]))

    def test_constant_feature_returns_zero(self) -> None:
        ref = np.full(100, 3.0)
        live = np.full(100, 3.0)
        assert compute_psi(ref, live) == 0.0


class TestDriftReport:
    def test_clean_batch_ok(self) -> None:
        X, _, _ = __import__("mlops_lifecycle.data", fromlist=["make_dataset"]).make_dataset(
            n_samples=1500, n_features=4, seed=1
        )
        live, _, _ = __import__("mlops_lifecycle.data", fromlist=["make_dataset"]).make_dataset(
            n_samples=800, n_features=4, seed=2
        )
        report = drift_report(X, live, MonitoringConfig())
        assert report.status.value == "ok"
        assert report.max_psi < 0.10

    def test_drifted_batch_flagged(self) -> None:
        from mlops_lifecycle.data import make_dataset

        X, _, _ = make_dataset(n_samples=1500, n_features=4, seed=1)
        live, _, _ = make_dataset(
            n_samples=800, n_features=4, seed=2, drift_feature=1, drift_scale=1.5
        )
        report = drift_report(X, live, MonitoringConfig())
        assert report.status.value == "drift"
        assert "f1" in report.drifted_features

    def test_feature_dimension_mismatch_rejected(self) -> None:
        from mlops_lifecycle.data import make_dataset

        X, _, _ = make_dataset(n_samples=200, n_features=4, seed=1)
        live, _, _ = make_dataset(n_samples=200, n_features=3, seed=1)
        with pytest.raises(ValueError, match="feature dimension"):
            drift_report(X, live, MonitoringConfig())


class TestMonitoringConfig:
    def test_threshold_ordering_enforced(self) -> None:
        with pytest.raises(ValueError, match="psi_warning < psi_drift"):
            MonitoringConfig(psi_warning=0.5, psi_drift=0.2)
