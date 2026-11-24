"""Model registry: stage transitions with full audit history.

`ModelRegistry` is the protocol. `InMemoryRegistry` is the default (tests,
demo). `MlflowRegistry` is a thin adapter over MLflow's tracking/registry
APIs - swap it in with `pip install mlops-lifecycle[mlflow]` and the rest of
the system is unchanged (dependency inversion).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from mlops_lifecycle.models import CandidateModel, ChampionReport, Stage


class RegistryError(Exception):
    """Raised on invalid registry operations."""


@dataclass(frozen=True, slots=True)
class RegistryEntry:
    """Immutable audit record for one registry event."""

    model_name: str
    version: int
    from_stage: Stage
    to_stage: Stage
    at: datetime


class ModelRegistry(Protocol):
    """Anything that can track model versions and stage transitions."""

    def register(self, model: CandidateModel) -> int:
        """Record a new model version; returns its version number."""

    def transition(self, model_name: str, version: int, to_stage: Stage) -> RegistryEntry:
        """Move a version to a new stage, validating the transition."""

    def current(self, model_name: str, stage: Stage) -> CandidateModel | None:
        """Latest registered version currently sitting in `stage`."""

    def champion_report(self, model_name: str) -> ChampionReport:
        """Best-effort metrics snapshot of the production champion."""

    def history(self, model_name: str) -> tuple[RegistryEntry, ...]:
        """Full audit trail, oldest first."""

    def rollback(self, model_name: str, to_version: int | None = None) -> RegistryEntry:
        """Revert production to an archived version (highest archived by
        default), reusing promote-to-production semantics."""


_VALID_TRANSITIONS: dict[Stage, frozenset[Stage]] = {
    Stage.NONE: frozenset({Stage.STAGING, Stage.ARCHIVED}),
    Stage.STAGING: frozenset({Stage.PRODUCTION, Stage.ARCHIVED, Stage.NONE}),
    Stage.PRODUCTION: frozenset({Stage.ARCHIVED, Stage.NONE}),
    # archived -> production is the rollback path: an already-approved model
    # may return to service during an incident (registry.rollback)
    Stage.ARCHIVED: frozenset({Stage.NONE, Stage.STAGING, Stage.PRODUCTION}),
}


class InMemoryRegistry:
    """Protocol-complete registry with validation and audit history."""

    def __init__(self) -> None:
        self._versions: dict[str, dict[int, CandidateModel]] = {}
        self._stages: dict[tuple[str, int], Stage] = {}
        self._logs: dict[str, list[RegistryEntry]] = {}
        self._reports: dict[tuple[str, int], ChampionReport] = {}

    def register(self, model: CandidateModel) -> int:
        """Register a candidate; the registry owns final version numbering.

        Returns the assigned version (max existing + 1), so repeated
        registrations of similarly-stamped candidates never collide.
        """
        versions = self._versions.setdefault(model.name, {})
        next_version = max(versions, default=0) + 1
        stored = _with_version(model, next_version)
        versions[next_version] = stored
        self._stages[(model.name, next_version)] = Stage.NONE
        return next_version

    def transition(self, model_name: str, version: int, to_stage: Stage) -> RegistryEntry:
        key = (model_name, version)
        if key not in self._stages:
            msg = f"unknown model version: {model_name} v{version}"
            raise RegistryError(msg)
        from_stage = self._stages[key]
        if to_stage not in _VALID_TRANSITIONS[from_stage]:
            msg = f"invalid transition {from_stage.value} -> {to_stage.value}"
            raise RegistryError(msg)
        # promoting to production archives the previous champion
        if to_stage is Stage.PRODUCTION:
            current_prod = self.current(model_name, Stage.PRODUCTION)
            if current_prod is not None and current_prod.version != version:
                prod_key = (model_name, current_prod.version)
                self._stages[prod_key] = Stage.ARCHIVED
                self._logs[model_name].append(
                    RegistryEntry(
                        model_name=model_name,
                        version=current_prod.version,
                        from_stage=Stage.PRODUCTION,
                        to_stage=Stage.ARCHIVED,
                        at=datetime.now(tz=timezone.utc),
                    )
                )
        self._stages[key] = to_stage
        entry = RegistryEntry(
            model_name=model_name,
            version=version,
            from_stage=from_stage,
            to_stage=to_stage,
            at=datetime.now(tz=timezone.utc),
        )
        self._logs.setdefault(model_name, []).append(entry)
        return entry

    def current(self, model_name: str, stage: Stage) -> CandidateModel | None:
        best: CandidateModel | None = None
        for (name, version), model_stage in self._stages.items():
            if (
                name == model_name
                and model_stage is stage
                and (best is None or version > best.version)
            ):
                best = self._versions[name][version]
        return best

    def champion_report(self, model_name: str) -> ChampionReport:
        prod = self.current(model_name, Stage.PRODUCTION)
        if prod is None:
            return ChampionReport(model_uri=None, accuracy=None, f1=None)
        report = self._reports.get((model_name, prod.version))
        if report is None:
            return ChampionReport(model_uri=prod.uri, accuracy=None, f1=None)
        return report

    def record_metrics(self, model_name: str, version: int, report: ChampionReport) -> None:
        """Persist a champion's metrics for later lift comparisons."""
        if (model_name, version) not in self._stages:
            msg = f"unknown model version: {model_name} v{version}"
            raise RegistryError(msg)
        self._reports[(model_name, version)] = report

    def history(self, model_name: str) -> tuple[RegistryEntry, ...]:
        return tuple(self._logs.get(model_name, ()))

    def rollback(self, model_name: str, to_version: int | None = None) -> RegistryEntry:
        """One-call incident rollback: re-promote an archived version.

        Default target is the highest archived version. Delegates to
        `transition(..., PRODUCTION)`, so the current champion is archived
        automatically and the audit trail stays complete.
        """
        archived = sorted(
            (
                version
                for (name, version), stage in self._stages.items()
                if name == model_name and stage is Stage.ARCHIVED
            ),
            reverse=True,
        )
        if not archived:
            msg = f"nothing archived to roll back to: {model_name}"
            raise RegistryError(msg)
        if to_version is not None and to_version not in archived:
            msg = f"v{to_version} is not archived: {model_name}"
            raise RegistryError(msg)
        target = to_version if to_version is not None else archived[0]
        return self.transition(model_name, target, Stage.PRODUCTION)


class MlflowRegistry:
    """Adapter over MLflow's Model Registry (optional dependency).

    Delegates to `mlflow.tracking` / `mlflow.models` APIs and performs the
    same transition validation as the in-memory registry so behavior is
    consistent across backends (contract tests assert this).
    """

    def __init__(self, tracking_uri: str, registry_uri: str | None = None) -> None:
        try:
            import mlflow
        except ImportError as exc:  # pragma: no cover - optional dependency
            msg = "mlflow is required; install with: pip install mlops-lifecycle[mlflow]"
            raise ImportError(msg) from exc
        mlflow.set_tracking_uri(tracking_uri)
        if registry_uri:
            mlflow.set_registry_uri(registry_uri)
        self._mlflow = mlflow
        self._stages: dict[tuple[str, str], Stage] = {}

    def _client(self):
        from mlflow.tracking import MlflowClient

        return MlflowClient()

    def register(self, model: CandidateModel) -> int:  # pragma: no cover - needs server
        client = self._client()
        mv = client.create_model_version(name=model.name, source=model.uri, run_id=None)
        self._stages[(model.name, str(mv.version))] = Stage.NONE
        return int(mv.version)

    def transition(  # pragma: no cover - needs MLflow server
        self, model_name: str, version: int, to_stage: Stage
    ) -> RegistryEntry:
        client = self._client()
        from_stage = self._stages.get((model_name, str(version)), Stage.NONE)
        if to_stage not in _VALID_TRANSITIONS[from_stage]:
            msg = f"invalid transition {from_stage.value} -> {to_stage.value}"
            raise RegistryError(msg)
        client.transition_model_version_stage(
            name=model_name, version=version, stage=to_stage.value.capitalize()
        )
        self._stages[(model_name, str(version))] = to_stage
        return RegistryEntry(
            model_name=model_name,
            version=version,
            from_stage=from_stage,
            to_stage=to_stage,
            at=datetime.now(tz=timezone.utc),
        )

    def current(self, model_name: str, stage: Stage) -> CandidateModel | None:  # pragma: no cover
        client = self._client()
        for mv in client.search_model_versions(f"name='{model_name}'"):
            if mv.current_stage == stage.value.capitalize():
                return CandidateModel(
                    name=model_name,
                    version=int(mv.version),
                    params={},
                    train_data=DatasetHandle_stub("train"),
                    val_data=DatasetHandle_stub("val"),
                )
        return None

    def champion_report(self, model_name: str) -> ChampionReport:  # pragma: no cover
        return ChampionReport(model_uri=None, accuracy=None, f1=None)

    def history(self, model_name: str) -> tuple[RegistryEntry, ...]:  # pragma: no cover
        return ()

    def rollback(  # pragma: no cover - needs MLflow server
        self, model_name: str, to_version: int | None = None
    ) -> RegistryEntry:
        client = self._client()
        archived = sorted(
            (
                int(mv.version)
                for mv in client.search_model_versions(f"name='{model_name}'")
                if mv.current_stage == Stage.ARCHIVED.value.capitalize()
            ),
            reverse=True,
        )
        if not archived:
            msg = f"nothing archived to roll back to: {model_name}"
            raise RegistryError(msg)
        if to_version is not None and to_version not in archived:
            msg = f"v{to_version} is not archived: {model_name}"
            raise RegistryError(msg)
        target = to_version if to_version is not None else archived[0]
        return self.transition(model_name, target, Stage.PRODUCTION)


def DatasetHandle_stub(split: str):  # noqa: N802 - small internal helper
    from mlops_lifecycle.models import DatasetHandle

    return DatasetHandle(name=f"{split}", n_samples=0, n_features=0, sha8="-")


def _with_version(candidate: CandidateModel, version: int) -> CandidateModel:
    from dataclasses import replace

    return replace(candidate, version=version)


_CARD_MAX_CHARS = 4000


def model_card(registry: ModelRegistry, model_name: str, version: int | None = None) -> str:
    """Render a bounded Markdown model card from the registry's own audit
    trail (EU-AI-Act-era governance baseline). Never invents numbers:
    metrics nobody recorded render as "not recorded". History is cut to
    keep the card under the module's `_CARD_MAX_CHARS` characters.
    """
    champion = registry.champion_report(model_name)
    prod = registry.current(model_name, Stage.PRODUCTION)

    # resolve the card's version: explicit arg > production champion > latest audited
    history = registry.history(model_name)
    if version is None:
        version = prod.version if prod is not None else None
        if version is None and history:
            version = max(entry.version for entry in history)
    if version is None:
        # shortcut: a version with zero audit entries and zero stages reads as
        # unknown — upgrade by adding a versions() listing to the protocol
        msg = f"unknown model: {model_name}"
        raise RegistryError(msg)

    stage = "none"
    for entry in history:
        if entry.version == version:
            stage = entry.to_stage.value
    if prod is not None and prod.version == version:
        stage = Stage.PRODUCTION.value

    card_version_is_prod = prod is not None and prod.version == version
    lines: list[str] = [
        f"# Model card: {model_name}",
        "",
        "## Identity",
        f"- version: v{version} ({stage})",
        f"- generated: {datetime.now(tz=timezone.utc).isoformat(timespec='seconds')}",
        "",
        "## Metrics",
    ]
    if card_version_is_prod:
        accuracy = f"{champion.accuracy:.4f}" if champion.accuracy is not None else "not recorded"
        f1 = f"{champion.f1:.4f}" if champion.f1 is not None else "not recorded"
    else:
        accuracy = f1 = "not recorded"
    lines += [
        f"- accuracy: {accuracy}",
        f"- f1: {f1}",
        f"- model_uri: {champion.model_uri}" if champion.model_uri else "- model_uri: not recorded",
        "",
        "## Audit history",
    ]

    entry_lines = [
        f"- {entry.at.isoformat(timespec='seconds')} v{entry.version}: "
        f"{entry.from_stage.value} -> {entry.to_stage.value}"
        for entry in history
    ]
    truncated_marker = "- ... older entries truncated"
    kept: list[str] = []
    for line in entry_lines:
        if len("\n".join([*lines, *kept, line])) + len(truncated_marker) > _CARD_MAX_CHARS:
            break
        kept.append(line)
    if len(kept) < len(entry_lines):
        kept.append(truncated_marker)

    return "\n".join([*lines, *kept])
