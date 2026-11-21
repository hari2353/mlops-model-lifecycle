"""Versioned model datasheet, following Harvard Edge's ML-Sensors template.

A datasheet travels with the registry artifact: intended use, limitations,
training data, evaluation, hardware, privacy and environmental footprint in
one deterministic document. Missing optional sections are rendered as "not
documented" instead of being silently dropped.
"""
from __future__ import annotations

from dataclasses import dataclass

from mlops_lifecycle.models import CandidateModel, EvaluationReport

_NOT_DOCUMENTED = "not documented"


@dataclass(frozen=True, slots=True)
class Datasheet:
    model: str
    version: int
    uri: str
    intended_use: str
    limitations: tuple[str, ...]
    training_data: dict[str, object]
    evaluation: dict[str, object]
    hardware: dict[str, object]
    privacy: dict[str, object]
    environmental: dict[str, object]

    def __post_init__(self) -> None:
        if not self.intended_use.strip():
            raise ValueError("intended_use must be non-empty")
        if not self.limitations or any(not item.strip() for item in self.limitations):
            raise ValueError("at least one non-empty limitation is required")

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "version": self.version,
            "uri": self.uri,
            "intended_use": self.intended_use,
            "limitations": list(self.limitations),
            "training_data": dict(self.training_data),
            "evaluation": dict(self.evaluation),
            "hardware": dict(self.hardware),
            "privacy": dict(self.privacy),
            "environmental": dict(self.environmental),
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Model datasheet: {self.model} v{self.version}",
            "",
            f"- Registry URI: `{self.uri}`",
            "- Datasheet version: 1",
            "",
            "## Intended use",
            "",
            self.intended_use,
            "",
            "## Limitations",
            "",
        ]
        lines.extend(f"- {item}" for item in self.limitations)
        for title, payload in (
            ("Training data", self.training_data),
            ("Evaluation", self.evaluation),
            ("Hardware", self.hardware),
            ("Privacy and security", self.privacy),
            ("Environmental footprint", self.environmental),
        ):
            lines.extend(["", f"## {title}", ""])
            if payload:
                lines.extend(f"- {key}: {value}" for key, value in payload.items())
            else:
                lines.append(_NOT_DOCUMENTED)
        lines.extend([
            "",
            "## Version history",
            "",
            f"- 1: initial datasheet for `{self.uri}`",
            "",
        ])
        return "\n".join(lines)


def build_datasheet(
    model: CandidateModel,
    evaluation: EvaluationReport,
    *,
    intended_use: str,
    limitations: tuple[str, ...],
    hardware: dict[str, object] | None = None,
    privacy: dict[str, object] | None = None,
    environmental: dict[str, object] | None = None,
) -> Datasheet:
    """Assemble a datasheet from the registry's own artifact + metrics."""
    if evaluation.model_uri != model.uri:
        raise ValueError(
            f"evaluation is for {evaluation.model_uri!r}, not {model.uri!r}"
        )
    return Datasheet(
        model=model.name,
        version=model.version,
        uri=model.uri,
        intended_use=intended_use,
        limitations=tuple(limitations),
        training_data={
            "name": model.train_data.name,
            "samples": model.train_data.n_samples,
            "features": model.train_data.n_features,
            "sha8": model.train_data.sha8,
            "validation_sha8": model.val_data.sha8,
        },
        evaluation={
            "accuracy": evaluation.accuracy,
            "f1": evaluation.f1,
            "roc_auc": evaluation.roc_auc,
            "n_samples": evaluation.n_samples,
        },
        hardware=dict(hardware or {}),
        privacy=dict(privacy or {}),
        environmental=dict(environmental or {}),
    )
