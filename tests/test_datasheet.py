from mlops_lifecycle.datasheet import build_datasheet
from mlops_lifecycle.models import CandidateModel, DatasetHandle, EvaluationReport


def _model() -> CandidateModel:
    return CandidateModel(
        name="churn",
        version=3,
        params={"algorithm": "hist-gb"},
        train_data=DatasetHandle("train-v3", 800, 12, "aaaaaaaa"),
        val_data=DatasetHandle("val-v3", 200, 12, "bbbbbbbb"),
    )


def _report(uri: str = "models:/churn/3") -> EvaluationReport:
    return EvaluationReport(uri, accuracy=0.91, f1=0.88, roc_auc=0.95, n_samples=200)


def test_build_datasheet_carries_registry_identity_and_metrics():
    sheet = build_datasheet(
        _model(), _report(),
        intended_use="Score churn risk for retention outreach.",
        limitations=("Not validated outside the 2025 cohort.",),
    )
    assert sheet.uri == "models:/churn/3"
    assert sheet.training_data["sha8"] == "aaaaaaaa"
    assert sheet.evaluation["f1"] == 0.88


def test_markdown_contains_every_section_and_missing_ones_are_explicit():
    sheet = build_datasheet(
        _model(), _report(),
        intended_use="Score churn risk.",
        limitations=("Small validation set.",),
    )
    text = sheet.to_markdown()
    for heading in (
        "## Intended use",
        "## Limitations",
        "## Training data",
        "## Evaluation",
        "## Hardware",
        "## Privacy and security",
        "## Environmental footprint",
        "## Version history",
    ):
        assert heading in text
    assert text.count("not documented") == 3
    assert "models:/churn/3" in text


def test_datasheet_is_deterministic():
    kwargs = {
        "intended_use": "Score churn risk.",
        "limitations": ("Small validation set.",),
        "hardware": {"gpu": "T4", "memory_gb": 16},
        "privacy": {"pii": "none stored"},
        "environmental": {"kwh": 0.4},
    }
    first = build_datasheet(_model(), _report(), **kwargs).to_markdown()
    second = build_datasheet(_model(), _report(), **kwargs).to_markdown()
    assert first == second


def test_datasheet_rejects_missing_intended_use_or_limitations():
    for kwargs in (
        {"intended_use": " ", "limitations": ("x",)},
        {"intended_use": "use", "limitations": ()},
        {"intended_use": "use", "limitations": (" ",)},
    ):
        try:
            build_datasheet(_model(), _report(), **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid datasheet: {kwargs}")


def test_datasheet_rejects_mismatched_evaluation():
    try:
        build_datasheet(
            _model(), _report("models:/other/1"),
            intended_use="use", limitations=("x",),
        )
    except ValueError as exc:
        assert "evaluation is for" in str(exc)
    else:
        raise AssertionError("mismatched evaluation accepted")
