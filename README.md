# MLOps Model Lifecycle: Training -> Gated Promotion -> Registry -> Monitoring

**A public portfolio replica of enterprise MLOps work** — the model operationalization and monitoring capability I built at Altria (MLflow + Azure DevOps + Databricks), condensed into a runnable, fully tested, zero-cloud-dependency repo.

It tells the complete production story in one command: a champion model is trained and promoted, a challenger is **automatically rejected** by a lift gate, and a drifted live batch is **caught by PSI monitoring** — with a full registry audit trail.

## The demo run (verified, reproducible with `--seed 42`)

```
$ python scripts/run_lifecycle.py --out artifacts

PHASE 1 - champion: train + promote baseline
  | v1 promoted to PRODUCTION
  champion metrics: acc=0.978 f1=0.979 auc=0.996

PHASE 2 - challenger: train + gate against champion
  | v2 rejected: lift over champion -0.030 < required +0.02
  challenger metrics: acc=0.948 f1=0.945 auc=0.985
  gate decision     : rejected

PHASE 3 - monitoring: drifted live batch through the promoted model
  drift status       : drift (max PSI 5.416, thresholds 0.10/0.25)
  drifted features   : ('f0',)
  live accuracy      : 0.680 (floor 0.70)
  ALERTS             : live accuracy 0.680 below floor 0.70; covariate drift on f0

REGISTRY HISTORY (audit trail)
  v1: none -> staging
  v1: staging -> production
```

The challenger is *good* (94.8% accuracy) — and the gate still rejects it, because a retrain that doesn't beat the production champion by the configured lift shouldn't ship. That's the point.

**Verified quality bar:** 70 tests (unit + registry contract + integration), 96% coverage, ruff + mypy clean, CI on Python 3.10-3.12 including a full lifecycle run.

## Architecture

```
deterministic synthetic data ──▶ Trainer ──▶ Evaluator ──▶ Promotion Gate ──▶ Registry ──▶ Monitoring
        (data.py)             (protocol)    (protocol)   (pure function)   (protocol)   (PSI + perf)
                                                                       │
                                                          champion/challenger story:
                                                          v1 promoted, v2 rejected,
                                                          drifted batch flagged
```

| Principle | Where |
|---|---|
| **S**ingle responsibility | `data` / `training` / `registry` / `monitoring` / `lifecycle` each own one stage |
| **O**pen/closed | New trainer (XGBoost, torch), registry backend, or exporter = new class implementing a protocol |
| **L**iskov | Registry contract tests assert identical semantics for any `ModelRegistry` implementation |
| **I**nterface segregation | Small focused protocols: `Trainer`, `Evaluator`, `ModelStore`, `ModelRegistry`, `ProgressReporter` |
| **D**ependency inversion | `LifecycleRunner` receives every dependency via constructor injection; the CLI is the composition root |

Deliberate production semantics baked in:
- **The registry owns version numbering** — a retrain can never collide with an existing version.
- **Promoting archives the old champion** automatically (mirrors MLflow stage semantics).
- **Invalid transitions rejected** (none→production directly, production→staging, etc.) — every change leaves an audit entry.
- **PSI on shared quantile edges** — deterministic, symmetric, CI-stable drift measurement.
- **Everything seeded** — identical metrics on every machine and every CI run.

## Quickstart

```bash
pip install -e .[dev]
pytest                                            # 70 tests
python scripts/run_lifecycle.py --out artifacts   # the demo above
# vary the story:
python scripts/run_lifecycle.py --live-drift-scale 0     # clean batch -> drift OK
python scripts/run_lifecycle.py --live-noise 0.25       # label noise -> perf decay alert
```

## Incident + governance layer (market-steal, 2026-11-18)

- **`registry.rollback(model_name, to_version=None)`** — one-call revert to
  the archived previous champion (highest archived by default), reusing
  promote-to-production semantics so the outgoing champion is archived and
  the audit trail stays complete. This makes `archived -> production` a
  legal transition: an already-approved model may return to service.
- **`registry.model_card(registry, model_name, version=None)`** — bounded
  (≤ 4k chars) Markdown card: identity, recorded metrics (never invented),
  full audit history, generated-at. Unknown model → `RegistryError`.
- **`policy.decide_action(event, ActionPolicy)`** — pure alert-to-action
  policy with documented precedence **PAGE > ROLLBACK > RETRAIN > NONE**:
  accuracy-below-floor ⇒ ROLLBACK, drift alone ⇒ RETRAIN, both ⇒ PAGE.
  Wired as `recommended_action` on every run result that monitored, and
  printed by the demo (`recommended action : PAGE`).

### Using MLflow as the registry backend

```bash
pip install -e .[mlflow]
python -c "
from mlops_lifecycle.registry import MlflowRegistry
registry = MlflowRegistry(tracking_uri='file://./mlruns')
# pass registry=registry to LifecycleRunner - everything else unchanged
"
```

## Repository layout

```
src/mlops_lifecycle/
  models.py       # typed frozen domain: candidates, reports, gates, drift
  data.py         # deterministic synthetic data + drift/noise injection
  training.py     # Trainer/Evaluator protocols, sklearn defaults, promotion gate
  registry.py     # ModelRegistry protocol, InMemoryRegistry, MLflow adapter
  monitoring.py   # PSI, drift reports, live-batch monitoring
  lifecycle.py    # LifecycleRunner orchestrator (all deps injected)
scripts/
  run_lifecycle.py   # champion -> challenger gate -> drifted monitoring demo
tests/
  test_models_data.py        # domain + data determinism
  test_training_gate.py      # training, evaluation, all gate branches
  test_registry_monitoring.py # registry contract + PSI/drift unit tests
  test_integration.py        # full lifecycle stories (promote/reject/drift)
```

## Why synthetic data?

Enterprise MLOps runs on proprietary models and data (Altria's production forecasting/ESG stack). This repo demonstrates the *lifecycle machinery* — gating, registry semantics, drift detection — on data anyone can regenerate with one seed.

## License

MIT
