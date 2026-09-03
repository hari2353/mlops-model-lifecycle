"""Run the full model lifecycle end-to-end on deterministic synthetic data.

Story told by this script (all numbers reproducible via --seed):

1. train a champion on batch 1 data, force-promote it as baseline;
2. train a challenger on fresh data, gate it against the champion
   (it must beat absolute thresholds AND the lift requirement);
3. simulate a drifted live batch; monitor the promoted model for
   covariate drift (PSI) and performance decay.

Usage:
    python scripts/run_lifecycle.py --out artifacts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlops_lifecycle.data import make_dataset, train_val_test_split
from mlops_lifecycle.lifecycle import LifecycleRunner
from mlops_lifecycle.models import ChampionReport, GateConfig, Stage
from mlops_lifecycle.monitoring import MonitoringConfig, monitor_batch
from mlops_lifecycle.registry import InMemoryRegistry
from mlops_lifecycle.training import InMemoryModelStore, SklearnEvaluator, SklearnTrainer


class PrintReporter:
    """Prints each lifecycle step so CI can assert on the story."""

    def step(self, message: str) -> None:
        print(f"  | {message}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--live-drift-scale", type=float, default=1.0,
                        help="0 = clean live batch, >0 shifts feature 0")
    parser.add_argument("--live-noise", type=float, default=0.0,
                        help="label noise in the live batch (performance decay)")
    args = parser.parse_args()

    reporter = PrintReporter()
    store = InMemoryModelStore()
    trainer = SklearnTrainer(store=store)
    evaluator = SklearnEvaluator(store=store)
    registry = InMemoryRegistry()
    runner = LifecycleRunner(
        trainer=trainer,
        evaluator=evaluator,
        registry=registry,
        store=store,
        gate=GateConfig(min_accuracy=0.70, min_f1=0.70, min_roc_auc=0.75,
                        min_lift_over_champion=0.02),
        monitoring_config=MonitoringConfig(),
        reporter=reporter,
    )

    print("=" * 68)
    print("PHASE 1 - champion: train + promote baseline")
    print("=" * 68)
    X, y, handle = make_dataset(n_samples=4000, n_features=8, seed=args.seed)
    train, val, _test = train_val_test_split(X, y, handle)
    champion_result = runner.run("churn-model", train, val)
    if champion_result.promotion.decision.value != "promoted":
        # no champion yet -> lift gate is vacuous; force-promote baseline
        version = registry.current("churn-model", Stage.STAGING) or champion_result.candidate
        registry.transition("churn-model", version.version, Stage.PRODUCTION)
        print(f"baseline v{version.version} force-promoted to PRODUCTION")
    registry.record_metrics(
        "churn-model",
        champion_result.candidate.version,
        ChampionReport(
            model_uri=champion_result.candidate.uri,
            accuracy=champion_result.report.accuracy,
            f1=champion_result.report.f1,
        ),
    )
    print(f"champion metrics: acc={champion_result.report.accuracy:.3f} "
          f"f1={champion_result.report.f1:.3f} auc={champion_result.report.roc_auc:.3f}")

    print()
    print("=" * 68)
    print("PHASE 2 - challenger: train + gate against champion")
    print("=" * 68)
    X2, y2, handle2 = make_dataset(n_samples=4000, n_features=8, seed=args.seed + 100)
    train2, val2, _ = train_val_test_split(X2, y2, handle2)
    # challenger must beat the lift gate; if the fresh batch is statistically
    # identical, the gate should REJECT (no free lunch) - we show both paths
    challenger_result = runner.run("churn-model", train2, val2)
    print(f"challenger metrics: acc={challenger_result.report.accuracy:.3f} "
          f"f1={challenger_result.report.f1:.3f} auc={challenger_result.report.roc_auc:.3f}")
    print(f"gate decision     : {challenger_result.promotion.decision.value}")
    print(f"gate reasons       : {'; '.join(challenger_result.promotion.reasons)}")

    print()
    print("=" * 68)
    print("PHASE 3 - monitoring: drifted live batch through the promoted model")
    print("=" * 68)
    # monitor whichever version is in production now
    prod = registry.current("churn-model", Stage.PRODUCTION)
    monitor_candidate = prod or champion_result.candidate
    X_live, y_live, _ = make_dataset(
        n_samples=1200, n_features=8, seed=args.seed + 7,
        drift_feature=0 if args.live_drift_scale > 0 else None,
        drift_scale=args.live_drift_scale,
        label_noise=args.live_noise,
    )
    monitor_event = monitor_batch(
        monitor_candidate, store.load(monitor_candidate), X, (X_live, y_live),
        MonitoringConfig(),
    )
    print(f"live batch         : n={len(X_live)} drift_scale={args.live_drift_scale} "
          f"label_noise={args.live_noise}")
    print(f"drift status       : {monitor_event.drift.status.value} "
          f"(max PSI {monitor_event.drift.max_psi:.3f}, thresholds 0.10/0.25)")
    print(f"drifted features   : {monitor_event.drift.drifted_features or 'none'}")
    print(f"live accuracy      : {monitor_event.performance_accuracy:.3f} "
          f"(floor 0.70)")
    if monitor_event.notes:
        print(f"ALERTS             : {'; '.join(monitor_event.notes)}")
    else:
        print("ALERTS             : none")

    print()
    print("=" * 68)
    print("REGISTRY HISTORY (audit trail)")
    print("=" * 68)
    for entry in registry.history("churn-model"):
        print(f"  {entry.at.isoformat(timespec='seconds')} "
              f"v{entry.version}: {entry.from_stage.value} -> {entry.to_stage.value}")

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {
        "seed": args.seed,
        "champion": {
            "version": champion_result.candidate.version,
            "accuracy": champion_result.report.accuracy,
            "f1": champion_result.report.f1,
            "roc_auc": champion_result.report.roc_auc,
        },
        "challenger": {
            "version": challenger_result.candidate.version,
            "decision": challenger_result.promotion.decision.value,
            "reasons": list(challenger_result.promotion.reasons),
            "accuracy": challenger_result.report.accuracy,
            "f1": challenger_result.report.f1,
            "roc_auc": challenger_result.report.roc_auc,
        },
        "monitoring": {
            "status": monitor_event.drift.status.value,
            "max_psi": monitor_event.drift.max_psi,
            "drifted_features": list(monitor_event.drift.drifted_features),
            "live_accuracy": monitor_event.performance_accuracy,
            "alerts": list(monitor_event.notes),
        },
        "registry_history": [
            {"version": e.version, "from": e.from_stage.value, "to": e.to_stage.value,
             "at": e.at.isoformat()}
            for e in registry.history("churn-model")
        ],
    }
    with (args.out / "lifecycle_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print()
    print(f"summary written to {args.out / 'lifecycle_summary.json'}")
    print()
    print("PHASE 2 gate outcome:", challenger_result.promotion.decision.value.upper())
    print("PHASE 3 drift outcome:", monitor_event.drift.status.value.upper())


if __name__ == "__main__":
    main()
