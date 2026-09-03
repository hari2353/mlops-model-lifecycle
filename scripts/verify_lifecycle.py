"""CI verification: check the lifecycle demo produced the expected story.

Run after scripts/run_lifecycle.py --out artifacts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    log = Path("artifacts/lifecycle.log")
    summary_path = Path("artifacts/lifecycle_summary.json")
    if not log.exists():
        print("FAIL: artifacts/lifecycle.log not found")
        return 1
    text = log.read_text(encoding="utf-8")

    assert "promoted to PRODUCTION" in text, "promotion missing from log"
    assert "DRIFT" in text or "drift" in text, "monitoring output missing from log"
    assert "REJECTED" in text or "rejected" in text, "challenger gate missing from log"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["champion"]["accuracy"] > 0.7, "champion accuracy implausible"
    assert summary["challenger"]["decision"] == "rejected", "challenger should be rejected"
    assert summary["monitoring"]["status"] == "drift", "live batch should show drift"
    assert len(summary["registry_history"]) >= 2, "audit trail incomplete"
    print("lifecycle check passed:", summary["challenger"]["decision"], "|",
          summary["monitoring"]["status"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
