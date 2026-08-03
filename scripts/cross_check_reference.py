#!/usr/bin/env python3
"""Cross-check v0.1 finite results against TS repair kernel v0.3.0."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ts_language_foundations.experiment import analyse_candidate
from ts_language_foundations.world import RotatingBitWorld


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_source", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reference_source = args.reference_source.resolve()
    sys.path.insert(0, str(reference_source))
    from ts_representation_repair import (  # noqa: PLC0415
        dynamic_closure,
        joint_repair,
        lumpability_residual,
        predictive_residual,
    )

    world = RotatingBitWorld()
    matrix = np.array([[float(value) for value in row] for row in world.transition_matrix])
    laws = np.array([[float(value) for value in row] for row in world.task_laws])
    candidates = {
        "one_state": ((0, 1, 2, 3),),
        "task_visible_bit": world.task_partition,
        "selected_hidden_phase": ((0, 2), (1, 3)),
        "alternative_parity": ((0, 3), (1, 2)),
        "full_repaired_state": world.identity_partition,
    }
    comparisons = []
    for name, partition in candidates.items():
        exact = analyse_candidate(name, partition, world)
        reference = {
            "predictive_residual_nats": predictive_residual(laws, partition),
            "lumpability_residual_nats": lumpability_residual(
                matrix,
                partition,
                weights=np.full(4, 0.25),
            ),
            "dynamic_closure": dynamic_closure(matrix, partition),
            "joint_repair": joint_repair(matrix, laws, partition),
        }
        passed = (
            abs(exact["predictive_residual_nats"] - reference["predictive_residual_nats"]) < 1e-12
            and abs(exact["lumpability_residual_nats"] - reference["lumpability_residual_nats"]) < 1e-12
            and exact["dynamic_closure"] == reference["dynamic_closure"]
            and exact["joint_repair"] == reference["joint_repair"]
        )
        comparisons.append(
            {
                "candidate": name,
                "exact_v0_1": {
                    key: exact[key]
                    for key in (
                        "predictive_residual_nats",
                        "lumpability_residual_nats",
                        "dynamic_closure",
                        "joint_repair",
                    )
                },
                "reference_v0_3": reference,
                "passed": passed,
            }
        )
    receipt = {
        "schema_version": "ts-language-foundations-cross-check-v0.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_source": str(reference_source),
        "reference_core_sha256": sha256(reference_source / "ts_representation_repair" / "core.py"),
        "comparison_count": len(comparisons),
        "comparisons": comparisons,
        "all_passed": all(item["passed"] for item in comparisons),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"reference cross-check: {sum(item['passed'] for item in comparisons)}/{len(comparisons)}")
    return 0 if receipt["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

