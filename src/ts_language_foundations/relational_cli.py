"""Run typed relational feature construction and representation repair."""

from __future__ import annotations

import argparse
from pathlib import Path

from .relational_experiment import (
    run_relational_experiment,
    write_relational_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--figure", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--maximum-steps", type=int, default=32768)
    parser.add_argument("--heldout-interactions", type=int, default=65536)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--merge-tolerance", type=float, default=0.2)
    parser.add_argument("--split-tolerance", type=float, default=0.35)
    parser.add_argument("--minimum-samples", type=int, default=32)
    parser.add_argument("--probe-cost", type=float, default=2.0)
    parser.add_argument("--monte-carlo-runs", type=int, default=50)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[2]
    receipt = run_relational_experiment(
        seed=args.seed,
        maximum_steps=args.maximum_steps,
        heldout_interactions=args.heldout_interactions,
        delta=args.delta,
        merge_tolerance=args.merge_tolerance,
        split_tolerance=args.split_tolerance,
        minimum_samples=args.minimum_samples,
        probe_cost=args.probe_cost,
        monte_carlo_runs=args.monte_carlo_runs,
        project_root=project_root,
    )
    write_relational_artifacts(
        receipt,
        output=args.output,
        report=args.report,
        figure=args.figure,
    )
    primary = receipt["primary_relational_case"]
    monte = receipt["monte_carlo"]
    print(
        "verified relational repair: "
        f"feature={primary['result']['constructed_feature']}, "
        f"predictive/autonomous="
        f"{primary['dynamics']['predictive_block_count']}/"
        f"{primary['dynamics']['autonomous_block_count']}, "
        f"active/uniform relation median="
        f"{monte['relational_witness_steps']['active']['median']:.0f}/"
        f"{monte['relational_witness_steps']['uniform']['median']:.0f}, "
        f"recovery={monte['active_full_recovery']['successes']}/"
        f"{monte['active_full_recovery']['trials']}, "
        f"overall={receipt['verification']['passed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

