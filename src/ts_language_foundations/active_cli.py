"""Run active-intervention representation repair."""

from __future__ import annotations

import argparse
from pathlib import Path

from .active_experiment import run_active_experiment, write_active_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--figure", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--maximum-steps", type=int, default=65536)
    parser.add_argument("--heldout-episodes", type=int, default=64)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--merge-tolerance", type=float, default=0.2)
    parser.add_argument("--split-tolerance", type=float, default=0.35)
    parser.add_argument("--minimum-samples", type=int, default=32)
    parser.add_argument("--probe-cost", type=float, default=2.0)
    parser.add_argument("--monte-carlo-runs", type=int, default=100)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[2]
    receipt = run_active_experiment(
        seed=args.seed,
        maximum_steps=args.maximum_steps,
        heldout_episode_count=args.heldout_episodes,
        delta=args.delta,
        merge_tolerance=args.merge_tolerance,
        split_tolerance=args.split_tolerance,
        minimum_samples=args.minimum_samples,
        probe_cost=args.probe_cost,
        monte_carlo_runs=args.monte_carlo_runs,
        project_root=project_root,
    )
    write_active_artifacts(
        receipt,
        output=args.output,
        report=args.report,
        figure=args.figure,
    )
    primary = receipt["primary_controlled_case"]
    monte = receipt["monte_carlo"]
    print(
        "verified active repair: "
        f"memory={primary['result']['memory_repair_trace']}, "
        f"active/uniform split={primary['policy_comparison']['active_first_repair_step']}/"
        f"{primary['policy_comparison']['uniform_first_repair_step']}, "
        f"predictive/autonomous={primary['result']['predictive_blocks']}/"
        f"{primary['result']['autonomous_blocks']}, "
        f"recovery={monte['active_discovery']['successes']}/"
        f"{monte['active_discovery']['trials']}, "
        f"false_splits={monte['null_false_splits']['events']}, "
        f"overall={receipt['verification']['passed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
