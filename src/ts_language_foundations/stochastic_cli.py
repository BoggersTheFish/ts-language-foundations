"""Run confidence-bounded stochastic representation repair."""

from __future__ import annotations

import argparse
from pathlib import Path

from .stochastic_experiment import (
    render_report,
    render_svg,
    run_stochastic_experiment,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--figure", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--episode-count", type=int, default=256)
    parser.add_argument("--episode-length", type=int, default=128)
    parser.add_argument("--heldout-episodes", type=int, default=64)
    parser.add_argument("--maximum-order", type=int, default=3)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--family-cell-budget", type=int, default=256)
    parser.add_argument("--merge-tolerance", type=float, default=0.12)
    parser.add_argument("--split-tolerance", type=float, default=0.25)
    parser.add_argument("--minimum-child-samples", type=int, default=64)
    parser.add_argument("--monte-carlo-runs", type=int, default=100)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[2]
    receipt = run_stochastic_experiment(
        seed=args.seed,
        episode_count=args.episode_count,
        episode_length=args.episode_length,
        heldout_episode_count=args.heldout_episodes,
        maximum_order=args.maximum_order,
        delta=args.delta,
        family_cell_budget=args.family_cell_budget,
        merge_tolerance=args.merge_tolerance,
        split_tolerance=args.split_tolerance,
        minimum_samples_per_child=args.minimum_child_samples,
        monte_carlo_runs=args.monte_carlo_runs,
        project_root=project_root,
    )
    write_json(receipt, args.output)
    render_svg(receipt, args.figure)
    render_report(receipt, args.report)
    primary = receipt["primary_stochastic_infinite_fibre_case"]
    monte = receipt["monte_carlo"]
    print(
        "verified stochastic repair: "
        f"memory={primary['result']['memory_repair_trace']}, "
        f"predictive_blocks={primary['result']['predictive_blocks']}, "
        f"autonomous_blocks={primary['result']['autonomous_blocks']}, "
        f"recovery={monte['primary_recovery']['successes']}/"
        f"{monte['primary_recovery']['trials']}, "
        f"false_splits={monte['first_order_false_splits']['events']}, "
        f"overall={receipt['verification']['passed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
