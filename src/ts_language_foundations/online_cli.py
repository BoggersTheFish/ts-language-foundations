"""Run the observation-only adaptive-interface experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from .online_experiment import render_report, render_svg, run_online_experiment, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--figure", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--maximum-order", type=int, default=8)
    parser.add_argument("--stable-episode-length", type=int, default=64)
    parser.add_argument("--stable-heldout-length", type=int, default=128)
    parser.add_argument("--countdown-tail", type=int, default=16)
    parser.add_argument("--minimum-context-support", type=int, default=8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[2]
    receipt = run_online_experiment(
        maximum_order=args.maximum_order,
        stable_episode_length=args.stable_episode_length,
        stable_heldout_length=args.stable_heldout_length,
        countdown_post_absorption_steps=args.countdown_tail,
        minimum_context_support=args.minimum_context_support,
        project_root=project_root,
    )
    write_json(receipt, args.output)
    render_svg(receipt, args.figure)
    render_report(receipt, args.report)
    stable = receipt["stable_infinite_fibre_case"]
    countdown = receipt["unbounded_countdown_case"]
    print(
        "verified observation-only repair: "
        f"order={stable['learned_interface']['final_order']}, "
        f"states={stable['result']['learned_state_count']}, "
        f"heldout={stable['heldout_evaluation']['correct']}/"
        f"{stable['heldout_evaluation']['eligible_transitions']}, "
        f"countdown_abstained={countdown['result']['all_finite_budgets_abstained']}, "
        f"overall={receipt['verification']['passed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
