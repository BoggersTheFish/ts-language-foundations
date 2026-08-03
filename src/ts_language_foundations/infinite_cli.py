"""Run the infinite-interior finite-interface experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from .infinite_experiment import render_report, render_svg, run_infinite_experiment, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--figure", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--trace-steps", type=int, default=24)
    parser.add_argument("--induction-depth", type=int, default=512)
    parser.add_argument("--states-checked", type=int, default=4096)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[2]
    receipt = run_infinite_experiment(
        trace_steps=args.trace_steps,
        induction_depth=args.induction_depth,
        states_checked=args.states_checked,
        project_root=project_root,
    )
    write_json(receipt, args.output)
    render_svg(receipt, args.figure)
    render_report(receipt, args.report)
    countdown = receipt["unbounded_countdown_case"]["verification_summary"]
    print(
        f"verified infinite interfaces: stable={receipt['stable_infinite_fibre_case']['computed_invariance_check']['passed']}, "
        f"predictive_checks={countdown['predictive_state_checks']}, "
        f"induction_instances={countdown['induction_instances_checked']}, "
        f"overall={receipt['verification']['passed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

