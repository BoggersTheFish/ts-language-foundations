"""Command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from .experiment import render_report, render_svg, run_experiment, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="JSON receipt path")
    parser.add_argument("--figure", type=Path, required=True, help="generated SVG path")
    parser.add_argument("--report", type=Path, required=True, help="technical report path")
    parser.add_argument("--renaming-count", type=int, default=24)
    parser.add_argument("--reference-kernel-archive-sha256")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[2]
    receipt = run_experiment(
        renaming_count=args.renaming_count,
        project_root=project_root,
        reference_kernel_archive_sha256=args.reference_kernel_archive_sha256,
    )
    write_json(receipt, args.output)
    render_svg(receipt, args.figure)
    render_report(receipt, args.report)
    print(
        f"verified ordered={receipt['protocol']['summary']['all_checks_passed_count']}/"
        f"{receipt['protocol']['summary']['trial_count']}, "
        f"unordered={receipt['protocol']['summary']['unordered_all_checks_passed_count']}/"
        f"{receipt['protocol']['summary']['unordered_trial_count']} protocol trials"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
