#!/usr/bin/env python3
"""Verify hashes and recompute the checked-in scientific result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ts_language_foundations.experiment import run_experiment
from ts_language_foundations.active_experiment import run_active_experiment
from ts_language_foundations.infinite_experiment import run_infinite_experiment
from ts_language_foundations.online_experiment import run_online_experiment
from ts_language_foundations.relational_experiment import run_relational_experiment
from ts_language_foundations.sequence_experiment import run_sequence_experiment
from ts_language_foundations.stochastic_experiment import run_stochastic_experiment

from build_release_manifest import included_files


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/release_manifest.json"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {entry["path"]: entry for entry in manifest["files"]}
    observed_paths = {path.relative_to(root).as_posix() for path in included_files(root)}
    if observed_paths != set(expected):
        missing = sorted(set(expected) - observed_paths)
        unexpected = sorted(observed_paths - set(expected))
        raise SystemExit(f"manifest membership mismatch: missing={missing}, unexpected={unexpected}")
    for relative, entry in expected.items():
        path = root / relative
        if path.stat().st_size != entry["size_bytes"]:
            raise SystemExit(f"size mismatch: {relative}")
        if sha256(path) != entry["sha256"]:
            raise SystemExit(f"SHA-256 mismatch: {relative}")

    stored = json.loads((root / "artifacts/run_receipt.json").read_text(encoding="utf-8"))
    rerun = run_experiment(
        renaming_count=stored["run"]["renaming_count"],
        project_root=root,
        reference_kernel_archive_sha256=stored["provenance"][
            "reference_kernel_archive_sha256"
        ],
    )
    compared_sections = (
        "world",
        "representation_repair",
        "factor_search",
        "protocol",
        "interpretation_boundary",
    )
    for section in compared_sections:
        if stored[section] != rerun[section]:
            raise SystemExit(f"scientific rerun mismatch in section: {section}")
    if stored["provenance"]["source_hashes"] != rerun["provenance"]["source_hashes"]:
        raise SystemExit("source hashes in run receipt do not match the current release")

    stored_infinite = json.loads(
        (root / "artifacts/infinite_interface_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    infinite_run = stored_infinite["run"]
    rerun_infinite = run_infinite_experiment(
        horizons=infinite_run["horizons"],
        probabilities=infinite_run["geometric_probabilities"],
        primary_horizon=infinite_run["primary_horizon"],
        primary_probability=infinite_run["primary_probability"],
        trace_steps=infinite_run["trace_steps"],
        induction_depth=infinite_run["induction_depth"],
        states_checked=infinite_run["states_checked_per_horizon"],
        project_root=root,
    )
    for section in (
        "stable_infinite_fibre_case",
        "unbounded_countdown_case",
        "joint_conclusion",
        "verification",
    ):
        if stored_infinite[section] != rerun_infinite[section]:
            raise SystemExit(f"infinite-interface rerun mismatch in section: {section}")
    if (
        stored_infinite["provenance"]["source_hashes"]
        != rerun_infinite["provenance"]["source_hashes"]
    ):
        raise SystemExit("infinite-interface source hashes do not match the release")

    stored_online = json.loads(
        (root / "artifacts/online_interface_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    online_run = stored_online["run"]
    rerun_online = run_online_experiment(
        maximum_order=online_run["maximum_suffix_order"],
        stable_episode_length=online_run["stable_episode_length"],
        stable_heldout_length=online_run["stable_heldout_length"],
        countdown_post_absorption_steps=online_run[
            "countdown_post_absorption_steps"
        ],
        minimum_context_support=online_run["minimum_context_support"],
        project_root=root,
    )
    for section in (
        "algorithm",
        "stable_infinite_fibre_case",
        "unbounded_countdown_case",
        "negative_control",
        "verification",
        "conclusion",
    ):
        if stored_online[section] != rerun_online[section]:
            raise SystemExit(f"online-interface rerun mismatch in section: {section}")
    if (
        stored_online["provenance"]["source_hashes"]
        != rerun_online["provenance"]["source_hashes"]
    ):
        raise SystemExit("online-interface source hashes do not match the release")

    stored_stochastic = json.loads(
        (root / "artifacts/stochastic_interface_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    stochastic_run = stored_stochastic["run"]
    rerun_stochastic = run_stochastic_experiment(
        seed=stochastic_run["seed"],
        episode_count=stochastic_run["episode_count"],
        episode_length=stochastic_run["episode_length"],
        heldout_episode_count=stochastic_run["heldout_episode_count"],
        maximum_order=stochastic_run["maximum_order"],
        delta=stochastic_run["delta"],
        family_cell_budget=stochastic_run["family_cell_budget"],
        merge_tolerance=stochastic_run["merge_tolerance"],
        split_tolerance=stochastic_run["split_tolerance"],
        minimum_samples_per_child=stochastic_run[
            "minimum_samples_per_child"
        ],
        monte_carlo_runs=stochastic_run["monte_carlo_runs"],
        project_root=root,
    )
    for section in (
        "mathematical_contract",
        "primary_stochastic_infinite_fibre_case",
        "controls",
        "sensitivity",
        "monte_carlo",
        "verification",
        "conclusion",
    ):
        if stored_stochastic[section] != rerun_stochastic[section]:
            raise SystemExit(f"stochastic-interface rerun mismatch in section: {section}")
    if (
        stored_stochastic["provenance"]["source_hashes"]
        != rerun_stochastic["provenance"]["source_hashes"]
    ):
        raise SystemExit("stochastic-interface source hashes do not match the release")

    stored_active = json.loads(
        (root / "artifacts/active_repair_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    active_run = stored_active["run"]
    rerun_active = run_active_experiment(
        seed=active_run["seed"],
        maximum_steps=active_run["maximum_steps"],
        heldout_episode_count=active_run["heldout_episode_count"],
        delta=active_run["delta"],
        merge_tolerance=active_run["merge_tolerance"],
        split_tolerance=active_run["split_tolerance"],
        minimum_samples=active_run["minimum_samples"],
        probe_cost=active_run["probe_cost"],
        monte_carlo_runs=active_run["monte_carlo_runs"],
        project_root=root,
    )
    for section in (
        "mathematical_contract",
        "primary_controlled_case",
        "controls",
        "monte_carlo",
        "verification",
        "conclusion",
    ):
        if stored_active[section] != rerun_active[section]:
            raise SystemExit(f"active-repair rerun mismatch in section: {section}")
    if (
        stored_active["provenance"]["source_hashes"]
        != rerun_active["provenance"]["source_hashes"]
    ):
        raise SystemExit("active-repair source hashes do not match the release")

    stored_relational = json.loads(
        (root / "artifacts/relational_repair_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    relational_run = stored_relational["run"]
    rerun_relational = run_relational_experiment(
        seed=relational_run["seed"],
        maximum_steps=relational_run["maximum_steps"],
        heldout_interactions=relational_run["heldout_interactions"],
        delta=relational_run["delta"],
        merge_tolerance=relational_run["merge_tolerance"],
        split_tolerance=relational_run["split_tolerance"],
        minimum_samples=relational_run["minimum_samples"],
        probe_cost=relational_run["probe_cost"],
        monte_carlo_runs=relational_run["monte_carlo_runs"],
        project_root=root,
    )
    for section in (
        "mathematical_contract",
        "primary_relational_case",
        "controls",
        "monte_carlo",
        "verification",
        "conclusion",
    ):
        if stored_relational[section] != rerun_relational[section]:
            raise SystemExit(f"relational-repair rerun mismatch in section: {section}")
    if (
        stored_relational["provenance"]["source_hashes"]
        != rerun_relational["provenance"]["source_hashes"]
    ):
        raise SystemExit("relational-repair source hashes do not match the release")

    stored_sequence = json.loads(
        (root / "artifacts/sequence_repair_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    sequence_run = stored_sequence["run"]
    rerun_sequence = run_sequence_experiment(
        seed=sequence_run["seed"],
        maximum_steps=sequence_run["maximum_steps"],
        heldout_interactions=sequence_run["heldout_interactions"],
        delta=sequence_run["delta"],
        merge_tolerance=sequence_run["merge_tolerance"],
        split_tolerance=sequence_run["split_tolerance"],
        minimum_samples=sequence_run["minimum_samples"],
        probe_cost=sequence_run["probe_cost"],
        monte_carlo_runs=sequence_run["monte_carlo_runs"],
        project_root=root,
    )
    for section in (
        "mathematical_contract",
        "primary_sequence_case",
        "controls",
        "monte_carlo",
        "verification",
        "conclusion",
    ):
        if stored_sequence[section] != rerun_sequence[section]:
            raise SystemExit(f"sequence-repair rerun mismatch in section: {section}")
    if (
        stored_sequence["provenance"]["source_hashes"]
        != rerun_sequence["provenance"]["source_hashes"]
    ):
        raise SystemExit("sequence-repair source hashes do not match the release")

    cross_check = json.loads(
        (root / "artifacts/reference_cross_check.json").read_text(encoding="utf-8")
    )
    if not cross_check["all_passed"] or cross_check["comparison_count"] != 5:
        raise SystemExit("reference-kernel cross-check is absent or failed")
    if not (root / "figures/finite_protocol.svg").read_text(encoding="utf-8").startswith("<svg"):
        raise SystemExit("generated SVG is absent or malformed")
    if "## Verifier results" not in (
        root / "docs/TECHNICAL_REPORT.md"
    ).read_text(encoding="utf-8"):
        raise SystemExit("technical report is absent or malformed")
    if not (root / "figures/infinite_interface.svg").read_text(
        encoding="utf-8"
    ).startswith("<svg"):
        raise SystemExit("infinite-interface SVG is absent or malformed")
    if "### Theorem — no finite strongly lumpable refinement" not in (
        root / "docs/INFINITE_INTERFACE_REPORT.md"
    ).read_text(encoding="utf-8"):
        raise SystemExit("infinite-interface report is absent or malformed")
    if not (root / "figures/online_interface_learning.svg").read_text(
        encoding="utf-8"
    ).startswith("<svg"):
        raise SystemExit("online-interface SVG is absent or malformed")
    if "## Verifier results" not in (
        root / "docs/ONLINE_INTERFACE_REPORT.md"
    ).read_text(encoding="utf-8"):
        raise SystemExit("online-interface report is absent or malformed")
    if not (root / "figures/stochastic_interface_learning.svg").read_text(
        encoding="utf-8"
    ).startswith("<svg"):
        raise SystemExit("stochastic-interface SVG is absent or malformed")
    if "## Verifier results" not in (
        root / "docs/STOCHASTIC_INTERFACE_REPORT.md"
    ).read_text(encoding="utf-8"):
        raise SystemExit("stochastic-interface report is absent or malformed")
    if not (root / "figures/active_repair.svg").read_text(
        encoding="utf-8"
    ).startswith("<svg"):
        raise SystemExit("active-repair SVG is absent or malformed")
    if "## Verifier results" not in (
        root / "docs/ACTIVE_REPAIR_REPORT.md"
    ).read_text(encoding="utf-8"):
        raise SystemExit("active-repair report is absent or malformed")
    if not (root / "figures/relational_repair.svg").read_text(
        encoding="utf-8"
    ).startswith("<svg"):
        raise SystemExit("relational-repair SVG is absent or malformed")
    if "## Verifier results" not in (
        root / "docs/RELATIONAL_REPAIR_REPORT.md"
    ).read_text(encoding="utf-8"):
        raise SystemExit("relational-repair report is absent or malformed")
    if not (root / "figures/sequence_repair.svg").read_text(
        encoding="utf-8"
    ).startswith("<svg"):
        raise SystemExit("sequence-repair SVG is absent or malformed")
    if "## Verifier results" not in (
        root / "docs/SEQUENCE_REPAIR_REPORT.md"
    ).read_text(encoding="utf-8"):
        raise SystemExit("sequence-repair report is absent or malformed")

    print(
        "release verification passed: "
        f"{len(expected)} hashes, "
        f"{stored['protocol']['summary']['trial_count']} rerun trials, "
        f"{stored_infinite['unbounded_countdown_case']['verification_summary']['predictive_state_checks']} infinite predictive checks, "
        f"{stored_infinite['unbounded_countdown_case']['verification_summary']['induction_instances_checked']} induction instances, "
        f"{stored_online['verification']['passed_count']} online-interface checks, "
        f"{stored_online['stable_infinite_fibre_case']['heldout_evaluation']['eligible_transitions']} observation-only holdout transitions, "
        f"{stored_stochastic['verification']['passed_count']} stochastic-interface checks, "
        f"{stored_stochastic['monte_carlo']['primary_recovery']['trials']} Monte Carlo recovery trials, "
        f"{stored_active['verification']['passed_count']} active-repair checks, "
        f"{stored_active['monte_carlo']['active_discovery']['trials']} active discovery trials, "
        f"{stored_relational['verification']['passed_count']} relational-repair checks, "
        f"{stored_relational['monte_carlo']['active_full_recovery']['trials']} relational construction trials, "
        f"{stored_sequence['verification']['passed_count']} sequence-repair checks, "
        f"{stored_sequence['monte_carlo']['active_recovery']['trials']} typed sequence trials, "
        "5 reference comparisons"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
