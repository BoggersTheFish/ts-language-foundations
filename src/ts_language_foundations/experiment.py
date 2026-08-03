"""End-to-end verifier-first experiment and receipt generation."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

from .factorization import (
    CoordinateSystemSearchResult,
    FactorSearchResult,
    search_binary_complement,
    search_coordinate_system,
)
from .protocol import (
    ProtocolTrial,
    UnorderedProtocolTrial,
    learn_decoders,
    learn_factorized_dynamics,
    learn_unordered_decoders,
    make_codebook,
    make_unordered_codebook,
)
from .repair import (
    Partition,
    dynamic_closure,
    is_strongly_lumpable,
    joint_repair,
    lumpability_residual,
    partition_entropy,
    predictive_partition,
    predictive_repair,
    predictive_residual,
)
from .world import RotatingBitWorld


def _jsonable(value: Any) -> Any:
    if isinstance(value, Fraction):
        return {"numerator": value.numerator, "denominator": value.denominator}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes(project_root: Path) -> dict[str, str]:
    patterns = ("src/**/*.py", "tests/**/*.py", "scripts/*.py", "pyproject.toml")
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(project_root.glob(pattern))
    return {
        str(path.relative_to(project_root)): sha256_file(path)
        for path in sorted(paths)
        if path.is_file()
    }


def analyse_candidate(
    name: str,
    partition: Partition,
    world: RotatingBitWorld,
) -> dict[str, object]:
    matrix = world.transition_matrix
    laws = world.task_laws
    task_partition = predictive_partition(laws)
    predictive = predictive_repair(partition, task_partition)
    dynamic, dynamic_trace = dynamic_closure(matrix, partition, return_trace=True)
    joint_pd, trace_pd = joint_repair(
        matrix,
        laws,
        partition,
        schedule="predictive_then_dynamic",
        return_trace=True,
    )
    joint_dp, trace_dp = joint_repair(
        matrix,
        laws,
        partition,
        schedule="dynamic_then_predictive",
        return_trace=True,
    )
    assert joint_pd == joint_dp
    return {
        "name": name,
        "input_partition": partition,
        "predictive_residual_nats": predictive_residual(laws, partition),
        "lumpability_residual_nats": lumpability_residual(matrix, partition),
        "predictive_repair": predictive,
        "dynamic_closure": dynamic,
        "joint_repair": joint_pd,
        "dynamic_trace": dynamic_trace,
        "joint_trace_predictive_then_dynamic": trace_pd,
        "joint_trace_dynamic_then_predictive": trace_dp,
        "closure_excess_blocks": len(dynamic) - len(partition),
        "joint_excess_blocks": len(joint_pd) - len(partition),
        "closure_excess_entropy_nats": (
            partition_entropy(dynamic) - partition_entropy(partition)
        ),
        "joint_excess_entropy_nats": (
            partition_entropy(joint_pd) - partition_entropy(partition)
        ),
        "predictive_fixed": predictive == partition,
        "dynamic_fixed": dynamic == partition,
        "joint_fixed": joint_pd == partition,
        "joint_schedule_independent": joint_pd == joint_dp,
        "joint_predictively_sufficient": (
            predictive_repair(joint_pd, task_partition) == joint_pd
        ),
        "joint_strongly_lumpable": is_strongly_lumpable(matrix, joint_pd),
    }


def _true_coordinate_rollout(
    world: RotatingBitWorld,
    state_index: int,
    coordinates: tuple[tuple[int, int], ...],
    steps: int,
) -> tuple[tuple[int, int], ...]:
    result = [coordinates[state_index]]
    current = state_index
    for _ in range(steps):
        current = world.successor_indices[current]
        result.append(coordinates[current])
    return tuple(result)


def run_protocol_trial(
    world: RotatingBitWorld,
    factor_search: FactorSearchResult,
    renaming_index: int,
    held_out_state: int,
) -> ProtocolTrial:
    selected = factor_search.selected
    if selected is None:
        raise RuntimeError(f"factor search abstained: {factor_search.abstention_reason}")
    coordinates = selected.coordinate_pairs
    codebook = make_codebook(renaming_index)
    messages = tuple(codebook.encode(coordinate) for coordinate in coordinates)
    training_states = [state for state in range(len(world.states)) if state != held_out_state]

    # The verifier supplies registered task consequences, not word meanings.
    examples = []
    for state in training_states:
        target = world.task_trace(world.states[state], horizon=2)
        if target != coordinates[state]:
            raise AssertionError("selected coordinates lack the declared operational grounding")
        examples.append((messages[state], target))
    decoders = learn_decoders(examples)
    decoded_correctly = (
        len(decoders) == 1
        and decoders[0].decode(messages[held_out_state]) == coordinates[held_out_state]
    )

    observations = [
        (coordinates[state], coordinates[world.successor_indices[state]])
        for state in training_states
    ]
    learned_dynamics = learn_factorized_dynamics(observations)
    dynamics_identified = learned_dynamics is not None
    successor_correct = False
    rollout_correct = False
    if learned_dynamics is not None and decoded_correctly:
        decoded = decoders[0].decode(messages[held_out_state])
        true_successor = coordinates[world.successor_indices[held_out_state]]
        successor_correct = learned_dynamics.successor(decoded) == true_successor
        rollout_correct = (
            learned_dynamics.rollout(decoded, 4)
            == _true_coordinate_rollout(world, held_out_state, coordinates, 4)
        )

    # A one-signal-per-state code has no grounded entry for the excluded state.
    holistic_signals = tuple(f"whole-{signal}" for signal in ("ka", "zu", "mi", "te"))
    training_holistic_dictionary = {
        holistic_signals[state]: coordinates[state] for state in training_states
    }
    holistic_abstained = holistic_signals[held_out_state] not in training_holistic_dictionary

    return ProtocolTrial(
        renaming_index=renaming_index,
        held_out_state=held_out_state,
        decoder_count=len(decoders),
        decoded_held_out_correctly=decoded_correctly,
        dynamics_identified=dynamics_identified,
        held_out_successor_correct=successor_correct,
        four_step_rollout_correct=rollout_correct,
        holistic_baseline_abstained=holistic_abstained,
    )


def run_unordered_protocol_trial(
    world: RotatingBitWorld,
    coordinate_search: CoordinateSystemSearchResult,
    renaming_index: int,
    held_out_state: int,
) -> UnorderedProtocolTrial:
    selected = coordinate_search.selected
    if selected is None:
        raise RuntimeError(
            f"coordinate search abstained: {coordinate_search.abstention_reason}"
        )
    if selected.radices != (2, 2):
        raise RuntimeError("the v0.1 grammar learner currently handles two binary factors")
    coordinates = selected.state_coordinates
    codebook = make_unordered_codebook(renaming_index, selected.radices)
    messages = tuple(codebook.encode(coordinate) for coordinate in coordinates)
    training_states = [state for state in range(len(world.states)) if state != held_out_state]
    examples = []
    for state in training_states:
        target = world.task_trace(world.states[state], horizon=2)
        if target != coordinates[state]:
            raise AssertionError("selected coordinates lack operational grounding")
        examples.append((messages[state], target))
    decoders = learn_unordered_decoders(examples, selected.radices)
    decoded_correctly = (
        len(decoders) == 1
        and decoders[0].decode(messages[held_out_state]) == coordinates[held_out_state]
    )
    inferred_types = len(decoders) == 1 and all(
        decoders[0].decode(messages[state]) == coordinates[state]
        for state in training_states
    )
    observations = [
        (coordinates[state], coordinates[world.successor_indices[state]])
        for state in training_states
    ]
    learned_dynamics = learn_factorized_dynamics(observations)  # type: ignore[arg-type]
    successor_correct = False
    rollout_correct = False
    if learned_dynamics is not None and decoded_correctly:
        decoded = decoders[0].decode(messages[held_out_state])
        true_successor = coordinates[world.successor_indices[held_out_state]]
        successor_correct = learned_dynamics.successor(decoded) == true_successor  # type: ignore[arg-type]
        rollout_correct = (
            learned_dynamics.rollout(decoded, 4)  # type: ignore[arg-type]
            == _true_coordinate_rollout(
                world,
                held_out_state,
                coordinates,  # type: ignore[arg-type]
                4,
            )
        )
    return UnorderedProtocolTrial(
        renaming_index=renaming_index,
        held_out_state=held_out_state,
        decoder_count=len(decoders),
        decoded_held_out_correctly=decoded_correctly,
        inferred_types_without_slots=inferred_types,
        held_out_successor_correct=successor_correct,
        four_step_rollout_correct=rollout_correct,
    )


def _coarse_next_task_accuracy(world: RotatingBitWorld) -> float:
    correct_mass = 0
    for block in world.task_partition:
        counts = [0, 0]
        for state in block:
            successor = world.successor_indices[state]
            counts[world.states[successor][0]] += 1
        correct_mass += max(counts)
    return correct_mass / len(world.states)


def run_experiment(
    *,
    renaming_count: int = 24,
    project_root: Path | None = None,
    reference_kernel_archive_sha256: str | None = None,
) -> dict[str, object]:
    if not 1 <= renaming_count <= 24:
        raise ValueError("renaming_count must be in 1..24 for the four-signal group")
    world = RotatingBitWorld()
    matrix = world.transition_matrix
    task_partition = predictive_partition(world.task_laws)
    dynamic = dynamic_closure(matrix, task_partition)
    assert isinstance(dynamic, tuple)
    factor_search = search_binary_complement(
        world.successor_indices,
        task_partition,
        dynamic,
    )
    if factor_search.selected is None:
        raise RuntimeError(f"fixture factor search abstained: {factor_search.abstention_reason}")
    coordinate_search = search_coordinate_system(
        world.successor_indices,
        tuple(state[0] for state in world.states),
        dynamic,
    )
    if coordinate_search.selected is None:
        raise RuntimeError(
            f"fixture coordinate search abstained: {coordinate_search.abstention_reason}"
        )
    if coordinate_search.selected.factors != (
        task_partition,
        factor_search.selected.complement,
    ):
        raise AssertionError("the two independently declared factor searches disagree")

    selected_complement = factor_search.selected.complement
    alternative = next(
        candidate.complement
        for candidate in factor_search.candidates
        if candidate.complement != selected_complement
    )
    candidates = (
        ("one_state", (tuple(range(len(world.states))),)),
        ("task_visible_bit", task_partition),
        ("selected_hidden_phase", selected_complement),
        ("alternative_parity", alternative),
        ("full_repaired_state", world.identity_partition),
    )
    candidate_results = {
        name: analyse_candidate(name, partition, world) for name, partition in candidates
    }

    trials = tuple(
        run_protocol_trial(world, factor_search, renaming_index, held_out_state)
        for renaming_index in range(renaming_count)
        for held_out_state in range(len(world.states))
    )
    trial_count = len(trials)
    passed_count = sum(trial.passed for trial in trials)
    exact_count = sum(trial.decoded_held_out_correctly for trial in trials)
    successor_count = sum(trial.held_out_successor_correct for trial in trials)
    rollout_count = sum(trial.four_step_rollout_correct for trial in trials)
    holistic_abstentions = sum(trial.holistic_baseline_abstained for trial in trials)
    unordered_trials = tuple(
        run_unordered_protocol_trial(
            world,
            coordinate_search,
            renaming_index,
            held_out_state,
        )
        for renaming_index in range(renaming_count)
        for held_out_state in range(len(world.states))
    )
    unordered_passed = sum(trial.passed for trial in unordered_trials)
    unordered_decoded = sum(
        trial.decoded_held_out_correctly for trial in unordered_trials
    )
    unordered_successor = sum(
        trial.held_out_successor_correct for trial in unordered_trials
    )
    unordered_rollout = sum(
        trial.four_step_rollout_correct for trial in unordered_trials
    )

    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]
    receipt: dict[str, object] = {
        "schema_version": "ts-language-foundations-receipt-v0.1",
        "experiment": {
            "name": "finite repaired-state compositional protocol",
            "version": "0.1.0",
            "status": "controlled_fixture",
            "question": (
                "Can an opaque compositional message be constructed from a "
                "predictively sufficient and dynamically closed state, then "
                "support held-out combination decoding and autonomous rollout?"
            ),
        },
        "run": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "renaming_count": renaming_count,
            "renaming_group_size": 24,
            "renaming_coverage_exhaustive": renaming_count == 24,
            "held_out_states_per_renaming": len(world.states),
            "python": sys.version,
            "platform": platform.platform(),
            "neural_components": [],
            "external_corpus": None,
            "tokenizer": None,
            "network_required": False,
        },
        "provenance": {
            "source_hashes": source_hashes(project_root),
            "reference_kernel_archive_sha256": reference_kernel_archive_sha256,
            "reference_relation": (
                "The exact v0.1 implementation independently instantiates the same "
                "finite predictive/dynamic repair definitions used by the v0.3.0 kernel."
            ),
        },
        "world": {
            "state_dictionary": {
                str(index): {"task_visible_a": state[0], "hidden_phase_b": state[1]}
                for index, state in enumerate(world.states)
            },
            "update": "(a,b) -> (b,1-a)",
            "successor_indices": world.successor_indices,
            "transition_matrix": matrix,
            "registered_task": "report the current value of a",
            "predictive_laws": world.task_laws,
            "uniform_state_weights": True,
        },
        "representation_repair": {
            "task_partition": task_partition,
            "task_partition_predictive_residual_nats": predictive_residual(
                world.task_laws, task_partition
            ),
            "task_partition_lumpability_residual_nats": lumpability_residual(
                matrix, task_partition
            ),
            "dynamic_closure": dynamic,
            "joint_repair": joint_repair(matrix, world.task_laws, task_partition),
            "candidate_abstractions": candidate_results,
        },
        "factor_search": factor_search.to_dict(),
        "coordinate_system_search": coordinate_search.to_dict(),
        "protocol": {
            "message_form": "unordered set of one opaque signal per inferred factor",
            "ordered_slot_ablation_message_form": "two ordered opaque signals",
            "operational_semantics": (
                "signal types and values are inferred from co-occurrence constraints "
                "plus exact current-task and next-task "
                "verifier consequences"
            ),
            "unordered_decoder_hypothesis_class": (
                "all 4! bijections from four observed signals to two binary "
                "factor/value meanings; message order is discarded"
            ),
            "ordered_ablation_decoder_hypothesis_class": (
                "two slot-to-coordinate permutations times two binary signal "
                "bijections per slot"
            ),
            "dynamics_hypothesis_class": "all 16 Boolean truth tables per output bit",
            "dynamics_selection_rule": "unique minimum essential-input count",
            "dynamics_sparse_evidence_policy": (
                "abstain unless at least three of four distinct input combinations "
                "have real transition observations"
            ),
            "leave_one_state_out_trials": [_jsonable(trial.to_dict()) for trial in trials],
            "unordered_leave_one_state_out_trials": [
                _jsonable(trial.to_dict()) for trial in unordered_trials
            ],
            "summary": {
                "trial_count": trial_count,
                "all_checks_passed_count": passed_count,
                "all_checks_passed_rate": passed_count / trial_count,
                "held_out_compositional_decode_correct_count": exact_count,
                "held_out_compositional_decode_accuracy": exact_count / trial_count,
                "held_out_successor_correct_count": successor_count,
                "held_out_successor_accuracy": successor_count / trial_count,
                "four_step_rollout_correct_count": rollout_count,
                "four_step_rollout_accuracy": rollout_count / trial_count,
                "holistic_baseline_abstention_count": holistic_abstentions,
                "holistic_baseline_held_out_coverage": 1 - holistic_abstentions / trial_count,
                "coarse_task_message_current_task_accuracy": 1.0,
                "coarse_task_message_best_next_task_accuracy": _coarse_next_task_accuracy(world),
                "unordered_trial_count": len(unordered_trials),
                "unordered_all_checks_passed_count": unordered_passed,
                "unordered_all_checks_passed_rate": unordered_passed / len(unordered_trials),
                "unordered_held_out_decode_correct_count": unordered_decoded,
                "unordered_held_out_decode_accuracy": unordered_decoded / len(unordered_trials),
                "unordered_held_out_successor_correct_count": unordered_successor,
                "unordered_held_out_successor_accuracy": unordered_successor / len(unordered_trials),
                "unordered_four_step_rollout_correct_count": unordered_rollout,
                "unordered_four_step_rollout_accuracy": unordered_rollout / len(unordered_trials),
            },
        },
        "interpretation_boundary": {
            "fixture_specific_finding": (
                "In this four-state world, closure restores one omitted bit; a "
                "declared lookup-table score selects its two-factor coordinates "
                "over a holistic code and two alternative products; and an "
                "unordered opaque protocol infers signal types and generalises to "
                "every held-out state under every tested signal renaming."
            ),
            "does_not_establish": [
                "a theory of human language",
                "general emergent communication",
                "that the selected factor score is universally correct",
                "that a bounded coordinate candidate family is unnecessary",
                "that one atomic signal per selected factor is unnecessary",
                "any claim about consciousness",
            ],
            "declared_scaffolds": [
                "the four-state deterministic world",
                "the registered immediate task",
                "balanced binary factor candidates",
                "a fixed-width lookup-table description-length score",
                "minimum essential-input selection",
                "a three-of-four transition-evidence threshold",
                "one signal per inferred factor",
                "exact verifier feedback",
            ],
        },
    }
    return _jsonable(receipt)


def write_json(receipt: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_svg(receipt: dict[str, object], path: Path) -> None:
    summary = receipt["protocol"]["summary"]  # type: ignore[index]
    repair = receipt["representation_repair"]  # type: ignore[assignment]
    lump = repair["task_partition_lumpability_residual_nats"]  # type: ignore[index]
    accuracy = summary["unordered_held_out_decode_accuracy"]  # type: ignore[index]
    holistic = summary["holistic_baseline_held_out_coverage"]  # type: ignore[index]
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="760" viewBox="0 0 1200 760">
<defs><style>
.bg{{fill:#07111f}} .panel{{fill:#0d1d31;stroke:#294767;stroke-width:2}} .t{{fill:#e8f1ff;font:22px system-ui}}
.h{{fill:#7ee7ff;font:bold 34px system-ui}} .s{{fill:#9bb3ce;font:17px system-ui}} .node{{fill:#173a55;stroke:#7ee7ff;stroke-width:3}}
.good{{fill:#75f0a7}} .warn{{fill:#ffcf70}} .arrow{{stroke:#9bb3ce;stroke-width:3;fill:none;marker-end:url(#m)}}
</style><marker id="m" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#9bb3ce"/></marker></defs>
<rect class="bg" width="1200" height="760"/><text class="h" x="55" y="60">Repaired state → opaque protocol — finite v0.1 receipt</text>
<rect class="panel" x="45" y="90" width="500" height="360" rx="18"/><text class="t" x="75" y="130">World dynamics: (a,b) → (b,1−a)</text>
<path class="arrow" d="M170 205 C260 145,350 145,440 205"/><path class="arrow" d="M455 230 C510 300,510 330,455 380"/>
<path class="arrow" d="M420 395 C330 455,240 455,150 395"/><path class="arrow" d="M135 370 C80 300,80 270,135 220"/>
<circle class="node" cx="145" cy="205" r="42"/><circle class="node" cx="455" cy="205" r="42"/><circle class="node" cx="455" cy="390" r="42"/><circle class="node" cx="145" cy="390" r="42"/>
<text class="t" x="126" y="213">00</text><text class="t" x="436" y="213">01</text><text class="t" x="436" y="398">11</text><text class="t" x="126" y="398">10</text>
<rect class="panel" x="575" y="90" width="580" height="360" rx="18"/><text class="t" x="605" y="130">Repair receipt</text>
<text class="s" x="605" y="175">Task partition</text><text class="t" x="820" y="175">{{00,01}} | {{10,11}}</text>
<text class="s" x="605" y="220">Predictive residual</text><text class="good t" x="820" y="220">0.000 nats</text>
<text class="s" x="605" y="265">Lumpability residual</text><text class="warn t" x="820" y="265">{float(lump):.6f} nats</text>
<text class="s" x="605" y="310">Dynamic closure</text><text class="t" x="820" y="310">4 singleton states</text>
<text class="s" x="605" y="355">Recovered coordinates</text><text class="t" x="820" y="355">a and hidden phase b</text>
<text class="s" x="605" y="400">Opaque message</text><text class="t" x="820" y="400">{{signal(a), signal(b)}}</text>
<rect class="panel" x="45" y="485" width="1110" height="220" rx="18"/><text class="t" x="75" y="530">Leave-one-state-out verifier result across signal renamings</text>
<text class="h good" x="75" y="600">{float(accuracy)*100:.0f}%</text><text class="s" x="75" y="635">slot-free type inference + autonomous rollout</text>
<text class="h warn" x="650" y="600">{float(holistic)*100:.0f}%</text><text class="s" x="650" y="635">held-out coverage for one-signal-per-state code</text>
<text class="s" x="75" y="680">Controlled fixture only. Bounded factors, exact feedback, one-signal-per-factor form, and the score remain scaffolds.</text>
</svg>'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def render_report(receipt: dict[str, object], path: Path) -> None:
    repair = receipt["representation_repair"]  # type: ignore[assignment]
    protocol = receipt["protocol"]  # type: ignore[assignment]
    summary = protocol["summary"]  # type: ignore[index]
    factor = receipt["factor_search"]  # type: ignore[assignment]
    selected = factor["selected"]  # type: ignore[index]
    coordinates = receipt["coordinate_system_search"]  # type: ignore[assignment]
    coordinate_selected = coordinates["selected"]  # type: ignore[index]
    text = f"""# Finite Repaired-State Compositional Protocol v0.1

## Question

Can a communication protocol be built from verifier-grounded state repair rather than from a language model or corpus?

## Construction

The controlled world has four states `(a,b)` and deterministic dynamics `(a,b) -> (b,1-a)`. The registered immediate task reports `a`. Its coarse partition is therefore predictively exact for the current task, but it discards `b`, which determines the next task value.

No LLM, neural network, tokenizer, corpus, embedding model, or external data source participates in the experiment.

## Representation-repair result

- Predictive residual of the task partition: `{repair['task_partition_predictive_residual_nats']:.12f}` nats.
- Lumpability residual of the task partition: `{repair['task_partition_lumpability_residual_nats']:.12f}` nats.
- Dynamic closure: `{repair['dynamic_closure']}`.
- Joint repair: `{repair['joint_repair']}`.

Thus the current-task label is sufficient for the immediate verifier but not an autonomous Markov state. Closure restores two block distinctions, yielding the four-state identity partition.

## Factor and protocol scaffold

The first diagnostic searches balanced binary complements of the task factor. It selects the unique candidate minimising the total number of essential inputs in the two coordinate-update truth tables. The selected complement is `{selected['complement_partition']}` with score `{selected['dependency_score']}`.

A second search does not preselect one versus two factors. It compares a four-valued holistic coordinate with all irredundant pairs of balanced binary coordinates. Its fixed-width task-plus-dynamics table score uniquely selects `{coordinate_selected['factors']}` at `{coordinate_selected['total_model_bits']}` bits; the other candidates require `{[candidate['total_model_bits'] for candidate in coordinates['candidates'][1:]]}` bits.

Each coordinate value receives an arbitrary opaque signal. A message is an **unordered set** containing one signal from each selected factor. A receiver is shown three of four states. Exact verifier receipts expose only the current and next registered task consequences. The receiver exhaustively assigns four signals to candidate factor/value meanings, so both signal type and value must be inferred without message positions. It then learns factorised Boolean dynamics with a declared minimal-dependency rule.

The learner is required to abstain with fewer than three distinct transition inputs; this blocks the simplest compatible constant law from being promoted on one-example evidence.

## Verifier results

- Trials: `{summary['trial_count']}` ({receipt['run']['renaming_count']} symbol renamings × four held-out states).
- Unique decoder and all checks passed: `{summary['all_checks_passed_count']}/{summary['trial_count']}`.
- Held-out compositional decoding: `{summary['held_out_compositional_decode_accuracy']:.3f}`.
- Held-out successor prediction: `{summary['held_out_successor_accuracy']:.3f}`.
- Four-step autonomous rollout: `{summary['four_step_rollout_accuracy']:.3f}`.
- Unordered signal-type inference and held-out decoding: `{summary['unordered_held_out_decode_accuracy']:.3f}`.
- Unordered held-out successor prediction: `{summary['unordered_held_out_successor_accuracy']:.3f}`.
- Unordered four-step autonomous rollout: `{summary['unordered_four_step_rollout_accuracy']:.3f}`.
- Holistic one-signal-per-state held-out coverage: `{summary['holistic_baseline_held_out_coverage']:.3f}`.
- Coarse current-task message accuracy: `{summary['coarse_task_message_current_task_accuracy']:.3f}`.
- Best coarse next-task accuracy: `{summary['coarse_task_message_best_next_task_accuracy']:.3f}`.

## Retained claim

In this fixture, language-like symbols can be introduced after representation repair as a serialization of verified distinctions. A bounded description-length search prefers the reusable product coordinates over the holistic state code. The resulting unordered factor code infers symbol types from co-occurrence plus verifier consequences, generalises to unseen combinations, and carries enough state for autonomous continuation; the coarse task label and ungrounded holistic code do not.

## What this does not establish

This is not a theory of human language, a result about consciousness, or evidence that factor discovery will scale. The world, exact feedback, bounded one-or-two-factor candidate family, one-signal-per-factor message form, and simplicity score are explicit scaffolds. The next falsifiable experiment must infer factor arities and message structure in a larger noisy world while preserving exact receipts and abstention on ambiguity.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
