"""Observation-only adaptive-interface experiment, receipts, and figures."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

from .experiment import source_hashes
from .online import (
    CompiledSuffixInterface,
    compile_suffix_interface,
    countdown_observation_episode,
    evaluate_majority_context_model,
    evaluate_suffix_interface,
    fit_majority_context_model,
    learn_suffix_interface,
    rotating_observation_episode,
)
from .repair import (
    Partition,
    as_fraction_laws,
    as_fraction_matrix,
    dynamic_closure,
    is_strongly_lumpable,
    joint_repair,
    lumpability_residual,
    partition_entropy,
    predictive_partition,
    predictive_repair,
    predictive_residual,
)


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


def _episode_hash(episodes: Iterable[Iterable[int]]) -> str:
    materialized = [list(episode) for episode in episodes]
    encoded = json.dumps(materialized, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _compact_evaluation(result: dict[str, object], limit: int = 8) -> dict[str, object]:
    compact = dict(result)
    errors = compact.pop("errors", [])
    compact["error_example_count"] = min(len(errors), limit)
    compact["error_examples"] = errors[:limit]
    compact["total_error_records"] = len(errors)
    return compact


def _partition_symbols(
    partition: Partition,
    compiled: CompiledSuffixInterface,
) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(compiled.symbols[index] for index in block) for block in partition)


def _candidate_record(
    name: str,
    partition: Partition,
    compiled: CompiledSuffixInterface,
    weights: tuple[Fraction, ...],
) -> dict[str, object]:
    matrix = as_fraction_matrix(compiled.transition_rows)
    laws = as_fraction_laws(compiled.predictive_laws)
    predictive = predictive_repair(partition, predictive_partition(laws))
    dynamic, dynamic_trace = dynamic_closure(matrix, partition, return_trace=True)
    joint, joint_trace = joint_repair(
        matrix,
        laws,
        partition,
        return_trace=True,
    )
    return {
        "name": name,
        "partition": partition,
        "partition_symbols": _partition_symbols(partition, compiled),
        "block_count": len(partition),
        "predictive_residual_nats": predictive_residual(laws, partition, weights),
        "lumpability_residual_nats": lumpability_residual(matrix, partition, weights),
        "partition_entropy_nats": partition_entropy(partition, weights),
        "predictive_repair": predictive,
        "predictive_repair_symbols": _partition_symbols(predictive, compiled),
        "dynamic_closure": dynamic,
        "dynamic_closure_symbols": _partition_symbols(dynamic, compiled),
        "joint_repair": joint,
        "joint_repair_symbols": _partition_symbols(joint, compiled),
        "dynamic_repair_trace": dynamic_trace,
        "joint_repair_trace": joint_trace,
        "dynamic_closure_excess_blocks": len(dynamic) - len(partition),
        "joint_repair_excess_blocks": len(joint) - len(partition),
        "predictive_fixed": predictive == partition,
        "dynamic_fixed": dynamic == partition,
        "joint_fixed": joint == partition,
        "strongly_lumpable": is_strongly_lumpable(matrix, partition),
    }


def _stable_case(
    maximum_order: int,
    episode_length: int,
    heldout_length: int,
    minimum_context_support: int,
) -> dict[str, object]:
    training_counter_values = (0, 1, 7, 31)
    training_specs = tuple(
        (a_value, b_value, counter)
        for counter in training_counter_values
        for a_value in (0, 1)
        for b_value in (0, 1)
    )
    training_episodes = tuple(
        rotating_observation_episode(a_value, b_value, counter, episode_length)
        for a_value, b_value, counter in training_specs
    )
    heldout_counter = 10**6
    heldout_specs = tuple(
        (a_value, b_value, heldout_counter)
        for a_value in (0, 1)
        for b_value in (0, 1)
    )
    heldout_episodes = tuple(
        rotating_observation_episode(a_value, b_value, counter, heldout_length)
        for a_value, b_value, counter in heldout_specs
    )
    learned = learn_suffix_interface(
        training_episodes,
        initial_order=1,
        maximum_order=maximum_order,
        minimum_context_support=minimum_context_support,
    )
    compiled = compile_suffix_interface(learned)
    support = learned.support_map
    support_total = sum(support.values())
    weights = tuple(
        Fraction(support[context], support_total) for context in compiled.contexts
    )

    state_count = len(compiled.contexts)
    identity = tuple((index,) for index in range(state_count))
    one_state = (tuple(range(state_count)),)
    current_observation = tuple(
        tuple(
            index
            for index, context in enumerate(compiled.contexts)
            if context[-1] == observation
        )
        for observation in compiled.alphabet
    )
    next_symbol = predictive_partition(as_fraction_laws(compiled.predictive_laws))
    candidates = {
        "one_state_negative_control": _candidate_record(
            "one-state negative control", one_state, compiled, weights
        ),
        "current_observation": _candidate_record(
            "current observed symbol", current_observation, compiled, weights
        ),
        "next_symbol_predictive_partition": _candidate_record(
            "equal next-symbol laws", next_symbol, compiled, weights
        ),
        "learned_suffix_identity": _candidate_record(
            "learned two-symbol suffix identity", identity, compiled, weights
        ),
    }

    learner_evaluation = _compact_evaluation(
        evaluate_suffix_interface(learned, heldout_episodes)
    )
    baselines = {}
    for name, order in (
        ("one_state_majority", 0),
        ("current_observation_majority", 1),
        ("fixed_order_two_sequence", 2),
    ):
        model = fit_majority_context_model(training_episodes, order)
        baselines[name] = _compact_evaluation(
            evaluate_majority_context_model(model, heldout_episodes)
        )

    budget_sensitivity = []
    for budget in range(1, maximum_order + 1):
        result = learn_suffix_interface(
            training_episodes,
            maximum_order=budget,
            minimum_context_support=minimum_context_support,
        )
        budget_sensitivity.append(
            {
                "maximum_order": budget,
                "final_order": result.final_order,
                "certified": result.certified,
                "status": result.status,
                "repair_event_count": len(result.repair_events),
                "terminal_counterexample": (
                    None
                    if result.terminal_counterexample is None
                    else result.terminal_counterexample.to_dict()
                ),
            }
        )

    symbol_by_context = dict(zip(compiled.contexts, compiled.symbols))
    hidden_alignment = []
    for a_value in (0, 1):
        for b_value in (0, 1):
            # The predecessor of (a,b) has observed bit 1-b, so the last two
            # observations at the current state are (1-b,a).  This is used only
            # by the evaluator after learning.
            context = (1 - b_value, a_value)
            hidden_alignment.append(
                {
                    "evaluator_hidden_macrostate": (a_value, b_value),
                    "learned_observation_context": context,
                    "learned_symbol": symbol_by_context[context],
                }
            )
    alignment_bijective = len(
        {record["learned_symbol"] for record in hidden_alignment}
    ) == 4

    predictive_candidate = candidates["next_symbol_predictive_partition"]
    identity_candidate = candidates["learned_suffix_identity"]
    return {
        "name": "finite autonomous interface inferred from observations only",
        "hidden_environment": {
            "state_space": "{0,1} x {0,1} x N_0",
            "update": "(a,b,n)->(b,1-a,n+1)",
            "observation": "a",
            "access_policy": "hidden states are used by the simulator and post-hoc evaluator only",
        },
        "learner_input_boundary": {
            "schema": "tuple[tuple[non-negative integer observation, ...], ...]",
            "fields_received": ("observation_episodes",),
            "hidden_state_received": False,
            "training_episode_count": len(training_episodes),
            "training_observation_count": sum(map(len, training_episodes)),
            "training_payload_sha256": _episode_hash(training_episodes),
            "heldout_episode_count": len(heldout_episodes),
            "heldout_observation_count": sum(map(len, heldout_episodes)),
            "heldout_payload_sha256": _episode_hash(heldout_episodes),
            "training_counter_values_evaluator_only": training_counter_values,
            "heldout_counter_value_evaluator_only": heldout_counter,
        },
        "learned_interface": learned.to_dict(),
        "compiled_autonomous_machine": compiled.to_dict(),
        "candidate_abstractions": candidates,
        "heldout_evaluation": learner_evaluation,
        "baselines": baselines,
        "memory_budget_sensitivity": budget_sensitivity,
        "posthoc_hidden_alignment": {
            "records": hidden_alignment,
            "bijective": alignment_bijective,
            "used_for_learning": False,
        },
        "result": {
            "one_bit_memory_rejected": budget_sensitivity[0]["certified"] is False,
            "learned_memory_order": learned.final_order,
            "learned_state_count": len(compiled.contexts),
            "heldout_exact": learner_evaluation["exact_on_eligible_transitions"],
            "heldout_counter_transfer": heldout_counter,
            "next_symbol_partition_predictive_residual_nats": predictive_candidate[
                "predictive_residual_nats"
            ],
            "next_symbol_partition_lumpability_residual_nats": predictive_candidate[
                "lumpability_residual_nats"
            ],
            "next_symbol_partition_block_count": predictive_candidate["block_count"],
            "joint_repaired_block_count": len(predictive_candidate["joint_repair"]),
            "learned_identity_joint_fixed": identity_candidate["joint_fixed"],
            "hidden_alignment_bijective_posthoc": alignment_bijective,
        },
    }


def _countdown_case(
    maximum_order: int,
    post_absorption_steps: int,
) -> dict[str, object]:
    training_states = tuple(range(1, maximum_order + 5))
    heldout_states = tuple(range(maximum_order + 5, maximum_order + 13))
    training_episodes = tuple(
        countdown_observation_episode(state, post_absorption_steps)
        for state in training_states
    )
    heldout_episodes = tuple(
        countdown_observation_episode(state, post_absorption_steps)
        for state in heldout_states
    )
    budget_values = tuple(sorted({1, 2, 4, maximum_order}))
    budget_sensitivity = []
    learned_by_budget = {}
    for budget in budget_values:
        learned = learn_suffix_interface(
            training_episodes,
            maximum_order=budget,
            minimum_context_support=1,
        )
        learned_by_budget[budget] = learned
        terminal = learned.terminal_counterexample
        budget_sensitivity.append(
            {
                "maximum_order": budget,
                "final_order": learned.final_order,
                "certified": learned.certified,
                "status": learned.status,
                "repair_event_count": len(learned.repair_events),
                "terminal_shared_context": None if terminal is None else terminal.context,
                "terminal_targets": (
                    None if terminal is None else (terminal.left.target, terminal.right.target)
                ),
                "terminal_counterexample_passed": (
                    False if terminal is None else terminal.passed
                ),
            }
        )
    primary = learned_by_budget[maximum_order]
    learner_evaluation = _compact_evaluation(
        evaluate_suffix_interface(primary, heldout_episodes)
    )
    baselines = {}
    for name, order in (
        ("one_state_majority", 0),
        ("current_observation_majority", 1),
        (f"fixed_order_{maximum_order}_sequence", maximum_order),
    ):
        model = fit_majority_context_model(training_episodes, order)
        baselines[name] = _compact_evaluation(
            evaluate_majority_context_model(model, heldout_episodes)
        )
    exact_baselines = [
        name
        for name, record in baselines.items()
        if record["exact_on_eligible_transitions"]
    ]
    return {
        "name": "finite observation-memory budget cannot close the countdown",
        "hidden_environment": {
            "state_space": "N_0",
            "update": "n->max(n-1,0)",
            "observation": "1[n=0]",
            "access_policy": "hidden countdown values are used by the simulator and evaluator only",
        },
        "learner_input_boundary": {
            "schema": "tuple[tuple[binary observation, ...], ...]",
            "fields_received": ("observation_episodes",),
            "hidden_state_received": False,
            "training_episode_count": len(training_episodes),
            "training_observation_count": sum(map(len, training_episodes)),
            "training_payload_sha256": _episode_hash(training_episodes),
            "heldout_episode_count": len(heldout_episodes),
            "heldout_observation_count": sum(map(len, heldout_episodes)),
            "heldout_payload_sha256": _episode_hash(heldout_episodes),
            "training_initial_states_evaluator_only": training_states,
            "heldout_initial_states_evaluator_only": heldout_states,
        },
        "primary_learner": primary.to_dict(),
        "memory_budget_sensitivity": budget_sensitivity,
        "heldout_evaluation": learner_evaluation,
        "baselines": baselines,
        "result": {
            "all_finite_budgets_abstained": all(
                not record["certified"] for record in budget_sensitivity
            ),
            "all_terminal_counterexamples_passed": all(
                record["terminal_counterexample_passed"]
                for record in budget_sensitivity
            ),
            "primary_status": primary.status,
            "primary_strict_coverage": learner_evaluation["coverage"],
            "exact_baselines": exact_baselines,
            "high_accuracy_not_exact": all(
                record["accuracy_on_covered"] > 0.9
                and not record["exact_on_eligible_transitions"]
                for name, record in baselines.items()
                if name != "one_state_majority"
            ),
            "boundary": (
                "the finite runs witness failure through the declared memory budget; "
                "the universal no-finite-suffix statement follows from the symbolic "
                "countdown argument retained in the v0.2 report"
            ),
        },
    }


def _contradictory_control(maximum_order: int) -> dict[str, object]:
    prefix = (0,) * (maximum_order + 1)
    episodes = (prefix + (0,), prefix + (1,))
    learned = learn_suffix_interface(episodes, maximum_order=maximum_order)
    return {
        "name": "directly contradictory observation control",
        "input_payload_sha256": _episode_hash(episodes),
        "learned_interface": learned.to_dict(),
        "passed": (
            not learned.certified
            and learned.status == "memory_budget_exhausted"
            and learned.terminal_counterexample is not None
            and learned.terminal_counterexample.passed
        ),
        "meaning": (
            "an exact learner must not silently turn irreconcilable observations "
            "into a deterministic autonomous law"
        ),
    }


def run_online_experiment(
    *,
    maximum_order: int = 8,
    stable_episode_length: int = 64,
    stable_heldout_length: int = 128,
    countdown_post_absorption_steps: int = 16,
    minimum_context_support: int = 8,
    project_root: Path | None = None,
) -> dict[str, object]:
    if maximum_order < 2:
        raise ValueError("maximum_order must be at least two for the positive fixture")
    if stable_episode_length <= maximum_order or stable_heldout_length <= maximum_order:
        raise ValueError("stable episodes must exceed the memory budget")
    if countdown_post_absorption_steps <= maximum_order:
        raise ValueError("countdown absorbing tail must exceed the memory budget")
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]

    stable = _stable_case(
        maximum_order,
        stable_episode_length,
        stable_heldout_length,
        minimum_context_support,
    )
    countdown = _countdown_case(maximum_order, countdown_post_absorption_steps)
    contradictory = _contradictory_control(maximum_order)
    stable_result = stable["result"]
    countdown_result = countdown["result"]
    verification_checks = {
        "stable_interface_certified": stable["learned_interface"]["certified"],
        "stable_memory_order_is_two": stable_result["learned_memory_order"] == 2,
        "stable_four_state_machine": stable_result["learned_state_count"] == 4,
        "stable_heldout_exact": stable_result["heldout_exact"],
        "stable_transfers_to_unseen_counter": (
            stable_result["heldout_counter_transfer"] == 10**6
        ),
        "predictive_partition_has_zero_predictive_residual": (
            stable_result["next_symbol_partition_predictive_residual_nats"] == 0.0
        ),
        "predictive_partition_is_not_dynamically_closed": (
            stable_result["next_symbol_partition_lumpability_residual_nats"] > 0.0
        ),
        "joint_repair_recovers_four_states": (
            stable_result["joint_repaired_block_count"] == 4
        ),
        "posthoc_hidden_alignment_is_bijective": stable_result[
            "hidden_alignment_bijective_posthoc"
        ],
        "countdown_all_budgets_abstain": countdown_result[
            "all_finite_budgets_abstained"
        ],
        "countdown_counterexamples_valid": countdown_result[
            "all_terminal_counterexamples_passed"
        ],
        "countdown_high_accuracy_baselines_not_exact": countdown_result[
            "high_accuracy_not_exact"
        ],
        "contradictory_control_abstains": contradictory["passed"],
    }
    receipt = {
        "schema_version": "ts-language-foundations-online-interface-v0.3",
        "experiment": {
            "name": "counterexample-guided observation-only interface learning",
            "version": "0.3.0",
            "question": (
                "Can a learner with no hidden-state access discover the finite "
                "memory needed for prediction and autonomous continuation, and "
                "abstain when its finite memory budget cannot close the process?"
            ),
        },
        "run": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "maximum_suffix_order": maximum_order,
            "minimum_context_support": minimum_context_support,
            "stable_episode_length": stable_episode_length,
            "stable_heldout_length": stable_heldout_length,
            "countdown_post_absorption_steps": countdown_post_absorption_steps,
            "random_seed": None,
            "stochastic_sampling": False,
            "python": sys.version,
            "platform": platform.platform(),
            "neural_components": [],
            "external_corpus": None,
            "tokenizer": None,
            "network_required": False,
        },
        "algorithm": {
            "input": "observation episodes only",
            "initial_interface": "last observed symbol",
            "verifier": "same represented suffix must not have two distinct observed next symbols",
            "repair": "increase suffix order by one and rebuild from admitted observations",
            "closure_test": "compiled context transitions remain inside the supported finite context set",
            "failure_policy": "abstain globally when an exact conflict survives the declared memory budget",
            "standard_connections": (
                "variable-order Markov modelling, deterministic automata inference, "
                "predictive equivalence, causal-state ideas, and partition refinement"
            ),
            "novelty_boundary": (
                "suffix-memory growth is standard machinery; this experiment tests "
                "its explicit integration with predictive repair, dynamic closure, "
                "fail-closed verification, and infinite-fibre semantics"
            ),
        },
        "provenance": {"source_hashes": source_hashes(project_root)},
        "stable_infinite_fibre_case": stable,
        "unbounded_countdown_case": countdown,
        "negative_control": contradictory,
        "verification": {
            "checks": verification_checks,
            "check_count": len(verification_checks),
            "passed_count": sum(bool(value) for value in verification_checks.values()),
            "passed": all(verification_checks.values()),
        },
        "conclusion": {
            "established_in_fixtures": (
                "observation-only counterexamples select a two-symbol history "
                "interface in the stable process; predictive repair alone gives two "
                "blocks, while dynamic closure forces the exact four-state machine"
            ),
            "failure_result": (
                "the same learner rejects every tested finite suffix budget on the "
                "countdown instead of fabricating a closed state"
            ),
            "not_established": (
                "robust stochastic learning, continuous observations, scalable "
                "active experiment selection, language acquisition, or superiority "
                "to trained neural sequence models"
            ),
            "next_falsifiable_experiment": (
                "replace exact conflicts by confidence-bounded distributional tests "
                "under controlled noise, while charging explicit memory and sampling costs"
            ),
        },
    }
    return _jsonable(receipt)


def write_json(receipt: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_report(receipt: dict[str, object], path: Path) -> None:
    stable = receipt["stable_infinite_fibre_case"]
    countdown = receipt["unbounded_countdown_case"]
    learned = stable["learned_interface"]
    stable_result = stable["result"]
    prediction_candidate = stable["candidate_abstractions"][
        "next_symbol_predictive_partition"
    ]
    current_baseline = stable["baselines"]["current_observation_majority"]
    order_two_baseline = stable["baselines"]["fixed_order_two_sequence"]
    countdown_current = countdown["baselines"]["current_observation_majority"]
    countdown_fixed = countdown["baselines"][f"fixed_order_{receipt['run']['maximum_suffix_order']}_sequence"]
    text = f"""# Observation-Only Adaptive Interface Learning — v0.3

## Question

Can a learner that never receives the hidden state discover enough finite memory to predict and evolve autonomously? Can it also refuse a false finite closure when every available memory budget still contains a verifier counterexample?

## Input and algorithm

The learner receives only finite binary observation episodes. Environment states are retained outside the learner for simulation and post-hoc evaluation. Its initial representation is the last observed symbol. Whenever one represented suffix has been followed by two different symbols, the verifier emits the two observation prefixes as a counterexample and increases suffix order by one.

This is an exact, deterministic variable-memory learner. Suffix growth itself is standard and is closely related to variable-order Markov models and automata inference. The experiment-specific connection is that its learned context machine is then subjected to predictive repair and strong-lumpability repair, with fail-closed abstention at a declared resource boundary.

## Positive fixture — the hidden state is never supplied

The simulator evolves

```text
(a,b,n) -> (b,1-a,n+1),    observation = a,
```

but the learner receives only bit strings. At suffix order one, the same observed bit has incompatible next symbols. One verifier event changes the represented history length from `{learned['initial_order']}` to `{learned['final_order']}`.

The four supported observation contexts are `00`, `01`, `10`, and `11`. They compile to the exact autonomous cycle

```text
00 -> 01 -> 11 -> 10 -> 00.
```

The learned machine is exact on `{stable['heldout_evaluation']['eligible_transitions']}` held-out transitions generated at the unseen counter value `{stable_result['heldout_counter_transfer']}`. Post-hoc evaluation finds a bijection between the four learned contexts and the four hidden `(a,b)` macrostates. That alignment was not used for learning.

## Prediction is still weaker than autonomous state

Grouping the four contexts only by their next-symbol law yields two blocks. Its predictive residual is `{prediction_candidate['predictive_residual_nats']:.12f}` nats, while its lumpability residual is `{prediction_candidate['lumpability_residual_nats']:.12f}` nats. Dynamic repair splits it into `{stable_result['joint_repaired_block_count']}` blocks.

Thus the observation-only run independently reconstructs the original distinction:

```text
enough state for the next answer != enough state for autonomous continuation.
```

## Baselines

On the infinite-fibre holdout:

| Model | Held-out accuracy | Exact? | Selects memory? |
|---|---:|---:|---:|
| One-state majority | {stable['baselines']['one_state_majority']['accuracy_on_covered']:.3f} | no | no |
| Current observation | {current_baseline['accuracy_on_covered']:.3f} | no | no |
| Fixed order two | {order_two_baseline['accuracy_on_covered']:.3f} | yes | no |
| Counterexample-guided interface | {stable['heldout_evaluation']['accuracy_on_covered']:.3f} | yes | yes |

The fixed order-two baseline shows that raw prediction does not require a new algorithm here. The additional result is automatic memory selection plus a separately verified autonomous quotient.

## Negative fixture — finite memory cannot identify the countdown boundary

For observation strings `0^n 1^...`, every tested suffix budget contains the context `0^K` followed by both `0` and `1`. Budgets `{', '.join(str(record['maximum_order']) for record in countdown['memory_budget_sensitivity'])}` all terminate with valid counterexamples. The exact learner therefore abstains.

This contrasts accuracy with certification. The current-observation majority baseline scores `{countdown_current['accuracy_on_covered']:.3f}` but makes `{countdown_current['wrong']}` boundary errors. The fixed order-{receipt['run']['maximum_suffix_order']} baseline scores `{countdown_fixed['accuracy_on_covered']:.3f}` and still makes `{countdown_fixed['wrong']}` boundary errors. Neither is exact.

The computation verifies the declared finite budgets. The universal no-finite-suffix conclusion is inherited from the symbolic countdown proof in the v0.2 report, not inferred from finite testing alone.

## Verifier results

- Checks passed: `{receipt['verification']['passed_count']}/{receipt['verification']['check_count']}`.
- Observation-only learned order: `{learned['final_order']}`.
- Learned autonomous states: `{stable_result['learned_state_count']}`.
- Stable holdout errors: `{stable['heldout_evaluation']['wrong']}`.
- Countdown strict coverage after failed closure: `{countdown['heldout_evaluation']['coverage']:.3f}`.
- Contradictory-control abstention: `{receipt['negative_control']['passed']}`.

## Interpretation boundary

This establishes a small deterministic mechanism: observable verifier failures can revise memory, and predictive state can be repaired into autonomous state without hidden-state supervision. It does not establish robust learning under noise, continuous state discovery, scalable active experimentation, language learning, or superiority to neural sequence models.

The next falsifiable experiment replaces exact equality by confidence-bounded comparison of next-symbol distributions under controlled noise, with explicit penalties for memory, samples, abstention, and mistaken merges.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_svg(receipt: dict[str, object], path: Path) -> None:
    stable = receipt["stable_infinite_fibre_case"]
    countdown = receipt["unbounded_countdown_case"]
    stable_bases = stable["baselines"]
    countdown_current = countdown["baselines"]["current_observation_majority"]
    countdown_fixed = countdown["baselines"][f"fixed_order_{receipt['run']['maximum_suffix_order']}_sequence"]
    bars = (
        ("one state", stable_bases["one_state_majority"]["accuracy_on_covered"], "#ffbf5b"),
        ("current bit", stable_bases["current_observation_majority"]["accuracy_on_covered"], "#ffbf5b"),
        ("fixed k=2", stable_bases["fixed_order_two_sequence"]["accuracy_on_covered"], "#65e6a5"),
        ("adaptive", stable["heldout_evaluation"]["accuracy_on_covered"], "#70ddff"),
    )
    bar_svg = []
    for index, (label, value, colour) in enumerate(bars):
        x_value = 80 + index * 135
        height = 150 * value
        y_value = 745 - height
        bar_svg.append(
            f'<rect x="{x_value}" y="{y_value:.1f}" width="92" height="{height:.1f}" rx="8" fill="{colour}" opacity="0.88"/>'
        )
        bar_svg.append(
            f'<text x="{x_value + 46}" y="{y_value - 10:.1f}" text-anchor="middle" class="value">{value:.2f}</text>'
        )
        bar_svg.append(
            f'<text x="{x_value + 46}" y="772" text-anchor="middle" class="small">{label}</text>'
        )
    budget_nodes = []
    for index, record in enumerate(countdown["memory_budget_sensitivity"]):
        x_value = 790 + index * 120
        budget = record["maximum_order"]
        budget_nodes.append(
            f'<circle cx="{x_value}" cy="655" r="36" fill="#402c58" stroke="#c58cff" stroke-width="3"/>'
        )
        budget_nodes.append(
            f'<text x="{x_value}" y="650" text-anchor="middle" class="value">k={budget}</text>'
        )
        budget_nodes.append(
            f'<text x="{x_value}" y="674" text-anchor="middle" class="tiny">conflict</text>'
        )
        if index:
            budget_nodes.append(
                f'<path d="M {x_value - 84} 655 L {x_value - 40} 655" stroke="#9fb5cc" stroke-width="2" marker-end="url(#arrow)"/>'
            )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="900" viewBox="0 0 1400 900">
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#9fb5cc"/></marker>
  <style>
    .title{{font:700 34px system-ui,sans-serif;fill:#70ddff}} .head{{font:600 23px system-ui,sans-serif;fill:#e6eef8}}
    .body{{font:18px system-ui,sans-serif;fill:#d4dfed}} .small{{font:15px system-ui,sans-serif;fill:#a8bdd3}}
    .tiny{{font:13px system-ui,sans-serif;fill:#cbb7df}} .value{{font:600 17px system-ui,sans-serif;fill:#f1f6fb}}
    .panel{{fill:#102136;stroke:#315a7d;stroke-width:2}} .node{{fill:#194563;stroke:#70ddff;stroke-width:3}}
  </style>
</defs>
<rect width="1400" height="900" fill="#071321"/>
<text x="48" y="58" class="title">Observation-only repair — finite interface, hidden infinite interior</text>

<rect x="40" y="90" width="410" height="410" rx="18" class="panel"/>
<text x="68" y="130" class="head">1. The learner sees only this</text>
<text x="68" y="167" class="small">hidden simulator: (a,b,n) → (b,1−a,n+1)</text>
<rect x="70" y="200" width="350" height="72" rx="12" fill="#0b1828" stroke="#4f7799"/>
<text x="245" y="245" text-anchor="middle" class="head">0 0 1 1 0 0 1 1 …</text>
<text x="68" y="310" class="body">No a, b, n labels. No state IDs.</text>
<text x="68" y="344" class="body">Only finite observation episodes.</text>
<text x="68" y="395" class="small">held-out fibre index</text>
<text x="68" y="430" class="head">n = 1,000,000</text>
<text x="68" y="466" class="body" fill="#65e6a5">{stable['heldout_evaluation']['correct']}/{stable['heldout_evaluation']['eligible_transitions']} transitions correct</text>

<rect x="470" y="90" width="890" height="410" rx="18" class="panel"/>
<text x="500" y="130" class="head">2. Counterexample → memory repair → autonomous closure</text>
<circle cx="555" cy="225" r="56" class="node"/><text x="555" y="218" text-anchor="middle" class="value">k = 1</text><text x="555" y="242" text-anchor="middle" class="tiny">0 → 0 and 1</text>
<path d="M 615 225 L 690 225" stroke="#9fb5cc" stroke-width="3" marker-end="url(#arrow)"/>
<text x="653" y="205" text-anchor="middle" class="tiny">split memory</text>
<circle cx="760" cy="225" r="70" class="node"/><text x="760" y="209" text-anchor="middle" class="value">k = 2</text><text x="760" y="235" text-anchor="middle" class="tiny">00 01 10 11</text><text x="760" y="258" text-anchor="middle" class="tiny">next symbol exact</text>
<path d="M 833 225 L 910 225" stroke="#9fb5cc" stroke-width="3" marker-end="url(#arrow)"/>
<text x="872" y="205" text-anchor="middle" class="tiny">predictive quotient</text>
<circle cx="985" cy="225" r="70" fill="#3f3155" stroke="#c58cff" stroke-width="3"/><text x="985" y="209" text-anchor="middle" class="value">2 blocks</text><text x="985" y="235" text-anchor="middle" class="tiny">R = 0</text><text x="985" y="258" text-anchor="middle" class="tiny">L = ln 2</text>
<path d="M 1058 225 L 1135 225" stroke="#9fb5cc" stroke-width="3" marker-end="url(#arrow)"/>
<text x="1097" y="205" text-anchor="middle" class="tiny">dynamic repair</text>
<circle cx="1210" cy="225" r="70" fill="#184b3d" stroke="#65e6a5" stroke-width="3"/><text x="1210" y="209" text-anchor="middle" class="value">4 states</text><text x="1210" y="235" text-anchor="middle" class="tiny">R = 0, L = 0</text><text x="1210" y="258" text-anchor="middle" class="tiny">autonomous</text>
<text x="500" y="342" class="body">00 → 01 → 11 → 10 → 00</text>
<text x="500" y="382" class="small">Prediction merges contexts that dynamics must keep separate.</text>
<text x="500" y="424" class="small">The four learned symbols align bijectively with (a,b) only after evaluation.</text>
<text x="500" y="463" class="body" fill="#65e6a5">Hidden-state supervision used: none</text>

<rect x="40" y="520" width="650" height="330" rx="18" class="panel"/>
<text x="68" y="560" class="head">3. Stable-process held-out accuracy</text>
<line x1="68" y1="745" x2="630" y2="745" stroke="#66829c" stroke-width="2"/>
{''.join(bar_svg)}
<text x="68" y="816" class="small">Fixed k=2 matches prediction; adaptive repair selects k=2 and certifies closure.</text>

<rect x="710" y="520" width="650" height="330" rx="18" class="panel"/>
<text x="740" y="560" class="head">4. Countdown: every finite budget fails closure</text>
{''.join(budget_nodes)}
<text x="740" y="735" class="body">current-bit accuracy = {countdown_current['accuracy_on_covered']:.3f}, errors = {countdown_current['wrong']}</text>
<text x="740" y="770" class="body">fixed k={receipt['run']['maximum_suffix_order']} accuracy = {countdown_fixed['accuracy_on_covered']:.3f}, errors = {countdown_fixed['wrong']}</text>
<text x="740" y="808" class="body" fill="#c58cff">Verifier result: abstain — no exact finite closure certified</text>
<text x="48" y="882" class="small">Exact deterministic fixtures. Finite runs test declared budgets; symbolic arguments carry universal claims.</text>
</svg>'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
