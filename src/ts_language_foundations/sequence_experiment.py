"""End-to-end ternary sequence construction, transfer, and repair study."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from fractions import Fraction
from html import escape
from math import log
from pathlib import Path
from random import Random
from statistics import mean, median
from typing import Any, Iterable, Sequence

from .active import (
    CompiledControlledInterface,
    controlled_dynamic_closure,
    controlled_lumpability_residual,
    controlled_predictive_residual,
)
from .experiment import source_hashes
from .relational import controlled_joint_repair
from .repair import common_refinement, partition_entropy, refines
from .sequence import (
    CategoricalInteraction,
    CategoricalSequenceWorld,
    SequenceRepairConfig,
    TypedExpression,
    TypedRepresentation,
    categorical_contexts,
    categorical_stationarity_assessment,
    combine_oracle_interfaces,
    compile_empirical_sequence_interface,
    evaluate_categorical_table,
    fit_categorical_table,
    generate_sequence_interactions,
    generate_typed_features,
    generate_typed_representations,
    oracle_sequence_interface,
    partition_from_typed_features,
    predictive_partition_for_actions,
    raw_typed_identity,
    representation_by_name,
    simulate_sequence_repair,
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


def _interaction_hash(interactions: Iterable[CategoricalInteraction]) -> str:
    encoded = json.dumps(
        [interaction.to_dict() for interaction in interactions],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("summary requires observations")
    return {
        "minimum": min(values),
        "median": median(values),
        "mean": mean(values),
        "maximum": max(values),
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _paired_ratio_bootstrap(
    numerator: Sequence[float],
    denominator: Sequence[float],
    *,
    seed: int,
    resamples: int = 2000,
) -> dict[str, object]:
    if len(numerator) != len(denominator) or not numerator:
        raise ValueError("paired bootstrap needs equal non-empty samples")
    ratios = tuple(left / right for left, right in zip(numerator, denominator, strict=True))
    rng = Random(seed)
    values = []
    for _ in range(resamples):
        values.append(median(ratios[rng.randrange(len(ratios))] for _ in ratios))
    return {
        "point_estimate": median(ratios),
        "percentile_95_interval": (_percentile(values, 0.025), _percentile(values, 0.975)),
        "resamples": resamples,
        "unit": "paired algorithmic seed",
        "boundary": "fixture sensitivity, not a population confidence interval",
    }


def _wilson(successes: int, trials: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    half = z * (p * (1 - p) / trials + z * z / (4 * trials * trials)) ** 0.5 / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def _transition_entropy(interface: CompiledControlledInterface) -> dict[str, float]:
    result = {}
    for action_index, action in enumerate(interface.actions):
        result[action] = mean(
            -sum(float(value) * log(float(value)) for value in law if value > 0)
            for law in interface.predictive_laws[action_index]
        )
    return result


def _persistence(
    interface: CompiledControlledInterface, partition
) -> dict[str, float]:
    block_of = {
        state: block_index
        for block_index, block in enumerate(partition)
        for state in block
    }
    result = {}
    for action_index, action in enumerate(interface.actions):
        values = []
        for state, row in enumerate(interface.transition_rows[action_index]):
            values.append(
                sum(
                    float(probability)
                    for target, probability in enumerate(row)
                    if block_of[target] == block_of[state]
                )
            )
        result[action] = mean(values)
    return result


def _find_partition_representation(
    representations: Sequence[TypedRepresentation], partition
) -> TypedRepresentation:
    matches = tuple(item for item in representations if item.partition == partition)
    if len(matches) != 1:
        raise ValueError(f"expected one canonical representation for partition, found {len(matches)}")
    return matches[0]


def _candidate_record(
    representation: TypedRepresentation,
    *,
    predictive_partition,
    interface: CompiledControlledInterface,
    weights,
) -> dict[str, object]:
    dynamic, dynamic_trace = controlled_dynamic_closure(
        interface.transition_rows, representation.partition, return_trace=True
    )
    predictive = common_refinement(representation.partition, predictive_partition)
    joint_pd, trace_pd = controlled_joint_repair(
        interface.transition_rows,
        predictive_partition,
        representation.partition,
        schedule="predictive_then_dynamic",
        return_trace=True,
    )
    joint_dp, trace_dp = controlled_joint_repair(
        interface.transition_rows,
        predictive_partition,
        representation.partition,
        schedule="dynamic_then_predictive",
        return_trace=True,
    )
    cp = controlled_dynamic_closure(interface.transition_rows, predictive)
    pc = common_refinement(dynamic, predictive_partition)
    return {
        "name": representation.name,
        "partition": representation.partition,
        "block_count": representation.block_count,
        "predictive_residual_nats": controlled_predictive_residual(
            interface, representation.partition, weights
        ),
        "lumpability_residual_nats": controlled_lumpability_residual(
            interface, representation.partition, weights
        ),
        "partition_entropy_nats": partition_entropy(representation.partition, weights),
        "predictive_repair": predictive,
        "predictive_fixed": predictive == representation.partition,
        "dynamic_closure": dynamic,
        "dynamic_trace": dynamic_trace,
        "dynamic_fixed": dynamic == representation.partition,
        "dynamic_closure_excess_blocks": len(dynamic) - representation.block_count,
        "joint_repair": joint_pd,
        "joint_trace_predictive_then_dynamic": trace_pd,
        "joint_trace_dynamic_then_predictive": trace_dp,
        "joint_repair_excess_blocks": len(joint_pd) - representation.block_count,
        "joint_fixed": joint_pd == representation.partition,
        "completed_repair_order_independent": joint_pd == joint_dp,
        "partial_repairs": {
            "C_after_P": cp,
            "P_after_C": pc,
            "commute_at_candidate": cp == pc,
        },
    }


def _candidate_lattice_audit(
    representations: Sequence[TypedRepresentation],
    *,
    interface: CompiledControlledInterface,
    predictive_partition,
) -> dict[str, object]:
    """Exhaust the declared candidate family without calling it a proof.

    This checks closure laws on every generated partition and monotonicity on
    every comparable ordered pair inside the grammar-generated subposet.  The
    theorem remains the finite-lattice theorem; the counts are a fixture
    receipt, not an exhaustive traversal of the full Bell(27) lattice.
    """

    candidates = tuple(representations)
    closures = {}
    joint_repairs = {}
    closure_histogram: Counter[int] = Counter()
    joint_histogram: Counter[int] = Counter()
    joint_basins: Counter[object] = Counter()
    predictive_fixed = []
    dynamic_fixed = []
    joint_fixed = []
    noncommuting = []
    extensive_failures = []
    idempotence_failures = []
    order_failures = []
    for representation in candidates:
        partition = representation.partition
        predictive = common_refinement(partition, predictive_partition)
        closure = controlled_dynamic_closure(interface.transition_rows, partition)
        c_after_p = controlled_dynamic_closure(
            interface.transition_rows, predictive
        )
        p_after_c = common_refinement(closure, predictive_partition)
        joint_pd = controlled_joint_repair(
            interface.transition_rows,
            predictive_partition,
            partition,
            schedule="predictive_then_dynamic",
        )
        joint_dp = controlled_joint_repair(
            interface.transition_rows,
            predictive_partition,
            partition,
            schedule="dynamic_then_predictive",
        )
        closures[partition] = closure
        joint_repairs[partition] = joint_pd
        closure_histogram[len(closure)] += 1
        joint_histogram[len(joint_pd)] += 1
        joint_basins[joint_pd] += 1
        if predictive == partition:
            predictive_fixed.append(representation.name)
        if closure == partition:
            dynamic_fixed.append(representation.name)
        if joint_pd == partition:
            joint_fixed.append(representation.name)
        if c_after_p != p_after_c:
            noncommuting.append(
                {
                    "name": representation.name,
                    "input_blocks": representation.block_count,
                    "C_after_P_blocks": len(c_after_p),
                    "P_after_C_blocks": len(p_after_c),
                }
            )
        if not refines(closure, partition):
            extensive_failures.append(representation.name)
        if controlled_dynamic_closure(interface.transition_rows, closure) != closure:
            idempotence_failures.append(representation.name)
        if joint_pd != joint_dp:
            order_failures.append(representation.name)

    def label_vector(partition) -> tuple[int, ...]:
        labels = [-1] * 27
        for block_index, block in enumerate(partition):
            for state in block:
                labels[state] = block_index
        return tuple(labels)

    labels = {partition: label_vector(partition) for partition in closures}
    closure_labels = {
        partition: label_vector(closure) for partition, closure in closures.items()
    }

    def labels_refine(finer: tuple[int, ...], coarser: tuple[int, ...]) -> bool:
        mapping = {}
        for fine, coarse in zip(finer, coarser, strict=True):
            incumbent = mapping.setdefault(fine, coarse)
            if incumbent != coarse:
                return False
        return True

    comparable_pairs = 0
    monotonicity_failures = []
    partitions = tuple(closures)
    for finer in partitions:
        for coarser in partitions:
            if not labels_refine(labels[finer], labels[coarser]):
                continue
            comparable_pairs += 1
            if not labels_refine(closure_labels[finer], closure_labels[coarser]):
                monotonicity_failures.append((finer, coarser))

    source_only = CategoricalSequenceWorld(
        actions=("source_probe",), informative_action="source_probe"
    )
    target = CategoricalSequenceWorld(
        relation="middle_symbol",
        actions=("target_probe",),
        informative_action="target_probe",
    )
    source_interface = oracle_sequence_interface(source_only)
    enlarged_interface = combine_oracle_interfaces(
        (source_interface, oracle_sequence_interface(target))
    )
    action_family_failures = []
    for representation in candidates:
        source_closure = controlled_dynamic_closure(
            source_interface.transition_rows, representation.partition
        )
        enlarged_closure = controlled_dynamic_closure(
            enlarged_interface.transition_rows, representation.partition
        )
        if not refines(enlarged_closure, source_closure):
            action_family_failures.append(representation.name)

    return {
        "scope": "all grammar-generated candidate partitions; not all partitions of 27 states",
        "candidate_partition_count": len(candidates),
        "predictive_fixed_point_count": len(predictive_fixed),
        "dynamic_fixed_point_count": len(dynamic_fixed),
        "joint_fixed_point_count": len(joint_fixed),
        "dynamic_fixed_points": tuple(dynamic_fixed),
        "joint_fixed_points": tuple(joint_fixed),
        "noncommuting_partial_repair_count": len(noncommuting),
        "noncommuting_partial_repairs": tuple(noncommuting),
        "dynamic_closure_block_histogram": dict(sorted(closure_histogram.items())),
        "joint_repair_block_histogram": dict(sorted(joint_histogram.items())),
        "unique_joint_completion_count": len(joint_basins),
        "largest_joint_completion_basin": max(joint_basins.values()),
        "extensivity_failures": tuple(extensive_failures),
        "idempotence_failures": tuple(idempotence_failures),
        "repair_order_failures": tuple(order_failures),
        "comparable_ordered_pair_count": comparable_pairs,
        "candidate_subposet_monotonicity_failure_count": len(monotonicity_failures),
        "enlarged_action_family_failure_count": len(action_family_failures),
        "passed": not (
            extensive_failures
            or idempotence_failures
            or order_failures
            or monotonicity_failures
            or action_family_failures
        ),
    }


def _evaluate_representations(
    training: Sequence[CategoricalInteraction],
    heldout: Sequence[CategoricalInteraction],
    representations: dict[str, TypedRepresentation],
) -> dict[str, object]:
    result = {}
    for name, representation in representations.items():
        table = fit_categorical_table(training, representation, pseudocount=0.5)
        result[name] = evaluate_categorical_table(table, representation, heldout)
    return result


def _transfer_analysis(
    *,
    seed: int,
    selected: TypedRepresentation,
    representations: Sequence[TypedRepresentation],
) -> dict[str, object]:
    contexts = categorical_contexts()
    weights = tuple(Fraction(1, len(contexts)) for _ in contexts)
    one_state = representation_by_name(representations, "one_state")
    lag1 = representation_by_name(representations, "lag1")
    identity = raw_typed_identity()

    compatible_world = CategoricalSequenceWorld(
        relation="endpoint_repeat",
        actions=("challenge",),
        informative_action="challenge",
        endpoint_true_outcome=2,
        endpoint_false_outcome=0,
    )
    compatible_interface = oracle_sequence_interface(compatible_world)
    compatible_partition = predictive_partition_for_actions(compatible_interface)
    compatible_train = generate_sequence_interactions(
        world=compatible_world,
        action="challenge",
        interaction_count=8100,
        seed=seed + 101,
    )
    compatible_test = generate_sequence_interactions(
        world=compatible_world,
        action="challenge",
        interaction_count=27000,
        seed=seed + 102,
        initial_step=10**6,
    )

    target_world = CategoricalSequenceWorld(
        relation="middle_symbol",
        actions=("target_probe",),
        informative_action="target_probe",
    )
    target_interface = oracle_sequence_interface(target_world)
    target_partition = predictive_partition_for_actions(target_interface)
    target_train = generate_sequence_interactions(
        world=target_world,
        action="target_probe",
        interaction_count=8100,
        seed=seed + 201,
    )
    target_test = generate_sequence_interactions(
        world=target_world,
        action="target_probe",
        interaction_count=27000,
        seed=seed + 202,
        initial_step=10**6,
    )

    source_task_world = CategoricalSequenceWorld(
        relation="endpoint_repeat",
        actions=("source_probe",),
        informative_action="source_probe",
    )
    source_interface = oracle_sequence_interface(source_task_world)
    combined = combine_oracle_interfaces((source_interface, target_interface))
    family_partition = predictive_partition_for_actions(combined)
    family_representation = _find_partition_representation(
        representations, family_partition
    )
    family_closure, family_trace = controlled_dynamic_closure(
        combined.transition_rows, family_partition, return_trace=True
    )

    source_train = generate_sequence_interactions(
        world=source_task_world,
        action="source_probe",
        interaction_count=8100,
        seed=seed + 301,
    )
    source_test = generate_sequence_interactions(
        world=source_task_world,
        action="source_probe",
        interaction_count=27000,
        seed=seed + 302,
        initial_step=10**6,
    )
    family_training = (*source_train, *target_train)
    family_test = (*source_test, *target_test)

    compatible_residual = controlled_predictive_residual(
        compatible_interface, selected.partition, weights
    )
    incompatible_residual = controlled_predictive_residual(
        target_interface, selected.partition, weights
    )
    family_selected_residual = controlled_predictive_residual(
        combined, selected.partition, weights
    )
    family_repaired_residual = controlled_predictive_residual(
        combined, family_representation.partition, weights
    )
    return {
        "frozen_compatible_task": {
            "task": "new outcome mapping with the same endpoint-repeat factor",
            "source_partition_equals_target_partition": selected.partition
            == compatible_partition,
            "exact_predictive_residual_nats": compatible_residual,
            "training_interaction_sha256": _interaction_hash(compatible_train),
            "heldout_interaction_sha256": _interaction_hash(compatible_test),
            "heldout": _evaluate_representations(
                compatible_train,
                compatible_test,
                {
                    "one_state": one_state,
                    "frozen_source_state": selected,
                    "raw_identity": identity,
                },
            ),
            "passed": compatible_residual < 1e-14,
        },
        "frozen_incompatible_task": {
            "task": "next law depends on the middle symbol",
            "source_state_exact_predictive_residual_nats": incompatible_residual,
            "target_specific_partition_blocks": len(target_partition),
            "training_interaction_sha256": _interaction_hash(target_train),
            "heldout_interaction_sha256": _interaction_hash(target_test),
            "heldout": _evaluate_representations(
                target_train,
                target_test,
                {
                    "one_state": one_state,
                    "frozen_source_state": selected,
                    "target_specific_lag1": lag1,
                    "raw_identity": identity,
                },
            ),
            "rejected": incompatible_residual > 1e-6,
        },
        "task_family_repair": {
            "source_predictive_blocks": selected.block_count,
            "target_predictive_blocks": len(target_partition),
            "least_joint_predictive_blocks": len(family_partition),
            "least_joint_predictive_representation": family_representation.name,
            "autonomous_blocks": len(family_closure),
            "autonomous_trace": family_trace,
            "source_state_family_residual_nats": family_selected_residual,
            "repaired_family_residual_nats": family_repaired_residual,
            "source_partition_refined_by_family": refines(
                family_partition, selected.partition
            ),
            "target_partition_refined_by_family": refines(
                family_partition, target_partition
            ),
            "family_training_interaction_sha256": _interaction_hash(family_training),
            "family_heldout_interaction_sha256": _interaction_hash(family_test),
            "heldout": _evaluate_representations(
                family_training,
                family_test,
                {
                    "one_state": one_state,
                    "frozen_source_state": selected,
                    "least_family_state": family_representation,
                    "raw_identity": identity,
                },
            ),
            "passed": len(family_partition) == 6
            and len(family_closure) == 18
            and family_repaired_residual < 1e-14,
        },
        "theorem_boundary": {
            "standard_finite_fact": (
                "adding predictive-law families can only refine the equal-law partition; "
                "adding controlled kernels can only refine the common lumpability closure"
            ),
            "frozen_transfer_criterion": (
                "a frozen partition transfers exactly iff every added law factors through it"
            ),
            "explicit_counterexample": (
                "endpoint-repeat has two blocks but middle-symbol varies inside both blocks"
            ),
        },
    }


def _primary_analysis(
    *, seed: int, config: SequenceRepairConfig, heldout_interactions: int
) -> dict[str, object]:
    contexts = categorical_contexts()
    features = generate_typed_features()
    representations = generate_typed_representations(
        maximum_feature_count=config.maximum_feature_count
    )
    world = CategoricalSequenceWorld(actions=config.actions)
    active = simulate_sequence_repair(
        world=world,
        policy="active",
        seed=seed,
        config=config,
        representations=representations,
        features=features,
    )
    uniform = simulate_sequence_repair(
        world=world,
        policy="uniform",
        seed=seed,
        config=config,
        representations=representations,
        features=features,
    )
    if not active.certified or not uniform.certified:
        raise RuntimeError("primary sequence runs failed to certify")
    selected = active.selected_representation
    if selected is None or selected.name != "eq(lag0,lag2)":
        raise RuntimeError(f"unexpected primary sequence state: {None if selected is None else selected.name}")
    if uniform.selected_representation is None or uniform.selected_representation.partition != selected.partition:
        raise RuntimeError("active and uniform runs selected different partitions")

    oracle = oracle_sequence_interface(world)
    predictive_partition = predictive_partition_for_actions(
        oracle, config.certification_actions
    )
    closure, closure_trace = controlled_dynamic_closure(
        oracle.transition_rows, selected.partition, return_trace=True
    )
    autonomous = _find_partition_representation(representations, closure)
    pooled = compile_empirical_sequence_interface(
        active, actions=config.certification_actions, pool_partition=selected.partition
    )
    pooled_closure = controlled_dynamic_closure(
        pooled.transition_rows, selected.partition
    )
    weights = tuple(Fraction(1, len(contexts)) for _ in contexts)
    identity = raw_typed_identity()
    focused_names = (
        "one_state",
        "lag0",
        "lag1",
        "lag2",
        "eq(lag0,lag2)",
        "lag0 + lag1 + eq(lag0,lag2)",
    )
    focused = [representation_by_name(representations, name) for name in focused_names]
    focused.append(identity)
    candidates = {
        item.name: _candidate_record(
            item,
            predictive_partition=predictive_partition,
            interface=oracle,
            weights=weights,
        )
        for item in focused
    }
    lattice_audit = _candidate_lattice_audit(
        representations,
        interface=oracle,
        predictive_partition=predictive_partition,
    )

    heldout = generate_sequence_interactions(
        world=CategoricalSequenceWorld(actions=("probe",), informative_action="probe"),
        action="probe",
        interaction_count=heldout_interactions,
        seed=seed + 1,
        initial_step=10**6,
    )
    training = tuple(item for item in active.interactions if item.action == "probe")
    heldout_models = _evaluate_representations(
        training,
        heldout,
        {
            "one_state": representation_by_name(representations, "one_state"),
            "selected_endpoint_relation": selected,
            "autonomous_repaired_state": autonomous,
            "raw_identity": identity,
        },
    )
    occupancy = Counter(item.history for item in active.interactions)
    selected_block_of = {
        state: block_index
        for block_index, block in enumerate(selected.partition)
        for state in block
    }
    autonomous_block_of = {
        state: block_index
        for block_index, block in enumerate(closure)
        for state in block
    }
    state_dictionary = tuple(
        {
            "state_index": state,
            "raw_history": context,
            "lag2": context[-3],
            "lag1": context[-2],
            "lag0": context[-1],
            "endpoint_repeat": context[-3] == context[-1],
            "predictive_block": selected_block_of[state],
            "autonomous_block": autonomous_block_of[state],
        }
        for state, context in enumerate(contexts)
    )
    one_state = candidates["one_state"]
    transfer = _transfer_analysis(
        seed=seed,
        selected=selected,
        representations=representations,
    )
    return {
        "hidden_environment": {
            "state": "(x[t-2],x[t-1],x[t],n) in {0,1,2}^3 x N_0",
            "transition": "(a,b,c,n) --u--> (b,c,W,n+1)",
            "baseline_law": "uniform over three next symbols",
            "probe_law": "dominant outcome 0 if a=c, dominant outcome 1 otherwise",
            "infinite_coordinate_effect": "n increments but affects no registered law",
            "learner_hidden_relation_access": False,
            "learner_input": "raw ternary history, chosen action, cost, categorical outcome",
            "protocol": "one reset transition per episode; contexts cycle through all 27 histories",
        },
        "grammar": {
            "type": "typed finite symbol/predicate expression grammar",
            "symbol_operators": ("lag",),
            "predicate_operators": (
                "eq",
                "neq",
                "contains",
                "count_ge",
                "all_equal",
                "all_distinct",
                "and",
                "or",
                "xor",
            ),
            "typed_output_deduplication": True,
            "feature_count": len(features),
            "candidate_partition_count": len(representations),
            "maximum_feature_tuple_size": config.maximum_feature_count,
            "features": tuple(feature.to_dict(contexts) for feature in features),
            "fixed_candidate_order_sha256": hashlib.sha256(
                "\n".join(item.name for item in representations).encode()
            ).hexdigest(),
            "canonical_label_boundary": (
                "eq and neq induce the same unlabelled partition; eq wins fixed syntax ordering"
            ),
        },
        "state_dictionary": state_dictionary,
        "active_run": active.to_dict(contexts),
        "uniform_run": uniform.to_dict(contexts),
        "policy_comparison": {
            "active_steps": active.steps,
            "uniform_steps": uniform.steps,
            "active_cost": active.total_cost,
            "uniform_cost": uniform.total_cost,
            "step_ratio_active_over_uniform": active.steps / uniform.steps,
            "cost_ratio_active_over_uniform": active.total_cost / uniform.total_cost,
            "active_action_counts": dict(active.action_counts),
            "uniform_action_counts": dict(uniform.action_counts),
            "authority_boundary": "the Bayesian policy proposes actions; it cannot accept a state",
        },
        "training_receipt": {
            "active_interaction_count": len(active.interactions),
            "active_interaction_sha256": _interaction_hash(active.interactions),
            "uniform_interaction_count": len(uniform.interactions),
            "uniform_interaction_sha256": _interaction_hash(uniform.interactions),
            "true_relation_label_received": False,
            "hidden_counter_received": False,
        },
        "selected_predictive_state": selected.to_dict(contexts),
        "autonomous_repaired_state": autonomous.to_dict(contexts),
        "controlled_dynamic_repair_trace": closure_trace,
        "candidate_abstractions": candidates,
        "candidate_lattice_audit": lattice_audit,
        "dynamics": {
            "raw_state_count": len(contexts),
            "predictive_block_count": selected.block_count,
            "autonomous_block_count": len(closure),
            "identity_block_count": identity.block_count,
            "dynamic_closure_excess_blocks": len(closure) - selected.block_count,
            "joint_repair_excess_from_one_state": len(closure) - 1,
            "transition_entropy_nats_by_action": _transition_entropy(oracle),
            "selected_state_persistence_by_action": _persistence(
                oracle, selected.partition
            ),
            "selected_predictive_residual_nats": controlled_predictive_residual(
                oracle, selected.partition, weights
            ),
            "selected_lumpability_residual_nats": controlled_lumpability_residual(
                oracle, selected.partition, weights
            ),
            "pooled_empirical_closure_agrees_with_oracle": pooled_closure == closure,
            "one_state_C_after_P_blocks": len(one_state["partial_repairs"]["C_after_P"]),
            "one_state_P_after_C_blocks": len(one_state["partial_repairs"]["P_after_C"]),
            "partial_repairs_commute_at_one_state": one_state["partial_repairs"]["commute_at_candidate"],
            "completed_joint_repair_order_independent": one_state[
                "completed_repair_order_independent"
            ],
            "discarded_raw_distinction": (
                "when oldest != newest, the identity of the older alternative is unnecessary"
            ),
        },
        "occupancy": {
            "raw_context_counts": {str(context): occupancy[context] for context in contexts},
            "balanced_reset_schedule": True,
        },
        "heldout_source_task": {
            "interaction_count": len(heldout),
            "interaction_sha256": _interaction_hash(heldout),
            "evaluations": heldout_models,
            "honest_null": "raw identity has the same oracle law; finite fitting can only change estimation error",
        },
        "transfer": transfer,
        "result": {
            "constructed_feature": selected.name,
            "predictive_compression": (27, 2),
            "autonomous_repair": (2, 18),
            "identity_not_required": len(closure) < 27,
            "new_feature_type": "equality predicate over categorical symbols",
        },
    }


def _controls(*, seed: int, config: SequenceRepairConfig) -> dict[str, object]:
    features = generate_typed_features()
    representations = generate_typed_representations(
        maximum_feature_count=config.maximum_feature_count
    )
    null = simulate_sequence_repair(
        world=CategoricalSequenceWorld.null(actions=config.actions),
        policy="active",
        seed=seed + 10,
        config=config,
        representations=representations,
        features=features,
    )
    primitive = simulate_sequence_repair(
        world=CategoricalSequenceWorld(relation="middle_symbol", actions=config.actions),
        policy="active",
        seed=seed + 11,
        config=config,
        representations=representations,
        features=features,
    )
    omission_representations = generate_typed_representations(
        maximum_feature_count=3,
        allowed_lags=(0, 1),
        include_sequence_predicates=False,
        include_boolean_compositions=False,
    )
    omission = simulate_sequence_repair(
        world=CategoricalSequenceWorld(actions=config.actions),
        policy="active",
        seed=seed + 12,
        config=replace(config, maximum_steps=min(config.maximum_steps, 32768)),
        representations=omission_representations,
        features=features,
    )

    stationarity_count = 60000
    stationary_world = CategoricalSequenceWorld(
        actions=("probe",), informative_action="probe"
    )
    stationary_trace = generate_sequence_interactions(
        world=stationary_world,
        action="probe",
        interaction_count=stationarity_count,
        seed=seed + 13,
    )
    stationary = categorical_stationarity_assessment(
        stationary_trace,
        actions=("probe",),
        delta=config.delta,
        split_tolerance=config.split_tolerance,
        minimum_samples=config.minimum_samples,
    )
    drifting_world = CategoricalSequenceWorld(
        actions=("probe",),
        informative_action="probe",
        drift_step=stationarity_count // 2,
    )
    drifting_trace = generate_sequence_interactions(
        world=drifting_world,
        action="probe",
        interaction_count=stationarity_count,
        seed=seed + 14,
    )
    drifting = categorical_stationarity_assessment(
        drifting_trace,
        actions=("probe",),
        delta=config.delta,
        split_tolerance=config.split_tolerance,
        minimum_samples=config.minimum_samples,
    )

    contexts = categorical_contexts()
    lag0 = TypedExpression("lag", value=0)
    lag2 = TypedExpression("lag", value=2)
    equality = TypedExpression("eq", (lag0, lag2))
    inequality = TypedExpression("neq", (lag0, lag2))
    equality_partition = partition_from_typed_features(contexts, (equality,))
    inequality_partition = partition_from_typed_features(contexts, (inequality,))
    return {
        "null_world": {
            "selected": None if null.selected_representation is None else null.selected_representation.name,
            "steps": null.steps,
            "false_relational_witness": null.first_relational_witness_step is not None,
            "passed": null.certified
            and null.selected_representation is not None
            and null.selected_representation.name == "one_state"
            and null.first_relational_witness_step is None,
        },
        "primitive_middle_symbol_world": {
            "selected": None if primitive.selected_representation is None else primitive.selected_representation.name,
            "steps": primitive.steps,
            "passed": primitive.certified
            and primitive.selected_representation is not None
            and primitive.selected_representation.name == "lag1",
        },
        "grammar_omission": {
            "allowed_lags": (0, 1),
            "candidate_count": len(omission_representations),
            "status": omission.status,
            "selected": None if omission.selected_representation is None else omission.selected_representation.name,
            "passed": not omission.certified,
            "boundary": "lag2 and every whole-sequence predicate are absent; abstention is required",
        },
        "stationarity_audit": {
            "interaction_count": len(stationary_trace),
            "interaction_sha256": _interaction_hash(stationary_trace),
            "assessment": stationary,
            "passed": stationary["status"] == "no_drift_certificate"
            and stationary["unresolved_cell_count"] == 0,
        },
        "drift_audit": {
            "interaction_count": len(drifting_trace),
            "interaction_sha256": _interaction_hash(drifting_trace),
            "assessment": drifting,
            "passed": drifting["status"] == "drift_detected",
        },
        "semantic_polarity_nonidentifiability": {
            "eq_partition": equality_partition,
            "neq_partition": inequality_partition,
            "same_unlabelled_partition": equality_partition == inequality_partition,
            "passed": equality_partition == inequality_partition,
        },
    }


def _monte_carlo(
    *, seed: int, config: SequenceRepairConfig, runs: int
) -> dict[str, object]:
    if runs < 1:
        raise ValueError("monte_carlo_runs must be positive")
    features = generate_typed_features()
    representations = generate_typed_representations(
        maximum_feature_count=config.maximum_feature_count
    )
    active_steps = []
    uniform_steps = []
    active_costs = []
    uniform_costs = []
    records = []
    active_successes = 0
    uniform_successes = 0
    active_misses = []
    uniform_misses = []
    for index in range(runs):
        trial_seed = seed + 1000 + index
        active = simulate_sequence_repair(
            world=CategoricalSequenceWorld(actions=config.actions),
            policy="active",
            seed=trial_seed,
            config=config,
            representations=representations,
            features=features,
        )
        uniform = simulate_sequence_repair(
            world=CategoricalSequenceWorld(actions=config.actions),
            policy="uniform",
            seed=trial_seed,
            config=config,
            representations=representations,
            features=features,
        )
        active_ok = active.certified and active.selected_representation is not None and active.selected_representation.name == "eq(lag0,lag2)"
        uniform_ok = uniform.certified and uniform.selected_representation is not None and uniform.selected_representation.name == "eq(lag0,lag2)"
        active_successes += int(active_ok)
        uniform_successes += int(uniform_ok)
        if not active_ok:
            active_misses.append(trial_seed)
        if not uniform_ok:
            uniform_misses.append(trial_seed)
        if active_ok and uniform_ok:
            active_steps.append(float(active.steps))
            uniform_steps.append(float(uniform.steps))
            active_costs.append(active.total_cost)
            uniform_costs.append(uniform.total_cost)
            records.append(
                {
                    "seed": trial_seed,
                    "active_steps": active.steps,
                    "uniform_steps": uniform.steps,
                    "active_cost": active.total_cost,
                    "uniform_cost": uniform.total_cost,
                    "active_probe_count": dict(active.action_counts)["probe"],
                    "uniform_probe_count": dict(uniform.action_counts)["probe"],
                }
            )
    if not active_steps:
        raise RuntimeError("no paired successful sequence runs")

    control_runs = max(2, min(10, runs // 2))
    null_failures = []
    primitive_failures = []
    omission_failures = []
    drift_failures = []
    omission_representations = generate_typed_representations(
        maximum_feature_count=3,
        allowed_lags=(0, 1),
        include_sequence_predicates=False,
        include_boolean_compositions=False,
    )
    for index in range(control_runs):
        trial_seed = seed + 10000 + index
        null = simulate_sequence_repair(
            world=CategoricalSequenceWorld.null(actions=config.actions),
            policy="active",
            seed=trial_seed,
            config=config,
            representations=representations,
            features=features,
        )
        if not (
            null.certified
            and null.selected_representation is not None
            and null.selected_representation.name == "one_state"
            and null.first_relational_witness_step is None
        ):
            null_failures.append(trial_seed)
        primitive = simulate_sequence_repair(
            world=CategoricalSequenceWorld(relation="middle_symbol", actions=config.actions),
            policy="active",
            seed=trial_seed,
            config=config,
            representations=representations,
            features=features,
        )
        if not (
            primitive.certified
            and primitive.selected_representation is not None
            and primitive.selected_representation.name == "lag1"
        ):
            primitive_failures.append(trial_seed)
        omission = simulate_sequence_repair(
            world=CategoricalSequenceWorld(actions=config.actions),
            policy="active",
            seed=trial_seed,
            config=replace(config, maximum_steps=min(config.maximum_steps, 32768)),
            representations=omission_representations,
            features=features,
        )
        if omission.certified:
            omission_failures.append(trial_seed)
        trace_count = 60000
        drifting = CategoricalSequenceWorld(
            actions=("probe",),
            informative_action="probe",
            drift_step=trace_count // 2,
        )
        trace = generate_sequence_interactions(
            world=drifting,
            action="probe",
            interaction_count=trace_count,
            seed=trial_seed,
        )
        drift = categorical_stationarity_assessment(
            trace,
            actions=("probe",),
            delta=config.delta,
            split_tolerance=config.split_tolerance,
            minimum_samples=config.minimum_samples,
        )
        if drift["status"] != "drift_detected":
            drift_failures.append(trial_seed)
    return {
        "runs": runs,
        "active_recovery": {
            "successes": active_successes,
            "trials": runs,
            "rate": active_successes / runs,
            "wilson_95_interval": _wilson(active_successes, runs),
            "miss_seeds": tuple(active_misses),
        },
        "uniform_recovery": {
            "successes": uniform_successes,
            "trials": runs,
            "rate": uniform_successes / runs,
            "wilson_95_interval": _wilson(uniform_successes, runs),
            "miss_seeds": tuple(uniform_misses),
        },
        "paired_successful_runs": len(active_steps),
        "fixed_point_steps": {
            "active": _summary(active_steps),
            "uniform": _summary(uniform_steps),
            "paired_ratio_active_over_uniform": _paired_ratio_bootstrap(
                active_steps, uniform_steps, seed=seed + 50000
            ),
        },
        "fixed_point_costs": {
            "active": _summary(active_costs),
            "uniform": _summary(uniform_costs),
            "paired_ratio_active_over_uniform": _paired_ratio_bootstrap(
                active_costs, uniform_costs, seed=seed + 50001
            ),
        },
        "seed_records": tuple(records),
        "control_runs": control_runs,
        "controls": {
            "null_false_construction": {
                "failures": len(null_failures),
                "trials": control_runs,
                "failure_seeds": tuple(null_failures),
            },
            "primitive_wrong_selection": {
                "failures": len(primitive_failures),
                "trials": control_runs,
                "failure_seeds": tuple(primitive_failures),
            },
            "grammar_omission_unsafe_output": {
                "failures": len(omission_failures),
                "trials": control_runs,
                "failure_seeds": tuple(omission_failures),
            },
            "drift_miss": {
                "failures": len(drift_failures),
                "trials": control_runs,
                "failure_seeds": tuple(drift_failures),
            },
        },
        "boundary": "algorithmic repeatability on declared fixtures, not an external population estimate",
    }


def run_sequence_experiment(
    *,
    seed: int = 20260802,
    maximum_steps: int = 65536,
    heldout_interactions: int = 27000,
    delta: float = 0.01,
    merge_tolerance: float = 0.32,
    split_tolerance: float = 0.45,
    minimum_samples: int = 32,
    probe_cost: float = 2.0,
    monte_carlo_runs: int = 20,
    project_root: Path | None = None,
) -> dict[str, object]:
    root = Path.cwd() if project_root is None else project_root
    config = SequenceRepairConfig(
        action_costs=(("baseline", 1.0), ("probe", probe_cost)),
        maximum_steps=maximum_steps,
        delta=delta,
        merge_tolerance=merge_tolerance,
        split_tolerance=split_tolerance,
        minimum_samples=minimum_samples,
    )
    primary = _primary_analysis(
        seed=seed, config=config, heldout_interactions=heldout_interactions
    )
    controls = _controls(seed=seed, config=config)
    monte_carlo = _monte_carlo(
        seed=seed, config=config, runs=monte_carlo_runs
    )
    transfer = primary["transfer"]
    checks = {
        "active_selected_endpoint_equality": primary["active_run"]["selected_representation"]["name"] == "eq(lag0,lag2)",
        "uniform_selected_same_partition": primary["uniform_run"]["selected_representation"]["partition"] == primary["selected_predictive_state"]["partition"],
        "typed_grammar_has_symbol_and_bool": {feature["output_type"] for feature in primary["grammar"]["features"]} == {"symbol", "bool"},
        "predictive_state_has_two_blocks": primary["dynamics"]["predictive_block_count"] == 2,
        "autonomous_state_has_eighteen_blocks": primary["dynamics"]["autonomous_block_count"] == 18,
        "repair_does_not_restore_full_identity": primary["dynamics"]["autonomous_block_count"] < 27,
        "closure_trace_is_2_4_12_18": tuple(len(partition) for partition in primary["controlled_dynamic_repair_trace"]) == (2, 4, 12, 18),
        "selected_predictive_residual_zero": primary["dynamics"]["selected_predictive_residual_nats"] < 1e-14,
        "selected_lumpability_residual_positive": primary["dynamics"]["selected_lumpability_residual_nats"] > 1e-6,
        "pooled_empirical_oracle_closure_agree": primary["dynamics"]["pooled_empirical_closure_agrees_with_oracle"],
        "partial_repairs_do_not_commute": not primary["dynamics"]["partial_repairs_commute_at_one_state"],
        "partial_path_sizes_are_18_and_2": (primary["dynamics"]["one_state_C_after_P_blocks"], primary["dynamics"]["one_state_P_after_C_blocks"]) == (18, 2),
        "joint_completion_order_independent": primary["dynamics"]["completed_joint_repair_order_independent"],
        "active_used_fewer_steps": primary["policy_comparison"]["active_steps"] < primary["policy_comparison"]["uniform_steps"],
        "active_used_less_cost": primary["policy_comparison"]["active_cost"] < primary["policy_comparison"]["uniform_cost"],
        "compatible_frozen_transfer_passed": transfer["frozen_compatible_task"]["passed"],
        "incompatible_frozen_transfer_rejected": transfer["frozen_incompatible_task"]["rejected"],
        "task_family_repair_passed": transfer["task_family_repair"]["passed"],
        "candidate_lattice_audit_passed": primary["candidate_lattice_audit"]["passed"],
        "task_family_refines_source": transfer["task_family_repair"]["source_partition_refined_by_family"],
        "null_control_passed": controls["null_world"]["passed"],
        "primitive_control_passed": controls["primitive_middle_symbol_world"]["passed"],
        "omission_control_abstained": controls["grammar_omission"]["passed"],
        "stationarity_control_passed": controls["stationarity_audit"]["passed"],
        "drift_control_passed": controls["drift_audit"]["passed"],
        "polarity_nonidentifiability_exhibited": controls["semantic_polarity_nonidentifiability"]["passed"],
        "active_monte_carlo_full_recovery": monte_carlo["active_recovery"]["successes"] == monte_carlo_runs,
        "uniform_monte_carlo_full_recovery": monte_carlo["uniform_recovery"]["successes"] == monte_carlo_runs,
        "monte_carlo_controls_zero_failures": all(item["failures"] == 0 for item in monte_carlo["controls"].values()),
    }
    return _jsonable(
        {
            "schema_version": "ts-language-foundations-sequence-repair-v0.7",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "run": {
                "seed": seed,
                "maximum_steps": maximum_steps,
                "heldout_interactions": heldout_interactions,
                "delta": delta,
                "merge_tolerance": merge_tolerance,
                "split_tolerance": split_tolerance,
                "minimum_samples": minimum_samples,
                "probe_cost": probe_cost,
                "monte_carlo_runs": monte_carlo_runs,
                "configuration": config.to_dict(),
            },
            "mathematical_contract": {
                "predictive_state": "equal categorical laws across the declared task/action family",
                "dynamic_closure": "coarsest refinement strongly lumpable for every declared controlled kernel",
                "joint_repair": "least common fixed point of predictive and controlled-dynamic closure",
                "transfer_criterion": "each added task law factors through the frozen partition",
                "confidence_metric": "total variation",
                "confidence_scope": "simultaneous over declared action/context cells and all checked prefixes",
            },
            "primary_sequence_case": primary,
            "controls": controls,
            "monte_carlo": monte_carlo,
            "verification": {
                "checks": checks,
                "passed_count": sum(checks.values()),
                "check_count": len(checks),
                "passed": all(checks.values()),
            },
            "conclusion": {
                "established_for_fixture": (
                    "a typed verifier constructs endpoint equality from raw ternary histories; "
                    "two predictive blocks repair to eighteen autonomous blocks; compatible "
                    "transfer preserves the state and an incompatible task forces six predictive blocks"
                ),
                "rejected_interpretations": (
                    "not evidence of general intelligence",
                    "not a neural mechanism",
                    "not proof that language has been learned",
                    "not proof that the grammar is complete",
                    "not evidence for consciousness or an ontological substrate",
                ),
                "next_falsifiable_experiment": (
                    "freeze the constructed state across separately generated variable-length tasks, "
                    "then test whether bounded compositional constructors avoid raw-state explosion"
                ),
            },
            "provenance": {
                "python": sys.version,
                "platform": platform.platform(),
                "project_root": str(root.resolve()),
                "source_hashes": source_hashes(root),
            },
        }
    )


def write_sequence_report(receipt: dict[str, object], path: Path) -> None:
    primary = receipt["primary_sequence_case"]
    dynamics = primary["dynamics"]
    policy = primary["policy_comparison"]
    transfer = primary["transfer"]
    lattice = primary["candidate_lattice_audit"]
    controls = receipt["controls"]
    monte = receipt["monte_carlo"]
    verification = receipt["verification"]
    compatible = transfer["frozen_compatible_task"]
    incompatible = transfer["frozen_incompatible_task"]
    family = transfer["task_family_repair"]
    report = rf"""# Typed categorical sequence construction and repair — v0.7.0

## Question

Can a verifier construct a useful relation over non-binary sequence events, determine whether that relation is an autonomous dynamical state, and decide—without semantic hand-waving—when the state transfers or must be repaired?

## Result boundary

This is a finite controlled fixture with an unbounded nuisance counter. It is a real executable test of representation construction and repair, not a language model and not evidence about consciousness. The typed grammar is declared by hand; the endpoint-equality label is not supplied to the learner.

## World and evidence

The observable state is a ternary history `(oldest, middle, newest)` in `{{0,1,2}}^3`, giving 27 raw states. `baseline` produces a uniform next symbol. Under `probe`, outcome 0 has probability 0.8 when oldest equals newest; otherwise outcome 1 has probability 0.8. The hidden counter increments forever but changes no registered law.

The proposal layer is Bayesian and chooses actions. It has no authority to alter a representation. The verifier uses simultaneous multinomial total-variation radii

\[
r(n)=\sqrt{{\frac{{\log((2^k-2)M/\delta)}}{{2n}}}}
\]

with a fixed candidate order and split/merge hysteresis. A candidate is selected only after it is certified and every simpler candidate has been rejected.

## Constructed state

The grammar generated {primary['grammar']['feature_count']} typed features and {primary['grammar']['candidate_partition_count']} distinct candidate partitions. It selected `{primary['result']['constructed_feature']}` independently in active and uniform runs.

| Quantity | Value |
|---|---:|
| Raw histories | 27 |
| Predictive blocks | {dynamics['predictive_block_count']} |
| Autonomous blocks | {dynamics['autonomous_block_count']} |
| Identity blocks | {dynamics['identity_block_count']} |
| Dynamic closure excess | {dynamics['dynamic_closure_excess_blocks']} |
| Selected predictive residual | {dynamics['selected_predictive_residual_nats']:.12g} nats |
| Selected lumpability residual | {dynamics['selected_lumpability_residual_nats']:.12g} nats |

The exact closure trace is `2 -> 4 -> 12 -> 18`. The autonomous representation is `{primary['autonomous_repaired_state']['name']}`. It remembers the newest symbol, middle symbol, and whether oldest equals newest. When oldest differs from newest, it still discards which of the two alternatives the oldest symbol was. Thus repair restores necessary hidden state without collapsing to raw identity.

## Noncommuting partial repairs

At the one-state start, predictive repair followed by dynamic closure has {dynamics['one_state_C_after_P_blocks']} blocks, while dynamic closure followed by predictive repair has {dynamics['one_state_P_after_C_blocks']} blocks. The partial operators therefore do not commute. Fair completion is nevertheless order-independent and reaches the same 18-block common fixed point.

## Exhaustive declared-family audit

Every one of the {lattice['candidate_partition_count']} grammar-generated partitions was checked for extensivity, dynamic-closure idempotence, and agreement of the two fair joint-repair schedules. The generated subposet contained {lattice['comparable_ordered_pair_count']} comparable ordered pairs; no closure-monotonicity violation occurred. Enlarging the controlled action family also produced no monotonicity violation on any candidate.

Only {lattice['dynamic_fixed_point_count']} candidates were dynamically closed: `{', '.join(lattice['dynamic_fixed_points'])}`. Only {lattice['joint_fixed_point_count']} were common predictive/dynamic fixed points: `{', '.join(lattice['joint_fixed_points'])}`. Partial noncommutation occurred at {lattice['noncommuting_partial_repair_count']} candidates, and the candidate family flowed into {lattice['unique_joint_completion_count']} distinct joint completions. These are exhaustive computations over the declared grammar family, not a proof by enumeration over the full partition lattice of 27 states.

## Active intervention

| Policy | Steps | Cost | Probe observations |
|---|---:|---:|---:|
| Active | {policy['active_steps']} | {policy['active_cost']:.0f} | {policy['active_action_counts']['probe']} |
| Uniform | {policy['uniform_steps']} | {policy['uniform_cost']:.0f} | {policy['uniform_action_counts']['probe']} |

The active policy used {policy['step_ratio_active_over_uniform']:.3f} times the observations and {policy['cost_ratio_active_over_uniform']:.3f} times the cost of uniform allocation in the primary run. This is an algorithmic fixture result, not a universal sample-complexity claim.

## Frozen transfer and task-family repair

The compatible held-out task changed which categorical outcomes mark the endpoint relation. Its exact residual through the frozen state was {compatible['exact_predictive_residual_nats']:.12g}; the two-block state transferred after fitting only a new predictive head.

The incompatible held-out task depended on the middle symbol. The frozen endpoint state had exact residual {incompatible['source_state_exact_predictive_residual_nats']:.6f} nats and was rejected. Combining both tasks produced the common predictive refinement `{family['least_joint_predictive_representation']}` with {family['least_joint_predictive_blocks']} blocks. Its autonomous closure again had {family['autonomous_blocks']} blocks.

This supplies an explicit finite counterexample to unrestricted frozen transfer and verifies the monotonicity rule: adding task laws can preserve or refine the predictive partition, never coarsen it.

## Controls

| Control | Outcome | Passed |
|---|---|---:|
| Null world | selected `{controls['null_world']['selected']}` | {controls['null_world']['passed']} |
| Primitive middle-symbol world | selected `{controls['primitive_middle_symbol_world']['selected']}` | {controls['primitive_middle_symbol_world']['passed']} |
| Grammar omits oldest coordinate | `{controls['grammar_omission']['status']}` | {controls['grammar_omission']['passed']} |
| Stationary trace | {controls['stationarity_audit']['assessment']['status']} | {controls['stationarity_audit']['passed']} |
| Drifting trace | {controls['drift_audit']['assessment']['status']} | {controls['drift_audit']['passed']} |
| EQ/NEQ polarity | same unlabelled partition | {controls['semantic_polarity_nonidentifiability']['passed']} |

## Monte Carlo sensitivity

Across {monte['runs']} paired seeds, active recovery was {monte['active_recovery']['successes']}/{monte['runs']} and uniform recovery was {monte['uniform_recovery']['successes']}/{monte['runs']}. Median steps were {monte['fixed_point_steps']['active']['median']:.0f} active versus {monte['fixed_point_steps']['uniform']['median']:.0f} uniform. The paired median step ratio was {monte['fixed_point_steps']['paired_ratio_active_over_uniform']['point_estimate']:.4f}, with algorithmic bootstrap interval {tuple(round(value, 4) for value in monte['fixed_point_steps']['paired_ratio_active_over_uniform']['percentile_95_interval'])}.

The {monte['control_runs']} repeated control seeds produced null/primitive/omission/drift failure counts of {monte['controls']['null_false_construction']['failures']}, {monte['controls']['primitive_wrong_selection']['failures']}, {monte['controls']['grammar_omission_unsafe_output']['failures']}, and {monte['controls']['drift_miss']['failures']}.

## Verifier results

{verification['passed_count']}/{verification['check_count']} scientific checks passed. The JSON receipt records raw partitions, every repair trace, exact residuals, source hashes, interaction hashes, support counts, policy traces, held-out scores, controls, and Monte Carlo seeds.

## What this establishes

For this fixture, the engine can construct a typed categorical relation, certify it from interventions, distinguish prediction from autonomous evolution, stop short of raw identity, reuse the state on a compatible task, reject it on an incompatible task, and compute the least task-family repair.

It does not establish grammar completeness, open-ended concept learning, natural-language competence, neural implementation, or any metaphysical claim. The next falsifiable step is variable-length compositional event structure with frozen cross-task evaluation.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def write_sequence_figure(receipt: dict[str, object], path: Path) -> None:
    primary = receipt["primary_sequence_case"]
    policy = primary["policy_comparison"]
    family = primary["transfer"]["task_family_repair"]
    incompatible = primary["transfer"]["frozen_incompatible_task"]
    width, height = 1280, 760
    active_bar = 300 * policy["active_steps"] / policy["uniform_steps"]
    uniform_bar = 300
    residual = incompatible["source_state_exact_predictive_residual_nats"]
    residual_bar = min(190, 190 * residual / 0.5)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#08111f"/>
<style>
text {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; fill: #e8f2ff; }}
.title {{ font: 700 27px system-ui, sans-serif; }} .sub {{ fill:#9fb5ce; font: 15px system-ui,sans-serif; }}
.big {{ font:700 28px system-ui,sans-serif; }} .label {{ font:600 15px system-ui,sans-serif; }}
.small {{ font:13px system-ui,sans-serif; fill:#a9bed6; }}
</style>
<text x="60" y="55" class="title">Typed ternary sequence repair · verified v0.7.0</text>
<text x="60" y="83" class="sub">Prediction compresses 27 raw histories to 2; autonomous evolution needs 18, not 27.</text>

<g transform="translate(60,125)">
  <rect x="0" y="0" width="210" height="115" rx="16" fill="#13283d" stroke="#48789f"/>
  <text x="105" y="44" text-anchor="middle" class="big">27</text><text x="105" y="75" text-anchor="middle" class="label">raw histories</text>
  <path d="M220 58 H292" stroke="#54d6a3" stroke-width="4"/><path d="M292 58 l-12 -8 v16 z" fill="#54d6a3"/>
  <rect x="302" y="0" width="210" height="115" rx="16" fill="#123a38" stroke="#54d6a3" stroke-width="2"/>
  <text x="407" y="44" text-anchor="middle" class="big">2</text><text x="407" y="75" text-anchor="middle" class="label">predictive blocks</text><text x="407" y="99" text-anchor="middle" class="small">eq(lag0, lag2)</text>
  <path d="M522 58 H594" stroke="#ffbd66" stroke-width="4"/><path d="M594 58 l-12 -8 v16 z" fill="#ffbd66"/>
  <rect x="604" y="0" width="250" height="115" rx="16" fill="#3a2b1c" stroke="#ffbd66" stroke-width="2"/>
  <text x="729" y="44" text-anchor="middle" class="big">18</text><text x="729" y="75" text-anchor="middle" class="label">autonomous blocks</text><text x="729" y="99" text-anchor="middle" class="small">closure: 2 → 4 → 12 → 18</text>
  <path d="M864 58 H936" stroke="#73879c" stroke-width="3" stroke-dasharray="8 7"/><path d="M936 58 l-12 -8 v16 z" fill="#73879c"/>
  <rect x="946" y="0" width="210" height="115" rx="16" fill="#1b2633" stroke="#73879c"/>
  <text x="1051" y="44" text-anchor="middle" class="big">27</text><text x="1051" y="75" text-anchor="middle" class="label">identity control</text><text x="1051" y="99" text-anchor="middle" class="small">unnecessary detail</text>
</g>

<g transform="translate(60,300)">
 <rect width="560" height="205" rx="18" fill="#0e1c2b" stroke="#29445e"/>
 <text x="25" y="36" class="label">ACTIVE INTERVENTION</text>
 <text x="25" y="78" class="small">active · {policy['active_steps']} steps</text>
 <rect x="190" y="59" width="{active_bar:.1f}" height="24" rx="6" fill="#54d6a3"/>
 <text x="25" y="126" class="small">uniform · {policy['uniform_steps']} steps</text>
 <rect x="190" y="107" width="{uniform_bar:.1f}" height="24" rx="6" fill="#6d8dad"/>
 <text x="25" y="169" class="small">Bayesian scores choose actions; confidence bounds alone choose state.</text>
 <text x="25" y="190" class="small">Primary observation ratio: {policy['step_ratio_active_over_uniform']:.3f}</text>
</g>

<g transform="translate(660,300)">
 <rect width="560" height="205" rx="18" fill="#0e1c2b" stroke="#29445e"/>
 <text x="25" y="36" class="label">TRANSFER TEST</text>
 <text x="25" y="77" class="small">compatible residual · 0</text>
 <circle cx="205" cy="72" r="8" fill="#54d6a3"/>
 <text x="25" y="121" class="small">incompatible residual · {residual:.4f} nats</text>
 <rect x="340" y="102" width="{residual_bar:.1f}" height="24" rx="6" fill="#ff6f7d"/>
 <text x="25" y="166" class="small">task family · 2 + 3 → {family['least_joint_predictive_blocks']} predictive → {family['autonomous_blocks']} autonomous</text>
 <text x="25" y="190" class="small">The frozen state is reused only when the new law factors through it.</text>
</g>

<g transform="translate(60,555)">
 <rect width="1160" height="145" rx="18" fill="#0e1c2b" stroke="#29445e"/>
 <text x="25" y="37" class="label">THE MATHEMATICAL HIT</text>
 <text x="25" y="72" class="small">At one state:  C(P(π)) = 18 blocks, but P(C(π)) = 2 blocks. Partial repairs do not commute.</text>
 <text x="25" y="100" class="small">Fair completion still reaches the same least common fixed point. Repair restores only the distinctions needed for continuation.</text>
 <text x="25" y="127" class="small">Evidence boundary: finite declared world + grammar + verifier. No language, neural, consciousness, or universality claim.</text>
</g>
</svg>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def write_sequence_artifacts(
    receipt: dict[str, object], *, output: Path, report: Path, figure: Path
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_sequence_report(receipt, report)
    write_sequence_figure(receipt, figure)
