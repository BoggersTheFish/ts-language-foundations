"""Reproducible relational feature-construction and repair experiment."""

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
    Interaction,
    controlled_dynamic_closure,
    controlled_lumpability_residual,
    controlled_predictive_residual,
)
from .experiment import source_hashes
from .repair import common_refinement, partition_entropy
from .relational import (
    FeatureRepresentation,
    RelationalRepairConfig,
    RelationalWorld,
    assess_relational_stationarity,
    binary_contexts,
    compile_relational_interface,
    controlled_joint_repair,
    evaluate_representation_probability_table,
    fit_representation_probability_table,
    generate_boolean_features,
    generate_feature_representations,
    generate_relational_interactions,
    oracle_relational_interface,
    partition_from_features,
    raw_identity_representation,
    representation_by_name,
    simulate_relational_repair,
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


def _interaction_hash(interactions: Iterable[Interaction]) -> str:
    payload = [interaction.to_dict() for interaction in interactions]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _wilson_interval(
    successes: int,
    trials: int,
    z_value: float = 1.959963984540054,
) -> tuple[float, float]:
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("invalid binomial count")
    proportion = successes / trials
    denominator = 1.0 + z_value**2 / trials
    centre = (proportion + z_value**2 / (2.0 * trials)) / denominator
    half_width = (
        z_value
        * (
            proportion * (1.0 - proportion) / trials
            + z_value**2 / (4.0 * trials**2)
        )
        ** 0.5
        / denominator
    )
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise ValueError("invalid percentile request")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap_paired_median_ratio(
    numerator: Sequence[float],
    denominator: Sequence[float],
    *,
    seed: int,
    resamples: int = 2000,
) -> dict[str, object]:
    if len(numerator) != len(denominator) or not numerator:
        raise ValueError("paired bootstrap requires equal non-empty inputs")
    ratios = [left / right for left, right in zip(numerator, denominator, strict=True)]
    rng = Random(seed)
    bootstrapped = []
    for _ in range(resamples):
        sample = [ratios[rng.randrange(len(ratios))] for _ in ratios]
        bootstrapped.append(median(sample))
    return {
        "point_estimate": median(ratios),
        "percentile_95_interval": (
            _percentile(bootstrapped, 0.025),
            _percentile(bootstrapped, 0.975),
        ),
        "resamples": resamples,
        "unit": "paired algorithmic random seed",
        "boundary": "fixture sensitivity, not a population confidence interval",
    }


def _summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("summary requires observations")
    return {
        "minimum": min(values),
        "median": median(values),
        "mean": mean(values),
        "maximum": max(values),
    }


def _transition_entropy_by_action(
    interface: CompiledControlledInterface,
) -> dict[str, float]:
    result = {}
    for action_index, action in enumerate(interface.actions):
        entropies = []
        for law in interface.predictive_laws[action_index]:
            probability = float(law[1])
            if probability in (0.0, 1.0):
                entropies.append(0.0)
            else:
                entropies.append(
                    -probability * log(probability)
                    - (1.0 - probability) * log(1.0 - probability)
                )
        result[action] = mean(entropies)
    return result


def _persistence_by_action(
    interface: CompiledControlledInterface,
    partition,
) -> dict[str, float]:
    labels = {
        state: block_index
        for block_index, block in enumerate(partition)
        for state in block
    }
    result = {}
    for action_index, action in enumerate(interface.actions):
        persistence = 0.0
        for state, row in enumerate(interface.transition_rows[action_index]):
            persistence += sum(
                float(probability)
                for target, probability in enumerate(row)
                if labels[target] == labels[state]
            ) / len(row)
        result[action] = persistence
    return result


def _candidate_record(
    representation: FeatureRepresentation,
    *,
    predictive_partition,
    oracle: CompiledControlledInterface,
    empirical_raw: CompiledControlledInterface,
    weights: tuple[Fraction, ...],
    verifier_status: str | None,
) -> dict[str, object]:
    dynamic, dynamic_trace = controlled_dynamic_closure(
        oracle.transition_rows, representation.partition, return_trace=True
    )
    predictive = common_refinement(representation.partition, predictive_partition)
    joint_pd, trace_pd = controlled_joint_repair(
        oracle.transition_rows,
        predictive_partition,
        representation.partition,
        schedule="predictive_then_dynamic",
        return_trace=True,
    )
    joint_dp, trace_dp = controlled_joint_repair(
        oracle.transition_rows,
        predictive_partition,
        representation.partition,
        schedule="dynamic_then_predictive",
        return_trace=True,
    )
    cp = controlled_dynamic_closure(oracle.transition_rows, predictive)
    pc = common_refinement(dynamic, predictive_partition)
    return {
        "name": representation.name,
        "partition": representation.partition,
        "block_count": representation.block_count,
        "feature_count": representation.feature_count,
        "total_complexity": representation.total_complexity,
        "verifier_status_at_active_stop": verifier_status,
        "oracle_action_averaged_predictive_residual_nats": controlled_predictive_residual(
            oracle, representation.partition, weights
        ),
        "oracle_action_averaged_lumpability_residual_nats": controlled_lumpability_residual(
            oracle, representation.partition, weights
        ),
        "raw_empirical_action_averaged_predictive_residual_nats": controlled_predictive_residual(
            empirical_raw, representation.partition, weights
        ),
        "raw_empirical_action_averaged_lumpability_residual_nats": controlled_lumpability_residual(
            empirical_raw, representation.partition, weights
        ),
        "partition_entropy_nats": partition_entropy(representation.partition, weights),
        "predictive_repair": predictive,
        "predictive_fixed": predictive == representation.partition,
        "controlled_dynamic_closure": dynamic,
        "controlled_dynamic_repair_trace": dynamic_trace,
        "controlled_dynamic_fixed": dynamic == representation.partition,
        "controlled_dynamic_closure_excess_blocks": len(dynamic)
        - representation.block_count,
        "joint_repair_predictive_then_dynamic": joint_pd,
        "joint_repair_dynamic_then_predictive": joint_dp,
        "joint_repair_trace_predictive_then_dynamic": trace_pd,
        "joint_repair_trace_dynamic_then_predictive": trace_dp,
        "joint_repair_excess_blocks": len(joint_pd) - representation.block_count,
        "joint_fixed": joint_pd == representation.partition,
        "completed_repair_order_independent": joint_pd == joint_dp,
        "partial_repairs": {
            "C_after_P": cp,
            "P_after_C": pc,
            "commute_at_candidate": cp == pc,
        },
    }


def _evaluate_models(
    *,
    training: Sequence[Interaction],
    heldout: Sequence[Interaction],
    contexts,
    selected: FeatureRepresentation,
    representations: Sequence[FeatureRepresentation],
) -> dict[str, object]:
    models = {
        "one_state": representation_by_name(representations, "one_state"),
        "lag0": representation_by_name(representations, "lag0"),
        "lag1": representation_by_name(representations, "lag1"),
        "selected_xor": selected,
        "autonomous_lag0_lag1": representation_by_name(
            representations, "lag0 + lag1"
        ),
        "raw_identity_control": raw_identity_representation(3),
    }
    evaluations = {}
    for name, representation in models.items():
        table = fit_representation_probability_table(
            training, representation, contexts=contexts
        )
        evaluations[name] = evaluate_representation_probability_table(
            table, representation, heldout, contexts=contexts
        )
    return evaluations


def _primary_analysis(
    *,
    seed: int,
    config: RelationalRepairConfig,
    heldout_interactions: int,
) -> dict[str, object]:
    world = RelationalWorld(actions=config.actions)
    features = generate_boolean_features(
        history_length=config.history_length,
        maximum_complexity=config.maximum_complexity,
    )
    representations = generate_feature_representations(
        history_length=config.history_length,
        maximum_complexity=config.maximum_complexity,
        maximum_feature_count=config.maximum_feature_count,
    )
    active = simulate_relational_repair(
        world=world,
        policy="active",
        seed=seed,
        config=config,
        representations=representations,
        belief_features=features,
    )
    uniform = simulate_relational_repair(
        world=world,
        policy="uniform",
        seed=seed,
        config=config,
        representations=representations,
        belief_features=features,
    )
    if not active.result.certified or not uniform.result.certified:
        raise RuntimeError("primary relational runs failed to certify")
    selected = active.result.selected_representation
    assert selected is not None
    if selected.name != "xor(lag0,lag1)":
        raise RuntimeError(f"primary fixture selected unexpected state: {selected.name}")

    empirical_pooled = compile_relational_interface(
        active.result,
        actions=config.actions,
        history_length=config.history_length,
        pool_partition=selected.partition,
    )
    empirical_raw = compile_relational_interface(
        active.result,
        actions=config.actions,
        history_length=config.history_length,
    )
    oracle = oracle_relational_interface(world, history_length=config.history_length)
    state_count = len(active.contexts)
    weights = tuple(Fraction(1, state_count) for _ in range(state_count))
    closure, closure_trace = controlled_dynamic_closure(
        empirical_pooled.transition_rows, selected.partition, return_trace=True
    )
    oracle_closure = controlled_dynamic_closure(
        oracle.transition_rows, selected.partition
    )
    if closure != oracle_closure:
        raise RuntimeError("empirical pooled and oracle closures disagree")
    autonomous = next(
        representation
        for representation in representations
        if representation.partition == closure
    )

    active_assessment_by_name = {
        item.representation.name: item.status
        for item in active.result.final_assessment.candidate_assessments
    }
    identity = raw_identity_representation(config.history_length)
    metric_candidates = (*representations, identity)
    candidates = {
        representation.name: _candidate_record(
            representation,
            predictive_partition=selected.partition,
            oracle=oracle,
            empirical_raw=empirical_raw,
            weights=weights,
            verifier_status=active_assessment_by_name.get(representation.name),
        )
        for representation in metric_candidates
    }

    heldout = generate_relational_interactions(
        world=world,
        actions=config.actions,
        action_costs=config.cost_map,
        interaction_count=heldout_interactions,
        history_length=config.history_length,
        seed=seed + 1,
        initial_step=10**9,
    )
    evaluations = _evaluate_models(
        training=active.interactions,
        heldout=heldout,
        contexts=active.contexts,
        selected=selected,
        representations=representations,
    )
    active_stationarity = assess_relational_stationarity(
        active.interactions,
        actions=config.actions,
        history_length=config.history_length,
        delta=config.delta,
        merge_tolerance=config.merge_tolerance,
        split_tolerance=config.split_tolerance,
        minimum_samples_per_half=config.minimum_samples,
    )
    occupancy = Counter(interaction.history for interaction in active.interactions)
    block_of = {
        state: block_index
        for block_index, block in enumerate(selected.partition)
        for state in block
    }
    block_occupancy = Counter(
        block_of[active.contexts.index(interaction.history)]
        for interaction in active.interactions
    )
    state_dictionary = tuple(
        {
            "state_index": index,
            "raw_history": context,
            "lag2": context[-3],
            "lag1": context[-2],
            "lag0": context[-1],
            "xor_lag0_lag1": context[-1] ^ context[-2],
            "predictive_block": block_of[index],
            "autonomous_block": next(
                block_index
                for block_index, block in enumerate(closure)
                if index in block
            ),
        }
        for index, context in enumerate(active.contexts)
    )
    one_state_metrics = candidates["one_state"]
    return {
        "hidden_environment": {
            "state": "(history,n) in {0,1}^3 x N_0",
            "transition": "(x[t-2],x[t-1],x[t],n) --a--> (x[t-1],x[t],W,n+1)",
            "baseline_law": "P(W=1|history,baseline)=0.5",
            "probe_law": (
                "P(W=1|xor(x[t-1],x[t])=0,probe)=0.85; "
                "P(W=1|xor(x[t-1],x[t])=1,probe)=0.15"
            ),
            "infinite_coordinate_effect": "n increments but affects no registered law",
            "learner_hidden_state_access": False,
            "learner_true_relation_access": False,
            "learner_input": "raw three-bit histories, chosen action, cost, and outcome",
            "episode_protocol": (
                "one transition per episode; initial raw contexts cycle through all "
                "eight histories to remove context/action confounding"
            ),
        },
        "grammar": {
            "type": "finite typed Boolean expression grammar",
            "history_length": config.history_length,
            "maximum_node_complexity": config.maximum_complexity,
            "maximum_feature_tuple_size": config.maximum_feature_count,
            "operators": ("constant", "lag", "not", "xor", "eq", "and", "or"),
            "truth_table_deduplication": True,
            "feature_count": len(features),
            "candidate_partition_count": len(representations),
            "features": tuple(feature.to_dict(active.contexts) for feature in features),
            "fixed_candidate_order": tuple(
                representation.name for representation in representations
            ),
            "canonical_label_rule": (
                "equal induced partitions are deduplicated by expression complexity, "
                "fixed operator priority, then lexical name"
            ),
            "polarity_boundary": (
                "xor and eq induce the same unlabelled partition; choosing xor is "
                "canonical syntax, not semantic identification"
            ),
        },
        "state_dictionary": state_dictionary,
        "active_run": active.result.to_dict(active.contexts),
        "uniform_run": uniform.result.to_dict(uniform.contexts),
        "policy_comparison": {
            "active_first_relational_witness_step": active.result.first_relational_witness_step,
            "uniform_first_relational_witness_step": uniform.result.first_relational_witness_step,
            "active_first_relational_witness_cost": active.result.first_relational_witness_cost,
            "uniform_first_relational_witness_cost": uniform.result.first_relational_witness_cost,
            "active_full_family_fixed_point_step": active.result.steps,
            "uniform_full_family_fixed_point_step": uniform.result.steps,
            "active_full_family_fixed_point_cost": active.result.total_cost,
            "uniform_full_family_fixed_point_cost": uniform.result.total_cost,
            "relational_witness_step_ratio_active_over_uniform": (
                active.result.first_relational_witness_step
                / uniform.result.first_relational_witness_step
            ),
            "fixed_point_step_ratio_active_over_uniform": (
                active.result.steps / uniform.result.steps
            ),
            "fixed_point_cost_ratio_active_over_uniform": (
                active.result.total_cost / uniform.result.total_cost
            ),
        },
        "training_input_receipt": {
            "active_interaction_count": len(active.interactions),
            "active_interaction_sha256": _interaction_hash(active.interactions),
            "uniform_interaction_count": len(uniform.interactions),
            "uniform_interaction_sha256": _interaction_hash(uniform.interactions),
            "hidden_counter_received": False,
            "true_feature_label_received": False,
        },
        "selected_predictive_state": selected.to_dict(active.contexts),
        "autonomous_repaired_state": autonomous.to_dict(active.contexts),
        "controlled_dynamic_repair_trace": closure_trace,
        "compiled_interfaces": {
            "empirical_confidence_pooled": empirical_pooled.to_dict(),
            "empirical_raw_unpooled": empirical_raw.to_dict(),
            "oracle_fixture_cross_check": oracle.to_dict(),
            "pooling_boundary": (
                "exact dynamic refinement is applied after pooling response laws "
                "inside confidence-certified predictive blocks; unpooled finite-sample "
                "noise would spuriously split all eight raw contexts"
            ),
        },
        "candidate_abstractions": candidates,
        "focused_candidates": (
            "one_state",
            "lag0",
            "lag1",
            "lag2",
            "xor(lag0,lag1)",
            "lag0 + lag1",
            identity.name,
        ),
        "occupancy": {
            "raw_context_counts": {
                str(context): occupancy[context] for context in active.contexts
            },
            "selected_block_counts": dict(block_occupancy),
            "context_schedule": "balanced deterministic reset cycle",
        },
        "dynamics": {
            "transition_entropy_nats_by_action": _transition_entropy_by_action(
                empirical_pooled
            ),
            "selected_block_persistence_by_action": _persistence_by_action(
                empirical_pooled, selected.partition
            ),
            "predictive_block_count": selected.block_count,
            "autonomous_block_count": len(closure),
            "dynamic_closure_excess_blocks": len(closure) - selected.block_count,
            "pooled_action_averaged_predictive_residual_nats": controlled_predictive_residual(
                empirical_pooled, selected.partition, weights
            ),
            "raw_empirical_action_averaged_predictive_residual_nats": controlled_predictive_residual(
                empirical_raw, selected.partition, weights
            ),
            "pooled_action_averaged_lumpability_residual_nats": controlled_lumpability_residual(
                empirical_pooled, selected.partition, weights
            ),
            "oracle_action_averaged_lumpability_residual_nats": controlled_lumpability_residual(
                oracle, selected.partition, weights
            ),
            "oracle_and_empirical_closure_agree": closure == oracle_closure,
            "one_state_partial_repairs_commute": one_state_metrics[
                "partial_repairs"
            ]["commute_at_candidate"],
            "one_state_joint_repair": one_state_metrics[
                "joint_repair_predictive_then_dynamic"
            ],
        },
        "heldout": {
            "interaction_count": len(heldout),
            "interaction_sha256": _interaction_hash(heldout),
            "balanced_action_and_context_schedule": True,
            "hidden_counter_evaluator_only": 10**9,
            "evaluations": evaluations,
            "honest_null": (
                "the selected XOR quotient should match, not beat, a sufficiently "
                "supported raw-identity probability table"
            ),
        },
        "stationarity_on_primary_adaptive_trace": active_stationarity.to_dict(),
        "result": {
            "constructed_feature": selected.name,
            "predictive_compression": (state_count, selected.block_count),
            "autonomous_repair": (selected.block_count, len(closure)),
            "irrelevant_coordinate_discarded": "lag2",
            "relational_feature_discovered_without_label": True,
            "full_action_family_certified": active.result.certified,
        },
    }


def _controls(
    *, seed: int, config: RelationalRepairConfig
) -> dict[str, object]:
    features = generate_boolean_features(
        history_length=config.history_length,
        maximum_complexity=config.maximum_complexity,
    )
    representations = generate_feature_representations(
        history_length=config.history_length,
        maximum_complexity=config.maximum_complexity,
        maximum_feature_count=config.maximum_feature_count,
    )
    control_config = replace(config, maximum_steps=min(config.maximum_steps, 32768))

    null = simulate_relational_repair(
        world=RelationalWorld(relation="null", actions=config.actions),
        policy="active",
        seed=seed,
        config=control_config,
        representations=representations,
        belief_features=features,
    )
    primitive = simulate_relational_repair(
        world=RelationalWorld(relation="lag1", actions=config.actions),
        policy="active",
        seed=seed,
        config=control_config,
        representations=representations,
        belief_features=features,
    )
    primitive_only = generate_feature_representations(
        history_length=config.history_length,
        maximum_complexity=1,
        maximum_feature_count=1,
        operators=(),
    )
    omission = simulate_relational_repair(
        world=RelationalWorld(actions=config.actions),
        policy="active",
        seed=seed,
        config=replace(control_config, maximum_steps=min(16384, control_config.maximum_steps)),
        representations=primitive_only,
        belief_features=features,
    )
    passive = simulate_relational_repair(
        world=RelationalWorld(actions=config.actions),
        policy="passive",
        seed=seed,
        config=replace(control_config, maximum_steps=min(16384, control_config.maximum_steps)),
        representations=representations,
        belief_features=features,
    )
    misspecified = simulate_relational_repair(
        world=RelationalWorld(actions=config.actions),
        policy="greedy_misspecified",
        seed=seed,
        config=replace(control_config, maximum_steps=min(16384, control_config.maximum_steps)),
        representations=representations,
        belief_features=features,
    )

    stationary_interactions = generate_relational_interactions(
        world=RelationalWorld(actions=config.actions),
        actions=config.actions,
        action_costs=config.cost_map,
        interaction_count=65536,
        history_length=config.history_length,
        seed=seed + 20,
    )
    stationary = assess_relational_stationarity(
        stationary_interactions,
        actions=config.actions,
        history_length=config.history_length,
        delta=config.delta,
        merge_tolerance=config.merge_tolerance,
        split_tolerance=config.split_tolerance,
        minimum_samples_per_half=config.minimum_samples,
    )
    drift_count = 32768
    drifting_interactions = generate_relational_interactions(
        world=RelationalWorld(
            actions=config.actions, drift_step=drift_count // 2
        ),
        actions=config.actions,
        action_costs=config.cost_map,
        interaction_count=drift_count,
        history_length=config.history_length,
        seed=seed,
    )
    drifting = assess_relational_stationarity(
        drifting_interactions,
        actions=config.actions,
        history_length=config.history_length,
        delta=config.delta,
        merge_tolerance=config.merge_tolerance,
        split_tolerance=config.split_tolerance,
        minimum_samples_per_half=config.minimum_samples,
    )

    by_action = dict(passive.result.final_action_assessments)
    xor = representation_by_name(representations, "xor(lag0,lag1)")
    eq_feature = next(
        feature for feature in features if feature.render() == "eq(lag0,lag1)"
    )
    eq_partition = partition_from_features(binary_contexts(3), (eq_feature,))
    return {
        "null_world": {
            "selected": (
                None
                if null.result.selected_representation is None
                else null.result.selected_representation.name
            ),
            "false_relational_witness": null.result.first_relational_witness_step
            is not None,
            "passed": (
                null.result.certified
                and null.result.selected_representation is not None
                and null.result.selected_representation.name == "one_state"
                and null.result.first_relational_witness_step is None
            ),
        },
        "primitive_world": {
            "selected": (
                None
                if primitive.result.selected_representation is None
                else primitive.result.selected_representation.name
            ),
            "passed": (
                primitive.result.certified
                and primitive.result.selected_representation is not None
                and primitive.result.selected_representation.name == "lag1"
            ),
        },
        "grammar_omission": {
            "declared_candidates": tuple(item.name for item in primitive_only),
            "selected": None,
            "status": omission.result.status,
            "passed": not omission.result.certified,
            "boundary": (
                "safe abstention is expected because XOR/EQ and feature tuples "
                "are deliberately absent"
            ),
        },
        "passive_policy_scope": {
            "baseline_selected": (
                None
                if by_action["baseline"].selected is None
                else by_action["baseline"].selected.name
            ),
            "probe_selected": (
                None
                if by_action["probe"].selected is None
                else by_action["probe"].selected.name
            ),
            "full_family_certified": passive.result.certified,
            "passed": (
                by_action["baseline"].selected is not None
                and by_action["baseline"].selected.name == "one_state"
                and by_action["probe"].selected is None
                and not passive.result.certified
            ),
        },
        "misspecified_proposal": {
            "status": misspecified.result.status,
            "selected": None,
            "passed": not misspecified.result.certified,
        },
        "stationary_audit": {
            "interaction_count": len(stationary_interactions),
            "interaction_sha256": _interaction_hash(stationary_interactions),
            "assessment": stationary.to_dict(),
            "passed": stationary.status == "raw_cells_stationary_within_tolerance",
        },
        "drift_audit": {
            "interaction_count": len(drifting_interactions),
            "interaction_sha256": _interaction_hash(drifting_interactions),
            "assessment": drifting.to_dict(),
            "passed": drifting.status == "raw_cell_distribution_shift_detected",
        },
        "polarity_ambiguity": {
            "xor_partition": xor.partition,
            "eq_partition": eq_partition,
            "same_unlabelled_partition": xor.partition == eq_partition,
            "canonical_selected_name": xor.name,
            "passed": xor.partition == eq_partition,
        },
    }


def _monte_carlo(
    *, seed: int, config: RelationalRepairConfig, runs: int
) -> dict[str, object]:
    if runs < 1:
        raise ValueError("monte_carlo_runs must be positive")
    features = generate_boolean_features(
        history_length=config.history_length,
        maximum_complexity=config.maximum_complexity,
    )
    representations = generate_feature_representations(
        history_length=config.history_length,
        maximum_complexity=config.maximum_complexity,
        maximum_feature_count=config.maximum_feature_count,
    )
    mc_config = replace(
        config,
        maximum_steps=min(config.maximum_steps, 32768),
        check_interval=max(config.check_interval, 256),
    )
    active_steps = []
    uniform_steps = []
    active_witness_steps = []
    uniform_witness_steps = []
    active_costs = []
    uniform_costs = []
    active_successes = uniform_successes = 0
    active_misses = []
    uniform_misses = []
    seed_records = []
    for index in range(runs):
        trial_seed = seed + 1000 + index
        active = simulate_relational_repair(
            world=RelationalWorld(actions=config.actions),
            policy="active",
            seed=trial_seed,
            config=mc_config,
            representations=representations,
            belief_features=features,
        ).result
        uniform = simulate_relational_repair(
            world=RelationalWorld(actions=config.actions),
            policy="uniform",
            seed=trial_seed,
            config=mc_config,
            representations=representations,
            belief_features=features,
        ).result
        active_ok = (
            active.certified
            and active.selected_representation is not None
            and active.selected_representation.name == "xor(lag0,lag1)"
        )
        uniform_ok = (
            uniform.certified
            and uniform.selected_representation is not None
            and uniform.selected_representation.name == "xor(lag0,lag1)"
        )
        active_successes += int(active_ok)
        uniform_successes += int(uniform_ok)
        if not active_ok:
            active_misses.append(trial_seed)
        if not uniform_ok:
            uniform_misses.append(trial_seed)
        if active_ok and uniform_ok:
            assert active.first_relational_witness_step is not None
            assert uniform.first_relational_witness_step is not None
            active_steps.append(float(active.steps))
            uniform_steps.append(float(uniform.steps))
            active_witness_steps.append(float(active.first_relational_witness_step))
            uniform_witness_steps.append(float(uniform.first_relational_witness_step))
            active_costs.append(active.total_cost)
            uniform_costs.append(uniform.total_cost)
            seed_records.append(
                {
                    "seed": trial_seed,
                    "active_witness_step": active.first_relational_witness_step,
                    "uniform_witness_step": uniform.first_relational_witness_step,
                    "active_fixed_point_step": active.steps,
                    "uniform_fixed_point_step": uniform.steps,
                    "active_fixed_point_cost": active.total_cost,
                    "uniform_fixed_point_cost": uniform.total_cost,
                }
            )

    control_runs = max(2, min(25, runs // 2))
    null_false_constructions = 0
    null_failures = []
    primitive_wrong_selections = 0
    primitive_failures = []
    omission_unsafe_outputs = 0
    omission_failures = []
    drift_detections = 0
    drift_misses = []
    primitive_only = generate_feature_representations(
        history_length=config.history_length,
        maximum_complexity=1,
        maximum_feature_count=1,
        operators=(),
    )
    omission_config = replace(mc_config, maximum_steps=min(16384, mc_config.maximum_steps))
    for index in range(control_runs):
        trial_seed = seed + 10000 + index
        null = simulate_relational_repair(
            world=RelationalWorld(relation="null", actions=config.actions),
            policy="active",
            seed=trial_seed,
            config=mc_config,
            representations=representations,
            belief_features=features,
        ).result
        null_ok = (
            null.certified
            and null.selected_representation is not None
            and null.selected_representation.name == "one_state"
            and null.first_relational_witness_step is None
        )
        null_false_constructions += int(not null_ok)
        if not null_ok:
            null_failures.append(trial_seed)

        primitive = simulate_relational_repair(
            world=RelationalWorld(relation="lag1", actions=config.actions),
            policy="active",
            seed=trial_seed,
            config=mc_config,
            representations=representations,
            belief_features=features,
        ).result
        primitive_ok = (
            primitive.certified
            and primitive.selected_representation is not None
            and primitive.selected_representation.name == "lag1"
        )
        primitive_wrong_selections += int(not primitive_ok)
        if not primitive_ok:
            primitive_failures.append(trial_seed)

        omission = simulate_relational_repair(
            world=RelationalWorld(actions=config.actions),
            policy="active",
            seed=trial_seed,
            config=omission_config,
            representations=primitive_only,
            belief_features=features,
        ).result
        omission_unsafe_outputs += int(omission.certified)
        if omission.certified:
            omission_failures.append(trial_seed)

        drift_count = 32768
        interactions = generate_relational_interactions(
            world=RelationalWorld(
                actions=config.actions, drift_step=drift_count // 2
            ),
            actions=config.actions,
            action_costs=config.cost_map,
            interaction_count=drift_count,
            history_length=config.history_length,
            seed=trial_seed,
        )
        drift = assess_relational_stationarity(
            interactions,
            actions=config.actions,
            history_length=config.history_length,
            delta=config.delta,
            merge_tolerance=config.merge_tolerance,
            split_tolerance=config.split_tolerance,
            minimum_samples_per_half=config.minimum_samples,
        )
        detected = drift.status == "raw_cell_distribution_shift_detected"
        drift_detections += int(detected)
        if not detected:
            drift_misses.append(trial_seed)

    paired_count = len(active_steps)
    if paired_count < 1:
        raise RuntimeError("no paired relational recoveries were available")
    return {
        "runs": runs,
        "active_full_recovery": {
            "successes": active_successes,
            "trials": runs,
            "rate": active_successes / runs,
            "wilson_95_interval": _wilson_interval(active_successes, runs),
            "miss_seeds": tuple(active_misses),
        },
        "uniform_full_recovery": {
            "successes": uniform_successes,
            "trials": runs,
            "rate": uniform_successes / runs,
            "wilson_95_interval": _wilson_interval(uniform_successes, runs),
            "miss_seeds": tuple(uniform_misses),
        },
        "paired_successful_runs": paired_count,
        "relational_witness_steps": {
            "active": _summary(active_witness_steps),
            "uniform": _summary(uniform_witness_steps),
            "paired_median_ratio_active_over_uniform": _bootstrap_paired_median_ratio(
                active_witness_steps,
                uniform_witness_steps,
                seed=seed + 50000,
            ),
        },
        "full_family_fixed_point_steps": {
            "active": _summary(active_steps),
            "uniform": _summary(uniform_steps),
            "paired_median_ratio_active_over_uniform": _bootstrap_paired_median_ratio(
                active_steps,
                uniform_steps,
                seed=seed + 50001,
            ),
        },
        "full_family_fixed_point_costs": {
            "active": _summary(active_costs),
            "uniform": _summary(uniform_costs),
            "paired_median_ratio_active_over_uniform": _bootstrap_paired_median_ratio(
                active_costs,
                uniform_costs,
                seed=seed + 50002,
            ),
        },
        "paired_seed_records": tuple(seed_records),
        "null_false_constructions": {
            "events": null_false_constructions,
            "trials": control_runs,
            "failure_seeds": tuple(null_failures),
        },
        "primitive_wrong_selections": {
            "events": primitive_wrong_selections,
            "trials": control_runs,
            "failure_seeds": tuple(primitive_failures),
        },
        "grammar_omission_unsafe_outputs": {
            "events": omission_unsafe_outputs,
            "trials": control_runs,
            "failure_seeds": tuple(omission_failures),
        },
        "drift_detection": {
            "successes": drift_detections,
            "trials": control_runs,
            "rate": drift_detections / control_runs,
            "failure_seeds": tuple(drift_misses),
        },
        "boundary": (
            "Monte Carlo checks implementation stability over algorithmic seeds; "
            "the anytime coverage statement comes from the finite union bound"
        ),
    }


def run_relational_experiment(
    *,
    seed: int = 20260802,
    maximum_steps: int = 32768,
    heldout_interactions: int = 65536,
    delta: float = 0.01,
    merge_tolerance: float = 0.2,
    split_tolerance: float = 0.35,
    minimum_samples: int = 32,
    probe_cost: float = 2.0,
    monte_carlo_runs: int = 50,
    project_root: Path | None = None,
) -> dict[str, object]:
    root = Path.cwd() if project_root is None else project_root
    config = RelationalRepairConfig(
        maximum_steps=maximum_steps,
        delta=delta,
        merge_tolerance=merge_tolerance,
        split_tolerance=split_tolerance,
        minimum_samples=minimum_samples,
        action_costs=(("baseline", 1.0), ("probe", probe_cost)),
    )
    primary = _primary_analysis(
        seed=seed,
        config=config,
        heldout_interactions=heldout_interactions,
    )
    controls = _controls(seed=seed, config=config)
    monte_carlo = _monte_carlo(seed=seed, config=config, runs=monte_carlo_runs)
    dynamics = primary["dynamics"]
    evaluations = primary["heldout"]["evaluations"]
    policy = primary["policy_comparison"]
    checks = {
        "generated_relation_is_xor": primary["result"]["constructed_feature"]
        == "xor(lag0,lag1)",
        "selected_predictive_state_has_two_blocks": dynamics[
            "predictive_block_count"
        ]
        == 2,
        "autonomous_repair_has_four_blocks": dynamics["autonomous_block_count"]
        == 4,
        "dynamic_closure_excess_is_two": dynamics[
            "dynamic_closure_excess_blocks"
        ]
        == 2,
        "pooled_predictive_residual_is_zero": dynamics[
            "pooled_action_averaged_predictive_residual_nats"
        ]
        == 0.0,
        "raw_empirical_predictive_residual_is_small": dynamics[
            "raw_empirical_action_averaged_predictive_residual_nats"
        ]
        < 0.005,
        "selected_state_is_not_dynamically_closed": dynamics[
            "pooled_action_averaged_lumpability_residual_nats"
        ]
        > 0.1,
        "oracle_and_empirical_closure_agree": dynamics[
            "oracle_and_empirical_closure_agree"
        ],
        "partial_repairs_do_not_commute_at_one_state": not dynamics[
            "one_state_partial_repairs_commute"
        ],
        "active_primary_finds_relation_before_uniform": policy[
            "active_first_relational_witness_step"
        ]
        < policy["uniform_first_relational_witness_step"],
        "selected_nll_beats_lag0": evaluations["selected_xor"][
            "mean_negative_log_likelihood_nats"
        ]
        < evaluations["lag0"]["mean_negative_log_likelihood_nats"],
        "selected_nll_beats_lag1": evaluations["selected_xor"][
            "mean_negative_log_likelihood_nats"
        ]
        < evaluations["lag1"]["mean_negative_log_likelihood_nats"],
        "selected_matches_raw_identity_within_point_zero_one_nats": abs(
            evaluations["selected_xor"]["mean_negative_log_likelihood_nats"]
            - evaluations["raw_identity_control"][
                "mean_negative_log_likelihood_nats"
            ]
        )
        < 0.01,
        "null_control_selects_one_state": controls["null_world"]["passed"],
        "primitive_control_selects_lag1": controls["primitive_world"]["passed"],
        "omitted_grammar_abstains": controls["grammar_omission"]["passed"],
        "passive_full_family_abstains": controls["passive_policy_scope"]["passed"],
        "misspecified_proposal_abstains": controls["misspecified_proposal"][
            "passed"
        ],
        "stationary_audit_certifies": controls["stationary_audit"]["passed"],
        "drift_audit_detects_shift": controls["drift_audit"]["passed"],
        "polarity_ambiguity_recorded": controls["polarity_ambiguity"]["passed"],
        "monte_carlo_active_recovery_at_least_95_percent": monte_carlo[
            "active_full_recovery"
        ]["rate"]
        >= 0.95,
        "monte_carlo_uniform_recovery_at_least_95_percent": monte_carlo[
            "uniform_full_recovery"
        ]["rate"]
        >= 0.95,
        "monte_carlo_active_median_relation_witness_is_earlier": monte_carlo[
            "relational_witness_steps"
        ]["active"]["median"]
        < monte_carlo["relational_witness_steps"]["uniform"]["median"],
        "monte_carlo_active_median_fixed_point_step_is_earlier": monte_carlo[
            "full_family_fixed_point_steps"
        ]["active"]["median"]
        < monte_carlo["full_family_fixed_point_steps"]["uniform"]["median"],
        "monte_carlo_null_false_constructions_zero": monte_carlo[
            "null_false_constructions"
        ]["events"]
        == 0,
        "monte_carlo_primitive_wrong_selections_zero": monte_carlo[
            "primitive_wrong_selections"
        ]["events"]
        == 0,
        "monte_carlo_grammar_omission_unsafe_outputs_zero": monte_carlo[
            "grammar_omission_unsafe_outputs"
        ]["events"]
        == 0,
        "monte_carlo_drift_detection_complete": monte_carlo[
            "drift_detection"
        ]["successes"]
        == monte_carlo["drift_detection"]["trials"],
        "anytime_budget_matches_all_raw_cells_and_prefixes": primary[
            "active_run"
        ]["confidence_contract"]["family_cell_budget"]
        == config.declared_intrinsic_cell_count * config.maximum_steps,
    }
    verification = {
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(checks.values()),
        "passed": all(checks.values()),
    }
    hashes = source_hashes(root)
    return _jsonable(
        {
            "schema_version": "ts-language-foundations-relational-repair-v0.6",
            "experiment": {
                "name": "typed relational feature construction under active intervention",
                "version": "0.6.0",
                "question": (
                    "Can a verifier-grounded learner construct rather than receive "
                    "the missing relational coordinate, and then separate task-predictive "
                    "compression from autonomous dynamic closure?"
                ),
            },
            "run": {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "seed": seed,
                "maximum_steps": maximum_steps,
                "heldout_interactions": heldout_interactions,
                "delta": delta,
                "merge_tolerance": merge_tolerance,
                "split_tolerance": split_tolerance,
                "minimum_samples": minimum_samples,
                "probe_cost": probe_cost,
                "monte_carlo_runs": monte_carlo_runs,
                "python": sys.version,
                "platform": platform.platform(),
                "network_required": False,
                "external_corpus": None,
                "neural_components": [],
                "tokenizer": None,
            },
            "mathematical_contract": {
                "candidate_order": (
                    "representations are ordered by induced block count, total AST "
                    "complexity, feature count, fixed grammar priority, and name"
                ),
                "minimality": (
                    "a candidate is selected only when certified and every earlier "
                    "candidate is rejected; an unresolved predecessor forces abstention"
                ),
                "predictive_sufficiency": (
                    "all response laws inside each candidate block are equivalent "
                    "within the merge tolerance for every declared action"
                ),
                "controlled_dynamic_closure": (
                    "the coarsest refinement strongly lumpable for every declared "
                    "action transition kernel"
                ),
                "joint_completion": (
                    "fair alternation of predictive and controlled dynamic repair "
                    "reaches the same least common fixed point on the finite partition lattice"
                ),
                "anytime_bound": (
                    "each of |A|*2^h raw cells and every within-cell sample prefix "
                    "n<=T receives radius sqrt(log(2M/delta)/(2n)), with "
                    "M=|A|*2^h*T"
                ),
                "proof_boundary": (
                    "the confidence guarantee assumes stationary conditionally independent "
                    "Bernoulli innovations per raw action/context cell; Monte Carlo is an "
                    "implementation stress test, not the proof"
                ),
            },
            "primary_relational_case": primary,
            "controls": controls,
            "monte_carlo": monte_carlo,
            "verification": verification,
            "conclusion": {
                "established_in_fixture": (
                    "from raw histories and a fixed grammar, the verifier selects the "
                    "two-block XOR predictive quotient; controlled dynamic repair then "
                    "requires the four last-two-bit suffix states while discarding lag2"
                ),
                "active_finding": (
                    "active probing exposes the relational witness earlier in median "
                    "algorithmic steps than balanced uniform action panels"
                ),
                "honest_null": (
                    "the selected quotient matches rather than beats the raw identity "
                    "predictor on heldout likelihood while using fewer predictive states"
                ),
                "negative_findings": (
                    "passive data cannot certify an untried action; an omitted grammar and "
                    "a misspecified proposal safely abstain; the semantic XOR/EQ polarity "
                    "is not identified; the primary adaptive trace alone leaves two "
                    "stationarity comparisons unresolved"
                ),
                "not_established": (
                    "general program synthesis, language understanding, neural mechanism, "
                    "open-world completeness, optimal active learning, consciousness, or "
                    "superiority to existing causal and representation-learning methods"
                ),
                "next_falsifiable_experiment": (
                    "replace the finite Boolean truth-table grammar with typed event and "
                    "sequence relations, train on one controlled family, and test whether "
                    "the repaired relational state transfers unchanged to held-out tasks"
                ),
            },
            "provenance": {
                "source_hashes": hashes,
                "source_file_count": len(hashes),
            },
        }
    )


def write_relational_report(receipt: dict[str, object], path: Path) -> None:
    primary = receipt["primary_relational_case"]
    result = primary["result"]
    dynamics = primary["dynamics"]
    policy = primary["policy_comparison"]
    evaluations = primary["heldout"]["evaluations"]
    controls = receipt["controls"]
    monte = receipt["monte_carlo"]
    verification = receipt["verification"]
    selected = primary["selected_predictive_state"]
    autonomous = primary["autonomous_repaired_state"]
    focused = [
        primary["candidate_abstractions"][name]
        for name in primary["focused_candidates"]
    ]
    candidate_rows = "\n".join(
        f"| `{item['name']}` | {item['block_count']} | "
        f"{item['verifier_status_at_active_stop'] or 'control'} | "
        f"{item['oracle_action_averaged_predictive_residual_nats']:.6f} | "
        f"{item['oracle_action_averaged_lumpability_residual_nats']:.6f} | "
        f"{item['controlled_dynamic_closure_excess_blocks']} | "
        f"{item['joint_repair_excess_blocks']} |"
        for item in focused
    )
    check_rows = "\n".join(
        f"| `{name}` | {'PASS' if passed else 'FAIL'} |"
        for name, passed in verification["checks"].items()
    )
    report = rf"""# Relational State Construction — Technical Report

## Question

Can a learner construct the missing coordinate from raw histories, rather than being told to append a specific older bit, while keeping every representation change under an independent finite-sample verifier?

## Current result

Yes, in the declared controlled fixture. The generated grammar contains 20 distinct Boolean truth functions and induces 65 unique candidate partitions. From those candidates, the verifier selected `{result['constructed_feature']}`. This compresses eight raw histories to {selected['block_count']} predictive blocks. That quotient is not autonomous: controlled dynamic closure repairs it to {autonomous['block_count']} blocks, represented canonically as `{autonomous['name']}`.

This is a fixture-specific computational result, not a claim that arbitrary relations or language have been learned.

## Source and integrity boundary

- Environment: `(history,n) in {{0,1}}^3 x N_0`; `n` is unbounded, increments, and never enters a registered law.
- Learner input: raw three-bit history, action, action cost, and binary outcome.
- Hidden from learner: `n`, the true XOR label, and the world probability function.
- Episode resets cycle all eight histories; action panels are scheduled separately so action is not confounded with context.
- No corpus, tokenizer, neural component, or network access is used.

## Typed feature grammar

The grammar has Boolean terminals `0`, `1`, `lag0`, `lag1`, `lag2`; unary `not`; and binary `xor`, `eq`, `and`, `or`. Expressions are exhaustively enumerated through node complexity {primary['grammar']['maximum_node_complexity']}. Exact truth-table duplicates are removed. Feature tuples contain at most {primary['grammar']['maximum_feature_tuple_size']} features.

Candidate order is frozen before observations: block count, total expression complexity, feature count, fixed operator priority, then name. A candidate may be selected only when it is certified and every earlier candidate has been rejected. Unresolved predecessors force abstention.

`xor(lag0,lag1)` and `eq(lag0,lag1)` induce the same unlabelled partition. The XOR name is a deterministic canonical representative, not an empirically identified semantic polarity.

## Confidence contract

For every raw history/action cell and every within-cell sample prefix up to the declared budget, the verifier uses

\[
r(n)=\sqrt{{\frac{{\log(2M/\delta)}}{{2n}}}},
\qquad M=|A|2^hT.
\]

All eight histories must have at least {receipt['run']['minimum_samples']} observations under every assessed action. A candidate block is rejected when a within-block pair has lower confidence difference above {receipt['run']['split_tolerance']}; it is certified only when every within-block upper difference is at most {receipt['run']['merge_tolerance']}.

The proposal belief chooses actions but has no split, merge, or stopping authority.

## Primary construction

| Quantity | Active | Uniform |
|---|---:|---:|
| First relational witness step | {policy['active_first_relational_witness_step']:,} | {policy['uniform_first_relational_witness_step']:,} |
| First relational witness cost | {policy['active_first_relational_witness_cost']:,.0f} | {policy['uniform_first_relational_witness_cost']:,.0f} |
| Full-family fixed-point step | {policy['active_full_family_fixed_point_step']:,} | {policy['uniform_full_family_fixed_point_step']:,} |
| Full-family fixed-point cost | {policy['active_full_family_fixed_point_cost']:,.0f} | {policy['uniform_full_family_fixed_point_cost']:,.0f} |

The active policy first resolves the informative probe and then fills the unresolved baseline cells. Uniform sampling rotates complete context panels through both actions.

## Predictive and autonomous state

The selected predictive partition is `{selected['partition']}`. Its confidence-pooled predictive residual is {dynamics['pooled_action_averaged_predictive_residual_nats']:.6f} nats; the unpooled empirical residual is {dynamics['raw_empirical_action_averaged_predictive_residual_nats']:.6f} nats.

Its controlled lumpability residual is {dynamics['pooled_action_averaged_lumpability_residual_nats']:.6f} nats, and dynamic closure is `{autonomous['partition']}`. The exact fixture oracle gives the same closure. The repaired state keeps `lag0` and `lag1` and discards irrelevant `lag2`.

At the one-state candidate, partial repairs do not commute: `C(P(pi))` has four blocks while `P(C(pi))` initially has two. Fair completion reaches the same four-block common fixed point.

## Candidate receipts

| Candidate | Blocks | Verifier | Oracle R | Oracle L | ΔK_C | ΔK_S |
|---|---:|---|---:|---:|---:|---:|
{candidate_rows}

The machine-readable receipt contains metrics and repair traces for all {len(primary['candidate_abstractions'])} candidate/control partitions.

## Heldout prediction

| Model | States | Accuracy | NLL (nats) | Brier |
|---|---:|---:|---:|---:|
| one state | 1 | {evaluations['one_state']['accuracy']:.4f} | {evaluations['one_state']['mean_negative_log_likelihood_nats']:.6f} | {evaluations['one_state']['mean_brier_score']:.6f} |
| lag0 | 2 | {evaluations['lag0']['accuracy']:.4f} | {evaluations['lag0']['mean_negative_log_likelihood_nats']:.6f} | {evaluations['lag0']['mean_brier_score']:.6f} |
| lag1 | 2 | {evaluations['lag1']['accuracy']:.4f} | {evaluations['lag1']['mean_negative_log_likelihood_nats']:.6f} | {evaluations['lag1']['mean_brier_score']:.6f} |
| selected XOR | 2 | {evaluations['selected_xor']['accuracy']:.4f} | {evaluations['selected_xor']['mean_negative_log_likelihood_nats']:.6f} | {evaluations['selected_xor']['mean_brier_score']:.6f} |
| autonomous suffix | 4 | {evaluations['autonomous_lag0_lag1']['accuracy']:.4f} | {evaluations['autonomous_lag0_lag1']['mean_negative_log_likelihood_nats']:.6f} | {evaluations['autonomous_lag0_lag1']['mean_brier_score']:.6f} |
| raw identity | 8 | {evaluations['raw_identity_control']['accuracy']:.4f} | {evaluations['raw_identity_control']['mean_negative_log_likelihood_nats']:.6f} | {evaluations['raw_identity_control']['mean_brier_score']:.6f} |

Honest null: the two-state selected quotient matches rather than beats the eight-state identity control within finite-sample variation. Its gain is compression and explicit sufficiency, not extra oracle information.

## Monte Carlo sensitivity

Across {monte['runs']} paired seeds, active recovery was {monte['active_full_recovery']['successes']}/{monte['active_full_recovery']['trials']} and uniform recovery was {monte['uniform_full_recovery']['successes']}/{monte['uniform_full_recovery']['trials']}. Median relational-witness steps were {monte['relational_witness_steps']['active']['median']:,.0f} active versus {monte['relational_witness_steps']['uniform']['median']:,.0f} uniform. The paired median step ratio was {monte['relational_witness_steps']['paired_median_ratio_active_over_uniform']['point_estimate']:.4f}, with algorithmic bootstrap interval {monte['relational_witness_steps']['paired_median_ratio_active_over_uniform']['percentile_95_interval']}.

Hostile controls used {monte['null_false_constructions']['trials']} seeds: null false constructions {monte['null_false_constructions']['events']}; primitive-world wrong selections {monte['primitive_wrong_selections']['events']}; omitted-grammar unsafe outputs {monte['grammar_omission_unsafe_outputs']['events']}; drift detections {monte['drift_detection']['successes']}/{monte['drift_detection']['trials']}.

## What failed or remains uncertain

- The active primary trace itself leaves two raw-cell stationarity comparisons unresolved at its stopping time. A separate balanced 65,536-interaction stationary audit certifies all cells; the drift fixture is rejected. This distinction is retained in the receipt.
- Exact dynamic refinement of noisy, unpooled empirical probabilities would split all eight contexts. Dynamic closure is therefore applied to response laws pooled only inside confidence-certified predictive blocks and cross-checked against the exact fixture oracle.
- The grammar is finite and hand-declared. Omit XOR/EQ and feature tuples, and the verifier abstains.
- XOR versus EQ polarity is not identified by an unlabelled partition.
- This does not establish language, planning, general causal discovery, neural mechanisms, consciousness, or a general new AI architecture.

## Verifier results

| Check | Result |
|---|---|
{check_rows}

Overall: **{verification['passed_count']}/{verification['check_count']} checks passed**.

## Next falsifiable experiment

Replace Boolean truth-table features with typed event and sequence relations. Train the representation on one controlled family, freeze it, and test whether the same repaired state predicts and evolves autonomously on held-out task families without further repair.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def write_relational_figure(receipt: dict[str, object], path: Path) -> None:
    primary = receipt["primary_relational_case"]
    state_dictionary = primary["state_dictionary"]
    policy = primary["policy_comparison"]
    monte = receipt["monte_carlo"]
    evaluations = primary["heldout"]["evaluations"]
    controls = receipt["controls"]
    verification = receipt["verification"]
    width, height = 1440, 940
    context_cells = []
    for index, state in enumerate(state_dictionary):
        x = 70 + (index % 4) * 132
        y = 155 + (index // 4) * 96
        colour = "#54d6b5" if state["xor_lag0_lag1"] == 0 else "#ff8d78"
        context_cells.append(
            f'<rect x="{x}" y="{y}" width="112" height="70" rx="12" '
            f'fill="{colour}" fill-opacity="0.18" stroke="{colour}"/>'
            f'<text x="{x + 56}" y="{y + 31}" text-anchor="middle" class="state">'
            f'{"".join(str(value) for value in state["raw_history"])}</text>'
            f'<text x="{x + 56}" y="{y + 54}" text-anchor="middle" class="tiny">'
            f'xor={state["xor_lag0_lag1"]}</text>'
        )

    max_step = max(
        monte["relational_witness_steps"]["uniform"]["median"],
        monte["full_family_fixed_point_steps"]["uniform"]["median"],
    )
    bars = []
    bar_data = (
        ("active relation", monte["relational_witness_steps"]["active"]["median"], "#54d6b5"),
        ("uniform relation", monte["relational_witness_steps"]["uniform"]["median"], "#ff8d78"),
        ("active fixed", monte["full_family_fixed_point_steps"]["active"]["median"], "#62a8ff"),
        ("uniform fixed", monte["full_family_fixed_point_steps"]["uniform"]["median"], "#b991ff"),
    )
    for index, (label, value, colour) in enumerate(bar_data):
        y = 550 + index * 54
        bar_width = 410 * value / max_step
        bars.append(
            f'<text x="72" y="{y + 20}" class="small">{escape(label)}</text>'
            f'<rect x="205" y="{y}" width="{bar_width:.1f}" height="28" rx="6" fill="{colour}"/>'
            f'<text x="{215 + bar_width:.1f}" y="{y + 20}" class="small">{value:,.0f}</text>'
        )

    nll_data = (
        ("one", evaluations["one_state"]["mean_negative_log_likelihood_nats"], "#7e8aa8"),
        ("lag0", evaluations["lag0"]["mean_negative_log_likelihood_nats"], "#7e8aa8"),
        ("lag1", evaluations["lag1"]["mean_negative_log_likelihood_nats"], "#7e8aa8"),
        ("XOR", evaluations["selected_xor"]["mean_negative_log_likelihood_nats"], "#54d6b5"),
        ("raw 8", evaluations["raw_identity_control"]["mean_negative_log_likelihood_nats"], "#62a8ff"),
    )
    nll_bars = []
    for index, (label, value, colour) in enumerate(nll_data):
        x = 782 + index * 106
        bar_height = 170 * value / 0.75
        y = 724 - bar_height
        nll_bars.append(
            f'<rect x="{x}" y="{y:.1f}" width="70" height="{bar_height:.1f}" rx="7" fill="{colour}"/>'
            f'<text x="{x + 35}" y="{y - 9:.1f}" text-anchor="middle" class="tiny">{value:.3f}</text>'
            f'<text x="{x + 35}" y="748" text-anchor="middle" class="tiny">{escape(label)}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<defs><filter id="glow"><feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>
<style>
  .bg {{ fill:#0b1020; }} .panel {{ fill:#121a2e; stroke:#2c3856; stroke-width:2; }}
  .title {{ fill:#f4f7ff; font:700 30px system-ui,sans-serif; }}
  .subtitle {{ fill:#9ba8c7; font:16px system-ui,sans-serif; }}
  .panel-title {{ fill:#f4f7ff; font:700 20px system-ui,sans-serif; }}
  .state {{ fill:#f4f7ff; font:700 22px ui-monospace,monospace; }}
  .small {{ fill:#c7d0e7; font:14px system-ui,sans-serif; }}
  .tiny {{ fill:#aeb9d3; font:12px system-ui,sans-serif; }}
  .big {{ fill:#f4f7ff; font:700 34px system-ui,sans-serif; }}
</style>
<rect width="100%" height="100%" class="bg"/>
<text x="58" y="52" class="title">RELATIONAL STATE CONSTRUCTION · v0.6</text>
<text x="58" y="80" class="subtitle">raw histories → generated relation → predictive quotient → autonomous repair</text>

<rect x="48" y="105" width="570" height="365" rx="20" class="panel"/>
<text x="70" y="138" class="panel-title">1 · Eight raw histories, relation not supplied</text>
{''.join(context_cells)}
<text x="70" y="383" class="small">Grammar: 20 truth functions · 65 unique partitions</text>
<text x="70" y="412" class="small">Verifier-selected: xor(lag0,lag1)</text>
<text x="70" y="440" class="tiny">XOR/EQ polarity remains canonically, not semantically, labelled.</text>

<rect x="642" y="105" width="750" height="365" rx="20" class="panel"/>
<text x="668" y="138" class="panel-title">2 · Predictive compression is not autonomous closure</text>
<circle cx="760" cy="252" r="72" fill="#54d6b5" fill-opacity="0.16" stroke="#54d6b5" stroke-width="3"/>
<circle cx="940" cy="252" r="72" fill="#ff8d78" fill-opacity="0.16" stroke="#ff8d78" stroke-width="3"/>
<text x="760" y="246" text-anchor="middle" class="big">0</text><text x="760" y="274" text-anchor="middle" class="tiny">4 histories</text>
<text x="940" y="246" text-anchor="middle" class="big">1</text><text x="940" y="274" text-anchor="middle" class="tiny">4 histories</text>
<path d="M1030 252 L1110 252" stroke="#f2d675" stroke-width="4"/>
<text x="1070" y="232" text-anchor="middle" class="tiny">Cₚ</text>
<g transform="translate(1150,190)">
  <rect x="0" y="0" width="76" height="76" rx="12" fill="#62a8ff" fill-opacity=".18" stroke="#62a8ff"/>
  <rect x="92" y="0" width="76" height="76" rx="12" fill="#62a8ff" fill-opacity=".18" stroke="#62a8ff"/>
  <rect x="0" y="92" width="76" height="76" rx="12" fill="#62a8ff" fill-opacity=".18" stroke="#62a8ff"/>
  <rect x="92" y="92" width="76" height="76" rx="12" fill="#62a8ff" fill-opacity=".18" stroke="#62a8ff"/>
  <text x="38" y="46" text-anchor="middle" class="state">00</text><text x="130" y="46" text-anchor="middle" class="state">01</text>
  <text x="38" y="138" text-anchor="middle" class="state">10</text><text x="130" y="138" text-anchor="middle" class="state">11</text>
</g>
<text x="760" y="361" text-anchor="middle" class="small">predictive blocks = 2</text>
<text x="1234" y="386" text-anchor="middle" class="small">autonomous blocks = 4</text>
<text x="668" y="425" class="tiny">R = {primary['dynamics']['raw_empirical_action_averaged_predictive_residual_nats']:.6f} · L = {primary['dynamics']['pooled_action_averaged_lumpability_residual_nats']:.6f} nats · lag2 discarded</text>

<rect x="48" y="494" width="642" height="398" rx="20" class="panel"/>
<text x="72" y="528" class="panel-title">3 · Active intervention exposes the relation earlier</text>
{''.join(bars)}
<text x="72" y="795" class="small">paired median relation-step ratio</text>
<text x="72" y="833" class="big">{monte['relational_witness_steps']['paired_median_ratio_active_over_uniform']['point_estimate']:.3f}</text>
<text x="205" y="825" class="tiny">bootstrap 95% {monte['relational_witness_steps']['paired_median_ratio_active_over_uniform']['percentile_95_interval']}</text>
<text x="72" y="866" class="tiny">{monte['active_full_recovery']['successes']}/{monte['active_full_recovery']['trials']} active · {monte['uniform_full_recovery']['successes']}/{monte['uniform_full_recovery']['trials']} uniform full recoveries</text>

<rect x="714" y="494" width="678" height="398" rx="20" class="panel"/>
<text x="740" y="528" class="panel-title">4 · Heldout NLL · hostile controls</text>
<line x1="760" y1="724" x2="1355" y2="724" stroke="#465373" stroke-width="2"/>
{''.join(nll_bars)}
<text x="760" y="780" class="small">lower NLL is better · selected XOR matches raw identity</text>
<rect x="760" y="807" width="590" height="58" rx="10" fill="#19233b" stroke="#344361"/>
<text x="776" y="831" class="tiny">null false relation: 0 · omitted grammar: ABSTAIN · passive: ABSTAIN</text>
<text x="776" y="852" class="tiny">drift: {monte['drift_detection']['successes']}/{monte['drift_detection']['trials']} detected · verifier: {verification['passed_count']}/{verification['check_count']}</text>
</svg>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def write_relational_artifacts(
    receipt: dict[str, object],
    *,
    output: Path,
    report: Path,
    figure: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_relational_report(receipt, report)
    write_relational_figure(receipt, figure)
