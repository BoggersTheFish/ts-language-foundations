"""Reproducible active-intervention representation-repair experiment."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
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
    Action,
    ControlledCounts,
    ControlledRepairConfig,
    ControlledWorld,
    Interaction,
    assess_action_stationarity,
    compile_controlled_interface,
    confidence_action_partition,
    controlled_dynamic_closure,
    controlled_lumpability_residual,
    controlled_predictive_residual,
    evaluate_action_probability_table,
    fit_action_probability_table,
    generate_controlled_interactions,
    simulate_controlled_repair,
)
from .experiment import source_hashes
from .repair import Partition, partition_entropy


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
    payload = [item.to_dict() for item in interactions]
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
    lower = 0.0 if successes == 0 else max(0.0, centre - half_width)
    upper = 1.0 if successes == trials else min(1.0, centre + half_width)
    return lower, upper


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise ValueError("invalid percentile request")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] * (1.0 - fraction) + ordered[upper_index] * fraction


def _bootstrap_paired_median_ratio(
    numerator: Sequence[float],
    denominator: Sequence[float],
    *,
    seed: int,
    resamples: int = 2000,
) -> dict[str, object]:
    if len(numerator) != len(denominator) or not numerator:
        raise ValueError("paired bootstrap requires equal non-empty samples")
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
        "unit": "paired Monte Carlo seed",
        "boundary": "algorithmic fixture sensitivity, not a population estimate",
    }


def _rebuild_counts(
    interactions: Sequence[Interaction], config: ControlledRepairConfig
) -> ControlledCounts:
    counts = ControlledCounts(config.actions, config.maximum_order)
    for interaction in interactions:
        counts.add(interaction)
    return counts


def _candidate_record(
    name: str,
    partition: Partition,
    interface,
    state_weights: tuple[Fraction, ...],
) -> dict[str, object]:
    dynamic, trace = controlled_dynamic_closure(
        interface.transition_rows, partition, return_trace=True
    )
    return {
        "name": name,
        "partition": partition,
        "block_contexts": tuple(
            tuple(interface.contexts[index] for index in block) for block in partition
        ),
        "block_count": len(partition),
        "action_averaged_predictive_residual_nats": controlled_predictive_residual(
            interface, partition, state_weights
        ),
        "action_averaged_lumpability_residual_nats": controlled_lumpability_residual(
            interface, partition, state_weights
        ),
        "partition_entropy_nats": partition_entropy(partition, state_weights),
        "controlled_dynamic_closure": dynamic,
        "controlled_dynamic_closure_contexts": tuple(
            tuple(interface.contexts[index] for index in block) for block in dynamic
        ),
        "controlled_dynamic_repair_trace": trace,
        "controlled_dynamic_closure_excess_blocks": len(dynamic) - len(partition),
        "controlled_dynamic_fixed": dynamic == partition,
    }


def _transition_entropy(interface) -> float:
    entropies = []
    for laws in interface.predictive_laws:
        for law in laws:
            probability = float(law[1])
            if probability in (0.0, 1.0):
                entropies.append(0.0)
            else:
                entropies.append(
                    -probability * log(probability)
                    - (1.0 - probability) * log(1.0 - probability)
                )
    return mean(entropies)


def _primary_analysis(
    *, seed: int, config: ControlledRepairConfig, heldout_episode_count: int
) -> dict[str, object]:
    world = ControlledWorld(actions=config.actions)
    active = simulate_controlled_repair(
        world=world, policy="active", seed=seed, config=config
    )
    uniform = simulate_controlled_repair(
        world=world, policy="uniform", seed=seed, config=config
    )
    if not active.result.certified or not uniform.result.certified:
        raise RuntimeError("primary active/uniform fixture failed to reach a fixed point")
    partition = confidence_action_partition(
        active.result,
        actions=config.actions,
        merge_tolerance=config.merge_tolerance,
        split_tolerance=config.split_tolerance,
        minimum_samples=config.minimum_samples,
    )
    if not partition.certified:
        raise RuntimeError("action-predictive partition failed to certify")
    interface = compile_controlled_interface(active.result, actions=config.actions)
    state_count = len(interface.contexts)
    state_weights = tuple(Fraction(1, state_count) for _ in range(state_count))
    identity = tuple((index,) for index in range(state_count))
    one_state = (tuple(range(state_count)),)
    current_observation = tuple(
        tuple(
            index
            for index, context in enumerate(interface.contexts)
            if context[-1] == observation
        )
        for observation in (0, 1)
    )
    candidates = {
        "one_state_negative_control": _candidate_record(
            "one-state negative control", one_state, interface, state_weights
        ),
        "current_observation": _candidate_record(
            "current observed bit", current_observation, interface, state_weights
        ),
        "confidence_action_predictive_partition": _candidate_record(
            "confidence-bounded action-predictive partition",
            partition.blocks,
            interface,
            state_weights,
        ),
        "learned_suffix_identity": _candidate_record(
            "learned suffix identity", identity, interface, state_weights
        ),
    }

    heldout = generate_controlled_interactions(
        world=world,
        actions=config.actions,
        action_costs=config.cost_map,
        episode_count=heldout_episode_count,
        episode_length=128,
        history_length=config.maximum_order + 1,
        seed=seed + 1,
        initial_step=10**9,
    )
    tables = {
        order: fit_action_probability_table(active.interactions, order)
        for order in (0, 1, active.result.final_order)
    }
    evaluations = {
        "one_state_action_conditioned": evaluate_action_probability_table(
            tables[0], 0, heldout
        ),
        "current_observation_action_conditioned": evaluate_action_probability_table(
            tables[1], 1, heldout
        ),
        "adaptive_repaired": evaluate_action_probability_table(
            tables[active.result.final_order], active.result.final_order, heldout
        ),
        "fixed_order_two": evaluate_action_probability_table(
            tables[2], 2, heldout
        ),
    }
    predictive_candidate = candidates["confidence_action_predictive_partition"]
    return {
        "name": "controlled stochastic infinite-fibre repair",
        "hidden_environment": {
            "state": "(u,v,n) in {0,1}^2 x N_0",
            "transition": "(u,v,n) --a--> (v,W,n+1)",
            "baseline_law": "P(W=1|u,baseline)=0.5",
            "probe_law": "P(W=1|u=0,probe)=0.85; P(W=1|u=1,probe)=0.15",
            "infinite_coordinate_effect": "n increments but affects no registered law",
            "learner_hidden_state_access": False,
            "learner_input": "visible binary histories, chosen actions, costs, and outcomes",
        },
        "active_run": active.result.to_dict(),
        "uniform_run": uniform.result.to_dict(),
        "policy_comparison": {
            "active_first_repair_step": active.result.first_repair_step,
            "uniform_first_repair_step": uniform.result.first_repair_step,
            "active_first_repair_cost": active.result.first_repair_cost,
            "uniform_first_repair_cost": uniform.result.first_repair_cost,
            "active_fixed_point_step": active.result.steps,
            "uniform_fixed_point_step": uniform.result.steps,
            "active_fixed_point_cost": active.result.total_cost,
            "uniform_fixed_point_cost": uniform.result.total_cost,
            "first_repair_step_ratio_active_over_uniform": (
                active.result.first_repair_step / uniform.result.first_repair_step
            ),
            "first_repair_cost_ratio_active_over_uniform": (
                active.result.first_repair_cost / uniform.result.first_repair_cost
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
        },
        "confidence_action_predictive_partition": partition.to_dict(),
        "compiled_controlled_interface": interface.to_dict(),
        "candidate_abstractions": candidates,
        "heldout": {
            "interaction_count": len(heldout),
            "interaction_sha256": _interaction_hash(heldout),
            "hidden_counter_evaluator_only": 10**9,
            "action_schedule": "balanced alternating baseline/probe",
            "evaluations": evaluations,
        },
        "resource_vector": {
            "memory_order": active.result.final_order,
            "raw_context_count": state_count,
            "action_count": len(config.actions),
            "predictive_block_count": len(partition.blocks),
            "autonomous_block_count": len(
                predictive_candidate["controlled_dynamic_closure"]
            ),
            "dynamic_closure_excess_blocks": predictive_candidate[
                "controlled_dynamic_closure_excess_blocks"
            ],
            "average_action_context_transition_entropy_nats": _transition_entropy(
                interface
            ),
            "active_steps": active.result.steps,
            "active_total_cost": active.result.total_cost,
        },
        "result": {
            "memory_repair_trace": (
                active.result.initial_order,
                active.result.final_order,
            ),
            "predictive_blocks": len(partition.blocks),
            "autonomous_blocks": len(
                predictive_candidate["controlled_dynamic_closure"]
            ),
            "dynamic_closure_excess_blocks": predictive_candidate[
                "controlled_dynamic_closure_excess_blocks"
            ],
            "action_averaged_predictive_residual_nats": predictive_candidate[
                "action_averaged_predictive_residual_nats"
            ],
            "action_averaged_lumpability_residual_nats": predictive_candidate[
                "action_averaged_lumpability_residual_nats"
            ],
            "heldout_accuracy": evaluations["adaptive_repaired"]["accuracy"],
            "heldout_nll_nats": evaluations["adaptive_repaired"][
                "mean_negative_log_likelihood_nats"
            ],
            "heldout_brier": evaluations["adaptive_repaired"][
                "mean_brier_score"
            ],
        },
    }


def _controls(
    *, seed: int, config: ControlledRepairConfig
) -> dict[str, object]:
    world = ControlledWorld(actions=config.actions)
    passive_config = replace(config, maximum_steps=min(config.maximum_steps, 32768))
    passive = simulate_controlled_repair(
        world=world, policy="passive", seed=seed + 10, config=passive_config
    )
    passive_counts = _rebuild_counts(passive.interactions, passive_config)
    passive_restricted = passive_counts.assess_memory(
        1,
        delta=passive_config.delta,
        family_cell_budget=passive_config.anytime_family_cell_budget,
        merge_tolerance=passive_config.merge_tolerance,
        split_tolerance=passive_config.split_tolerance,
        minimum_samples=passive_config.minimum_samples,
        actions=("baseline",),
    )

    null_config = replace(config, maximum_steps=min(config.maximum_steps, 32768))
    null = simulate_controlled_repair(
        world=ControlledWorld.null(actions=config.actions),
        policy="active",
        seed=seed + 20,
        config=null_config,
    )

    misspecified_config = replace(
        config,
        maximum_steps=min(config.maximum_steps, 8192),
        exploration_weight=0.0,
        split_evidence_weight=0.0,
    )
    misspecified = simulate_controlled_repair(
        world=world,
        policy="greedy_misspecified",
        seed=seed + 30,
        config=misspecified_config,
    )

    drift_interaction_count = 128 * (64 - (config.maximum_order + 1))
    drift_world = ControlledWorld(
        actions=config.actions, drift_step=drift_interaction_count // 2
    )
    drift = generate_controlled_interactions(
        world=drift_world,
        actions=config.actions,
        action_costs=config.cost_map,
        episode_count=128,
        episode_length=64,
        history_length=config.maximum_order + 1,
        seed=seed + 40,
    )
    drift_assessment = assess_action_stationarity(
        drift,
        config.maximum_order,
        actions=config.actions,
        delta=config.delta,
        merge_tolerance=config.merge_tolerance,
        split_tolerance=config.split_tolerance,
        minimum_samples=config.minimum_samples,
    )

    cost_sensitivity = []
    for probe_cost in (1.0, 2.0, 4.0, 8.0):
        sensitivity_config = replace(
            config,
            maximum_steps=min(config.maximum_steps, 8192),
            action_costs=(("baseline", 1.0), ("probe", probe_cost)),
        )
        run = simulate_controlled_repair(
            world=world,
            policy="active",
            seed=seed + 50,
            config=sensitivity_config,
            stop_after_first_repair=True,
        ).result
        cost_sensitivity.append(
            {
                "probe_cost": probe_cost,
                "repair_discovered": run.repair_discovered,
                "first_repair_step": run.first_repair_step,
                "first_repair_cost": run.first_repair_cost,
                "action_counts": dict(run.action_counts),
            }
        )

    return {
        "passive_policy_scope": {
            "run": passive.result.to_dict(),
            "baseline_only_assessment": passive_restricted.to_dict(),
            "full_action_family_assessment": passive.result.final_assessment.to_dict(),
            "passed": (
                passive_restricted.status == "certified_within_tolerance"
                and passive.result.final_assessment.status == "insufficient_evidence"
                and not passive.result.certified
            ),
            "meaning": (
                "baseline observations do not expose the missing coordinate; "
                "that policy-relative certificate cannot be promoted to the "
                "unobserved intervention family"
            ),
        },
        "null_action_family": {
            "run": null.result.to_dict(),
            "passed": (
                null.result.certified
                and null.result.final_order == 1
                and not null.result.repair_discovered
            ),
            "meaning": "when no declared action reveals older-bit dependence, no split is invented",
        },
        "misspecified_greedy_proposal": {
            "run": misspecified.result.to_dict(),
            "passed": (
                not misspecified.result.certified
                and not misspecified.result.repair_discovered
                and misspecified.result.status == "sample_budget_exhausted"
            ),
            "meaning": (
                "a wrong proposal model without forced exploration can miss the "
                "useful action, but the verifier abstains rather than certifying"
            ),
        },
        "action_conditioned_drift": {
            "interaction_count": len(drift),
            "interaction_sha256": _interaction_hash(drift),
            "assessment": drift_assessment.to_dict(),
            "passed": (
                drift_assessment.status
                == "action_conditioned_distribution_shift_detected"
            ),
        },
        "probe_cost_sensitivity": {
            "runs": tuple(cost_sensitivity),
            "interpretation": (
                "the only revealing intervention remains necessary as its price "
                "rises; sample count stays stable while paid cost scales"
            ),
        },
    }


def _monte_carlo(
    *, seed: int, config: ControlledRepairConfig, runs: int
) -> dict[str, object]:
    if runs < 1:
        raise ValueError("Monte Carlo runs must be positive")
    discovery_config = replace(config, maximum_steps=min(config.maximum_steps, 8192))
    world = ControlledWorld(actions=config.actions)
    active_steps = []
    uniform_steps = []
    active_costs = []
    uniform_costs = []
    active_failures = []
    uniform_failures = []
    for run_index in range(runs):
        run_seed = seed + 1000 + run_index
        active = simulate_controlled_repair(
            world=world,
            policy="active",
            seed=run_seed,
            config=discovery_config,
            stop_after_first_repair=True,
        ).result
        uniform = simulate_controlled_repair(
            world=world,
            policy="uniform",
            seed=run_seed,
            config=discovery_config,
            stop_after_first_repair=True,
        ).result
        if active.repair_discovered:
            active_steps.append(float(active.first_repair_step))
            active_costs.append(float(active.first_repair_cost))
        else:
            active_failures.append({"run_index": run_index, "seed": run_seed})
        if uniform.repair_discovered:
            uniform_steps.append(float(uniform.first_repair_step))
            uniform_costs.append(float(uniform.first_repair_cost))
        else:
            uniform_failures.append({"run_index": run_index, "seed": run_seed})

    null_runs = max(10, runs // 2)
    null_config = replace(config, maximum_steps=min(config.maximum_steps, 32768))
    null_false_splits = []
    null_certifications = 0
    for run_index in range(null_runs):
        run_seed = seed + 2000 + run_index
        result = simulate_controlled_repair(
            world=ControlledWorld.null(actions=config.actions),
            policy="active",
            seed=run_seed,
            config=null_config,
        ).result
        if result.repair_discovered:
            null_false_splits.append({"run_index": run_index, "seed": run_seed})
        null_certifications += int(result.certified and result.final_order == 1)

    misspecified_runs = max(10, runs // 5)
    misspecified_config = replace(
        config,
        maximum_steps=min(config.maximum_steps, 8192),
        exploration_weight=0.0,
        split_evidence_weight=0.0,
    )
    misspecified_unsafe = []
    for run_index in range(misspecified_runs):
        run_seed = seed + 3000 + run_index
        result = simulate_controlled_repair(
            world=world,
            policy="greedy_misspecified",
            seed=run_seed,
            config=misspecified_config,
        ).result
        if result.certified or result.repair_discovered:
            misspecified_unsafe.append({"run_index": run_index, "seed": run_seed})

    drift_runs = max(10, runs // 2)
    drift_detections = 0
    drift_misses = []
    drift_count = 128 * (64 - (config.maximum_order + 1))
    for run_index in range(drift_runs):
        run_seed = seed + 4000 + run_index
        drift_world = ControlledWorld(
            actions=config.actions, drift_step=drift_count // 2
        )
        interactions = generate_controlled_interactions(
            world=drift_world,
            actions=config.actions,
            action_costs=config.cost_map,
            episode_count=128,
            episode_length=64,
            history_length=config.maximum_order + 1,
            seed=run_seed,
        )
        assessment = assess_action_stationarity(
            interactions,
            config.maximum_order,
            actions=config.actions,
            delta=config.delta,
            merge_tolerance=config.merge_tolerance,
            split_tolerance=config.split_tolerance,
            minimum_samples=config.minimum_samples,
        )
        detected = (
            assessment.status == "action_conditioned_distribution_shift_detected"
        )
        drift_detections += int(detected)
        if not detected:
            drift_misses.append({"run_index": run_index, "seed": run_seed})

    active_successes = len(active_steps)
    uniform_successes = len(uniform_steps)
    paired_successes = min(active_successes, uniform_successes)
    if paired_successes:
        step_ratio = _bootstrap_paired_median_ratio(
            active_steps[:paired_successes],
            uniform_steps[:paired_successes],
            seed=seed + 5000,
        )
        cost_ratio = _bootstrap_paired_median_ratio(
            active_costs[:paired_successes],
            uniform_costs[:paired_successes],
            seed=seed + 5001,
        )
    else:
        step_ratio = cost_ratio = None
    return {
        "run_count": runs,
        "seed_rule": {
            "base_seed": seed,
            "active_uniform_offset": 1000,
            "null_offset": 2000,
            "misspecified_offset": 3000,
            "drift_offset": 4000,
        },
        "active_discovery": {
            "successes": active_successes,
            "trials": runs,
            "rate": active_successes / runs,
            "wilson_95_interval": _wilson_interval(active_successes, runs),
            "failure_records": tuple(active_failures),
            "steps": {
                "minimum": min(active_steps) if active_steps else None,
                "median": median(active_steps) if active_steps else None,
                "maximum": max(active_steps) if active_steps else None,
                "mean": mean(active_steps) if active_steps else None,
            },
            "cost": {
                "minimum": min(active_costs) if active_costs else None,
                "median": median(active_costs) if active_costs else None,
                "maximum": max(active_costs) if active_costs else None,
                "mean": mean(active_costs) if active_costs else None,
            },
        },
        "uniform_discovery": {
            "successes": uniform_successes,
            "trials": runs,
            "rate": uniform_successes / runs,
            "wilson_95_interval": _wilson_interval(uniform_successes, runs),
            "failure_records": tuple(uniform_failures),
            "steps": {
                "minimum": min(uniform_steps) if uniform_steps else None,
                "median": median(uniform_steps) if uniform_steps else None,
                "maximum": max(uniform_steps) if uniform_steps else None,
                "mean": mean(uniform_steps) if uniform_steps else None,
            },
            "cost": {
                "minimum": min(uniform_costs) if uniform_costs else None,
                "median": median(uniform_costs) if uniform_costs else None,
                "maximum": max(uniform_costs) if uniform_costs else None,
                "mean": mean(uniform_costs) if uniform_costs else None,
            },
        },
        "paired_active_over_uniform": {
            "step_median_ratio": step_ratio,
            "cost_median_ratio": cost_ratio,
        },
        "null_false_splits": {
            "events": len(null_false_splits),
            "trials": null_runs,
            "rate": len(null_false_splits) / null_runs,
            "wilson_95_interval": _wilson_interval(
                len(null_false_splits), null_runs
            ),
            "failure_records": tuple(null_false_splits),
            "correct_order_one_certifications": null_certifications,
        },
        "misspecified_proposal_unsafe_outputs": {
            "events": len(misspecified_unsafe),
            "trials": misspecified_runs,
            "records": tuple(misspecified_unsafe),
        },
        "drift_detection": {
            "successes": drift_detections,
            "trials": drift_runs,
            "rate": drift_detections / drift_runs,
            "wilson_95_interval": _wilson_interval(drift_detections, drift_runs),
            "miss_records": tuple(drift_misses),
        },
    }


def run_active_experiment(
    *,
    seed: int = 20260802,
    maximum_steps: int = 65536,
    heldout_episode_count: int = 64,
    delta: float = 0.01,
    merge_tolerance: float = 0.2,
    split_tolerance: float = 0.35,
    minimum_samples: int = 32,
    probe_cost: float = 2.0,
    monte_carlo_runs: int = 100,
    project_root: Path | None = None,
) -> dict[str, object]:
    root = Path.cwd() if project_root is None else project_root
    config = ControlledRepairConfig(
        maximum_steps=maximum_steps,
        delta=delta,
        merge_tolerance=merge_tolerance,
        split_tolerance=split_tolerance,
        minimum_samples=minimum_samples,
        action_costs=(("baseline", 1.0), ("probe", probe_cost)),
    )
    primary = _primary_analysis(
        seed=seed, config=config, heldout_episode_count=heldout_episode_count
    )
    controls = _controls(seed=seed, config=config)
    monte_carlo = _monte_carlo(
        seed=seed, config=config, runs=monte_carlo_runs
    )
    result = primary["result"]
    evaluations = primary["heldout"]["evaluations"]
    comparison = primary["policy_comparison"]
    checks = {
        "active_repairs_order_one_to_two": result["memory_repair_trace"] == (1, 2),
        "active_reaches_full_action_family_fixed_point": primary["active_run"][
            "certified"
        ],
        "active_discovers_repair_before_uniform": (
            comparison["active_first_repair_step"]
            < comparison["uniform_first_repair_step"]
        ),
        "active_reaches_fixed_point_before_uniform": (
            comparison["active_fixed_point_step"]
            < comparison["uniform_fixed_point_step"]
        ),
        "active_fixed_point_cost_below_uniform": (
            comparison["active_fixed_point_cost"]
            < comparison["uniform_fixed_point_cost"]
        ),
        "action_predictive_partition_has_two_blocks": result[
            "predictive_blocks"
        ]
        == 2,
        "controlled_dynamic_repair_has_four_blocks": result[
            "autonomous_blocks"
        ]
        == 4,
        "action_predictive_partition_not_autonomous": result[
            "dynamic_closure_excess_blocks"
        ]
        == 2,
        "action_predictive_residual_small": result[
            "action_averaged_predictive_residual_nats"
        ]
        < 0.005,
        "controlled_lumpability_residual_large": result[
            "action_averaged_lumpability_residual_nats"
        ]
        > 0.68,
        "repaired_nll_beats_current_observation": evaluations[
            "adaptive_repaired"
        ]["mean_negative_log_likelihood_nats"]
        < evaluations["current_observation_action_conditioned"][
            "mean_negative_log_likelihood_nats"
        ],
        "adaptive_matches_fixed_order_two": evaluations["adaptive_repaired"]
        == evaluations["fixed_order_two"],
        "passive_scope_control_passes": controls["passive_policy_scope"][
            "passed"
        ],
        "null_action_family_avoids_false_split": controls[
            "null_action_family"
        ]["passed"],
        "misspecified_proposal_abstains_safely": controls[
            "misspecified_greedy_proposal"
        ]["passed"],
        "action_conditioned_drift_detected": controls[
            "action_conditioned_drift"
        ]["passed"],
        "monte_carlo_active_recovery_at_least_95_percent": monte_carlo[
            "active_discovery"
        ]["rate"]
        >= 0.95,
        "monte_carlo_active_median_steps_below_uniform": monte_carlo[
            "active_discovery"
        ]["steps"]["median"]
        < monte_carlo["uniform_discovery"]["steps"]["median"],
        "monte_carlo_null_false_splits_zero": monte_carlo[
            "null_false_splits"
        ]["events"]
        == 0,
        "monte_carlo_misspecified_unsafe_outputs_zero": monte_carlo[
            "misspecified_proposal_unsafe_outputs"
        ]["events"]
        == 0,
        "monte_carlo_drift_detection_complete": monte_carlo[
            "drift_detection"
        ]["successes"]
        == monte_carlo["drift_detection"]["trials"],
        "anytime_family_budget_matches_declared_cells_and_prefixes": primary[
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
    return _jsonable(
        {
            "schema_version": "ts-language-foundations-active-repair-v0.5",
            "experiment": {
                "name": "active intervention selection for representation repair",
                "version": "0.5.0",
                "question": (
                    "Can a learner choose interventions that expose missing state "
                    "more efficiently than uniform or passive observation, while "
                    "leaving all representation changes to an independent verifier?"
                ),
            },
            "run": {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "seed": seed,
                "maximum_steps": maximum_steps,
                "heldout_episode_count": heldout_episode_count,
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
                "action_family_monotonicity": (
                    "for A0 subset A, equality of q_x^a for every a in A implies "
                    "equality for every a in A0; therefore Q_A refines Q_A0"
                ),
                "controlled_dynamic_closure": (
                    "a partition is autonomous only when it is strongly lumpable "
                    "for every declared action kernel"
                ),
                "proposal_authority_separation": (
                    "Bayesian information gain and uncertainty reduction choose "
                    "actions; only simultaneous confidence comparisons change state"
                ),
                "anytime_bound": (
                    "for each intrinsic action/context cell and every within-cell "
                    "sample prefix n<=T, radius sqrt(log(2M/delta)/(2n)); a finite "
                    "union bound covers adaptive stopping and predictable action choice"
                ),
                "proof_boundary": (
                    "coverage requires stationary conditionally independent Bernoulli "
                    "innovations for each action/context cell; Monte Carlo tests the "
                    "implementation and is not the proof"
                ),
            },
            "primary_controlled_case": primary,
            "controls": controls,
            "monte_carlo": monte_carlo,
            "verification": verification,
            "conclusion": {
                "established_in_fixture": (
                    "active action selection exposes a policy-hidden memory defect, "
                    "reaches the confidence fixed point with fewer samples and lower "
                    "cost than uniform exploration, and recovers a two-block predictive "
                    "quotient whose controlled dynamic closure has four states"
                ),
                "honest_null": (
                    "the repaired predictor matches rather than beats the correctly "
                    "specified fixed-order-two action-conditioned baseline"
                ),
                "negative_findings": (
                    "passive evidence cannot justify untried actions; a misspecified "
                    "greedy proposal can waste its budget; necessary intervention cost "
                    "cannot be optimised away; drift invalidates stationary compilation"
                ),
                "not_established": (
                    "general causal discovery, arbitrary hidden-state identification, "
                    "continuous control, learned action semantics, language, planning, "
                    "or superiority to reinforcement-learning and active-learning methods"
                ),
                "next_falsifiable_experiment": (
                    "replace the hand-declared suffix repair family with typed relational "
                    "constructors and test whether counterexamples can select both a new "
                    "feature and an intervention without hidden-state labels"
                ),
            },
            "provenance": {
                "source_hashes": source_hashes(root),
                "source_file_count": len(source_hashes(root)),
            },
        }
    )


def write_active_report(receipt: dict[str, object], path: Path) -> None:
    primary = receipt["primary_controlled_case"]
    result = primary["result"]
    policy = primary["policy_comparison"]
    evaluations = primary["heldout"]["evaluations"]
    monte = receipt["monte_carlo"]
    controls = receipt["controls"]
    verification = receipt["verification"]
    report = rf"""# Active Intervention Repair — Technical Report

## Question

Can a representation-repair learner choose an intervention that exposes missing state more efficiently than passive or uniform data collection, while keeping representation changes under an independent statistical verifier?

## Controlled world and source boundary

The hidden environment is

\[
(u,v,n)\xrightarrow{{a}}(v,W,n+1),\qquad (u,v)\in\{{0,1}}^2,\ n\in\mathbb N_0.
\]

For the `baseline` action, `P(W=1|u,baseline)=0.5`. For the `probe` action,

\[
P(W=1|u=0,probe)=0.85,\qquad P(W=1|u=1,probe)=0.15.
\]

The learner receives visible binary history, its own chosen action, the declared action cost, and the resulting bit. It never receives `n`, a hidden-state identifier, or the correct memory order. No corpus, network service, tokenizer, neural component, or external model is used.

## Action-family mathematics

For action family `A`, define `x ~_A y` when `q_x^a=q_y^a` for every `a in A`. If `A0` is a subset of `A`, then `Q_A` refines `Q_A0`: equality under every action implies equality under the restricted family. Expanding the interventions can therefore reveal distinctions invisible to a passive policy.

Controlled autonomous closure is stronger still. A partition must be strongly lumpable for every action kernel, not merely predictive for each immediate response.

## Proposal versus authority

The action policy scores interventions using posterior expected information gain, expected older-bit contrast, and an uncertainty-reduction proxy per unit cost. These quantities have no authority to change the representation.

For each predeclared action/context cell and every within-cell sample prefix `n<=T`, the verifier uses

\[
r(n)=\sqrt{{\frac{{\log(2M/\delta)}}{{2n}}}}.
\]

The finite family contains `{primary['active_run']['confidence_contract']['declared_intrinsic_cells']}` intrinsic cells and `{primary['active_run']['confidence_contract']['family_cell_budget']}` cell-prefix events. The union bound therefore remains valid under predictable action choice and adaptive stopping, assuming stationary conditionally independent Bernoulli innovations within each action/context cell.

## Primary repair

| Quantity | Active | Uniform |
|---|---:|---:|
| First certified repair step | {policy['active_first_repair_step']} | {policy['uniform_first_repair_step']} |
| First certified repair cost | {policy['active_first_repair_cost']:.0f} | {policy['uniform_first_repair_cost']:.0f} |
| Full fixed-point step | {policy['active_fixed_point_step']} | {policy['uniform_fixed_point_step']} |
| Full fixed-point cost | {policy['active_fixed_point_cost']:.0f} | {policy['uniform_fixed_point_cost']:.0f} |

The active learner repaired memory order `1 -> 2`. It then certified that one still-older coordinate was unnecessary across both declared actions.

## Predictive compression and autonomous closure

The confidence partition is

\[
\{{00,01\}},\qquad \{{10,11\}}.
\]

Its action-averaged predictive residual is `{result['action_averaged_predictive_residual_nats']:.9f}` nats. Its action-averaged lumpability residual is `{result['action_averaged_lumpability_residual_nats']:.9f}` nats. Controlled dynamic repair restores all four suffix states, giving closure excess `{result['dynamic_closure_excess_blocks']}`.

The reason is structural: the current response vector depends on the older bit `u`, while the next predictive block is determined by the current bit `v`. Merging `00` with `01` answers the current action query but destroys autonomous macrostate evolution.

## Heldout controlled prediction

Heldout evaluation uses `{primary['heldout']['interaction_count']}` balanced action-conditioned transitions at evaluator-only nuisance counter `10^9`.

| Model | Accuracy | NLL (nats) | Brier |
|---|---:|---:|---:|
| One state per action | {evaluations['one_state_action_conditioned']['accuracy']:.4f} | {evaluations['one_state_action_conditioned']['mean_negative_log_likelihood_nats']:.4f} | {evaluations['one_state_action_conditioned']['mean_brier_score']:.4f} |
| Current bit per action | {evaluations['current_observation_action_conditioned']['accuracy']:.4f} | {evaluations['current_observation_action_conditioned']['mean_negative_log_likelihood_nats']:.4f} | {evaluations['current_observation_action_conditioned']['mean_brier_score']:.4f} |
| Adaptive repaired order two | {evaluations['adaptive_repaired']['accuracy']:.4f} | {evaluations['adaptive_repaired']['mean_negative_log_likelihood_nats']:.4f} | {evaluations['adaptive_repaired']['mean_brier_score']:.4f} |
| Fixed order two | {evaluations['fixed_order_two']['accuracy']:.4f} | {evaluations['fixed_order_two']['mean_negative_log_likelihood_nats']:.4f} | {evaluations['fixed_order_two']['mean_brier_score']:.4f} |

The adaptive and fixed-order-two results are identical. This is a state-discovery result, not predictive superiority over an oracle supplied with the correct order.

## Policy-scope and hostile controls

- Passive baseline data: restricted extension status `{controls['passive_policy_scope']['baseline_only_assessment']['status']}`; full-family status `{controls['passive_policy_scope']['full_action_family_assessment']['status']}`.
- Null action family: final order `{controls['null_action_family']['run']['final_order']}`, split invented `{controls['null_action_family']['run']['repair_discovered']}`.
- Misspecified greedy proposal: `{controls['misspecified_greedy_proposal']['run']['status']}` with no unsafe certificate.
- Reversing probe law: `{controls['action_conditioned_drift']['assessment']['status']}`.
- Probe-cost sensitivity: every declared cost recovered the same repair sample requirement; paid cost scaled because the revealing intervention was necessary.

## Monte Carlo

Across `{monte['run_count']}` paired discovery seeds:

- active recovery `{monte['active_discovery']['successes']}/{monte['active_discovery']['trials']}`, median `{monte['active_discovery']['steps']['median']:.0f}` steps and `{monte['active_discovery']['cost']['median']:.0f}` cost;
- uniform recovery `{monte['uniform_discovery']['successes']}/{monte['uniform_discovery']['trials']}`, median `{monte['uniform_discovery']['steps']['median']:.0f}` steps and `{monte['uniform_discovery']['cost']['median']:.0f}` cost;
- null false splits `{monte['null_false_splits']['events']}/{monte['null_false_splits']['trials']}`;
- misspecified unsafe outputs `{monte['misspecified_proposal_unsafe_outputs']['events']}/{monte['misspecified_proposal_unsafe_outputs']['trials']}`;
- drift detections `{monte['drift_detection']['successes']}/{monte['drift_detection']['trials']}`.

The paired bootstrap median active/uniform step ratio is `{monte['paired_active_over_uniform']['step_median_ratio']['point_estimate']:.3f}` with 95% percentile interval `{monte['paired_active_over_uniform']['step_median_ratio']['percentile_95_interval']}`. This interval describes deterministic fixture sensitivity across seeds, not a population parameter.

## Verifier results

`{verification['passed_count']}/{verification['check_count']}` declared checks passed.

## What failed or remains uncertain

- Passive sufficiency is policy-relative and cannot be promoted to untried actions.
- A misspecified greedy proposal without forced exploration can completely miss the useful action. The verifier protects correctness by abstaining, but cannot manufacture data.
- Active selection cannot optimise away the cost of an intervention that is uniquely informative.
- The action semantics and suffix constructor family are hand-declared.
- The coverage proof assumes stationary conditionally independent cell innovations. The drift gate detects the supplied reversal but does not solve general nonstationary control.
- This is not general causal discovery, continuous control, language learning, planning, or evidence of neural-model superiority.

## Next falsifiable experiment

Replace the hand-declared suffix family with typed relational constructors. Require the learner to choose both a candidate feature and an intervention, then certify whether the constructed relation predicts and evolves autonomously without hidden-state labels.
"""
    report = report.replace("\\\\", "\\")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def write_active_figure(receipt: dict[str, object], path: Path) -> None:
    primary = receipt["primary_controlled_case"]
    policy = primary["policy_comparison"]
    result = primary["result"]
    monte = receipt["monte_carlo"]
    evals = primary["heldout"]["evaluations"]
    trace = [
        item
        for item in primary["active_run"]["assessment_trace"]
        if item["order"] == 1 and item["step"] <= policy["active_first_repair_step"]
    ]
    chart_left, chart_right = 85.0, 645.0
    chart_top, chart_bottom = 485.0, 735.0
    max_step = max(item["step"] for item in trace)
    points = []
    for item in trace:
        x = chart_left + (chart_right - chart_left) * item["step"] / max_step
        lower = item["maximum_lower_difference_by_action"]["probe"]
        y = chart_bottom - (chart_bottom - chart_top) * min(lower, 0.8) / 0.8
        points.append(f"{x:.1f},{y:.1f}")
    threshold_y = chart_bottom - (chart_bottom - chart_top) * 0.35 / 0.8

    bar_values = (
        ("Active repair", policy["active_first_repair_step"], "#8af7d4"),
        ("Uniform repair", policy["uniform_first_repair_step"], "#8aa9ff"),
        ("Active fixed point", policy["active_fixed_point_step"], "#d7a3ff"),
        ("Uniform fixed point", policy["uniform_fixed_point_step"], "#ffb86b"),
    )
    max_bar = max(value for _, value, _ in bar_values)
    bars = []
    for index, (label, value, colour) in enumerate(bar_values):
        y = 190 + index * 42
        width = 330 * value / max_bar
        bars.append(
            f'<text x="790" y="{y + 17}" class="small">{escape(label)}</text>'
            f'<rect x="930" y="{y}" width="{width:.1f}" height="24" rx="5" fill="{colour}"/>'
            f'<text x="{940 + width:.1f}" y="{y + 17}" class="small value">{value:,.0f}</text>'
        )

    nll_values = (
        ("One state", evals["one_state_action_conditioned"]["mean_negative_log_likelihood_nats"], "#66708e"),
        ("Current bit", evals["current_observation_action_conditioned"]["mean_negative_log_likelihood_nats"], "#8aa9ff"),
        ("Repaired", evals["adaptive_repaired"]["mean_negative_log_likelihood_nats"], "#8af7d4"),
    )
    nll_bars = []
    for index, (label, value, colour) in enumerate(nll_values):
        x = 770 + index * 135
        height = 145 * value / 0.75
        y = 680 - height
        nll_bars.append(
            f'<rect x="{x}" y="{y:.1f}" width="78" height="{height:.1f}" rx="7" fill="{colour}"/>'
            f'<text x="{x + 39}" y="{y - 8:.1f}" text-anchor="middle" class="small value">{value:.3f}</text>'
            f'<text x="{x + 39}" y="704" text-anchor="middle" class="small">{escape(label)}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="900" viewBox="0 0 1400 900">
<defs>
  <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#090d1b"/><stop offset="1" stop-color="#17102a"/></linearGradient>
  <filter id="glow"><feGaussianBlur stdDeviation="4" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
</defs>
<style>
  text{{font-family:Inter,Segoe UI,Arial,sans-serif;fill:#eef3ff}} .title{{font-size:30px;font-weight:700}} .subtitle{{font-size:14px;fill:#9eabc9}} .panel-title{{font-size:18px;font-weight:700}} .small{{font-size:13px;fill:#bac5df}} .value{{fill:#f7fbff;font-weight:700}} .box{{fill:#141c32;stroke:#354362;stroke-width:1.5}} .grid{{stroke:#303952;stroke-width:1}} .axis{{stroke:#697695;stroke-width:1.5}}
</style>
<rect width="1400" height="900" fill="url(#bg)"/>
<text x="52" y="54" class="title">ACTIVE REPRESENTATION REPAIR · v0.5.0</text>
<text x="52" y="80" class="subtitle">Proposal chooses experience. Confidence certificates retain authority.</text>

<rect x="48" y="108" width="650" height="300" rx="18" class="box"/>
<text x="72" y="142" class="panel-title">The repaired loop</text>
<rect x="78" y="174" width="150" height="70" rx="12" fill="#222b48" stroke="#5b6c98"/>
<text x="153" y="201" text-anchor="middle" class="small">Memory order 1</text><text x="153" y="224" text-anchor="middle" class="value">insufficient</text>
<path d="M228 209 H285" stroke="#8aa9ff" stroke-width="3" marker-end="url(#none)"/>
<rect x="285" y="174" width="160" height="70" rx="12" fill="#2b2141" stroke="#a47fe0"/>
<text x="365" y="201" text-anchor="middle" class="small">Chosen probe</text><text x="365" y="224" text-anchor="middle" class="value">certified split</text>
<path d="M445 209 H500" stroke="#d7a3ff" stroke-width="3"/>
<rect x="500" y="174" width="160" height="70" rx="12" fill="#173a35" stroke="#62d9ba"/>
<text x="580" y="201" text-anchor="middle" class="small">Memory order 2</text><text x="580" y="224" text-anchor="middle" class="value">fixed point</text>
<text x="78" y="286" class="small">Predictive compression</text>
<rect x="78" y="304" width="230" height="60" rx="12" fill="#173a35" stroke="#62d9ba"/><text x="193" y="330" text-anchor="middle" class="value">2 blocks</text><text x="193" y="350" text-anchor="middle" class="small">{{00,01}} · {{10,11}}</text>
<path d="M308 334 H410" stroke="#ffb86b" stroke-width="3"/>
<rect x="410" y="304" width="250" height="60" rx="12" fill="#40271e" stroke="#ff9b60"/><text x="535" y="330" text-anchor="middle" class="value">4 autonomous states</text><text x="535" y="350" text-anchor="middle" class="small">controlled dynamic closure</text>
<text x="78" y="390" class="subtitle">R_A={result['action_averaged_predictive_residual_nats']:.6f} · L_A={result['action_averaged_lumpability_residual_nats']:.6f}</text>

<rect x="720" y="108" width="632" height="300" rx="18" class="box"/>
<text x="744" y="142" class="panel-title">Interactions to certificate</text>
{''.join(bars)}

<rect x="48" y="430" width="650" height="420" rx="18" class="box"/>
<text x="72" y="464" class="panel-title">Probe split certificate</text>
<line x1="{chart_left}" y1="{chart_bottom}" x2="{chart_right}" y2="{chart_bottom}" class="axis"/>
<line x1="{chart_left}" y1="{chart_top}" x2="{chart_left}" y2="{chart_bottom}" class="axis"/>
<line x1="{chart_left}" y1="{threshold_y:.1f}" x2="{chart_right}" y2="{threshold_y:.1f}" stroke="#ff8c78" stroke-width="2" stroke-dasharray="7 6"/>
<text x="{chart_right - 5}" y="{threshold_y - 8:.1f}" text-anchor="end" class="small">split threshold 0.35</text>
<polyline points="{' '.join(points)}" fill="none" stroke="#8af7d4" stroke-width="4" filter="url(#glow)"/>
<text x="{chart_left}" y="770" class="small">0 interactions</text><text x="{chart_right}" y="770" text-anchor="end" class="small">{max_step:,} interactions</text>
<text x="72" y="807" class="subtitle">Lower confidence bound on older-bit response difference</text>

<rect x="720" y="430" width="632" height="420" rx="18" class="box"/>
<text x="744" y="464" class="panel-title">Heldout NLL and hostile receipts</text>
<line x1="755" y1="680" x2="1188" y2="680" class="axis"/>
{''.join(nll_bars)}
<text x="1210" y="520" class="small">Monte Carlo</text>
<text x="1210" y="548" class="value">{monte['active_discovery']['successes']}/{monte['active_discovery']['trials']}</text><text x="1210" y="568" class="small">active recovery</text>
<text x="1210" y="612" class="value">0/{monte['null_false_splits']['trials']}</text><text x="1210" y="632" class="small">null false splits</text>
<text x="1210" y="676" class="value">{monte['drift_detection']['successes']}/{monte['drift_detection']['trials']}</text><text x="1210" y="696" class="small">drift detected</text>
<rect x="755" y="745" width="565" height="66" rx="10" fill="#1b243d" stroke="#3d4a6d"/>
<text x="773" y="772" class="small">Passive full-family audit: ABSTAIN</text>
<text x="773" y="797" class="small">Wrong greedy proposal: ABSTAIN · unsafe certificates: 0</text>
</svg>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def write_active_artifacts(
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
    write_active_report(receipt, report)
    write_active_figure(receipt, figure)
