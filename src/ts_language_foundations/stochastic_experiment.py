"""Statistical representation-repair experiment and reproducible artifacts."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from fractions import Fraction
from math import log, sqrt
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

from .experiment import source_hashes
from .repair import (
    Partition,
    dynamic_closure,
    lumpability_residual,
    partition_entropy,
    predictive_partition,
    predictive_residual,
)
from .stochastic import (
    DistributionalLearningResult,
    assess_episode_stationarity,
    compile_stochastic_interface,
    confidence_predictive_partition,
    evaluate_probability_table,
    fit_probability_table,
    generate_first_order_episodes,
    generate_second_order_episodes,
    learn_distributional_suffix_interface,
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
    encoded = json.dumps(
        [list(episode) for episode in episodes], separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _batch_schedule(episode_count: int) -> tuple[int, ...]:
    values = [16, 32, 64, 128, 256, 512, 1024]
    selected = [value for value in values if value < episode_count]
    return tuple((*selected, episode_count))


def _wilson_interval(successes: int, trials: int, z_value: float = 1.959963984540054) -> tuple[float, float]:
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("invalid binomial count")
    proportion = successes / trials
    denominator = 1.0 + z_value**2 / trials
    centre = (proportion + z_value**2 / (2.0 * trials)) / denominator
    half_width = (
        z_value
        * sqrt(
            proportion * (1.0 - proportion) / trials
            + z_value**2 / (4.0 * trials**2)
        )
        / denominator
    )
    lower = 0.0 if successes == 0 else max(0.0, centre - half_width)
    upper = 1.0 if successes == trials else min(1.0, centre + half_width)
    return lower, upper


def _binary_entropy(probability: float) -> float:
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return -probability * log(probability) - (1.0 - probability) * log(1.0 - probability)


def _candidate_record(
    name: str,
    partition: Partition,
    compiled,
    weights: tuple[Fraction, ...],
    *,
    predictive_status: str,
) -> dict[str, object]:
    dynamic, trace = dynamic_closure(
        compiled.transition_rows,
        partition,
        return_trace=True,
    )
    exact_empirical_predictive = predictive_partition(compiled.predictive_laws)
    return {
        "name": name,
        "partition": partition,
        "block_contexts": tuple(
            tuple(compiled.contexts[index] for index in block) for block in partition
        ),
        "block_count": len(partition),
        "predictive_status": predictive_status,
        "predictive_residual_nats": predictive_residual(
            compiled.predictive_laws, partition, weights
        ),
        "lumpability_residual_nats": lumpability_residual(
            compiled.transition_rows, partition, weights
        ),
        "entropy_nats": partition_entropy(partition, weights),
        "dynamic_closure": dynamic,
        "dynamic_closure_contexts": tuple(
            tuple(compiled.contexts[index] for index in block) for block in dynamic
        ),
        "dynamic_repair_trace": trace,
        "dynamic_closure_excess_blocks": len(dynamic) - len(partition),
        "dynamic_fixed": dynamic == partition,
        "exact_empirical_predictive_partition": exact_empirical_predictive,
        "exact_empirical_predictive_fixed": exact_empirical_predictive == partition,
    }


def _learning_kwargs(
    *,
    episode_count: int,
    maximum_order: int,
    delta: float,
    family_cell_budget: int,
    merge_tolerance: float,
    split_tolerance: float,
    minimum_samples_per_child: int,
) -> dict[str, object]:
    return {
        "initial_order": 1,
        "maximum_order": maximum_order,
        "batch_schedule": _batch_schedule(episode_count),
        "delta": delta,
        "family_cell_budget": family_cell_budget,
        "merge_tolerance": merge_tolerance,
        "split_tolerance": split_tolerance,
        "minimum_samples_per_child": minimum_samples_per_child,
    }


def _primary_case(
    *,
    seed: int,
    episode_count: int,
    episode_length: int,
    heldout_episode_count: int,
    maximum_order: int,
    delta: float,
    family_cell_budget: int,
    merge_tolerance: float,
    split_tolerance: float,
    minimum_samples_per_child: int,
) -> tuple[dict[str, object], DistributionalLearningResult]:
    high_probability = 0.8
    low_probability = 0.2
    training = generate_second_order_episodes(
        probability_one_if_older_zero=high_probability,
        probability_one_if_older_one=low_probability,
        episode_count=episode_count,
        episode_length=episode_length,
        seed=seed,
        initial_counter=0,
    )
    heldout_counter = 10**9
    heldout = generate_second_order_episodes(
        probability_one_if_older_zero=high_probability,
        probability_one_if_older_one=low_probability,
        episode_count=heldout_episode_count,
        episode_length=episode_length,
        seed=seed + 1,
        initial_counter=heldout_counter,
    )
    learning_kwargs = _learning_kwargs(
        episode_count=episode_count,
        maximum_order=maximum_order,
        delta=delta,
        family_cell_budget=family_cell_budget,
        merge_tolerance=merge_tolerance,
        split_tolerance=split_tolerance,
        minimum_samples_per_child=minimum_samples_per_child,
    )
    learned = learn_distributional_suffix_interface(training, **learning_kwargs)
    if not learned.certified:
        raise RuntimeError("primary stochastic fixture failed to certify")
    admitted_training = training[: learned.episodes_admitted]
    stationarity = assess_episode_stationarity(
        admitted_training,
        learned.final_order,
        delta=delta,
        family_cell_budget=family_cell_budget,
        merge_tolerance=merge_tolerance,
        split_tolerance=split_tolerance,
        minimum_samples_per_half=minimum_samples_per_child,
    )
    confidence_partition = confidence_predictive_partition(
        learned.estimate_map,
        merge_tolerance=merge_tolerance,
        split_tolerance=split_tolerance,
        minimum_samples=minimum_samples_per_child,
    )
    if not confidence_partition.certified:
        raise RuntimeError("primary confidence partition failed to certify")
    compiled = compile_stochastic_interface(learned)
    total_support = sum(compiled.support_counts)
    weights = tuple(Fraction(count, total_support) for count in compiled.support_counts)
    state_count = len(compiled.contexts)
    identity = tuple((index,) for index in range(state_count))
    one_state = (tuple(range(state_count)),)
    current_observation = tuple(
        tuple(
            index
            for index, context in enumerate(compiled.contexts)
            if context[-1] == observation
        )
        for observation in (0, 1)
    )
    candidates = {
        "one_state_negative_control": _candidate_record(
            "one-state negative control",
            one_state,
            compiled,
            weights,
            predictive_status="rejected by empirical residual",
        ),
        "current_observation": _candidate_record(
            "current observed bit",
            current_observation,
            compiled,
            weights,
            predictive_status="rejected by extension verifier",
        ),
        "confidence_predictive_partition": _candidate_record(
            "confidence-bounded predictive partition",
            confidence_partition.blocks,
            compiled,
            weights,
            predictive_status="certified within declared tolerance",
        ),
        "learned_suffix_identity": _candidate_record(
            "learned suffix identity",
            identity,
            compiled,
            weights,
            predictive_status="trivially exact for empirical laws",
        ),
    }

    baseline_evaluations = {}
    for name, order in (
        ("one_state_probability", 0),
        ("current_observation_probability", 1),
        ("fixed_order_two_probability", 2),
    ):
        table = fit_probability_table(admitted_training, order, pseudocount=0.5)
        baseline_evaluations[name] = evaluate_probability_table(table, order, heldout)
    adaptive_table = fit_probability_table(
        admitted_training, learned.final_order, pseudocount=0.5
    )
    adaptive_evaluation = evaluate_probability_table(
        adaptive_table, learned.final_order, heldout
    )

    interval_coverage = []
    squared_parameter_errors = []
    for context, estimate in learned.estimate_map.items():
        truth = high_probability if context[0] == 0 else low_probability
        interval_coverage.append(
            {
                "context": context,
                "true_probability_one": truth,
                "interval_contains_truth": (
                    estimate.lower_probability_one
                    <= truth
                    <= estimate.upper_probability_one
                ),
            }
        )
        squared_parameter_errors.append(
            (estimate.empirical_probability_one - truth) ** 2
        )

    confidence_cell_queries = (
        sum(
            2 * trace_record["assessment"]["comparison_count"]
            for trace_record in learned.assessment_trace
        )
        + len(learned.estimates)
        + 2 * len(stationarity.comparisons)
    )
    declared_worst_case_cell_queries = (
        len(learning_kwargs["batch_schedule"])
        * sum(2 ** (order + 1) for order in range(1, maximum_order + 1))
        + 3 * 2**maximum_order
    )
    average_transition_entropy = sum(
        float(weight)
        * _binary_entropy(estimate.empirical_probability_one)
        for weight, (_, estimate) in zip(weights, learned.estimates)
    )
    predictive_candidate = candidates["confidence_predictive_partition"]
    result = {
        "name": "confidence-bounded observation-only repair with infinite nuisance fibre",
        "hidden_environment": {
            "state": "(u,v,n) in {0,1}^2 x N_0",
            "transition": "(u,v,n)->(v,W,n+1)",
            "law": "P(W=1|u=0)=0.8; P(W=1|u=1)=0.2",
            "observation": "the binary trajectory only",
            "infinite_coordinate_effect": "n increments but does not affect the registered laws",
            "learner_hidden_state_access": False,
        },
        "input_receipt": {
            "schema": "tuple[tuple[binary observation, ...], ...]",
            "training_episode_count": len(training),
            "training_observation_count": sum(map(len, training)),
            "training_sha256": _episode_hash(training),
            "admitted_episode_count": learned.episodes_admitted,
            "admitted_observation_count": learned.observations_admitted,
            "heldout_episode_count": len(heldout),
            "heldout_observation_count": sum(map(len, heldout)),
            "heldout_sha256": _episode_hash(heldout),
            "heldout_counter_evaluator_only": heldout_counter,
        },
        "confidence_contract": {
            "inequality": "two-sided Hoeffding plus finite union bound",
            "radius": "sqrt(log(2*M/delta)/(2*n))",
            "delta": delta,
            "declared_family_cell_budget": family_cell_budget,
            "actual_primary_cell_queries": confidence_cell_queries,
            "declared_worst_case_cell_queries": declared_worst_case_cell_queries,
            "cell_budget_respected": (
                confidence_cell_queries <= family_cell_budget
                and declared_worst_case_cell_queries <= family_cell_budget
            ),
            "merge_rule": f"upper bound on |p1-p2| <= {merge_tolerance}",
            "split_rule": f"lower bound on |p1-p2| > {split_tolerance}",
            "uncertainty_band": (merge_tolerance, split_tolerance),
            "adaptive_validity_boundary": (
                "the batch schedule and finite query family are declared in advance; "
                "the union bound covers every possible queried cell-stage pair"
            ),
            "fixture_assumption": (
                "conditional on a context visit, each next bit uses an independent "
                "Bernoulli innovation with a stationary context law"
            ),
        },
        "learner": learned.to_dict(),
        "stationarity_gate": stationarity.to_dict(),
        "confidence_predictive_partition": confidence_partition.to_dict(),
        "compiled_interface": compiled.to_dict(),
        "candidate_abstractions": candidates,
        "baselines": baseline_evaluations,
        "adaptive_heldout_evaluation": adaptive_evaluation,
        "parameter_receipt": {
            "contexts": tuple(
                {
                    **estimate.to_dict(),
                    "true_probability_one_evaluator_only": (
                        high_probability if context[0] == 0 else low_probability
                    ),
                }
                for context, estimate in learned.estimates
            ),
            "root_mean_square_parameter_error": sqrt(mean(squared_parameter_errors)),
            "simultaneous_selected_intervals_cover_truth": all(
                record["interval_contains_truth"] for record in interval_coverage
            ),
            "interval_coverage_records": interval_coverage,
        },
        "resource_vector": {
            "episodes_admitted": learned.episodes_admitted,
            "observations_admitted": learned.observations_admitted,
            "memory_order": learned.final_order,
            "raw_context_count": len(compiled.contexts),
            "predictive_block_count": len(confidence_partition.blocks),
            "autonomous_block_count": len(predictive_candidate["dynamic_closure"]),
            "confidence_cell_queries": confidence_cell_queries,
            "average_transition_entropy_nats": average_transition_entropy,
        },
        "result": {
            "certified": learned.certified
            and stationarity.status == "stationary_within_tolerance",
            "memory_repair_trace": (
                learned.initial_order,
                *(event["order_after"] for event in learned.repair_events),
            ),
            "predictive_blocks": len(confidence_partition.blocks),
            "predictive_residual_nats": predictive_candidate[
                "predictive_residual_nats"
            ],
            "predictive_lumpability_residual_nats": predictive_candidate[
                "lumpability_residual_nats"
            ],
            "autonomous_blocks": len(predictive_candidate["dynamic_closure"]),
            "dynamic_closure_excess_blocks": predictive_candidate[
                "dynamic_closure_excess_blocks"
            ],
            "heldout_accuracy": adaptive_evaluation["accuracy"],
            "heldout_nll_nats": adaptive_evaluation[
                "mean_negative_log_likelihood_nats"
            ],
            "heldout_brier": adaptive_evaluation["mean_brier_score"],
            "heldout_counter_transfer": heldout_counter,
        },
    }
    return result, learned


def _controls(
    *,
    seed: int,
    episode_count: int,
    episode_length: int,
    maximum_order: int,
    delta: float,
    family_cell_budget: int,
    merge_tolerance: float,
    split_tolerance: float,
    minimum_samples_per_child: int,
) -> dict[str, object]:
    kwargs = _learning_kwargs(
        episode_count=episode_count,
        maximum_order=maximum_order,
        delta=delta,
        family_cell_budget=family_cell_budget,
        merge_tolerance=merge_tolerance,
        split_tolerance=split_tolerance,
        minimum_samples_per_child=minimum_samples_per_child,
    )
    first_order = generate_first_order_episodes(
        probability_one_if_current_zero=0.25,
        probability_one_if_current_one=0.75,
        episode_count=episode_count,
        episode_length=episode_length,
        seed=seed + 10,
    )
    first_order_result = learn_distributional_suffix_interface(first_order, **kwargs)

    iid = generate_first_order_episodes(
        probability_one_if_current_zero=0.5,
        probability_one_if_current_one=0.5,
        episode_count=episode_count,
        episode_length=episode_length,
        seed=seed + 20,
    )
    iid_result = learn_distributional_suffix_interface(iid, **kwargs)
    iid_partition = confidence_predictive_partition(
        iid_result.estimate_map,
        merge_tolerance=merge_tolerance,
        split_tolerance=split_tolerance,
        minimum_samples=minimum_samples_per_child,
    )

    weak = generate_second_order_episodes(
        probability_one_if_older_zero=0.6,
        probability_one_if_older_one=0.4,
        episode_count=episode_count,
        episode_length=episode_length,
        seed=seed + 30,
    )
    weak_result = learn_distributional_suffix_interface(weak, **kwargs)

    half_count = episode_count // 2
    drift_first = generate_second_order_episodes(
        probability_one_if_older_zero=0.8,
        probability_one_if_older_one=0.2,
        episode_count=half_count,
        episode_length=episode_length,
        seed=seed + 40,
    )
    drift_second = generate_second_order_episodes(
        probability_one_if_older_zero=0.2,
        probability_one_if_older_one=0.8,
        episode_count=episode_count - half_count,
        episode_length=episode_length,
        seed=seed + 41,
    )
    drift_episodes = drift_first + drift_second
    drift_memory = learn_distributional_suffix_interface(drift_episodes, **kwargs)
    drift_stationarity = assess_episode_stationarity(
        drift_episodes,
        2,
        delta=delta,
        family_cell_budget=family_cell_budget,
        merge_tolerance=merge_tolerance,
        split_tolerance=split_tolerance,
        minimum_samples_per_half=minimum_samples_per_child,
    )

    return {
        "first_order_null": {
            "true_order": 1,
            "learner": first_order_result.to_dict(),
            "passed": (
                first_order_result.certified
                and first_order_result.final_order == 1
                and not first_order_result.repair_events
            ),
        },
        "iid_collapse_control": {
            "true_law": "P(next=1)=0.5 independent of history",
            "learner": iid_result.to_dict(),
            "confidence_partition": iid_partition.to_dict(),
            "passed": (
                iid_result.certified
                and iid_result.final_order == 1
                and iid_partition.certified
                and len(iid_partition.blocks) == 1
            ),
        },
        "weak_signal_uncertainty_control": {
            "true_extension_difference": 0.2,
            "uncertainty_band": (merge_tolerance, split_tolerance),
            "learner": weak_result.to_dict(),
            "passed": (
                not weak_result.certified
                and weak_result.status == "sample_budget_exhausted"
                and not weak_result.repair_events
            ),
            "meaning": "the verifier waits rather than forcing either merge or split",
        },
        "regime_drift_control": {
            "first_half_law": "p0=0.8, p1=0.2",
            "second_half_law": "p0=0.2, p1=0.8",
            "naive_stationary_memory_learner": drift_memory.to_dict(),
            "stationarity_gate": drift_stationarity.to_dict(),
            "final_policy": "reject stationary certification",
            "passed": drift_stationarity.status == "distribution_shift_detected",
        },
    }


def _sensitivity(
    *,
    seed: int,
    episode_count: int,
    episode_length: int,
    maximum_order: int,
    delta: float,
    family_cell_budget: int,
    merge_tolerance: float,
    split_tolerance: float,
    minimum_samples_per_child: int,
) -> dict[str, object]:
    primary = generate_second_order_episodes(
        probability_one_if_older_zero=0.8,
        probability_one_if_older_one=0.2,
        episode_count=episode_count,
        episode_length=episode_length,
        seed=seed,
    )
    sample_records = []
    for budget in _batch_schedule(episode_count):
        prefix = primary[:budget]
        result = learn_distributional_suffix_interface(
            prefix,
            initial_order=1,
            maximum_order=maximum_order,
            batch_schedule=_batch_schedule(budget),
            delta=delta,
            family_cell_budget=family_cell_budget,
            merge_tolerance=merge_tolerance,
            split_tolerance=split_tolerance,
            minimum_samples_per_child=minimum_samples_per_child,
        )
        sample_records.append(
            {
                "episode_budget": budget,
                "observation_budget": budget * episode_length,
                "status": result.status,
                "final_order": result.final_order,
                "certified": result.certified,
                "repair_event_count": len(result.repair_events),
            }
        )

    signal_records = []
    for high_probability, low_probability in (
        (0.9, 0.1),
        (0.8, 0.2),
        (0.7, 0.3),
        (0.65, 0.35),
        (0.6, 0.4),
        (0.55, 0.45),
    ):
        episodes = generate_second_order_episodes(
            probability_one_if_older_zero=high_probability,
            probability_one_if_older_one=low_probability,
            episode_count=episode_count,
            episode_length=episode_length,
            seed=seed,
        )
        result = learn_distributional_suffix_interface(
            episodes,
            **_learning_kwargs(
                episode_count=episode_count,
                maximum_order=maximum_order,
                delta=delta,
                family_cell_budget=family_cell_budget,
                merge_tolerance=merge_tolerance,
                split_tolerance=split_tolerance,
                minimum_samples_per_child=minimum_samples_per_child,
            ),
        )
        signal_records.append(
            {
                "probability_one_if_older_zero": high_probability,
                "probability_one_if_older_one": low_probability,
                "true_extension_difference": high_probability - low_probability,
                "status": result.status,
                "final_order": result.final_order,
                "certified": result.certified,
                "repair_event_count": len(result.repair_events),
            }
        )
    return {
        "sample_budget": sample_records,
        "signal_gap": signal_records,
        "interpretation": (
            "strong signals split and then certify; intermediate signals inside "
            "the declared hysteresis band remain unresolved at the fixed sample budget"
        ),
    }


def _monte_carlo(
    *,
    runs: int,
    seed: int,
    episode_count: int,
    episode_length: int,
    maximum_order: int,
    delta: float,
    family_cell_budget: int,
    merge_tolerance: float,
    split_tolerance: float,
    minimum_samples_per_child: int,
) -> dict[str, object]:
    if runs < 1:
        raise ValueError("monte_carlo_runs must be positive")
    kwargs = _learning_kwargs(
        episode_count=episode_count,
        maximum_order=maximum_order,
        delta=delta,
        family_cell_budget=family_cell_budget,
        merge_tolerance=merge_tolerance,
        split_tolerance=split_tolerance,
        minimum_samples_per_child=minimum_samples_per_child,
    )
    recovered = 0
    safe_abstentions = 0
    unsafe_certifications = 0
    selected_interval_coverage = 0
    false_splits = 0
    null_certifications = 0
    episodes_to_certify = []
    nonrecovery_records = []
    interval_failure_records = []
    false_split_records = []
    for run_index in range(runs):
        primary = generate_second_order_episodes(
            probability_one_if_older_zero=0.8,
            probability_one_if_older_one=0.2,
            episode_count=episode_count,
            episode_length=episode_length,
            seed=seed + 1000 + run_index,
        )
        learned = learn_distributional_suffix_interface(primary, **kwargs)
        partition = confidence_predictive_partition(
            learned.estimate_map,
            merge_tolerance=merge_tolerance,
            split_tolerance=split_tolerance,
            minimum_samples=minimum_samples_per_child,
        )
        recovered_this_run = (
            learned.certified
            and learned.final_order == 2
            and partition.certified
            and partition.blocks == ((0, 1), (2, 3))
        )
        recovered += int(recovered_this_run)
        if learned.certified:
            episodes_to_certify.append(learned.episodes_admitted)
        if not recovered_this_run:
            safe = not learned.certified
            safe_abstentions += int(safe)
            unsafe_certifications += int(not safe)
            nonrecovery_records.append(
                {
                    "run_index": run_index,
                    "seed": seed + 1000 + run_index,
                    "learner_status": learned.status,
                    "final_order": learned.final_order,
                    "episodes_admitted": learned.episodes_admitted,
                    "final_memory_assessment": learned.final_assessment.status,
                    "predictive_partition_status": partition.status,
                    "predictive_partition_blocks": partition.blocks,
                    "safe_abstention": safe,
                }
            )
        intervals_cover = all(
                estimate.lower_probability_one
                <= (0.8 if context[0] == 0 else 0.2)
                <= estimate.upper_probability_one
                for context, estimate in learned.estimate_map.items()
        )
        selected_interval_coverage += int(intervals_cover)
        if not intervals_cover:
            interval_failure_records.append(
                {"run_index": run_index, "seed": seed + 1000 + run_index}
            )

        null = generate_first_order_episodes(
            probability_one_if_current_zero=0.25,
            probability_one_if_current_one=0.75,
            episode_count=episode_count,
            episode_length=episode_length,
            seed=seed + 2000 + run_index,
        )
        null_result = learn_distributional_suffix_interface(null, **kwargs)
        false_split = bool(null_result.repair_events)
        false_splits += int(false_split)
        if false_split:
            false_split_records.append(
                {
                    "run_index": run_index,
                    "seed": seed + 2000 + run_index,
                    "learner_status": null_result.status,
                    "final_order": null_result.final_order,
                    "repair_event_count": len(null_result.repair_events),
                }
            )
        null_certifications += int(
            null_result.certified and null_result.final_order == 1
        )

    drift_runs = min(runs, 50)
    drift_detections = 0
    drift_miss_records = []
    half_count = episode_count // 2
    for run_index in range(drift_runs):
        first = generate_second_order_episodes(
            probability_one_if_older_zero=0.8,
            probability_one_if_older_one=0.2,
            episode_count=half_count,
            episode_length=episode_length,
            seed=seed + 3000 + 2 * run_index,
        )
        second = generate_second_order_episodes(
            probability_one_if_older_zero=0.2,
            probability_one_if_older_one=0.8,
            episode_count=episode_count - half_count,
            episode_length=episode_length,
            seed=seed + 3001 + 2 * run_index,
        )
        assessment = assess_episode_stationarity(
            first + second,
            2,
            delta=delta,
            family_cell_budget=family_cell_budget,
            merge_tolerance=merge_tolerance,
            split_tolerance=split_tolerance,
            minimum_samples_per_half=minimum_samples_per_child,
        )
        detected = assessment.status == "distribution_shift_detected"
        drift_detections += int(detected)
        if not detected:
            drift_miss_records.append(
                {
                    "run_index": run_index,
                    "first_seed": seed + 3000 + 2 * run_index,
                    "second_seed": seed + 3001 + 2 * run_index,
                    "status": assessment.status,
                }
            )

    return {
        "run_count": runs,
        "primary_recovery": {
            "successes": recovered,
            "trials": runs,
            "rate": recovered / runs,
            "wilson_95_interval": _wilson_interval(recovered, runs),
            "safe_abstentions": safe_abstentions,
            "unsafe_certifications": unsafe_certifications,
            "nonrecovery_records": nonrecovery_records,
        },
        "selected_interval_coverage": {
            "successes": selected_interval_coverage,
            "trials": runs,
            "rate": selected_interval_coverage / runs,
            "wilson_95_interval": _wilson_interval(selected_interval_coverage, runs),
            "boundary": "empirical coverage of selected context intervals, not a replacement for the Hoeffding guarantee",
            "failure_records": interval_failure_records,
        },
        "first_order_false_splits": {
            "events": false_splits,
            "trials": runs,
            "rate": false_splits / runs,
            "wilson_95_interval": _wilson_interval(false_splits, runs),
            "failure_records": false_split_records,
        },
        "first_order_correct_certifications": {
            "successes": null_certifications,
            "trials": runs,
            "rate": null_certifications / runs,
            "wilson_95_interval": _wilson_interval(null_certifications, runs),
        },
        "drift_detection": {
            "successes": drift_detections,
            "trials": drift_runs,
            "rate": drift_detections / drift_runs,
            "wilson_95_interval": _wilson_interval(drift_detections, drift_runs),
            "miss_records": drift_miss_records,
        },
        "episodes_to_primary_certification": {
            "values": episodes_to_certify,
            "minimum": min(episodes_to_certify) if episodes_to_certify else None,
            "median": median(episodes_to_certify) if episodes_to_certify else None,
            "maximum": max(episodes_to_certify) if episodes_to_certify else None,
        },
        "seed_rule": {
            "base_seed": seed,
            "primary_offset": 1000,
            "null_offset": 2000,
            "drift_offset": 3000,
        },
    }


def run_stochastic_experiment(
    *,
    seed: int = 20260802,
    episode_count: int = 256,
    episode_length: int = 128,
    heldout_episode_count: int = 64,
    maximum_order: int = 3,
    delta: float = 0.01,
    family_cell_budget: int = 256,
    merge_tolerance: float = 0.12,
    split_tolerance: float = 0.25,
    minimum_samples_per_child: int = 64,
    monte_carlo_runs: int = 100,
    project_root: Path | None = None,
) -> dict[str, object]:
    if episode_count < 32 or episode_count % 2:
        raise ValueError("episode_count must be an even integer of at least 32")
    if episode_length < 16:
        raise ValueError("episode_length must be at least 16")
    if heldout_episode_count < 1:
        raise ValueError("heldout_episode_count must be positive")
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]

    primary, primary_learning = _primary_case(
        seed=seed,
        episode_count=episode_count,
        episode_length=episode_length,
        heldout_episode_count=heldout_episode_count,
        maximum_order=maximum_order,
        delta=delta,
        family_cell_budget=family_cell_budget,
        merge_tolerance=merge_tolerance,
        split_tolerance=split_tolerance,
        minimum_samples_per_child=minimum_samples_per_child,
    )
    controls = _controls(
        seed=seed,
        episode_count=episode_count,
        episode_length=episode_length,
        maximum_order=maximum_order,
        delta=delta,
        family_cell_budget=family_cell_budget,
        merge_tolerance=merge_tolerance,
        split_tolerance=split_tolerance,
        minimum_samples_per_child=minimum_samples_per_child,
    )
    sensitivity = _sensitivity(
        seed=seed,
        episode_count=episode_count,
        episode_length=episode_length,
        maximum_order=maximum_order,
        delta=delta,
        family_cell_budget=family_cell_budget,
        merge_tolerance=merge_tolerance,
        split_tolerance=split_tolerance,
        minimum_samples_per_child=minimum_samples_per_child,
    )
    monte_carlo = _monte_carlo(
        runs=monte_carlo_runs,
        seed=seed,
        episode_count=episode_count,
        episode_length=episode_length,
        maximum_order=maximum_order,
        delta=delta,
        family_cell_budget=family_cell_budget,
        merge_tolerance=merge_tolerance,
        split_tolerance=split_tolerance,
        minimum_samples_per_child=minimum_samples_per_child,
    )
    primary_result = primary["result"]
    primary_baselines = primary["baselines"]
    verification_checks = {
        "learner_has_no_hidden_state_access": not primary["hidden_environment"][
            "learner_hidden_state_access"
        ],
        "confidence_family_budget_respected": primary["confidence_contract"][
            "cell_budget_respected"
        ],
        "primary_memory_repair_is_one_to_two": primary_result[
            "memory_repair_trace"
        ] == (1, 2),
        "primary_stationarity_certified": primary["stationarity_gate"]["status"]
        == "stationary_within_tolerance",
        "primary_predictive_partition_has_two_blocks": primary_result[
            "predictive_blocks"
        ] == 2,
        "primary_predictive_residual_is_small": primary_result[
            "predictive_residual_nats"
        ] < 1e-4,
        "primary_predictive_partition_not_dynamically_closed": primary_result[
            "predictive_lumpability_residual_nats"
        ] > 0.68,
        "primary_dynamic_repair_has_four_blocks": primary_result[
            "autonomous_blocks"
        ] == 4,
        "selected_intervals_cover_primary_truth": primary["parameter_receipt"][
            "simultaneous_selected_intervals_cover_truth"
        ],
        "adaptive_log_loss_beats_current_observation": primary_result[
            "heldout_nll_nats"
        ]
        < primary_baselines["current_observation_probability"][
            "mean_negative_log_likelihood_nats"
        ],
        "adaptive_matches_fixed_order_two": abs(
            primary_result["heldout_nll_nats"]
            - primary_baselines["fixed_order_two_probability"][
                "mean_negative_log_likelihood_nats"
            ]
        )
        < 1e-15,
        "first_order_null_avoids_false_split": controls["first_order_null"][
            "passed"
        ],
        "iid_control_collapses_predictive_partition": controls[
            "iid_collapse_control"
        ]["passed"],
        "weak_signal_abstains": controls["weak_signal_uncertainty_control"][
            "passed"
        ],
        "regime_drift_rejected": controls["regime_drift_control"]["passed"],
        "monte_carlo_primary_recovery_at_least_95_percent": monte_carlo[
            "primary_recovery"
        ]["rate"]
        >= 0.95,
        "monte_carlo_primary_has_no_unsafe_certification": monte_carlo[
            "primary_recovery"
        ]["unsafe_certifications"]
        == 0,
        "monte_carlo_false_splits_zero": monte_carlo["first_order_false_splits"][
            "events"
        ]
        == 0,
        "monte_carlo_drift_detection_complete": monte_carlo["drift_detection"][
            "successes"
        ]
        == monte_carlo["drift_detection"]["trials"],
    }
    receipt = {
        "schema_version": "ts-language-foundations-stochastic-interface-v0.4",
        "experiment": {
            "name": "confidence-bounded stochastic representation repair",
            "version": "0.4.0",
            "question": (
                "Can observation-only repair distinguish stochastic variation "
                "from missing state, wait under insufficient evidence, reject "
                "distribution drift, and recover an approximately predictive but "
                "dynamically closed finite interface?"
            ),
        },
        "run": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "episode_count": episode_count,
            "episode_length": episode_length,
            "heldout_episode_count": heldout_episode_count,
            "maximum_order": maximum_order,
            "delta": delta,
            "family_cell_budget": family_cell_budget,
            "merge_tolerance": merge_tolerance,
            "split_tolerance": split_tolerance,
            "minimum_samples_per_child": minimum_samples_per_child,
            "batch_schedule": _batch_schedule(episode_count),
            "monte_carlo_runs": monte_carlo_runs,
            "python": sys.version,
            "platform": platform.platform(),
            "neural_components": [],
            "external_corpus": None,
            "tokenizer": None,
            "network_required": False,
        },
        "mathematical_contract": {
            "simultaneous_bound": (
                "for every declared Bernoulli cell i with n_i observations, "
                "|p_hat_i-p_i| <= sqrt(log(2M/delta)/(2n_i)) simultaneously "
                "with probability at least 1-delta"
            ),
            "split_certificate": (
                "if |p_hat_i-p_hat_j|-r_i-r_j > epsilon_split, then on the "
                "simultaneous coverage event |p_i-p_j| > epsilon_split"
            ),
            "merge_certificate": (
                "if |p_hat_i-p_hat_j|+r_i+r_j <= epsilon_merge, then on the "
                "simultaneous coverage event |p_i-p_j| <= epsilon_merge"
            ),
            "uncertainty_policy": (
                "when neither certificate fires, acquire the next declared batch; "
                "at budget exhaustion abstain"
            ),
            "stationarity_policy": (
                "compare context laws across chronological episode halves and "
                "reject a stationary quotient when a split certificate fires"
            ),
            "proof_boundary": (
                "the probability statement follows from Hoeffding and a finite "
                "union bound under the fixture's conditional-independence assumption; "
                "Monte Carlo is an implementation stress test, not the proof"
            ),
        },
        "provenance": {"source_hashes": source_hashes(project_root)},
        "primary_stochastic_infinite_fibre_case": primary,
        "controls": controls,
        "sensitivity": sensitivity,
        "monte_carlo": monte_carlo,
        "verification": {
            "checks": verification_checks,
            "check_count": len(verification_checks),
            "passed_count": sum(bool(value) for value in verification_checks.values()),
            "passed": all(verification_checks.values()),
        },
        "conclusion": {
            "established_in_fixture": (
                "the confidence verifier distinguishes stochasticity from missing "
                "memory, repairs order one to order two, confidence-clusters four "
                "contexts into two approximately predictive blocks, and dynamic "
                "repair restores four autonomous states"
            ),
            "negative_findings": (
                "weak signals inside the hysteresis band remain unresolved; drift "
                "invalidates stationary certification; adaptive prediction matches "
                "rather than exceeds the correctly specified fixed-order-two baseline"
            ),
            "next_falsifiable_experiment": (
                "introduce actions and let the verifier select interventions that "
                "maximise expected reduction in representation uncertainty under a budget"
            ),
            "not_established": (
                "general hidden-state identification, arbitrary dependent noise, "
                "continuous observations, language acquisition, or superiority to "
                "modern neural sequence models"
            ),
        },
    }
    return _jsonable(receipt)


def write_json(receipt: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_report(receipt: dict[str, object], path: Path) -> None:
    primary = receipt["primary_stochastic_infinite_fibre_case"]
    result = primary["result"]
    confidence = primary["confidence_contract"]
    baselines = primary["baselines"]
    controls = receipt["controls"]
    monte = receipt["monte_carlo"]
    predictive = primary["candidate_abstractions"]["confidence_predictive_partition"]
    text = f"""# Confidence-Bounded Stochastic Representation Repair — v0.4

## Question

When observations are stochastic, a differing next symbol is not itself a contradiction. When is the evidence strong enough to change the represented state, when is a merge statistically defensible, and when must the machine wait or abstain?

## Statistical contract

For at most `M={confidence['declared_family_cell_budget']}` predeclared Bernoulli cell-stage queries and family error `delta={confidence['delta']}`, each empirical probability receives radius

```text
r(n) = sqrt(log(2M/delta)/(2n)).
```

Hoeffding's inequality plus a finite union bound gives simultaneous coverage at least `1-delta`, under the fixture assumption that each context visit uses an independent Bernoulli innovation with a stationary context law.

For two cells `i,j`, the verifier computes

```text
lower = max(0, |p_hat_i-p_hat_j|-r_i-r_j),
upper = min(1, |p_hat_i-p_hat_j|+r_i+r_j).
```

It merges only when `upper <= {receipt['run']['merge_tolerance']}`, splits only when `lower > {receipt['run']['split_tolerance']}`, and otherwise requests the next declared sample batch. This hysteresis band prevents a noisy near-threshold estimate from forcing either operation.

The primary run used `{confidence['actual_primary_cell_queries']}` confidence cells. Its precomputed worst-case query family contains `{confidence['declared_worst_case_cell_queries']}` cells, within the declared budget `{confidence['declared_family_cell_budget']}`.

## Hidden infinite-fibre process

The evaluator simulates

```text
(u,v,n) -> (v,W,n+1),
P(W=1|u=0)=0.8,
P(W=1|u=1)=0.2,
```

but supplies only binary observation episodes. The unbounded coordinate `n` is never passed to the learner and is irrelevant to the registered transition laws.

At order one, conditioning on one older observation exposes a certified difference, so memory repairs from `{result['memory_repair_trace'][0]}` to `{result['memory_repair_trace'][-1]}`. At order two, one further historical coordinate is certified irrelevant within tolerance.

The four observed contexts have estimated next-one probabilities:

| Context | Estimate | Simultaneous interval | Evaluator truth |
|---|---:|---:|---:|
"""
    for record in primary["parameter_receipt"]["contexts"]:
        text += (
            f"| `{''.join(map(str, record['context']))}` | "
            f"{record['empirical_probability_one']:.4f} | "
            f"[{record['lower_probability_one']:.4f}, {record['upper_probability_one']:.4f}] | "
            f"{record['true_probability_one_evaluator_only']:.1f} |\n"
        )
    text += f"""

All four selected intervals contain their evaluator truths. Parameter RMSE is `{primary['parameter_receipt']['root_mean_square_parameter_error']:.6f}`.

## Predictive repair versus autonomous repair

Confidence-bounded predictive equivalence groups

```text
{{00,01}} and {{10,11}}.
```

This two-block representation has empirical predictive residual `{predictive['predictive_residual_nats']:.8f}` nats, but lumpability residual `{predictive['lumpability_residual_nats']:.8f}` nats. Dynamic repair restores four context states. Thus stochastic sampling preserves the central distinction: approximate sufficiency for the selected next-symbol law does not imply autonomous macro-dynamics.

Raw empirical probabilities are almost never exactly equal, so exact equality would spuriously return four predictive blocks. The confidence relation is explicitly tolerance-dependent and carries a finite-sample certificate.

## Held-out prediction

| Model | Accuracy | NLL (nats) | Brier |
|---|---:|---:|---:|
| One state | {baselines['one_state_probability']['accuracy']:.4f} | {baselines['one_state_probability']['mean_negative_log_likelihood_nats']:.4f} | {baselines['one_state_probability']['mean_brier_score']:.4f} |
| Current observation | {baselines['current_observation_probability']['accuracy']:.4f} | {baselines['current_observation_probability']['mean_negative_log_likelihood_nats']:.4f} | {baselines['current_observation_probability']['mean_brier_score']:.4f} |
| Fixed order two | {baselines['fixed_order_two_probability']['accuracy']:.4f} | {baselines['fixed_order_two_probability']['mean_negative_log_likelihood_nats']:.4f} | {baselines['fixed_order_two_probability']['mean_brier_score']:.4f} |
| Adaptive repair | {primary['adaptive_heldout_evaluation']['accuracy']:.4f} | {primary['adaptive_heldout_evaluation']['mean_negative_log_likelihood_nats']:.4f} | {primary['adaptive_heldout_evaluation']['mean_brier_score']:.4f} |

Adaptive repair selects the correct order and matches the correctly specified fixed-order-two model. It does not beat that oracle-capacity baseline.

## Hostile controls

- First-order null: certified order one with no false split — `{controls['first_order_null']['passed']}`.
- IID collapse: the two observed symbols merge into one predictive block — `{controls['iid_collapse_control']['passed']}`.
- Weak signal `0.6 versus 0.4`: remained unresolved at the sample budget rather than forcing a decision — `{controls['weak_signal_uncertainty_control']['passed']}`.
- Regime reversal: chronological stationarity gate detected the change and rejected a stationary quotient — `{controls['regime_drift_control']['passed']}`.

## Monte Carlo receipt

- Primary recovery: `{monte['primary_recovery']['successes']}/{monte['primary_recovery']['trials']}`; Wilson 95% interval `{monte['primary_recovery']['wilson_95_interval']}`.
- Primary nonrecoveries: `{monte['primary_recovery']['safe_abstentions']}` safe abstentions and `{monte['primary_recovery']['unsafe_certifications']}` unsafe certifications. Full seeds and statuses are retained in the machine receipt.
- First-order false splits: `{monte['first_order_false_splits']['events']}/{monte['first_order_false_splits']['trials']}`; Wilson 95% interval `{monte['first_order_false_splits']['wilson_95_interval']}`.
- First-order correct certifications: `{monte['first_order_correct_certifications']['successes']}/{monte['first_order_correct_certifications']['trials']}`.
- Drift detections: `{monte['drift_detection']['successes']}/{monte['drift_detection']['trials']}`.
- Selected-interval empirical coverage: `{monte['selected_interval_coverage']['successes']}/{monte['selected_interval_coverage']['trials']}`.

The Monte Carlo study is a deterministic seeded implementation stress test. The probability guarantee comes from the stated inequality and assumptions, not from observing a favourable finite run.

## Verifier results

- Checks passed: `{receipt['verification']['passed_count']}/{receipt['verification']['check_count']}`.
- Episodes admitted: `{primary['resource_vector']['episodes_admitted']}`.
- Observations admitted: `{primary['resource_vector']['observations_admitted']}`.
- Predictive blocks: `{result['predictive_blocks']}`.
- Autonomous blocks: `{result['autonomous_blocks']}`.
- Held-out nuisance counter: `{result['heldout_counter_transfer']}`.

## Boundaries and next experiment

This release establishes a confidence-bounded stochastic repair mechanism for a finite family of binary stationary Markov fixtures. It does not solve arbitrary hidden-state inference, dependence beyond the declared process, continuous observations, language acquisition, or neural scaling.

The next falsifiable extension is active repair: introduce actions and let the verifier select interventions that maximise expected reduction in representation uncertainty per unit cost.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_svg(receipt: dict[str, object], path: Path) -> None:
    primary = receipt["primary_stochastic_infinite_fibre_case"]
    result = primary["result"]
    estimates = primary["parameter_receipt"]["contexts"]
    monte = receipt["monte_carlo"]
    controls = receipt["controls"]
    sample_records = receipt["sensitivity"]["sample_budget"]
    baselines = primary["baselines"]
    adaptive = primary["adaptive_heldout_evaluation"]

    interval_svg = []
    for index, record in enumerate(estimates):
        y_value = 360 + index * 40
        x_low = 340 + 300 * record["lower_probability_one"]
        x_high = 340 + 300 * record["upper_probability_one"]
        x_point = 340 + 300 * record["empirical_probability_one"]
        context_label = "".join(map(str, record["context"]))
        interval_svg.append(
            f'<text x="295" y="{y_value + 5}" class="value">{context_label}</text>'
            f'<line x1="{x_low:.1f}" y1="{y_value}" x2="{x_high:.1f}" y2="{y_value}" stroke="#70ddff" stroke-width="6" stroke-linecap="round"/>'
            f'<circle cx="{x_point:.1f}" cy="{y_value}" r="7" fill="#f4f7fb"/>'
        )

    trace_svg = []
    for index, record in enumerate(sample_records):
        x_value = 95 + index * 112
        certified = record["certified"]
        colour = "#65e6a5" if certified else "#c58cff"
        label = "certify" if certified else ("split/wait" if record["repair_event_count"] else "wait")
        trace_svg.append(
            f'<circle cx="{x_value}" cy="675" r="31" fill="#24364d" stroke="{colour}" stroke-width="3"/>'
            f'<text x="{x_value}" y="671" text-anchor="middle" class="value">{record["episode_budget"]}</text>'
            f'<text x="{x_value}" y="692" text-anchor="middle" class="tiny">{label}</text>'
        )
        if index:
            trace_svg.append(
                f'<path d="M {x_value - 78} 675 L {x_value - 35} 675" stroke="#93abc3" stroke-width="2" marker-end="url(#arrow)"/>'
            )

    nll_values = (
        ("one state", baselines["one_state_probability"]["mean_negative_log_likelihood_nats"], "#ffbf5b"),
        ("current", baselines["current_observation_probability"]["mean_negative_log_likelihood_nats"], "#ffbf5b"),
        ("fixed k=2", baselines["fixed_order_two_probability"]["mean_negative_log_likelihood_nats"], "#65e6a5"),
        ("adaptive", adaptive["mean_negative_log_likelihood_nats"], "#70ddff"),
    )
    nll_svg = []
    for index, (label, value, colour) in enumerate(nll_values):
        x_value = 780 + index * 125
        height = 135 * value / 0.75
        y_value = 780 - height
        nll_svg.append(
            f'<rect x="{x_value}" y="{y_value:.1f}" width="82" height="{height:.1f}" rx="7" fill="{colour}" opacity="0.88"/>'
            f'<text x="{x_value + 41}" y="{y_value - 9:.1f}" text-anchor="middle" class="value">{value:.3f}</text>'
            f'<text x="{x_value + 41}" y="800" text-anchor="middle" class="small">{label}</text>'
        )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="900" viewBox="0 0 1400 900">
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#93abc3"/></marker>
  <style>
    .title{{font:700 34px system-ui,sans-serif;fill:#70ddff}} .head{{font:600 23px system-ui,sans-serif;fill:#e6eef8}}
    .body{{font:18px system-ui,sans-serif;fill:#d6e0ec}} .small{{font:15px system-ui,sans-serif;fill:#a9bdd1}}
    .tiny{{font:13px system-ui,sans-serif;fill:#cdbbe0}} .value{{font:600 16px system-ui,sans-serif;fill:#f3f7fb}}
    .panel{{fill:#102136;stroke:#315a7d;stroke-width:2}} .node{{fill:#194563;stroke:#70ddff;stroke-width:3}}
  </style>
</defs>
<rect width="1400" height="900" fill="#071321"/>
<text x="48" y="58" class="title">Statistical repair — knowing when evidence can change the state</text>

<rect x="40" y="90" width="1320" height="170" rx="18" class="panel"/>
<text x="68" y="130" class="head">Confidence-bounded repair pipeline</text>
<circle cx="175" cy="192" r="44" class="node"/><text x="175" y="187" text-anchor="middle" class="value">bits only</text><text x="175" y="208" text-anchor="middle" class="tiny">no hidden state</text>
<path d="M 223 192 L 315 192" stroke="#93abc3" stroke-width="3" marker-end="url(#arrow)"/>
<circle cx="375" cy="192" r="52" class="node"/><text x="375" y="187" text-anchor="middle" class="value">CI verifier</text><text x="375" y="208" text-anchor="middle" class="tiny">merge / split / wait</text>
<path d="M 431 192 L 525 192" stroke="#93abc3" stroke-width="3" marker-end="url(#arrow)"/>
<circle cx="590" cy="192" r="52" class="node"/><text x="590" y="187" text-anchor="middle" class="value">k: 1 → 2</text><text x="590" y="208" text-anchor="middle" class="tiny">memory repair</text>
<path d="M 646 192 L 740 192" stroke="#93abc3" stroke-width="3" marker-end="url(#arrow)"/>
<circle cx="805" cy="192" r="52" fill="#402f55" stroke="#c58cff" stroke-width="3"/><text x="805" y="187" text-anchor="middle" class="value">2 blocks</text><text x="805" y="208" text-anchor="middle" class="tiny">predictive</text>
<path d="M 861 192 L 955 192" stroke="#93abc3" stroke-width="3" marker-end="url(#arrow)"/>
<circle cx="1020" cy="192" r="52" fill="#184b3d" stroke="#65e6a5" stroke-width="3"/><text x="1020" y="187" text-anchor="middle" class="value">4 states</text><text x="1020" y="208" text-anchor="middle" class="tiny">autonomous</text>
<path d="M 1076 192 L 1170 192" stroke="#93abc3" stroke-width="3" marker-end="url(#arrow)"/>
<circle cx="1230" cy="192" r="52" fill="#24364d" stroke="#ffbf5b" stroke-width="3"/><text x="1230" y="187" text-anchor="middle" class="value">receipt</text><text x="1230" y="208" text-anchor="middle" class="tiny">or abstain</text>

<rect x="40" y="280" width="650" height="270" rx="18" class="panel"/>
<text x="68" y="320" class="head">Primary simultaneous intervals for P(next=1)</text>
<line x1="340" y1="340" x2="340" y2="500" stroke="#66829c"/><line x1="640" y1="340" x2="640" y2="500" stroke="#66829c"/>
<text x="340" y="522" text-anchor="middle" class="tiny">0</text><text x="640" y="522" text-anchor="middle" class="tiny">1</text>
{''.join(interval_svg)}
<text x="68" y="505" class="small">{{00,01}} ≈ 0.8; {{10,11}} ≈ 0.2</text>

<rect x="710" y="280" width="650" height="270" rx="18" class="panel"/>
<text x="740" y="320" class="head">Hostile controls</text>
<text x="740" y="365" class="body" fill="#65e6a5">✓ first-order null: no false split</text>
<text x="740" y="405" class="body" fill="#65e6a5">✓ IID process: collapse to one predictive block</text>
<text x="740" y="445" class="body" fill="#ffbf5b">○ weak signal: wait at sample-budget boundary</text>
<text x="740" y="485" class="body" fill="#c58cff">× regime reversal: stationary quotient rejected</text>
<text x="740" y="525" class="small">Monte Carlo: {monte['primary_recovery']['successes']}/{monte['primary_recovery']['trials']} recoveries · {monte['first_order_false_splits']['events']}/{monte['first_order_false_splits']['trials']} false splits</text>

<rect x="40" y="570" width="650" height="280" rx="18" class="panel"/>
<text x="68" y="612" class="head">Declared sample schedule</text>
{''.join(trace_svg)}
<text x="68" y="744" class="body">Split evidence arrives early; merge certification needs more samples.</text>
<text x="68" y="782" class="small">Family confidence cells: {primary['confidence_contract']['actual_primary_cell_queries']}/{primary['confidence_contract']['declared_family_cell_budget']}</text>
<text x="68" y="818" class="small">Uncertainty is an operational state—not silently rounded into certainty.</text>

<rect x="710" y="570" width="650" height="280" rx="18" class="panel"/>
<text x="740" y="612" class="head">Held-out negative log likelihood (lower is better)</text>
<line x1="755" y1="780" x2="1305" y2="780" stroke="#66829c" stroke-width="2"/>
{''.join(nll_svg)}
<text x="740" y="830" class="small">Adaptive repair matches the correctly specified fixed-k=2 baseline.</text>
<text x="48" y="882" class="small">Finite stochastic fixtures with declared assumptions and error budget. Monte Carlo tests implementation; inequalities carry the guarantee.</text>
</svg>'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
