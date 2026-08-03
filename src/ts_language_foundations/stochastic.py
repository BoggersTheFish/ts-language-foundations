"""Confidence-bounded repair of finite observation-history interfaces.

The exact learner in :mod:`ts_language_foundations.online` treats two distinct
next symbols as a contradiction.  That is unsuitable for a stochastic world.
Here the verifier compares conditional Bernoulli laws with simultaneous
Hoeffding confidence bounds.

For a declared finite family of at most ``M`` Bernoulli cells, each empirical
mean with ``n`` samples receives radius

    sqrt(log(2 M / delta) / (2 n)).

By Hoeffding's inequality and a union bound, all declared cell means lie in
their intervals simultaneously with probability at least ``1-delta``.  The
same uniform event supports adaptive inspection of those cells.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, product
from math import log, sqrt
from random import Random
from typing import Iterable, Sequence

from .online import Context, Episode, normalize_episodes


def normalize_binary_episodes(
    episodes: Iterable[Iterable[int]],
) -> tuple[Episode, ...]:
    materialized = normalize_episodes(episodes)
    if any(symbol not in (0, 1) for episode in materialized for symbol in episode):
        raise ValueError("the Bernoulli verifier requires binary observations")
    return materialized


def hoeffding_radius(sample_count: int, delta: float, family_cell_budget: int) -> float:
    if sample_count < 1:
        return float("inf")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie strictly between zero and one")
    if family_cell_budget < 1:
        raise ValueError("family_cell_budget must be positive")
    return sqrt(log(2.0 * family_cell_budget / delta) / (2.0 * sample_count))


@dataclass(frozen=True)
class BernoulliEstimate:
    context: Context
    zero_count: int
    one_count: int
    empirical_probability_one: float
    radius: float
    lower_probability_one: float
    upper_probability_one: float

    @property
    def sample_count(self) -> int:
        return self.zero_count + self.one_count

    def to_dict(self) -> dict[str, object]:
        return {
            "context": self.context,
            "zero_count": self.zero_count,
            "one_count": self.one_count,
            "sample_count": self.sample_count,
            "empirical_probability_one": self.empirical_probability_one,
            "radius": self.radius,
            "lower_probability_one": self.lower_probability_one,
            "upper_probability_one": self.upper_probability_one,
        }


def bernoulli_context_counts(
    episodes: Sequence[Episode],
    order: int,
) -> dict[Context, Counter[int]]:
    if order < 0:
        raise ValueError("context order must be non-negative")
    counts: dict[Context, Counter[int]] = {}
    for episode in episodes:
        start = 0 if order == 0 else order - 1
        for time_index in range(start, len(episode) - 1):
            context = () if order == 0 else episode[time_index - order + 1 : time_index + 1]
            counts.setdefault(context, Counter())[episode[time_index + 1]] += 1
    return counts


def estimate_bernoulli_cells(
    episodes: Sequence[Episode],
    order: int,
    *,
    delta: float,
    family_cell_budget: int,
) -> dict[Context, BernoulliEstimate]:
    counts = bernoulli_context_counts(episodes, order)
    estimates = {}
    for context, target_counts in counts.items():
        zero_count = target_counts[0]
        one_count = target_counts[1]
        sample_count = zero_count + one_count
        probability = one_count / sample_count
        radius = hoeffding_radius(sample_count, delta, family_cell_budget)
        estimates[context] = BernoulliEstimate(
            context=context,
            zero_count=zero_count,
            one_count=one_count,
            empirical_probability_one=probability,
            radius=radius,
            lower_probability_one=max(0.0, probability - radius),
            upper_probability_one=min(1.0, probability + radius),
        )
    return estimates


@dataclass(frozen=True)
class DistributionComparison:
    label: str
    left: BernoulliEstimate | None
    right: BernoulliEstimate | None
    empirical_difference: float | None
    lower_difference: float | None
    upper_difference: float | None
    decision: str

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "left": None if self.left is None else self.left.to_dict(),
            "right": None if self.right is None else self.right.to_dict(),
            "empirical_difference": self.empirical_difference,
            "lower_difference": self.lower_difference,
            "upper_difference": self.upper_difference,
            "decision": self.decision,
        }


def compare_bernoulli_estimates(
    left: BernoulliEstimate | None,
    right: BernoulliEstimate | None,
    *,
    label: str,
    merge_tolerance: float,
    split_tolerance: float,
    minimum_samples: int,
) -> DistributionComparison:
    if not 0.0 <= merge_tolerance < split_tolerance <= 1.0:
        raise ValueError("require 0 <= merge_tolerance < split_tolerance <= 1")
    if minimum_samples < 1:
        raise ValueError("minimum_samples must be positive")
    if (
        left is None
        or right is None
        or left.sample_count < minimum_samples
        or right.sample_count < minimum_samples
    ):
        return DistributionComparison(label, left, right, None, None, None, "insufficient")
    empirical = abs(
        left.empirical_probability_one - right.empirical_probability_one
    )
    uncertainty = left.radius + right.radius
    lower = max(0.0, empirical - uncertainty)
    upper = min(1.0, empirical + uncertainty)
    if lower > split_tolerance:
        decision = "separated"
    elif upper <= merge_tolerance:
        decision = "equivalent_within_tolerance"
    else:
        decision = "uncertain"
    return DistributionComparison(label, left, right, empirical, lower, upper, decision)


@dataclass(frozen=True)
class MemoryAssessment:
    order: int
    status: str
    comparisons: tuple[DistributionComparison, ...]
    split_witness: DistributionComparison | None
    all_extension_pairs_supported: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "order": self.order,
            "status": self.status,
            "comparisons": tuple(comparison.to_dict() for comparison in self.comparisons),
            "comparison_count": len(self.comparisons),
            "split_witness": (
                None if self.split_witness is None else self.split_witness.to_dict()
            ),
            "all_extension_pairs_supported": self.all_extension_pairs_supported,
        }


def assess_suffix_memory(
    episodes: Iterable[Iterable[int]],
    order: int,
    *,
    delta: float,
    family_cell_budget: int,
    merge_tolerance: float,
    split_tolerance: float,
    minimum_samples_per_child: int,
) -> MemoryAssessment:
    """Test whether one older symbol changes a suffix's predictive law."""

    materialized = normalize_binary_episodes(episodes)
    if order < 1:
        raise ValueError("memory order must be positive")
    child_estimates = estimate_bernoulli_cells(
        materialized,
        order + 1,
        delta=delta,
        family_cell_budget=family_cell_budget,
    )
    comparisons = []
    for parent in product((0, 1), repeat=order):
        parent_context = tuple(parent)
        left_context = (0, *parent_context)
        right_context = (1, *parent_context)
        comparisons.append(
            compare_bernoulli_estimates(
                child_estimates.get(left_context),
                child_estimates.get(right_context),
                label=f"older-bit extension of {parent_context}",
                merge_tolerance=merge_tolerance,
                split_tolerance=split_tolerance,
                minimum_samples=minimum_samples_per_child,
            )
        )
    separated = [comparison for comparison in comparisons if comparison.decision == "separated"]
    if separated:
        status = "split_required"
    elif comparisons and all(
        comparison.decision == "equivalent_within_tolerance"
        for comparison in comparisons
    ):
        status = "certified_within_tolerance"
    else:
        status = "insufficient_evidence"
    return MemoryAssessment(
        order=order,
        status=status,
        comparisons=tuple(comparisons),
        split_witness=None if not separated else separated[0],
        all_extension_pairs_supported=all(
            comparison.decision != "insufficient" for comparison in comparisons
        ),
    )


@dataclass(frozen=True)
class DistributionalLearningResult:
    initial_order: int
    final_order: int
    maximum_order: int
    status: str
    certified: bool
    episodes_admitted: int
    observations_admitted: int
    batch_schedule: tuple[int, ...]
    assessment_trace: tuple[dict[str, object], ...]
    repair_events: tuple[dict[str, object], ...]
    final_assessment: MemoryAssessment
    estimates: tuple[tuple[Context, BernoulliEstimate], ...]
    delta: float
    family_cell_budget: int
    merge_tolerance: float
    split_tolerance: float
    minimum_samples_per_child: int

    @property
    def estimate_map(self) -> dict[Context, BernoulliEstimate]:
        return dict(self.estimates)

    def predict_probability_one(self, history: Sequence[int]) -> float | None:
        if not self.certified or len(history) < self.final_order:
            return None
        context = tuple(int(symbol) for symbol in history[-self.final_order :])
        estimate = self.estimate_map.get(context)
        return None if estimate is None else estimate.empirical_probability_one

    def to_dict(self) -> dict[str, object]:
        return {
            "initial_order": self.initial_order,
            "final_order": self.final_order,
            "maximum_order": self.maximum_order,
            "status": self.status,
            "certified": self.certified,
            "episodes_admitted": self.episodes_admitted,
            "observations_admitted": self.observations_admitted,
            "batch_schedule": self.batch_schedule,
            "assessment_trace": self.assessment_trace,
            "assessment_count": len(self.assessment_trace),
            "repair_events": self.repair_events,
            "repair_event_count": len(self.repair_events),
            "final_assessment": self.final_assessment.to_dict(),
            "estimates": tuple(estimate.to_dict() for _, estimate in self.estimates),
            "delta": self.delta,
            "family_cell_budget": self.family_cell_budget,
            "merge_tolerance": self.merge_tolerance,
            "split_tolerance": self.split_tolerance,
            "minimum_samples_per_child": self.minimum_samples_per_child,
        }


def learn_distributional_suffix_interface(
    episodes: Iterable[Iterable[int]],
    *,
    initial_order: int = 1,
    maximum_order: int = 3,
    batch_schedule: Iterable[int] = (16, 32, 64, 128, 256),
    delta: float = 0.01,
    family_cell_budget: int = 256,
    merge_tolerance: float = 0.12,
    split_tolerance: float = 0.25,
    minimum_samples_per_child: int = 64,
) -> DistributionalLearningResult:
    materialized = normalize_binary_episodes(episodes)
    if not 1 <= initial_order <= maximum_order:
        raise ValueError("require 1 <= initial_order <= maximum_order")
    schedule = tuple(sorted(set(int(value) for value in batch_schedule)))
    if not schedule or schedule[0] < 1 or schedule[-1] > len(materialized):
        raise ValueError("batch schedule must lie inside the supplied episode count")
    if schedule[-1] != len(materialized):
        schedule = (*schedule, len(materialized))

    order = initial_order
    assessment_trace: list[dict[str, object]] = []
    repair_events: list[dict[str, object]] = []
    final_assessment: MemoryAssessment | None = None
    admitted_count = 0
    status = "sample_budget_exhausted"

    for admitted_count in schedule:
        admitted = materialized[:admitted_count]
        while True:
            assessment = assess_suffix_memory(
                admitted,
                order,
                delta=delta,
                family_cell_budget=family_cell_budget,
                merge_tolerance=merge_tolerance,
                split_tolerance=split_tolerance,
                minimum_samples_per_child=minimum_samples_per_child,
            )
            final_assessment = assessment
            assessment_trace.append(
                {
                    "episodes_admitted": admitted_count,
                    "observation_count": sum(len(episode) for episode in admitted),
                    "order": order,
                    "status": assessment.status,
                    "assessment": assessment.to_dict(),
                }
            )
            if assessment.status == "split_required":
                if order >= maximum_order:
                    status = "memory_budget_exhausted"
                    break
                repair_events.append(
                    {
                        "event_index": len(repair_events),
                        "episodes_admitted": admitted_count,
                        "order_before": order,
                        "order_after": order + 1,
                        "repair": "register one additional observation-history coordinate",
                        "confidence_witness": assessment.split_witness.to_dict(),
                    }
                )
                order += 1
                continue
            if assessment.status == "certified_within_tolerance":
                status = "certified_stationary_memory_within_tolerance"
            break
        if status in {
            "memory_budget_exhausted",
            "certified_stationary_memory_within_tolerance",
        }:
            break

    assert final_assessment is not None
    admitted = materialized[:admitted_count]
    estimates = estimate_bernoulli_cells(
        admitted,
        order,
        delta=delta,
        family_cell_budget=family_cell_budget,
    )
    certified = status == "certified_stationary_memory_within_tolerance"
    return DistributionalLearningResult(
        initial_order=initial_order,
        final_order=order,
        maximum_order=maximum_order,
        status=status,
        certified=certified,
        episodes_admitted=admitted_count,
        observations_admitted=sum(len(episode) for episode in admitted),
        batch_schedule=schedule,
        assessment_trace=tuple(assessment_trace),
        repair_events=tuple(repair_events),
        final_assessment=final_assessment,
        estimates=tuple(sorted(estimates.items())),
        delta=delta,
        family_cell_budget=family_cell_budget,
        merge_tolerance=merge_tolerance,
        split_tolerance=split_tolerance,
        minimum_samples_per_child=minimum_samples_per_child,
    )


@dataclass(frozen=True)
class ConfidencePartitionResult:
    contexts: tuple[Context, ...]
    blocks: tuple[tuple[int, ...], ...]
    comparisons: tuple[DistributionComparison, ...]
    status: str
    certified: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "contexts": self.contexts,
            "blocks": self.blocks,
            "block_contexts": tuple(
                tuple(self.contexts[index] for index in block) for block in self.blocks
            ),
            "comparisons": tuple(comparison.to_dict() for comparison in self.comparisons),
            "comparison_count": len(self.comparisons),
            "status": self.status,
            "certified": self.certified,
        }


def confidence_predictive_partition(
    estimates: dict[Context, BernoulliEstimate],
    *,
    merge_tolerance: float,
    split_tolerance: float,
    minimum_samples: int,
) -> ConfidencePartitionResult:
    contexts = tuple(sorted(estimates))
    comparisons = []
    decisions: dict[tuple[int, int], str] = {}
    for left_index, right_index in combinations(range(len(contexts)), 2):
        comparison = compare_bernoulli_estimates(
            estimates[contexts[left_index]],
            estimates[contexts[right_index]],
            label=f"{contexts[left_index]} versus {contexts[right_index]}",
            merge_tolerance=merge_tolerance,
            split_tolerance=split_tolerance,
            minimum_samples=minimum_samples,
        )
        comparisons.append(comparison)
        decisions[(left_index, right_index)] = comparison.decision

    if any(comparison.decision in {"uncertain", "insufficient"} for comparison in comparisons):
        return ConfidencePartitionResult(
            contexts,
            (),
            tuple(comparisons),
            "pairwise_evidence_incomplete",
            False,
        )

    parents = list(range(len(contexts)))

    def find(value: int) -> int:
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for (left_index, right_index), decision in decisions.items():
        if decision == "equivalent_within_tolerance":
            union(left_index, right_index)
    groups: dict[int, list[int]] = {}
    for index in range(len(contexts)):
        groups.setdefault(find(index), []).append(index)
    blocks = tuple(sorted((tuple(group) for group in groups.values()), key=lambda block: block[0]))

    coherent = True
    block_of = {index: block_index for block_index, block in enumerate(blocks) for index in block}
    for (left_index, right_index), decision in decisions.items():
        same_block = block_of[left_index] == block_of[right_index]
        if same_block != (decision == "equivalent_within_tolerance"):
            coherent = False
            break
    return ConfidencePartitionResult(
        contexts,
        blocks if coherent else (),
        tuple(comparisons),
        "certified_pairwise_partition" if coherent else "nontransitive_equivalence_evidence",
        coherent,
    )


@dataclass(frozen=True)
class StationarityAssessment:
    status: str
    comparisons: tuple[DistributionComparison, ...]
    change_witness: DistributionComparison | None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "comparisons": tuple(comparison.to_dict() for comparison in self.comparisons),
            "comparison_count": len(self.comparisons),
            "change_witness": (
                None if self.change_witness is None else self.change_witness.to_dict()
            ),
        }


def assess_episode_stationarity(
    episodes: Iterable[Iterable[int]],
    order: int,
    *,
    delta: float,
    family_cell_budget: int,
    merge_tolerance: float,
    split_tolerance: float,
    minimum_samples_per_half: int,
) -> StationarityAssessment:
    materialized = normalize_binary_episodes(episodes)
    midpoint = len(materialized) // 2
    if midpoint < 1 or midpoint == len(materialized):
        raise ValueError("stationarity check requires at least two episodes")
    left = estimate_bernoulli_cells(
        materialized[:midpoint], order, delta=delta, family_cell_budget=family_cell_budget
    )
    right = estimate_bernoulli_cells(
        materialized[midpoint:], order, delta=delta, family_cell_budget=family_cell_budget
    )
    comparisons = []
    for context in product((0, 1), repeat=order):
        context = tuple(context)
        comparisons.append(
            compare_bernoulli_estimates(
                left.get(context),
                right.get(context),
                label=f"first half versus second half at {context}",
                merge_tolerance=merge_tolerance,
                split_tolerance=split_tolerance,
                minimum_samples=minimum_samples_per_half,
            )
        )
    changes = [comparison for comparison in comparisons if comparison.decision == "separated"]
    if changes:
        status = "distribution_shift_detected"
    elif all(
        comparison.decision == "equivalent_within_tolerance"
        for comparison in comparisons
    ):
        status = "stationary_within_tolerance"
    else:
        status = "stationarity_uncertain"
    return StationarityAssessment(
        status,
        tuple(comparisons),
        None if not changes else changes[0],
    )


def generate_second_order_episodes(
    *,
    probability_one_if_older_zero: float,
    probability_one_if_older_one: float,
    episode_count: int,
    episode_length: int,
    seed: int,
    initial_counter: int = 0,
) -> tuple[Episode, ...]:
    """Generate observations from ``(u,v,n)->(v,W,n+1)``.

    ``P(W=1 | u=0)=probability_one_if_older_zero`` and likewise for
    ``u=1``.  The unbounded counter is simulated but never returned.
    """

    probabilities = (probability_one_if_older_zero, probability_one_if_older_one)
    if any(not 0.0 < value < 1.0 for value in probabilities):
        raise ValueError("transition probabilities must lie strictly inside (0,1)")
    if episode_count < 1 or episode_length < 4 or initial_counter < 0:
        raise ValueError("invalid episode configuration")
    rng = Random(seed)
    initial_contexts = ((0, 0), (0, 1), (1, 0), (1, 1))
    episodes = []
    for episode_index in range(episode_count):
        older, current = initial_contexts[episode_index % len(initial_contexts)]
        counter = initial_counter + episode_index * episode_length
        observations = [older, current]
        while len(observations) < episode_length:
            probability_one = probabilities[older]
            nxt = int(rng.random() < probability_one)
            older, current, counter = current, nxt, counter + 1
            observations.append(nxt)
        episodes.append(tuple(observations))
    return tuple(episodes)


def generate_first_order_episodes(
    *,
    probability_one_if_current_zero: float,
    probability_one_if_current_one: float,
    episode_count: int,
    episode_length: int,
    seed: int,
) -> tuple[Episode, ...]:
    probabilities = (probability_one_if_current_zero, probability_one_if_current_one)
    if any(not 0.0 < value < 1.0 for value in probabilities):
        raise ValueError("transition probabilities must lie strictly inside (0,1)")
    if episode_count < 1 or episode_length < 4:
        raise ValueError("invalid episode configuration")
    rng = Random(seed)
    episodes = []
    for episode_index in range(episode_count):
        current = episode_index % 2
        observations = [1 - current, current]
        while len(observations) < episode_length:
            nxt = int(rng.random() < probabilities[current])
            current = nxt
            observations.append(nxt)
        episodes.append(tuple(observations))
    return tuple(episodes)


def fit_probability_table(
    episodes: Iterable[Iterable[int]],
    order: int,
    *,
    pseudocount: float = 0.5,
) -> dict[Context, float]:
    materialized = normalize_binary_episodes(episodes)
    if pseudocount < 0.0:
        raise ValueError("pseudocount must be non-negative")
    counts = bernoulli_context_counts(materialized, order)
    result = {}
    for context, target_counts in counts.items():
        result[context] = (target_counts[1] + pseudocount) / (
            target_counts[0] + target_counts[1] + 2.0 * pseudocount
        )
    return result


def evaluate_probability_table(
    probabilities: dict[Context, float],
    order: int,
    episodes: Iterable[Iterable[int]],
) -> dict[str, object]:
    materialized = normalize_binary_episodes(episodes)
    correct = 0
    total = 0
    abstained = 0
    negative_log_likelihood = 0.0
    brier = 0.0
    for episode in materialized:
        start = 0 if order == 0 else order - 1
        for time_index in range(start, len(episode) - 1):
            context = () if order == 0 else episode[time_index - order + 1 : time_index + 1]
            probability = probabilities.get(context)
            if probability is None:
                abstained += 1
                continue
            target = episode[time_index + 1]
            clipped = min(max(probability, 1e-12), 1.0 - 1e-12)
            negative_log_likelihood -= target * log(clipped) + (1 - target) * log(1 - clipped)
            brier += (probability - target) ** 2
            correct += int((probability >= 0.5) == bool(target))
            total += 1
    eligible = total + abstained
    return {
        "order": order,
        "eligible_transitions": eligible,
        "covered_transitions": total,
        "abstained": abstained,
        "coverage": 0.0 if eligible == 0 else total / eligible,
        "accuracy": None if total == 0 else correct / total,
        "mean_negative_log_likelihood_nats": (
            None if total == 0 else negative_log_likelihood / total
        ),
        "mean_brier_score": None if total == 0 else brier / total,
    }


@dataclass(frozen=True)
class CompiledStochasticInterface:
    contexts: tuple[Context, ...]
    transition_rows: tuple[tuple[Fraction, ...], ...]
    predictive_laws: tuple[tuple[Fraction, ...], ...]
    support_counts: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "contexts": self.contexts,
            "transition_rows": self.transition_rows,
            "predictive_laws": self.predictive_laws,
            "support_counts": self.support_counts,
        }


def compile_stochastic_interface(
    result: DistributionalLearningResult,
) -> CompiledStochasticInterface:
    if not result.certified:
        raise ValueError("cannot compile an uncertified distributional interface")
    estimates = result.estimate_map
    contexts = tuple(sorted(estimates))
    expected_contexts = tuple(product((0, 1), repeat=result.final_order))
    if contexts != expected_contexts:
        raise ValueError("certified support does not contain every binary context")
    indices = {context: index for index, context in enumerate(contexts)}
    rows = []
    laws = []
    supports = []
    for context in contexts:
        estimate = estimates[context]
        sample_count = estimate.sample_count
        probability_one = Fraction(estimate.one_count, sample_count)
        row = [Fraction(0) for _ in contexts]
        zero_successor = (*context[1:], 0)
        one_successor = (*context[1:], 1)
        row[indices[zero_successor]] += 1 - probability_one
        row[indices[one_successor]] += probability_one
        rows.append(tuple(row))
        laws.append((1 - probability_one, probability_one))
        supports.append(sample_count)
    return CompiledStochasticInterface(
        contexts,
        tuple(rows),
        tuple(laws),
        tuple(supports),
    )
