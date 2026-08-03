"""Active, verifier-authorised repair for controlled stochastic interfaces.

The proposal policy and the statistical authority are deliberately separate.
An information-gain heuristic chooses an action before the next observation;
an anytime finite-family Hoeffding certificate is the only component allowed to
split, merge, or certify a representation.

The controlled fixture has hidden state ``(u, v, n)`` and visible binary
history.  The nuisance coordinate ``n`` is unbounded and never supplied to the
learner.  The baseline action has a context-independent Bernoulli law.  The
probe action reveals whether the older visible bit ``u`` must be registered.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, product
from math import exp, log, sqrt
from random import Random
from typing import Iterable, Mapping, Sequence

from .repair import Partition, lumpability_residual, predictive_residual
from .stochastic import (
    BernoulliEstimate,
    DistributionComparison,
    compare_bernoulli_estimates,
    hoeffding_radius,
)

Action = str
Context = tuple[int, ...]


def _binary_entropy(probability: float) -> float:
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return -probability * log(probability) - (1.0 - probability) * log(
        1.0 - probability
    )


def _validate_binary_context(context: Sequence[int]) -> Context:
    result = tuple(int(value) for value in context)
    if not result or any(value not in (0, 1) for value in result):
        raise ValueError("contexts must be non-empty binary sequences")
    return result


@dataclass(frozen=True)
class ControlledWorld:
    """Second-order binary world with a passive and an informative action."""

    actions: tuple[Action, ...] = ("baseline", "probe")
    baseline_probability_one: float = 0.5
    probe_probability_one_if_older_zero: float = 0.85
    probe_probability_one_if_older_one: float = 0.15
    probe_action: Action = "probe"
    drift_step: int | None = None

    def __post_init__(self) -> None:
        if len(set(self.actions)) != len(self.actions) or not self.actions:
            raise ValueError("actions must be non-empty and unique")
        if self.probe_action not in self.actions:
            raise ValueError("probe action must occur in actions")
        probabilities = (
            self.baseline_probability_one,
            self.probe_probability_one_if_older_zero,
            self.probe_probability_one_if_older_one,
        )
        if any(not 0.0 < value < 1.0 for value in probabilities):
            raise ValueError("controlled probabilities must lie inside (0,1)")
        if self.drift_step is not None and self.drift_step < 1:
            raise ValueError("drift_step must be positive")

    def probability_one(self, older_bit: int, action: Action, step: int) -> float:
        if older_bit not in (0, 1) or action not in self.actions:
            raise ValueError("invalid controlled transition query")
        if action != self.probe_action:
            return self.baseline_probability_one
        high = self.probe_probability_one_if_older_zero
        low = self.probe_probability_one_if_older_one
        if self.drift_step is not None and step >= self.drift_step:
            high, low = low, high
        return high if older_bit == 0 else low

    @classmethod
    def null(cls, *, actions: tuple[Action, ...] = ("baseline", "probe")) -> "ControlledWorld":
        return cls(
            actions=actions,
            baseline_probability_one=0.5,
            probe_probability_one_if_older_zero=0.5,
            probe_probability_one_if_older_one=0.5,
            probe_action="probe",
        )


@dataclass(frozen=True)
class Interaction:
    """One learner-visible action/observation record."""

    history: Context
    action: Action
    outcome: int
    cost: float
    episode_index: int
    step: int

    def __post_init__(self) -> None:
        _validate_binary_context(self.history)
        if self.outcome not in (0, 1) or self.cost <= 0.0:
            raise ValueError("invalid interaction")

    def to_dict(self) -> dict[str, object]:
        return {
            "history": self.history,
            "action": self.action,
            "outcome": self.outcome,
            "cost": self.cost,
            "episode_index": self.episode_index,
            "step": self.step,
        }


@dataclass(frozen=True)
class ActionHypothesis:
    name: str
    informative_action: Action | None
    high_probability: float = 0.85
    low_probability: float = 0.15
    null_probability: float = 0.5

    def probability_one(self, older_bit: int, action: Action) -> float:
        if older_bit not in (0, 1):
            raise ValueError("older_bit must be binary")
        if action != self.informative_action:
            return self.null_probability
        return self.high_probability if older_bit == 0 else self.low_probability


class BayesianBelief:
    """A proposal-only belief state; it has no certification authority."""

    def __init__(self, hypotheses: Sequence[ActionHypothesis]) -> None:
        if not hypotheses or len({item.name for item in hypotheses}) != len(hypotheses):
            raise ValueError("hypotheses must be non-empty and uniquely named")
        self.hypotheses = tuple(hypotheses)
        initial = -log(len(self.hypotheses))
        self._log_weights = [initial for _ in self.hypotheses]

    def probabilities(self) -> tuple[float, ...]:
        maximum = max(self._log_weights)
        unnormalised = [exp(value - maximum) for value in self._log_weights]
        total = sum(unnormalised)
        return tuple(value / total for value in unnormalised)

    def expected_information_gain(self, older_bit: int, action: Action) -> float:
        weights = self.probabilities()
        laws = tuple(
            hypothesis.probability_one(older_bit, action)
            for hypothesis in self.hypotheses
        )
        mixture = sum(weight * law for weight, law in zip(weights, laws, strict=True))
        return max(
            0.0,
            _binary_entropy(mixture)
            - sum(
                weight * _binary_entropy(law)
                for weight, law in zip(weights, laws, strict=True)
            ),
        )

    def expected_extension_gap(self, action: Action) -> float:
        """Posterior mean magnitude of the older-bit response contrast."""

        return sum(
            weight
            * abs(
                hypothesis.probability_one(0, action)
                - hypothesis.probability_one(1, action)
            )
            for weight, hypothesis in zip(
                self.probabilities(), self.hypotheses, strict=True
            )
        )

    def update(self, older_bit: int, action: Action, outcome: int) -> None:
        if outcome not in (0, 1):
            raise ValueError("outcome must be binary")
        for index, hypothesis in enumerate(self.hypotheses):
            probability = hypothesis.probability_one(older_bit, action)
            likelihood = probability if outcome else 1.0 - probability
            self._log_weights[index] += log(max(likelihood, 1e-300))
        maximum = max(self._log_weights)
        self._log_weights = [value - maximum for value in self._log_weights]

    def to_dict(self) -> dict[str, object]:
        probabilities = self.probabilities()
        return {
            "hypotheses": tuple(
                {
                    "name": hypothesis.name,
                    "informative_action": hypothesis.informative_action,
                    "probability": probability,
                }
                for hypothesis, probability in zip(
                    self.hypotheses, probabilities, strict=True
                )
            )
        }


@dataclass(frozen=True)
class ActionMemoryAssessment:
    order: int
    status: str
    actions: tuple[Action, ...]
    action_statuses: tuple[tuple[Action, str], ...]
    comparisons: tuple[DistributionComparison, ...]
    split_witness: DistributionComparison | None

    @property
    def action_status_map(self) -> dict[Action, str]:
        return dict(self.action_statuses)

    def to_dict(self) -> dict[str, object]:
        return {
            "order": self.order,
            "status": self.status,
            "actions": self.actions,
            "action_statuses": dict(self.action_statuses),
            "comparisons": tuple(item.to_dict() for item in self.comparisons),
            "comparison_count": len(self.comparisons),
            "split_witness": (
                None if self.split_witness is None else self.split_witness.to_dict()
            ),
        }


class ControlledCounts:
    """Incremental action/context sufficient statistics."""

    def __init__(self, actions: Sequence[Action], maximum_order: int) -> None:
        if maximum_order < 1:
            raise ValueError("maximum_order must be positive")
        if not actions or len(set(actions)) != len(actions):
            raise ValueError("actions must be non-empty and unique")
        self.actions = tuple(actions)
        self.maximum_order = maximum_order
        self._counts: dict[tuple[int, Action, Context], list[int]] = defaultdict(
            lambda: [0, 0]
        )
        self.action_counts: Counter[Action] = Counter()
        self.total_records = 0

    def add(self, interaction: Interaction) -> None:
        if interaction.action not in self.actions:
            raise ValueError("interaction uses an undeclared action")
        self.action_counts[interaction.action] += 1
        self.total_records += 1
        maximum_length = min(len(interaction.history), self.maximum_order + 1)
        for length in range(0, maximum_length + 1):
            context = () if length == 0 else interaction.history[-length:]
            self._counts[(length, interaction.action, context)][interaction.outcome] += 1

    def target_counts(self, length: int, action: Action, context: Context) -> tuple[int, int]:
        values = self._counts.get((length, action, context), [0, 0])
        return values[0], values[1]

    def sample_count(self, length: int, action: Action, context: Context) -> int:
        zero, one = self.target_counts(length, action, context)
        return zero + one

    def estimate(
        self,
        length: int,
        action: Action,
        context: Context,
        *,
        delta: float,
        family_cell_budget: int,
    ) -> BernoulliEstimate | None:
        zero, one = self.target_counts(length, action, context)
        sample_count = zero + one
        if sample_count == 0:
            return None
        probability = one / sample_count
        radius = hoeffding_radius(sample_count, delta, family_cell_budget)
        return BernoulliEstimate(
            context=context,
            zero_count=zero,
            one_count=one,
            empirical_probability_one=probability,
            radius=radius,
            lower_probability_one=max(0.0, probability - radius),
            upper_probability_one=min(1.0, probability + radius),
        )

    def estimates(
        self,
        length: int,
        *,
        delta: float,
        family_cell_budget: int,
    ) -> dict[tuple[Action, Context], BernoulliEstimate]:
        result = {}
        for action in self.actions:
            for context in product((0, 1), repeat=length):
                context = tuple(context)
                estimate = self.estimate(
                    length,
                    action,
                    context,
                    delta=delta,
                    family_cell_budget=family_cell_budget,
                )
                if estimate is not None:
                    result[(action, context)] = estimate
        return result

    def assess_memory(
        self,
        order: int,
        *,
        delta: float,
        family_cell_budget: int,
        merge_tolerance: float,
        split_tolerance: float,
        minimum_samples: int,
        actions: Sequence[Action] | None = None,
    ) -> ActionMemoryAssessment:
        if order < 1 or order > self.maximum_order:
            raise ValueError("assessment order is outside the declared family")
        selected_actions = self.actions if actions is None else tuple(actions)
        if not selected_actions or any(action not in self.actions for action in selected_actions):
            raise ValueError("assessment actions must be declared")
        comparisons: list[DistributionComparison] = []
        action_statuses: list[tuple[Action, str]] = []
        separated: list[DistributionComparison] = []
        length = order + 1
        for action in selected_actions:
            action_comparisons = []
            for parent in product((0, 1), repeat=order):
                parent_context = tuple(parent)
                left_context = (0, *parent_context)
                right_context = (1, *parent_context)
                comparison = compare_bernoulli_estimates(
                    self.estimate(
                        length,
                        action,
                        left_context,
                        delta=delta,
                        family_cell_budget=family_cell_budget,
                    ),
                    self.estimate(
                        length,
                        action,
                        right_context,
                        delta=delta,
                        family_cell_budget=family_cell_budget,
                    ),
                    label=(
                        f"action={action}; older-bit extension of {parent_context}"
                    ),
                    merge_tolerance=merge_tolerance,
                    split_tolerance=split_tolerance,
                    minimum_samples=minimum_samples,
                )
                action_comparisons.append(comparison)
                comparisons.append(comparison)
            action_separated = [
                item for item in action_comparisons if item.decision == "separated"
            ]
            separated.extend(action_separated)
            if action_separated:
                action_status = "split_required"
            elif action_comparisons and all(
                item.decision == "equivalent_within_tolerance"
                for item in action_comparisons
            ):
                action_status = "certified_within_tolerance"
            else:
                action_status = "insufficient_evidence"
            action_statuses.append((action, action_status))
        if separated:
            status = "split_required"
        elif action_statuses and all(
            value == "certified_within_tolerance" for _, value in action_statuses
        ):
            status = "certified_within_tolerance"
        else:
            status = "insufficient_evidence"
        return ActionMemoryAssessment(
            order=order,
            status=status,
            actions=selected_actions,
            action_statuses=tuple(action_statuses),
            comparisons=tuple(comparisons),
            split_witness=None if not separated else separated[0],
        )

    def maximum_empirical_extension_gap(
        self,
        order: int,
        action: Action,
        *,
        minimum_samples: int = 8,
    ) -> float:
        maximum = 0.0
        length = order + 1
        for parent in product((0, 1), repeat=order):
            parent = tuple(parent)
            left = self.target_counts(length, action, (0, *parent))
            right = self.target_counts(length, action, (1, *parent))
            left_n, right_n = sum(left), sum(right)
            if left_n < minimum_samples or right_n < minimum_samples:
                continue
            maximum = max(maximum, abs(left[1] / left_n - right[1] / right_n))
        return maximum

    def intrinsic_cell_count(self, minimum_length: int = 2) -> int:
        return sum(
            1
            for (length, _, _), counts in self._counts.items()
            if length >= minimum_length and sum(counts) > 0
        )


@dataclass(frozen=True)
class ControlledRepairConfig:
    actions: tuple[Action, ...] = ("baseline", "probe")
    action_costs: tuple[tuple[Action, float], ...] = (
        ("baseline", 1.0),
        ("probe", 2.0),
    )
    initial_order: int = 1
    maximum_order: int = 2
    maximum_steps: int = 65536
    episode_length: int = 64
    check_interval: int = 64
    delta: float = 0.01
    merge_tolerance: float = 0.2
    split_tolerance: float = 0.35
    minimum_samples: int = 32
    exploration_weight: float = 0.005
    split_evidence_weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.actions or len(set(self.actions)) != len(self.actions):
            raise ValueError("actions must be non-empty and unique")
        costs = dict(self.action_costs)
        if set(costs) != set(self.actions) or any(value <= 0.0 for value in costs.values()):
            raise ValueError("every action requires one positive cost")
        if not 1 <= self.initial_order <= self.maximum_order:
            raise ValueError("invalid memory-order bounds")
        if self.maximum_steps < 1 or self.episode_length <= self.maximum_order + 1:
            raise ValueError("invalid controlled sample budget")
        if self.check_interval < 1:
            raise ValueError("check_interval must be positive")
        if not 0.0 < self.delta < 1.0:
            raise ValueError("delta must lie inside (0,1)")
        if not 0.0 <= self.merge_tolerance < self.split_tolerance <= 1.0:
            raise ValueError("invalid split/merge hysteresis")
        if self.minimum_samples < 1:
            raise ValueError("minimum_samples must be positive")
        if self.exploration_weight < 0.0 or self.split_evidence_weight < 0.0:
            raise ValueError("policy weights must be non-negative")

    @property
    def cost_map(self) -> dict[Action, float]:
        return dict(self.action_costs)

    @property
    def anytime_family_cell_budget(self) -> int:
        cells = len(self.actions) * sum(
            2 ** (order + 1)
            for order in range(self.initial_order, self.maximum_order + 1)
        )
        return cells * self.maximum_steps

    @property
    def declared_intrinsic_cell_count(self) -> int:
        return len(self.actions) * sum(
            2 ** (order + 1)
            for order in range(self.initial_order, self.maximum_order + 1)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "actions": self.actions,
            "action_costs": dict(self.action_costs),
            "initial_order": self.initial_order,
            "maximum_order": self.maximum_order,
            "maximum_steps": self.maximum_steps,
            "episode_length": self.episode_length,
            "check_interval": self.check_interval,
            "delta": self.delta,
            "merge_tolerance": self.merge_tolerance,
            "split_tolerance": self.split_tolerance,
            "minimum_samples": self.minimum_samples,
            "exploration_weight": self.exploration_weight,
            "split_evidence_weight": self.split_evidence_weight,
            "declared_intrinsic_cell_count": self.declared_intrinsic_cell_count,
            "anytime_family_cell_budget": self.anytime_family_cell_budget,
        }


def _uncertainty_proxy_gain(sample_count: int) -> float:
    return 1.0 / sqrt(sample_count + 1.0) - 1.0 / sqrt(sample_count + 2.0)


def select_active_action(
    history: Sequence[int],
    order: int,
    counts: ControlledCounts,
    assessment: ActionMemoryAssessment,
    belief: BayesianBelief,
    config: ControlledRepairConfig,
) -> tuple[Action, dict[str, object]]:
    history = _validate_binary_context(history)
    statuses = assessment.action_status_map
    unresolved = tuple(
        action
        for action in config.actions
        if statuses.get(action) != "certified_within_tolerance"
    )
    candidates = unresolved or config.actions
    costs = config.cost_map
    scores = []
    for action in candidates:
        information_gain = (
            belief.expected_information_gain(history[-2], action) if order == 1 else 0.0
        )
        empirical_gap = (
            counts.maximum_empirical_extension_gap(order, action)
            if order == 1
            else 0.0
        )
        expected_gap = belief.expected_extension_gap(action) if order == 1 else 0.0
        child_context = history[-(order + 1) :]
        count = counts.sample_count(order + 1, action, child_context)
        radius_gain = _uncertainty_proxy_gain(count)
        score = (
            information_gain
            + config.split_evidence_weight * max(empirical_gap, expected_gap)
            + config.exploration_weight * radius_gain
        ) / costs[action]
        scores.append(
            {
                "action": action,
                "score": score,
                "information_gain_nats": information_gain,
                "empirical_extension_gap": empirical_gap,
                "posterior_expected_extension_gap": expected_gap,
                "uncertainty_proxy_gain": radius_gain,
                "cost": costs[action],
            }
        )
    selected = max(scores, key=lambda item: (item["score"], -config.actions.index(item["action"])))
    return str(selected["action"]), {
        "selected_action": selected["action"],
        "candidates": tuple(scores),
        "unresolved_actions": unresolved,
    }


def select_greedy_information_action(
    history: Sequence[int],
    belief: BayesianBelief,
    config: ControlledRepairConfig,
) -> Action:
    history = _validate_binary_context(history)
    costs = config.cost_map
    return max(
        config.actions,
        key=lambda action: (
            belief.expected_information_gain(history[-2], action) / costs[action],
            -config.actions.index(action),
        ),
    )


@dataclass(frozen=True)
class ControlledRepairResult:
    policy: str
    initial_order: int
    final_order: int
    status: str
    certified: bool
    repair_discovered: bool
    steps: int
    total_cost: float
    first_repair_step: int | None
    first_repair_cost: float | None
    action_counts: tuple[tuple[Action, int], ...]
    repair_events: tuple[dict[str, object], ...]
    assessment_trace: tuple[dict[str, object], ...]
    final_assessment: ActionMemoryAssessment
    final_estimates: tuple[tuple[Action, Context, BernoulliEstimate], ...]
    posterior: dict[str, object] | None
    confidence_contract: dict[str, object]

    @property
    def estimate_map(self) -> dict[tuple[Action, Context], BernoulliEstimate]:
        return {
            (action, context): estimate
            for action, context, estimate in self.final_estimates
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "initial_order": self.initial_order,
            "final_order": self.final_order,
            "status": self.status,
            "certified": self.certified,
            "repair_discovered": self.repair_discovered,
            "steps": self.steps,
            "total_cost": self.total_cost,
            "first_repair_step": self.first_repair_step,
            "first_repair_cost": self.first_repair_cost,
            "action_counts": dict(self.action_counts),
            "repair_events": self.repair_events,
            "repair_event_count": len(self.repair_events),
            "assessment_trace": self.assessment_trace,
            "assessment_count": len(self.assessment_trace),
            "final_assessment": self.final_assessment.to_dict(),
            "final_estimates": tuple(
                {
                    "action": action,
                    **estimate.to_dict(),
                }
                for action, _, estimate in self.final_estimates
            ),
            "posterior": self.posterior,
            "confidence_contract": self.confidence_contract,
        }


@dataclass(frozen=True)
class ControlledSimulation:
    result: ControlledRepairResult
    interactions: tuple[Interaction, ...]


def _assessment_summary(
    assessment: ActionMemoryAssessment,
    step: int,
    cost: float,
    action_counts: Mapping[Action, int],
) -> dict[str, object]:
    lower_by_action: dict[Action, float] = {action: 0.0 for action in assessment.actions}
    upper_by_action: dict[Action, float] = {action: 1.0 for action in assessment.actions}
    for comparison in assessment.comparisons:
        action = comparison.label.split(";", 1)[0].split("=", 1)[1]
        if comparison.lower_difference is not None:
            lower_by_action[action] = max(
                lower_by_action[action], comparison.lower_difference
            )
        if comparison.upper_difference is not None:
            upper_by_action[action] = min(
                upper_by_action[action], comparison.upper_difference
            )
    return {
        "step": step,
        "cost": cost,
        "order": assessment.order,
        "status": assessment.status,
        "action_statuses": dict(assessment.action_statuses),
        "action_counts": dict(action_counts),
        "maximum_lower_difference_by_action": lower_by_action,
        "minimum_upper_difference_by_action": upper_by_action,
    }


def simulate_controlled_repair(
    *,
    world: ControlledWorld,
    policy: str,
    seed: int,
    config: ControlledRepairConfig,
    hypotheses: Sequence[ActionHypothesis] | None = None,
    stop_after_first_repair: bool = False,
) -> ControlledSimulation:
    """Run one controlled repair trajectory.

    Supported policies are ``active``, ``uniform``, ``passive``, and
    ``greedy_misspecified``.  The last is a negative control with no forced
    exploration; its proposal error cannot override the verifier.
    """

    if tuple(world.actions) != tuple(config.actions):
        raise ValueError("world and repair action families must match")
    if policy not in {"active", "uniform", "passive", "greedy_misspecified"}:
        raise ValueError("unknown action policy")
    if hypotheses is None:
        informative_action = (
            "baseline" if policy == "greedy_misspecified" else world.probe_action
        )
        hypotheses = (
            ActionHypothesis("order_one_null", None),
            ActionHypothesis("one_action_reveals_memory", informative_action),
        )
    belief = BayesianBelief(hypotheses)
    counts = ControlledCounts(config.actions, config.maximum_order)
    rng = Random(seed)
    costs = config.cost_map
    family_budget = config.anytime_family_cell_budget
    order = config.initial_order
    interactions: list[Interaction] = []
    repair_events: list[dict[str, object]] = []
    assessment_trace: list[dict[str, object]] = []
    total_cost = 0.0
    first_repair_step: int | None = None
    first_repair_cost: float | None = None
    status = "sample_budget_exhausted"

    assessment = counts.assess_memory(
        order,
        delta=config.delta,
        family_cell_budget=family_budget,
        merge_tolerance=config.merge_tolerance,
        split_tolerance=config.split_tolerance,
        minimum_samples=config.minimum_samples,
    )
    initial_history_length = config.maximum_order + 1
    initial_histories = tuple(product((0, 1), repeat=initial_history_length))
    episode_index = 0
    history = list(initial_histories[episode_index % len(initial_histories)])
    steps_in_episode = 0
    available_steps = config.episode_length - initial_history_length

    for step_index in range(config.maximum_steps):
        if steps_in_episode >= available_steps:
            episode_index += 1
            history = list(initial_histories[episode_index % len(initial_histories)])
            steps_in_episode = 0
        if policy == "active":
            action, _ = select_active_action(
                history, order, counts, assessment, belief, config
            )
        elif policy == "uniform":
            action = config.actions[step_index % len(config.actions)]
        elif policy == "passive":
            action = config.actions[0]
        else:
            action = select_greedy_information_action(history, belief, config)

        older_bit = history[-2]
        probability = world.probability_one(older_bit, action, step_index)
        outcome = int(rng.random() < probability)
        interaction = Interaction(
            history=tuple(history[-initial_history_length:]),
            action=action,
            outcome=outcome,
            cost=costs[action],
            episode_index=episode_index,
            step=step_index + 1,
        )
        interactions.append(interaction)
        counts.add(interaction)
        belief.update(older_bit, action, outcome)
        total_cost += costs[action]
        history.append(outcome)
        steps_in_episode += 1

        should_check = (step_index + 1) % config.check_interval == 0
        if not should_check and step_index + 1 != config.maximum_steps:
            continue
        while True:
            assessment = counts.assess_memory(
                order,
                delta=config.delta,
                family_cell_budget=family_budget,
                merge_tolerance=config.merge_tolerance,
                split_tolerance=config.split_tolerance,
                minimum_samples=config.minimum_samples,
            )
            assessment_trace.append(
                _assessment_summary(
                    assessment,
                    step_index + 1,
                    total_cost,
                    counts.action_counts,
                )
            )
            if assessment.status == "split_required":
                if order >= config.maximum_order:
                    status = "memory_budget_exhausted"
                    break
                if first_repair_step is None:
                    first_repair_step = step_index + 1
                    first_repair_cost = total_cost
                repair_events.append(
                    {
                        "event_index": len(repair_events),
                        "step": step_index + 1,
                        "cost": total_cost,
                        "order_before": order,
                        "order_after": order + 1,
                        "repair": "register one additional observation-history coordinate",
                        "action_conditioned_witness": assessment.split_witness.to_dict(),
                    }
                )
                order += 1
                if stop_after_first_repair:
                    status = "repair_witness_certified"
                    break
                continue
            if assessment.status == "certified_within_tolerance":
                status = "certified_action_family_memory_within_tolerance"
            break
        if status in {
            "repair_witness_certified",
            "memory_budget_exhausted",
            "certified_action_family_memory_within_tolerance",
        }:
            break

    final_step = len(interactions)
    if not assessment_trace or assessment_trace[-1]["step"] != final_step:
        assessment = counts.assess_memory(
            order,
            delta=config.delta,
            family_cell_budget=family_budget,
            merge_tolerance=config.merge_tolerance,
            split_tolerance=config.split_tolerance,
            minimum_samples=config.minimum_samples,
        )
        assessment_trace.append(
            _assessment_summary(
                assessment, final_step, total_cost, counts.action_counts
            )
        )
    estimates = counts.estimates(
        order,
        delta=config.delta,
        family_cell_budget=family_budget,
    )
    confidence_contract = {
        "kind": "anytime finite-family Hoeffding union",
        "radius": "sqrt(log(2*M/delta)/(2*n))",
        "declared_intrinsic_cells": config.declared_intrinsic_cell_count,
        "maximum_sample_prefixes_per_cell": config.maximum_steps,
        "family_cell_budget": family_budget,
        "actual_intrinsic_cells_observed": counts.intrinsic_cell_count(),
        "predictable_action_boundary": (
            "each action is chosen from past learner-visible records before the "
            "current Bernoulli innovation"
        ),
        "coverage_boundary": (
            "union over every intrinsic cell and every within-cell sample prefix "
            "up to the declared budget; valid under stationary conditionally "
            "independent Bernoulli innovations"
        ),
    }
    result = ControlledRepairResult(
        policy=policy,
        initial_order=config.initial_order,
        final_order=order,
        status=status,
        certified=status == "certified_action_family_memory_within_tolerance",
        repair_discovered=first_repair_step is not None,
        steps=final_step,
        total_cost=total_cost,
        first_repair_step=first_repair_step,
        first_repair_cost=first_repair_cost,
        action_counts=tuple((action, counts.action_counts[action]) for action in config.actions),
        repair_events=tuple(repair_events),
        assessment_trace=tuple(assessment_trace),
        final_assessment=assessment,
        final_estimates=tuple(
            (action, context, estimate)
            for (action, context), estimate in sorted(estimates.items())
        ),
        posterior=belief.to_dict(),
        confidence_contract=confidence_contract,
    )
    return ControlledSimulation(result=result, interactions=tuple(interactions))


@dataclass(frozen=True)
class ActionPartitionComparison:
    left_context: Context
    right_context: Context
    action_comparisons: tuple[tuple[Action, DistributionComparison], ...]
    decision: str

    def to_dict(self) -> dict[str, object]:
        return {
            "left_context": self.left_context,
            "right_context": self.right_context,
            "action_comparisons": tuple(
                {"action": action, **comparison.to_dict()}
                for action, comparison in self.action_comparisons
            ),
            "decision": self.decision,
        }


@dataclass(frozen=True)
class ActionPartitionResult:
    contexts: tuple[Context, ...]
    blocks: Partition
    comparisons: tuple[ActionPartitionComparison, ...]
    status: str
    certified: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "contexts": self.contexts,
            "blocks": self.blocks,
            "block_contexts": tuple(
                tuple(self.contexts[index] for index in block) for block in self.blocks
            ),
            "comparisons": tuple(item.to_dict() for item in self.comparisons),
            "comparison_count": len(self.comparisons),
            "status": self.status,
            "certified": self.certified,
        }


def confidence_action_partition(
    result: ControlledRepairResult,
    *,
    actions: Sequence[Action],
    merge_tolerance: float,
    split_tolerance: float,
    minimum_samples: int,
) -> ActionPartitionResult:
    if not result.certified:
        raise ValueError("cannot partition an uncertified controlled interface")
    actions = tuple(actions)
    contexts = tuple(product((0, 1), repeat=result.final_order))
    estimates = result.estimate_map
    comparisons = []
    decisions: dict[tuple[int, int], str] = {}
    for left_index, right_index in combinations(range(len(contexts)), 2):
        per_action = []
        for action in actions:
            comparison = compare_bernoulli_estimates(
                estimates.get((action, contexts[left_index])),
                estimates.get((action, contexts[right_index])),
                label=(
                    f"action={action}; {contexts[left_index]} versus "
                    f"{contexts[right_index]}"
                ),
                merge_tolerance=merge_tolerance,
                split_tolerance=split_tolerance,
                minimum_samples=minimum_samples,
            )
            per_action.append((action, comparison))
        action_decisions = tuple(item.decision for _, item in per_action)
        if "separated" in action_decisions:
            decision = "separated"
        elif all(value == "equivalent_within_tolerance" for value in action_decisions):
            decision = "equivalent_within_tolerance"
        else:
            decision = "uncertain"
        comparisons.append(
            ActionPartitionComparison(
                contexts[left_index], contexts[right_index], tuple(per_action), decision
            )
        )
        decisions[(left_index, right_index)] = decision
    if any(value == "uncertain" for value in decisions.values()):
        return ActionPartitionResult(
            contexts, (), tuple(comparisons), "pairwise_evidence_incomplete", False
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

    for pair, decision in decisions.items():
        if decision == "equivalent_within_tolerance":
            union(*pair)
    groups: dict[int, list[int]] = {}
    for index in range(len(contexts)):
        groups.setdefault(find(index), []).append(index)
    blocks = tuple(
        sorted((tuple(group) for group in groups.values()), key=lambda block: block[0])
    )
    block_of = {
        state: block_index
        for block_index, block in enumerate(blocks)
        for state in block
    }
    coherent = all(
        (block_of[left] == block_of[right])
        == (decision == "equivalent_within_tolerance")
        for (left, right), decision in decisions.items()
    )
    return ActionPartitionResult(
        contexts,
        blocks if coherent else (),
        tuple(comparisons),
        "certified_action_predictive_partition"
        if coherent
        else "nontransitive_equivalence_evidence",
        coherent,
    )


@dataclass(frozen=True)
class CompiledControlledInterface:
    actions: tuple[Action, ...]
    contexts: tuple[Context, ...]
    transition_rows: tuple[tuple[tuple[Fraction, ...], ...], ...]
    predictive_laws: tuple[tuple[tuple[Fraction, ...], ...], ...]
    support_counts: tuple[tuple[int, ...], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "actions": self.actions,
            "contexts": self.contexts,
            "transition_rows_by_action": {
                action: self.transition_rows[index]
                for index, action in enumerate(self.actions)
            },
            "predictive_laws_by_action": {
                action: self.predictive_laws[index]
                for index, action in enumerate(self.actions)
            },
            "support_counts_by_action": {
                action: self.support_counts[index]
                for index, action in enumerate(self.actions)
            },
        }


def compile_controlled_interface(
    result: ControlledRepairResult,
    *,
    actions: Sequence[Action],
) -> CompiledControlledInterface:
    if not result.certified:
        raise ValueError("cannot compile an uncertified controlled interface")
    actions = tuple(actions)
    contexts = tuple(product((0, 1), repeat=result.final_order))
    indices = {context: index for index, context in enumerate(contexts)}
    estimates = result.estimate_map
    all_rows = []
    all_laws = []
    all_supports = []
    for action in actions:
        rows = []
        laws = []
        supports = []
        for context in contexts:
            estimate = estimates.get((action, context))
            if estimate is None:
                raise ValueError("certified interface is missing an action/context cell")
            sample_count = estimate.sample_count
            probability_one = Fraction(estimate.one_count, sample_count)
            row = [Fraction(0) for _ in contexts]
            row[indices[(*context[1:], 0)]] += 1 - probability_one
            row[indices[(*context[1:], 1)]] += probability_one
            rows.append(tuple(row))
            laws.append((1 - probability_one, probability_one))
            supports.append(sample_count)
        all_rows.append(tuple(rows))
        all_laws.append(tuple(laws))
        all_supports.append(tuple(supports))
    return CompiledControlledInterface(
        actions,
        contexts,
        tuple(all_rows),
        tuple(all_laws),
        tuple(all_supports),
    )


def controlled_dynamic_closure(
    transition_rows: Sequence[Sequence[Sequence[Fraction]]],
    partition: Partition,
    *,
    return_trace: bool = False,
) -> Partition | tuple[Partition, tuple[Partition, ...]]:
    matrices = tuple(tuple(tuple(value for value in row) for row in matrix) for matrix in transition_rows)
    if not matrices:
        raise ValueError("at least one controlled transition matrix is required")
    state_count = len(matrices[0])
    if any(len(matrix) != state_count for matrix in matrices):
        raise ValueError("controlled transition matrices must share a state space")
    current = tuple(tuple(sorted(block)) for block in partition)
    if sorted(state for block in current for state in block) != list(range(state_count)):
        raise ValueError("partition must cover each state exactly once")
    trace = [current]
    while True:
        refined = []
        for block in current:
            groups: dict[tuple[Fraction, ...], list[int]] = {}
            for state in block:
                signature = []
                for matrix in matrices:
                    signature.extend(
                        sum(matrix[state][target] for target in target_block)
                        for target_block in current
                    )
                groups.setdefault(tuple(signature), []).append(state)
            refined.extend(tuple(group) for group in groups.values())
        next_partition = tuple(sorted(refined, key=lambda block: block[0]))
        if next_partition == current:
            break
        current = next_partition
        trace.append(current)
    if return_trace:
        return current, tuple(trace)
    return current


def controlled_predictive_residual(
    interface: CompiledControlledInterface,
    partition: Partition,
    state_weights: Sequence[Fraction],
    action_weights: Sequence[Fraction] | None = None,
) -> float:
    if action_weights is None:
        action_weights = tuple(
            Fraction(1, len(interface.actions)) for _ in interface.actions
        )
    if len(action_weights) != len(interface.actions) or sum(action_weights) != 1:
        raise ValueError("action weights must form a probability vector")
    return sum(
        float(action_weight)
        * predictive_residual(laws, partition, state_weights)
        for action_weight, laws in zip(
            action_weights, interface.predictive_laws, strict=True
        )
    )


def controlled_lumpability_residual(
    interface: CompiledControlledInterface,
    partition: Partition,
    state_weights: Sequence[Fraction],
    action_weights: Sequence[Fraction] | None = None,
) -> float:
    if action_weights is None:
        action_weights = tuple(
            Fraction(1, len(interface.actions)) for _ in interface.actions
        )
    if len(action_weights) != len(interface.actions) or sum(action_weights) != 1:
        raise ValueError("action weights must form a probability vector")
    return sum(
        float(action_weight)
        * lumpability_residual(rows, partition, state_weights)
        for action_weight, rows in zip(
            action_weights, interface.transition_rows, strict=True
        )
    )


def generate_controlled_interactions(
    *,
    world: ControlledWorld,
    actions: Sequence[Action],
    action_costs: Mapping[Action, float],
    episode_count: int,
    episode_length: int,
    history_length: int,
    seed: int,
    initial_step: int = 0,
) -> tuple[Interaction, ...]:
    actions = tuple(actions)
    if not actions or any(action not in world.actions for action in actions):
        raise ValueError("heldout actions must belong to the world")
    if episode_count < 1 or episode_length <= history_length or history_length < 2:
        raise ValueError("invalid heldout episode configuration")
    rng = Random(seed)
    initial_histories = tuple(product((0, 1), repeat=history_length))
    records = []
    global_step = initial_step
    for episode_index in range(episode_count):
        history = list(initial_histories[episode_index % len(initial_histories)])
        for _ in range(episode_length - history_length):
            action = actions[len(records) % len(actions)]
            probability = world.probability_one(history[-2], action, global_step)
            outcome = int(rng.random() < probability)
            records.append(
                Interaction(
                    history=tuple(history[-history_length:]),
                    action=action,
                    outcome=outcome,
                    cost=action_costs[action],
                    episode_index=episode_index,
                    step=len(records) + 1,
                )
            )
            history.append(outcome)
            global_step += 1
    return tuple(records)


def fit_action_probability_table(
    interactions: Iterable[Interaction],
    order: int,
    *,
    pseudocount: float = 0.5,
) -> dict[tuple[Action, Context], float]:
    if order < 0 or pseudocount < 0.0:
        raise ValueError("invalid probability-table configuration")
    counts: dict[tuple[Action, Context], list[int]] = defaultdict(lambda: [0, 0])
    for interaction in interactions:
        if order > len(interaction.history):
            continue
        context = () if order == 0 else interaction.history[-order:]
        counts[(interaction.action, context)][interaction.outcome] += 1
    return {
        key: (values[1] + pseudocount) / (sum(values) + 2.0 * pseudocount)
        for key, values in counts.items()
    }


def evaluate_action_probability_table(
    probabilities: Mapping[tuple[Action, Context], float],
    order: int,
    interactions: Iterable[Interaction],
) -> dict[str, object]:
    correct = total = abstained = 0
    negative_log_likelihood = brier = 0.0
    for interaction in interactions:
        context = () if order == 0 else interaction.history[-order:]
        probability = probabilities.get((interaction.action, context))
        if probability is None:
            abstained += 1
            continue
        clipped = min(max(probability, 1e-12), 1.0 - 1e-12)
        target = interaction.outcome
        negative_log_likelihood -= target * log(clipped) + (1 - target) * log(
            1.0 - clipped
        )
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
class ActionStationarityAssessment:
    status: str
    comparisons: tuple[tuple[Action, DistributionComparison], ...]
    change_witness: tuple[Action, DistributionComparison] | None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "comparisons": tuple(
                {"action": action, **comparison.to_dict()}
                for action, comparison in self.comparisons
            ),
            "comparison_count": len(self.comparisons),
            "change_witness": (
                None
                if self.change_witness is None
                else {
                    "action": self.change_witness[0],
                    **self.change_witness[1].to_dict(),
                }
            ),
        }


def assess_action_stationarity(
    interactions: Sequence[Interaction],
    order: int,
    *,
    actions: Sequence[Action],
    delta: float,
    merge_tolerance: float,
    split_tolerance: float,
    minimum_samples: int,
) -> ActionStationarityAssessment:
    midpoint = len(interactions) // 2
    if midpoint < 1 or midpoint == len(interactions):
        raise ValueError("stationarity assessment requires two non-empty halves")
    left = ControlledCounts(actions, max(1, order))
    right = ControlledCounts(actions, max(1, order))
    for item in interactions[:midpoint]:
        left.add(item)
    for item in interactions[midpoint:]:
        right.add(item)
    family_budget = 2 * len(tuple(actions)) * 2**order
    comparisons = []
    changes = []
    for action in actions:
        for context in product((0, 1), repeat=order):
            context = tuple(context)
            comparison = compare_bernoulli_estimates(
                left.estimate(
                    order,
                    action,
                    context,
                    delta=delta,
                    family_cell_budget=family_budget,
                ),
                right.estimate(
                    order,
                    action,
                    context,
                    delta=delta,
                    family_cell_budget=family_budget,
                ),
                label=f"first half versus second half at {context}",
                merge_tolerance=merge_tolerance,
                split_tolerance=split_tolerance,
                minimum_samples=minimum_samples,
            )
            comparisons.append((action, comparison))
            if comparison.decision == "separated":
                changes.append((action, comparison))
    if changes:
        status = "action_conditioned_distribution_shift_detected"
    elif comparisons and all(
        comparison.decision == "equivalent_within_tolerance"
        for _, comparison in comparisons
    ):
        status = "action_conditioned_stationary_within_tolerance"
    else:
        status = "action_conditioned_stationarity_uncertain"
    return ActionStationarityAssessment(
        status=status,
        comparisons=tuple(comparisons),
        change_witness=None if not changes else changes[0],
    )
