"""Verifier-authorised construction of finite relational state.

This module removes the hand-written ``add one older bit`` repair used by the
v0.5 active fixture.  A small typed Boolean grammar constructs candidate
features from raw three-bit histories.  Candidate representations are ordered
before data are observed, an action policy proposes interventions, and an
anytime confidence verifier is the only component allowed to select a state.

The primary fixture hides the law behind ``xor(lag0, lag1)``.  The label XOR is
only a canonical syntactic representative: XOR and EQ induce the same
unlabelled two-block partition, so the semantic polarity is not identifiable
from partition evidence alone.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, product
from math import exp, log, sqrt
from random import Random
from typing import Iterable, Mapping, Sequence

from .active import (
    Action,
    CompiledControlledInterface,
    Context,
    Interaction,
    controlled_dynamic_closure,
)
from .repair import Partition, common_refinement, normalize_partition
from .stochastic import (
    BernoulliEstimate,
    DistributionComparison,
    compare_bernoulli_estimates,
    hoeffding_radius,
)


_COMMUTATIVE_OPERATORS = frozenset({"xor", "eq", "and", "or"})
_BOOLEAN_OPERATORS = frozenset({"const", "lag", "not", *_COMMUTATIVE_OPERATORS})
_OPERATOR_PRIORITY = {
    "lag": 0,
    "xor": 1,
    "eq": 2,
    "and": 3,
    "or": 4,
    "not": 5,
    "const": 6,
}


def binary_contexts(history_length: int) -> tuple[Context, ...]:
    if history_length < 1:
        raise ValueError("history_length must be positive")
    return tuple(tuple(value) for value in product((0, 1), repeat=history_length))


@dataclass(frozen=True)
class BooleanExpression:
    """A typed Boolean expression over finite observation histories."""

    operator: str
    arguments: tuple["BooleanExpression", ...] = ()
    value: int | None = None

    def __post_init__(self) -> None:
        if self.operator not in _BOOLEAN_OPERATORS:
            raise ValueError(f"unknown Boolean operator: {self.operator}")
        if self.operator in {"const", "lag"}:
            if self.arguments or self.value is None:
                raise ValueError("terminal expressions require one integer value")
            if self.operator == "const" and self.value not in (0, 1):
                raise ValueError("Boolean constants must be zero or one")
            if self.operator == "lag" and self.value < 0:
                raise ValueError("lag indices must be non-negative")
        elif self.operator == "not":
            if len(self.arguments) != 1 or self.value is not None:
                raise ValueError("not requires exactly one argument")
        elif len(self.arguments) != 2 or self.value is not None:
            raise ValueError("binary Boolean operators require two arguments")

    @property
    def type_name(self) -> str:
        return "bool"

    @property
    def complexity(self) -> int:
        return 1 + sum(argument.complexity for argument in self.arguments)

    @property
    def dependencies(self) -> tuple[int, ...]:
        if self.operator == "lag":
            assert self.value is not None
            return (self.value,)
        return tuple(
            sorted(
                {
                    dependency
                    for argument in self.arguments
                    for dependency in argument.dependencies
                }
            )
        )

    def render(self) -> str:
        if self.operator == "const":
            return str(self.value)
        if self.operator == "lag":
            return f"lag{self.value}"
        if self.operator == "not":
            return f"not({self.arguments[0].render()})"
        return (
            f"{self.operator}({self.arguments[0].render()},"
            f"{self.arguments[1].render()})"
        )

    def evaluate(self, context: Sequence[int]) -> int:
        materialized = tuple(int(value) for value in context)
        if not materialized or any(value not in (0, 1) for value in materialized):
            raise ValueError("Boolean features require non-empty binary contexts")
        if self.operator == "const":
            assert self.value is not None
            return self.value
        if self.operator == "lag":
            assert self.value is not None
            if self.value >= len(materialized):
                raise ValueError("feature lag exceeds the supplied history")
            return materialized[-1 - self.value]
        values = tuple(argument.evaluate(materialized) for argument in self.arguments)
        if self.operator == "not":
            return 1 - values[0]
        left, right = values
        if self.operator == "xor":
            return left ^ right
        if self.operator == "eq":
            return int(left == right)
        if self.operator == "and":
            return left & right
        if self.operator == "or":
            return left | right
        raise AssertionError("unreachable Boolean operator")

    def truth_table(self, contexts: Sequence[Context]) -> tuple[int, ...]:
        return tuple(self.evaluate(context) for context in contexts)

    def to_dict(self, contexts: Sequence[Context] | None = None) -> dict[str, object]:
        result: dict[str, object] = {
            "type": self.type_name,
            "expression": self.render(),
            "operator": self.operator,
            "complexity": self.complexity,
            "dependencies": self.dependencies,
        }
        if contexts is not None:
            result["truth_table"] = self.truth_table(contexts)
        return result


def _expression_key(expression: BooleanExpression) -> tuple[object, ...]:
    return (
        expression.complexity,
        _OPERATOR_PRIORITY[expression.operator],
        expression.render(),
    )


def _terminal_expressions(history_length: int) -> tuple[BooleanExpression, ...]:
    return (
        BooleanExpression("const", value=0),
        BooleanExpression("const", value=1),
        *(BooleanExpression("lag", value=index) for index in range(history_length)),
    )


def generate_boolean_features(
    *,
    history_length: int = 3,
    maximum_complexity: int = 3,
    operators: Sequence[str] = ("not", "xor", "eq", "and", "or"),
) -> tuple[BooleanExpression, ...]:
    """Enumerate canonical expressions, deduplicated by finite truth table.

    Enumeration is exhaustive for the declared node-complexity budget.  When
    two expressions have the same truth table, the lower-complexity expression
    wins, followed by the fixed grammar priority and lexical rendering.
    """

    if maximum_complexity < 1:
        raise ValueError("maximum_complexity must be positive")
    contexts = binary_contexts(history_length)
    selected_operators = tuple(operators)
    if any(operator not in {"not", *_COMMUTATIVE_OPERATORS} for operator in selected_operators):
        raise ValueError("feature grammar contains an unsupported operator")
    if len(set(selected_operators)) != len(selected_operators):
        raise ValueError("feature grammar operators must be unique")

    best_by_truth: dict[tuple[int, ...], BooleanExpression] = {}
    by_complexity: dict[int, list[BooleanExpression]] = defaultdict(list)

    def register(expression: BooleanExpression) -> None:
        if expression.complexity > maximum_complexity:
            return
        truth = expression.truth_table(contexts)
        incumbent = best_by_truth.get(truth)
        if incumbent is None or _expression_key(expression) < _expression_key(incumbent):
            best_by_truth[truth] = expression

    for expression in _terminal_expressions(history_length):
        register(expression)

    # Rebuild exact-complexity buckets after each layer.  A replaced expression
    # can only become simpler or earlier in the fixed ordering.
    for complexity in range(2, maximum_complexity + 1):
        by_complexity.clear()
        for expression in best_by_truth.values():
            by_complexity[expression.complexity].append(expression)
        if "not" in selected_operators:
            for argument in tuple(by_complexity.get(complexity - 1, ())):
                register(BooleanExpression("not", (argument,)))
        for operator in selected_operators:
            if operator not in _COMMUTATIVE_OPERATORS:
                continue
            for left_complexity in range(1, complexity - 1):
                right_complexity = complexity - 1 - left_complexity
                for left in tuple(by_complexity.get(left_complexity, ())):
                    for right in tuple(by_complexity.get(right_complexity, ())):
                        left_key = _expression_key(left)
                        right_key = _expression_key(right)
                        if left_key > right_key:
                            continue
                        register(BooleanExpression(operator, (left, right)))

    return tuple(sorted(best_by_truth.values(), key=_expression_key))


def partition_from_features(
    contexts: Sequence[Context], features: Sequence[BooleanExpression]
) -> Partition:
    groups: dict[tuple[int, ...], list[int]] = {}
    for index, context in enumerate(contexts):
        signature = tuple(feature.evaluate(context) for feature in features)
        groups.setdefault(signature, []).append(index)
    return normalize_partition(groups.values())


@dataclass(frozen=True)
class FeatureRepresentation:
    """A finite partition induced by a tuple of Boolean features."""

    features: tuple[BooleanExpression, ...]
    partition: Partition
    search_enabled: bool = True

    @property
    def name(self) -> str:
        if not self.features:
            return "one_state"
        return " + ".join(feature.render() for feature in self.features)

    @property
    def feature_count(self) -> int:
        return len(self.features)

    @property
    def block_count(self) -> int:
        return len(self.partition)

    @property
    def total_complexity(self) -> int:
        return sum(feature.complexity for feature in self.features)

    @property
    def contains_relation(self) -> bool:
        return any(
            feature.operator in _COMMUTATIVE_OPERATORS
            for feature in self.features
        )

    def block_index(self, context_index: int) -> int:
        for block_index, block in enumerate(self.partition):
            if context_index in block:
                return block_index
        raise ValueError("context index is outside the representation")

    def to_dict(self, contexts: Sequence[Context]) -> dict[str, object]:
        return {
            "name": self.name,
            "features": tuple(feature.to_dict(contexts) for feature in self.features),
            "feature_count": self.feature_count,
            "block_count": self.block_count,
            "total_complexity": self.total_complexity,
            "partition": self.partition,
            "block_contexts": tuple(
                tuple(contexts[index] for index in block) for block in self.partition
            ),
            "contains_relation": self.contains_relation,
            "search_enabled": self.search_enabled,
        }


def _representation_tie_key(
    representation: FeatureRepresentation,
) -> tuple[object, ...]:
    return (
        representation.total_complexity,
        representation.feature_count,
        tuple(_OPERATOR_PRIORITY[feature.operator] for feature in representation.features),
        representation.name,
    )


def representation_search_key(
    representation: FeatureRepresentation,
) -> tuple[object, ...]:
    return (representation.block_count, *_representation_tie_key(representation))


def generate_feature_representations(
    *,
    history_length: int = 3,
    maximum_complexity: int = 3,
    maximum_feature_count: int = 2,
    operators: Sequence[str] = ("not", "xor", "eq", "and", "or"),
) -> tuple[FeatureRepresentation, ...]:
    """Generate a fixed candidate family, deduplicated by induced partition."""

    if maximum_feature_count < 0:
        raise ValueError("maximum_feature_count must be non-negative")
    contexts = binary_contexts(history_length)
    features = generate_boolean_features(
        history_length=history_length,
        maximum_complexity=maximum_complexity,
        operators=operators,
    )
    best_by_partition: dict[Partition, FeatureRepresentation] = {}

    def register(feature_tuple: tuple[BooleanExpression, ...]) -> None:
        partition = partition_from_features(contexts, feature_tuple)
        candidate = FeatureRepresentation(feature_tuple, partition)
        incumbent = best_by_partition.get(partition)
        if incumbent is None or _representation_tie_key(candidate) < _representation_tie_key(
            incumbent
        ):
            best_by_partition[partition] = candidate

    register(())
    nonconstant = tuple(
        feature
        for feature in features
        if len(set(feature.truth_table(contexts))) > 1
    )
    for feature_count in range(1, maximum_feature_count + 1):
        for feature_tuple in combinations(nonconstant, feature_count):
            register(feature_tuple)
    return tuple(sorted(best_by_partition.values(), key=representation_search_key))


def raw_identity_representation(history_length: int = 3) -> FeatureRepresentation:
    contexts = binary_contexts(history_length)
    features = tuple(
        BooleanExpression("lag", value=index)
        for index in reversed(range(history_length))
    )
    return FeatureRepresentation(
        features,
        partition_from_features(contexts, features),
        search_enabled=False,
    )


def representation_by_name(
    representations: Sequence[FeatureRepresentation], name: str
) -> FeatureRepresentation:
    matches = [item for item in representations if item.name == name]
    if len(matches) != 1:
        raise ValueError(f"expected one representation named {name!r}, found {len(matches)}")
    return matches[0]


@dataclass(frozen=True)
class RelationalWorld:
    """Controlled binary world whose probe law is hidden behind a relation."""

    relation: str = "xor_recent"
    actions: tuple[Action, ...] = ("baseline", "probe")
    baseline_probability_one: float = 0.5
    high_probability_one: float = 0.85
    low_probability_one: float = 0.15
    probe_action: Action = "probe"
    drift_step: int | None = None

    def __post_init__(self) -> None:
        if self.relation not in {"xor_recent", "lag1", "null"}:
            raise ValueError("unknown relational fixture")
        if not self.actions or len(set(self.actions)) != len(self.actions):
            raise ValueError("actions must be non-empty and unique")
        if self.probe_action not in self.actions:
            raise ValueError("probe_action must be declared")
        probabilities = (
            self.baseline_probability_one,
            self.high_probability_one,
            self.low_probability_one,
        )
        if any(not 0.0 < probability < 1.0 for probability in probabilities):
            raise ValueError("fixture probabilities must lie inside (0,1)")
        if self.drift_step is not None and self.drift_step < 1:
            raise ValueError("drift_step must be positive")

    def relation_value(self, context: Sequence[int]) -> int:
        materialized = tuple(int(value) for value in context)
        if len(materialized) < 2 or any(value not in (0, 1) for value in materialized):
            raise ValueError("world requires at least two binary history values")
        if self.relation == "xor_recent":
            return materialized[-1] ^ materialized[-2]
        if self.relation == "lag1":
            return materialized[-2]
        return 0

    def probability_one(self, context: Sequence[int], action: Action, step: int) -> float:
        if action not in self.actions:
            raise ValueError("action is outside the declared family")
        if action != self.probe_action or self.relation == "null":
            return self.baseline_probability_one
        high, low = self.high_probability_one, self.low_probability_one
        if self.drift_step is not None and step >= self.drift_step:
            high, low = low, high
        return high if self.relation_value(context) == 0 else low


class RelationalCounts:
    """Incremental sufficient statistics for raw context/action cells."""

    def __init__(self, actions: Sequence[Action], history_length: int) -> None:
        if not actions or len(set(actions)) != len(actions):
            raise ValueError("actions must be non-empty and unique")
        if history_length < 1:
            raise ValueError("history_length must be positive")
        self.actions = tuple(actions)
        self.history_length = history_length
        self.contexts = binary_contexts(history_length)
        self._counts: dict[tuple[Action, Context], list[int]] = defaultdict(
            lambda: [0, 0]
        )
        self.action_counts: Counter[Action] = Counter()
        self.total_records = 0

    def add(self, interaction: Interaction) -> None:
        if interaction.action not in self.actions:
            raise ValueError("interaction uses an undeclared action")
        if len(interaction.history) < self.history_length:
            raise ValueError("interaction history is shorter than the declared context")
        context = tuple(interaction.history[-self.history_length :])
        self._counts[(interaction.action, context)][interaction.outcome] += 1
        self.action_counts[interaction.action] += 1
        self.total_records += 1

    def target_counts(self, action: Action, context: Context) -> tuple[int, int]:
        values = self._counts.get((action, context), [0, 0])
        return values[0], values[1]

    def sample_count(self, action: Action, context: Context) -> int:
        return sum(self.target_counts(action, context))

    def estimate(
        self,
        action: Action,
        context: Context,
        *,
        delta: float,
        family_cell_budget: int,
    ) -> BernoulliEstimate | None:
        zero, one = self.target_counts(action, context)
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
        self, *, delta: float, family_cell_budget: int
    ) -> dict[tuple[Action, Context], BernoulliEstimate]:
        result = {}
        for action in self.actions:
            for context in self.contexts:
                estimate = self.estimate(
                    action,
                    context,
                    delta=delta,
                    family_cell_budget=family_cell_budget,
                )
                if estimate is not None:
                    result[(action, context)] = estimate
        return result


@dataclass(frozen=True)
class FeatureActionAssessment:
    action: Action
    status: str
    all_contexts_supported: bool
    within_block_pair_count: int
    decision_counts: tuple[tuple[str, int], ...]
    split_witness: DistributionComparison | None

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "status": self.status,
            "all_contexts_supported": self.all_contexts_supported,
            "within_block_pair_count": self.within_block_pair_count,
            "decision_counts": dict(self.decision_counts),
            "split_witness": (
                None if self.split_witness is None else self.split_witness.to_dict()
            ),
        }


@dataclass(frozen=True)
class RepresentationAssessment:
    representation: FeatureRepresentation
    status: str
    action_assessments: tuple[FeatureActionAssessment, ...]

    def to_dict(self, contexts: Sequence[Context]) -> dict[str, object]:
        return {
            "representation": self.representation.to_dict(contexts),
            "status": self.status,
            "action_assessments": tuple(item.to_dict() for item in self.action_assessments),
        }


@dataclass(frozen=True)
class FeatureFamilyAssessment:
    actions: tuple[Action, ...]
    candidate_assessments: tuple[RepresentationAssessment, ...]
    status: str
    selected_index: int | None
    unresolved_predecessor_indices: tuple[int, ...]

    @property
    def selected(self) -> FeatureRepresentation | None:
        if self.selected_index is None:
            return None
        return self.candidate_assessments[self.selected_index].representation

    def to_dict(self, contexts: Sequence[Context]) -> dict[str, object]:
        selected = self.selected
        return {
            "actions": self.actions,
            "status": self.status,
            "selected_index": self.selected_index,
            "selected_representation": (
                None if selected is None else selected.to_dict(contexts)
            ),
            "unresolved_predecessor_indices": self.unresolved_predecessor_indices,
            "candidate_count": len(self.candidate_assessments),
            "candidate_assessments": tuple(
                item.to_dict(contexts) for item in self.candidate_assessments
            ),
        }


def _finalize_family_assessment(
    actions: tuple[Action, ...],
    assessments: Sequence[RepresentationAssessment],
) -> FeatureFamilyAssessment:
    materialized = tuple(assessments)
    selected_index = None
    unresolved_predecessors: tuple[int, ...] = ()
    status = "no_candidate_certified"
    for index, assessment in enumerate(materialized):
        if assessment.status != "certified":
            continue
        unresolved_predecessors = tuple(
            predecessor
            for predecessor in range(index)
            if materialized[predecessor].status != "rejected"
        )
        if unresolved_predecessors:
            status = "simpler_candidate_unresolved"
        else:
            selected_index = index
            status = "least_supported_representation_certified"
        break
    return FeatureFamilyAssessment(
        actions=actions,
        candidate_assessments=materialized,
        status=status,
        selected_index=selected_index,
        unresolved_predecessor_indices=unresolved_predecessors,
    )


def project_action_assessment(
    assessment: FeatureFamilyAssessment, action: Action
) -> FeatureFamilyAssessment:
    """Project one already-computed family assessment to one declared action."""

    if action not in assessment.actions:
        raise ValueError("action was not included in the source assessment")
    projected = []
    for candidate in assessment.candidate_assessments:
        matches = tuple(
            item for item in candidate.action_assessments if item.action == action
        )
        if len(matches) != 1:
            raise ValueError("source assessment has malformed action evidence")
        projected.append(
            RepresentationAssessment(
                representation=candidate.representation,
                status=matches[0].status,
                action_assessments=matches,
            )
        )
    return _finalize_family_assessment((action,), projected)


def assess_feature_family(
    counts: RelationalCounts,
    representations: Sequence[FeatureRepresentation],
    *,
    actions: Sequence[Action],
    delta: float,
    family_cell_budget: int,
    merge_tolerance: float,
    split_tolerance: float,
    minimum_samples: int,
) -> FeatureFamilyAssessment:
    """Assess a predeclared representation family without data-driven reordering."""

    candidates = tuple(representations)
    if not candidates or tuple(sorted(candidates, key=representation_search_key)) != candidates:
        raise ValueError("representations must be non-empty and in fixed search order")
    selected_actions = tuple(actions)
    if not selected_actions or any(action not in counts.actions for action in selected_actions):
        raise ValueError("assessment actions must be declared")
    contexts = counts.contexts
    estimates = counts.estimates(
        delta=delta, family_cell_budget=family_cell_budget
    )

    # Each raw pair is compared once per action.  Candidate partitions only
    # select subsets of this fixed evidence table.
    pair_evidence: dict[tuple[Action, int, int], DistributionComparison] = {}
    for action in selected_actions:
        for left, right in combinations(range(len(contexts)), 2):
            pair_evidence[(action, left, right)] = compare_bernoulli_estimates(
                estimates.get((action, contexts[left])),
                estimates.get((action, contexts[right])),
                label=f"action={action}; {contexts[left]} versus {contexts[right]}",
                merge_tolerance=merge_tolerance,
                split_tolerance=split_tolerance,
                minimum_samples=minimum_samples,
            )

    assessments = []
    for representation in candidates:
        action_assessments = []
        for action in selected_actions:
            supported = all(
                (estimate := estimates.get((action, context))) is not None
                and estimate.sample_count >= minimum_samples
                for context in contexts
            )
            comparisons = tuple(
                pair_evidence[(action, min(left, right), max(left, right))]
                for block in representation.partition
                for left, right in combinations(block, 2)
            )
            decision_counts = Counter(item.decision for item in comparisons)
            split_witness = next(
                (item for item in comparisons if item.decision == "separated"),
                None,
            )
            if not supported:
                status = "unresolved"
            elif split_witness is not None:
                status = "rejected"
            elif all(
                comparison.decision == "equivalent_within_tolerance"
                for comparison in comparisons
            ):
                status = "certified"
            else:
                status = "unresolved"
            action_assessments.append(
                FeatureActionAssessment(
                    action=action,
                    status=status,
                    all_contexts_supported=supported,
                    within_block_pair_count=len(comparisons),
                    decision_counts=tuple(sorted(decision_counts.items())),
                    split_witness=split_witness,
                )
            )
        action_statuses = tuple(item.status for item in action_assessments)
        if "rejected" in action_statuses:
            overall_status = "rejected"
        elif action_statuses and all(status == "certified" for status in action_statuses):
            overall_status = "certified"
        else:
            overall_status = "unresolved"
        assessments.append(
            RepresentationAssessment(
                representation=representation,
                status=overall_status,
                action_assessments=tuple(action_assessments),
            )
        )

    return _finalize_family_assessment(selected_actions, assessments)


@dataclass(frozen=True)
class RelationalHypothesis:
    name: str
    truth_table: tuple[int, ...] | None
    polarity: int = 0
    informative_action: Action = "probe"
    high_probability: float = 0.85
    low_probability: float = 0.15
    null_probability: float = 0.5

    def probability_one(self, context_index: int, action: Action) -> float:
        if self.truth_table is None or action != self.informative_action:
            return self.null_probability
        value = self.truth_table[context_index] ^ self.polarity
        return self.high_probability if value == 0 else self.low_probability


class RelationalBelief:
    """Proposal-only Bayesian belief over feature hypotheses and polarities."""

    def __init__(self, hypotheses: Sequence[RelationalHypothesis]) -> None:
        if not hypotheses or len({item.name for item in hypotheses}) != len(hypotheses):
            raise ValueError("hypotheses must be non-empty and uniquely named")
        self.hypotheses = tuple(hypotheses)
        initial = -log(len(self.hypotheses))
        self._log_weights = [initial for _ in self.hypotheses]

    def probabilities(self) -> tuple[float, ...]:
        maximum = max(self._log_weights)
        values = [exp(weight - maximum) for weight in self._log_weights]
        total = sum(values)
        return tuple(value / total for value in values)

    def expected_information_gain(self, context_index: int, action: Action) -> float:
        weights = self.probabilities()
        laws = tuple(
            hypothesis.probability_one(context_index, action)
            for hypothesis in self.hypotheses
        )
        mixture = sum(weight * law for weight, law in zip(weights, laws, strict=True))

        def entropy(probability: float) -> float:
            if probability <= 0.0 or probability >= 1.0:
                return 0.0
            return -probability * log(probability) - (1.0 - probability) * log(
                1.0 - probability
            )

        return max(
            0.0,
            entropy(mixture)
            - sum(
                weight * entropy(law)
                for weight, law in zip(weights, laws, strict=True)
            ),
        )

    def expected_heterogeneity(self, action: Action) -> float:
        return sum(
            weight
            * (
                max(
                    hypothesis.probability_one(index, action)
                    for index in range(
                        1
                        if hypothesis.truth_table is None
                        else len(hypothesis.truth_table)
                    )
                )
                - min(
                    hypothesis.probability_one(index, action)
                    for index in range(
                        1
                        if hypothesis.truth_table is None
                        else len(hypothesis.truth_table)
                    )
                )
            )
            for weight, hypothesis in zip(
                self.probabilities(), self.hypotheses, strict=True
            )
        )

    def update(self, context_index: int, action: Action, outcome: int) -> None:
        if outcome not in (0, 1):
            raise ValueError("outcome must be binary")
        for index, hypothesis in enumerate(self.hypotheses):
            probability = hypothesis.probability_one(context_index, action)
            likelihood = probability if outcome else 1.0 - probability
            self._log_weights[index] += log(max(likelihood, 1e-300))
        maximum = max(self._log_weights)
        self._log_weights = [weight - maximum for weight in self._log_weights]

    def to_dict(self, limit: int = 12) -> dict[str, object]:
        ranked = sorted(
            zip(self.probabilities(), self.hypotheses, strict=True),
            key=lambda item: (-item[0], item[1].name),
        )
        return {
            "hypothesis_count": len(ranked),
            "top_hypotheses": tuple(
                {
                    "name": hypothesis.name,
                    "posterior_probability": probability,
                }
                for probability, hypothesis in ranked[:limit]
            ),
            "reported_probability_mass": sum(
                probability for probability, _ in ranked[:limit]
            ),
            "authority": "proposal only; no representation-changing authority",
        }


def build_relational_hypotheses(
    features: Sequence[BooleanExpression], contexts: Sequence[Context]
) -> tuple[RelationalHypothesis, ...]:
    hypotheses = [RelationalHypothesis("null", None)]
    seen_partitions: set[Partition] = set()
    for feature in features:
        truth = feature.truth_table(contexts)
        if len(set(truth)) < 2:
            continue
        partition = partition_from_features(contexts, (feature,))
        if partition in seen_partitions:
            continue
        seen_partitions.add(partition)
        for polarity in (0, 1):
            hypotheses.append(
                RelationalHypothesis(
                    name=f"{feature.render()}:polarity={polarity}",
                    truth_table=truth,
                    polarity=polarity,
                )
            )
    return tuple(hypotheses)


@dataclass(frozen=True)
class RelationalRepairConfig:
    actions: tuple[Action, ...] = ("baseline", "probe")
    action_costs: tuple[tuple[Action, float], ...] = (
        ("baseline", 1.0),
        ("probe", 2.0),
    )
    history_length: int = 3
    maximum_complexity: int = 3
    maximum_feature_count: int = 2
    maximum_steps: int = 65536
    episode_length: int = 4
    check_interval: int = 128
    delta: float = 0.01
    merge_tolerance: float = 0.2
    split_tolerance: float = 0.35
    minimum_samples: int = 32
    information_weight: float = 1.0
    heterogeneity_weight: float = 0.25
    exploration_weight: float = 0.01

    def __post_init__(self) -> None:
        if not self.actions or len(set(self.actions)) != len(self.actions):
            raise ValueError("actions must be non-empty and unique")
        costs = dict(self.action_costs)
        if set(costs) != set(self.actions) or any(cost <= 0.0 for cost in costs.values()):
            raise ValueError("every action requires exactly one positive cost")
        if self.history_length < 2:
            raise ValueError("history_length must be at least two")
        if self.maximum_complexity < 1 or self.maximum_feature_count < 0:
            raise ValueError("invalid feature grammar budget")
        if self.maximum_steps < 1 or self.episode_length <= self.history_length:
            raise ValueError("invalid relational sample budget")
        if self.check_interval < 1:
            raise ValueError("check_interval must be positive")
        if not 0.0 < self.delta < 1.0:
            raise ValueError("delta must lie inside (0,1)")
        if not 0.0 <= self.merge_tolerance < self.split_tolerance <= 1.0:
            raise ValueError("invalid split/merge hysteresis")
        if self.minimum_samples < 1:
            raise ValueError("minimum_samples must be positive")
        if any(
            weight < 0.0
            for weight in (
                self.information_weight,
                self.heterogeneity_weight,
                self.exploration_weight,
            )
        ):
            raise ValueError("proposal weights must be non-negative")

    @property
    def cost_map(self) -> dict[Action, float]:
        return dict(self.action_costs)

    @property
    def declared_intrinsic_cell_count(self) -> int:
        return len(self.actions) * 2**self.history_length

    @property
    def anytime_family_cell_budget(self) -> int:
        return self.declared_intrinsic_cell_count * self.maximum_steps

    def to_dict(self) -> dict[str, object]:
        return {
            "actions": self.actions,
            "action_costs": dict(self.action_costs),
            "history_length": self.history_length,
            "maximum_complexity": self.maximum_complexity,
            "maximum_feature_count": self.maximum_feature_count,
            "maximum_steps": self.maximum_steps,
            "episode_length": self.episode_length,
            "check_interval": self.check_interval,
            "delta": self.delta,
            "merge_tolerance": self.merge_tolerance,
            "split_tolerance": self.split_tolerance,
            "minimum_samples": self.minimum_samples,
            "information_weight": self.information_weight,
            "heterogeneity_weight": self.heterogeneity_weight,
            "exploration_weight": self.exploration_weight,
            "declared_intrinsic_cell_count": self.declared_intrinsic_cell_count,
            "anytime_family_cell_budget": self.anytime_family_cell_budget,
        }


@dataclass(frozen=True)
class RelationalRepairResult:
    policy: str
    status: str
    certified: bool
    steps: int
    total_cost: float
    action_counts: tuple[tuple[Action, int], ...]
    selected_representation: FeatureRepresentation | None
    first_relational_witness_step: int | None
    first_relational_witness_cost: float | None
    selection_trace: tuple[dict[str, object], ...]
    final_assessment: FeatureFamilyAssessment
    final_action_assessments: tuple[tuple[Action, FeatureFamilyAssessment], ...]
    final_estimates: tuple[tuple[Action, Context, BernoulliEstimate], ...]
    posterior: dict[str, object]
    confidence_contract: dict[str, object]

    @property
    def estimate_map(self) -> dict[tuple[Action, Context], BernoulliEstimate]:
        return {
            (action, context): estimate
            for action, context, estimate in self.final_estimates
        }

    def to_dict(self, contexts: Sequence[Context]) -> dict[str, object]:
        return {
            "policy": self.policy,
            "status": self.status,
            "certified": self.certified,
            "steps": self.steps,
            "total_cost": self.total_cost,
            "action_counts": dict(self.action_counts),
            "selected_representation": (
                None
                if self.selected_representation is None
                else self.selected_representation.to_dict(contexts)
            ),
            "first_relational_witness_step": self.first_relational_witness_step,
            "first_relational_witness_cost": self.first_relational_witness_cost,
            "selection_trace": self.selection_trace,
            "selection_check_count": len(self.selection_trace),
            "final_assessment": self.final_assessment.to_dict(contexts),
            "final_action_assessments": {
                action: assessment.to_dict(contexts)
                for action, assessment in self.final_action_assessments
            },
            "final_estimates": tuple(
                {"action": action, **estimate.to_dict()}
                for action, _, estimate in self.final_estimates
            ),
            "posterior": self.posterior,
            "confidence_contract": self.confidence_contract,
        }


@dataclass(frozen=True)
class RelationalSimulation:
    result: RelationalRepairResult
    interactions: tuple[Interaction, ...]
    contexts: tuple[Context, ...]
    representations: tuple[FeatureRepresentation, ...]


def _uncertainty_gain(sample_count: int) -> float:
    return 1.0 / sqrt(sample_count + 1.0) - 1.0 / sqrt(sample_count + 2.0)


def _select_relational_action(
    *,
    context_index: int,
    counts: RelationalCounts,
    belief: RelationalBelief,
    action_assessments: Mapping[Action, FeatureFamilyAssessment],
    config: RelationalRepairConfig,
) -> tuple[Action, tuple[dict[str, object], ...]]:
    unresolved = tuple(
        action
        for action in config.actions
        if action_assessments[action].selected is None
    )
    candidates = unresolved or config.actions
    context = counts.contexts[context_index]
    costs = config.cost_map
    score_records = []
    for action in candidates:
        information = belief.expected_information_gain(context_index, action)
        heterogeneity = belief.expected_heterogeneity(action)
        uncertainty = _uncertainty_gain(counts.sample_count(action, context))
        score = (
            config.information_weight * information
            + config.heterogeneity_weight * heterogeneity
            + config.exploration_weight * uncertainty
        ) / costs[action]
        score_records.append(
            {
                "action": action,
                "score": score,
                "expected_information_gain_nats": information,
                "posterior_expected_context_heterogeneity": heterogeneity,
                "uncertainty_reduction_proxy": uncertainty,
                "cost": costs[action],
            }
        )
    selected = max(
        score_records,
        key=lambda item: (
            item["score"],
            -config.actions.index(str(item["action"])),
        ),
    )
    return str(selected["action"]), tuple(score_records)


def _assessment_trace_record(
    *,
    step: int,
    cost: float,
    counts: RelationalCounts,
    family: FeatureFamilyAssessment,
    by_action: Mapping[Action, FeatureFamilyAssessment],
) -> dict[str, object]:
    return {
        "step": step,
        "cost": cost,
        "action_counts": dict(counts.action_counts),
        "full_family_status": family.status,
        "full_family_selected": None if family.selected is None else family.selected.name,
        "action_statuses": {
            action: {
                "status": assessment.status,
                "selected": None if assessment.selected is None else assessment.selected.name,
            }
            for action, assessment in by_action.items()
        },
    }


def simulate_relational_repair(
    *,
    world: RelationalWorld,
    policy: str,
    seed: int,
    config: RelationalRepairConfig,
    representations: Sequence[FeatureRepresentation] | None = None,
    belief_features: Sequence[BooleanExpression] | None = None,
    stop_after_relational_witness: bool = False,
) -> RelationalSimulation:
    """Run active, uniform, passive, or deliberately misspecified construction."""

    if tuple(world.actions) != tuple(config.actions):
        raise ValueError("world and repair action families must match")
    if policy not in {"active", "uniform", "passive", "greedy_misspecified"}:
        raise ValueError("unknown relational action policy")
    contexts = binary_contexts(config.history_length)
    context_indices = {context: index for index, context in enumerate(contexts)}
    if representations is None:
        representations = generate_feature_representations(
            history_length=config.history_length,
            maximum_complexity=config.maximum_complexity,
            maximum_feature_count=config.maximum_feature_count,
        )
    candidates = tuple(representations)
    if belief_features is None:
        belief_features = generate_boolean_features(
            history_length=config.history_length,
            maximum_complexity=config.maximum_complexity,
        )
    belief = RelationalBelief(build_relational_hypotheses(belief_features, contexts))
    counts = RelationalCounts(config.actions, config.history_length)
    rng = Random(seed)
    costs = config.cost_map
    family_budget = config.anytime_family_cell_budget

    def assess_all() -> FeatureFamilyAssessment:
        return assess_feature_family(
            counts,
            candidates,
            actions=config.actions,
            delta=config.delta,
            family_cell_budget=family_budget,
            merge_tolerance=config.merge_tolerance,
            split_tolerance=config.split_tolerance,
            minimum_samples=config.minimum_samples,
        )

    family = assess_all()
    by_action = {
        action: project_action_assessment(family, action)
        for action in config.actions
    }
    interactions: list[Interaction] = []
    selection_trace: list[dict[str, object]] = []
    total_cost = 0.0
    first_relational_witness_step: int | None = None
    first_relational_witness_cost: float | None = None
    episode_index = 0
    history = list(contexts[0])
    steps_in_episode = 0
    available_steps = config.episode_length - config.history_length

    for step_index in range(config.maximum_steps):
        if steps_in_episode >= available_steps:
            episode_index += 1
            history = list(contexts[episode_index % len(contexts)])
            steps_in_episode = 0
        context = tuple(history[-config.history_length :])
        context_index = context_indices[context]
        if policy == "active":
            action, _ = _select_relational_action(
                context_index=context_index,
                counts=counts,
                belief=belief,
                action_assessments=by_action,
                config=config,
            )
        elif policy == "uniform":
            # Cycle complete raw-context panels through the action family.  A
            # stepwise alternation would confound action with context because
            # the deterministic reset schedule also cycles contexts.
            action = config.actions[
                (step_index // len(contexts)) % len(config.actions)
            ]
        elif policy == "passive":
            action = config.actions[0]
        else:
            # Negative control: a misspecified proposal insists that baseline is
            # informative.  The verifier still sees the full declared family.
            action = config.actions[0]
        probability = world.probability_one(context, action, step_index)
        outcome = int(rng.random() < probability)
        interaction = Interaction(
            history=context,
            action=action,
            outcome=outcome,
            cost=costs[action],
            episode_index=episode_index,
            step=step_index + 1,
        )
        interactions.append(interaction)
        counts.add(interaction)
        belief.update(context_index, action, outcome)
        total_cost += costs[action]
        history.append(outcome)
        steps_in_episode += 1

        should_check = (step_index + 1) % config.check_interval == 0
        if not should_check and step_index + 1 != config.maximum_steps:
            continue
        family = assess_all()
        by_action = {
            action: project_action_assessment(family, action)
            for action in config.actions
        }
        selection_trace.append(
            _assessment_trace_record(
                step=step_index + 1,
                cost=total_cost,
                counts=counts,
                family=family,
                by_action=by_action,
            )
        )
        if first_relational_witness_step is None:
            relational_selections = [
                assessment.selected
                for assessment in by_action.values()
                if assessment.selected is not None
                and assessment.selected.contains_relation
            ]
            if relational_selections:
                first_relational_witness_step = step_index + 1
                first_relational_witness_cost = total_cost
                if stop_after_relational_witness:
                    break
        if family.selected is not None:
            break

    final_step = len(interactions)
    if not selection_trace or selection_trace[-1]["step"] != final_step:
        family = assess_all()
        by_action = {
            action: project_action_assessment(family, action)
            for action in config.actions
        }
        selection_trace.append(
            _assessment_trace_record(
                step=final_step,
                cost=total_cost,
                counts=counts,
                family=family,
                by_action=by_action,
            )
        )
    estimates = counts.estimates(
        delta=config.delta, family_cell_budget=family_budget
    )
    selected = family.selected
    if selected is not None:
        status = "least_supported_relational_state_certified"
    elif stop_after_relational_witness and first_relational_witness_step is not None:
        status = "action_specific_relational_witness_certified"
    elif policy in {"passive", "greedy_misspecified"}:
        status = "declared_action_family_unresolved_safe_abstention"
    else:
        status = "sample_budget_exhausted_safe_abstention"
    confidence_contract = {
        "kind": "anytime finite-family Hoeffding union",
        "radius": "sqrt(log(2*M/delta)/(2*n))",
        "declared_raw_action_cells": config.declared_intrinsic_cell_count,
        "maximum_within_cell_prefixes": config.maximum_steps,
        "family_cell_budget": family_budget,
        "candidate_order_frozen_before_observation": True,
        "minimum_samples_required_in_every_raw_context_action_cell": config.minimum_samples,
        "minimality_rule": (
            "select a certified candidate only when every earlier candidate "
            "in the fixed order is rejected; unresolved predecessors force abstention"
        ),
        "adaptive_sampling_boundary": (
            "actions are predictable from past records; each stationary raw-cell "
            "innovation sequence is conditionally independent Bernoulli"
        ),
        "proposal_authority": "none",
        "verifier_authority": "exclusive",
    }
    result = RelationalRepairResult(
        policy=policy,
        status=status,
        certified=selected is not None,
        steps=final_step,
        total_cost=total_cost,
        action_counts=tuple(
            (action, counts.action_counts[action]) for action in config.actions
        ),
        selected_representation=selected,
        first_relational_witness_step=first_relational_witness_step,
        first_relational_witness_cost=first_relational_witness_cost,
        selection_trace=tuple(selection_trace),
        final_assessment=family,
        final_action_assessments=tuple(
            (action, by_action[action]) for action in config.actions
        ),
        final_estimates=tuple(
            (action, context, estimate)
            for (action, context), estimate in sorted(estimates.items())
        ),
        posterior=belief.to_dict(),
        confidence_contract=confidence_contract,
    )
    return RelationalSimulation(
        result=result,
        interactions=tuple(interactions),
        contexts=contexts,
        representations=candidates,
    )


def compile_relational_interface(
    result: RelationalRepairResult,
    *,
    actions: Sequence[Action],
    history_length: int,
    pool_partition: Partition | None = None,
) -> CompiledControlledInterface:
    """Compile the empirical context process, optionally pooling certified laws."""

    if not result.certified:
        raise ValueError("cannot compile an uncertified relational interface")
    contexts = binary_contexts(history_length)
    indices = {context: index for index, context in enumerate(contexts)}
    estimates = result.estimate_map
    actions = tuple(actions)
    if pool_partition is None:
        pool_partition = raw_identity_representation(history_length).partition
    block_of = {
        state: block_index
        for block_index, block in enumerate(pool_partition)
        for state in block
    }
    if sorted(block_of) != list(range(len(contexts))):
        raise ValueError("pool_partition must cover every raw context")
    pooled: dict[tuple[Action, int], tuple[int, int]] = {}
    for action in actions:
        for block_index, block in enumerate(pool_partition):
            zero = sum(
                estimates[(action, contexts[index])].zero_count for index in block
            )
            one = sum(
                estimates[(action, contexts[index])].one_count for index in block
            )
            pooled[(action, block_index)] = (zero, one)
    all_rows = []
    all_laws = []
    all_supports = []
    for action in actions:
        rows = []
        laws = []
        supports = []
        for state_index, context in enumerate(contexts):
            estimate = estimates[(action, context)]
            supports.append(estimate.sample_count)
            zero, one = pooled[(action, block_of[state_index])]
            total = zero + one
            probability_one = Fraction(one, total)
            law = (1 - probability_one, probability_one)
            laws.append(law)
            row = [Fraction(0) for _ in contexts]
            row[indices[(*context[1:], 0)]] += 1 - probability_one
            row[indices[(*context[1:], 1)]] += probability_one
            rows.append(tuple(row))
        all_rows.append(tuple(rows))
        all_laws.append(tuple(laws))
        all_supports.append(tuple(supports))
    return CompiledControlledInterface(
        actions=actions,
        contexts=contexts,
        transition_rows=tuple(all_rows),
        predictive_laws=tuple(all_laws),
        support_counts=tuple(all_supports),
    )


def controlled_joint_repair(
    transition_rows: Sequence[Sequence[Sequence[Fraction]]],
    predictive_partition: Partition,
    partition: Partition,
    *,
    schedule: str = "predictive_then_dynamic",
    return_trace: bool = False,
) -> Partition | tuple[Partition, tuple[Partition, ...]]:
    """Least common predictive/dynamic refinement for an action family."""

    current = normalize_partition(partition)
    predictive_partition = normalize_partition(predictive_partition)
    trace = [current]
    state_count = sum(len(block) for block in current)
    for _ in range(1 + state_count):
        if schedule == "predictive_then_dynamic":
            sufficient = common_refinement(current, predictive_partition)
            nxt = controlled_dynamic_closure(transition_rows, sufficient)
        elif schedule == "dynamic_then_predictive":
            closed = controlled_dynamic_closure(transition_rows, current)
            nxt = common_refinement(closed, predictive_partition)
        else:
            raise ValueError("unknown controlled repair schedule")
        assert isinstance(nxt, tuple)
        if nxt == current:
            if return_trace:
                return current, tuple(trace)
            return current
        current = nxt
        trace.append(current)
    raise RuntimeError("controlled joint repair failed to converge")


def oracle_relational_interface(
    world: RelationalWorld, *, history_length: int = 3
) -> CompiledControlledInterface:
    if world.drift_step is not None:
        raise ValueError("a drifting world has no single stationary oracle interface")
    contexts = binary_contexts(history_length)
    indices = {context: index for index, context in enumerate(contexts)}
    all_rows = []
    all_laws = []
    for action in world.actions:
        rows = []
        laws = []
        for context in contexts:
            probability = Fraction(str(world.probability_one(context, action, 0)))
            law = (1 - probability, probability)
            laws.append(law)
            row = [Fraction(0) for _ in contexts]
            row[indices[(*context[1:], 0)]] += 1 - probability
            row[indices[(*context[1:], 1)]] += probability
            rows.append(tuple(row))
        all_rows.append(tuple(rows))
        all_laws.append(tuple(laws))
    supports = tuple(tuple(0 for _ in contexts) for _ in world.actions)
    return CompiledControlledInterface(
        actions=world.actions,
        contexts=contexts,
        transition_rows=tuple(all_rows),
        predictive_laws=tuple(all_laws),
        support_counts=supports,
    )


def generate_relational_interactions(
    *,
    world: RelationalWorld,
    actions: Sequence[Action],
    action_costs: Mapping[Action, float],
    interaction_count: int,
    history_length: int,
    seed: int,
    initial_step: int = 0,
) -> tuple[Interaction, ...]:
    """Generate a balanced one-transition-per-episode evaluation sequence."""

    if interaction_count < 1:
        raise ValueError("interaction_count must be positive")
    contexts = binary_contexts(history_length)
    actions = tuple(actions)
    if not actions or any(action not in world.actions for action in actions):
        raise ValueError("evaluation actions must belong to the world")
    rng = Random(seed)
    records = []
    for index in range(interaction_count):
        context = contexts[index % len(contexts)]
        action = actions[(index // len(contexts)) % len(actions)]
        probability = world.probability_one(context, action, initial_step + index)
        outcome = int(rng.random() < probability)
        records.append(
            Interaction(
                history=context,
                action=action,
                outcome=outcome,
                cost=action_costs[action],
                episode_index=index,
                step=index + 1,
            )
        )
    return tuple(records)


def fit_representation_probability_table(
    interactions: Iterable[Interaction],
    representation: FeatureRepresentation,
    *,
    contexts: Sequence[Context],
    pseudocount: float = 0.5,
) -> dict[tuple[Action, int], float]:
    if pseudocount < 0.0:
        raise ValueError("pseudocount must be non-negative")
    context_indices = {context: index for index, context in enumerate(contexts)}
    counts: dict[tuple[Action, int], list[int]] = defaultdict(lambda: [0, 0])
    for interaction in interactions:
        context = tuple(interaction.history[-len(contexts[0]) :])
        state_index = context_indices[context]
        block_index = representation.block_index(state_index)
        counts[(interaction.action, block_index)][interaction.outcome] += 1
    return {
        key: (values[1] + pseudocount) / (sum(values) + 2.0 * pseudocount)
        for key, values in counts.items()
    }


def evaluate_representation_probability_table(
    probabilities: Mapping[tuple[Action, int], float],
    representation: FeatureRepresentation,
    interactions: Iterable[Interaction],
    *,
    contexts: Sequence[Context],
) -> dict[str, object]:
    context_indices = {context: index for index, context in enumerate(contexts)}
    correct = covered = abstained = 0
    nll = brier = 0.0
    for interaction in interactions:
        context = tuple(interaction.history[-len(contexts[0]) :])
        block = representation.block_index(context_indices[context])
        probability = probabilities.get((interaction.action, block))
        if probability is None:
            abstained += 1
            continue
        clipped = min(max(probability, 1e-12), 1.0 - 1e-12)
        outcome = interaction.outcome
        nll -= outcome * log(clipped) + (1 - outcome) * log(1.0 - clipped)
        brier += (probability - outcome) ** 2
        correct += int((probability >= 0.5) == bool(outcome))
        covered += 1
    eligible = covered + abstained
    return {
        "representation": representation.name,
        "eligible_transitions": eligible,
        "covered_transitions": covered,
        "abstained": abstained,
        "coverage": 0.0 if eligible == 0 else covered / eligible,
        "accuracy": None if covered == 0 else correct / covered,
        "mean_negative_log_likelihood_nats": None if covered == 0 else nll / covered,
        "mean_brier_score": None if covered == 0 else brier / covered,
    }


@dataclass(frozen=True)
class RelationalStationarityAssessment:
    status: str
    cell_comparisons: tuple[tuple[Action, Context, DistributionComparison], ...]
    change_witness: tuple[Action, Context, DistributionComparison] | None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "cell_comparison_count": len(self.cell_comparisons),
            "cell_comparisons": tuple(
                {
                    "action": action,
                    "context": context,
                    **comparison.to_dict(),
                }
                for action, context, comparison in self.cell_comparisons
            ),
            "change_witness": (
                None
                if self.change_witness is None
                else {
                    "action": self.change_witness[0],
                    "context": self.change_witness[1],
                    **self.change_witness[2].to_dict(),
                }
            ),
            "split_rule": (
                "each raw action/context cell is split at its own sample-order "
                "midpoint, so adaptive action timing is not confounded with time"
            ),
        }


def assess_relational_stationarity(
    interactions: Sequence[Interaction],
    *,
    actions: Sequence[Action],
    history_length: int,
    delta: float,
    merge_tolerance: float,
    split_tolerance: float,
    minimum_samples_per_half: int,
) -> RelationalStationarityAssessment:
    """Compare the first and second sample-order halves of every raw cell."""

    actions = tuple(actions)
    contexts = binary_contexts(history_length)
    sequences: dict[tuple[Action, Context], list[int]] = defaultdict(list)
    for interaction in interactions:
        if interaction.action not in actions:
            continue
        context = tuple(interaction.history[-history_length:])
        sequences[(interaction.action, context)].append(interaction.outcome)
    family_budget = 2 * len(actions) * len(contexts)

    def estimate(values: Sequence[int], context: Context) -> BernoulliEstimate | None:
        if not values:
            return None
        one = sum(values)
        zero = len(values) - one
        probability = one / len(values)
        radius = hoeffding_radius(len(values), delta, family_budget)
        return BernoulliEstimate(
            context=context,
            zero_count=zero,
            one_count=one,
            empirical_probability_one=probability,
            radius=radius,
            lower_probability_one=max(0.0, probability - radius),
            upper_probability_one=min(1.0, probability + radius),
        )

    comparisons = []
    changes = []
    for action in actions:
        for context in contexts:
            values = sequences.get((action, context), [])
            midpoint = len(values) // 2
            comparison = compare_bernoulli_estimates(
                estimate(values[:midpoint], context),
                estimate(values[midpoint:], context),
                label=f"cell-order halves for action={action}; context={context}",
                merge_tolerance=merge_tolerance,
                split_tolerance=split_tolerance,
                minimum_samples=minimum_samples_per_half,
            )
            item = (action, context, comparison)
            comparisons.append(item)
            if comparison.decision == "separated":
                changes.append(item)
    if changes:
        status = "raw_cell_distribution_shift_detected"
    elif comparisons and all(item[2].decision == "equivalent_within_tolerance" for item in comparisons):
        status = "raw_cells_stationary_within_tolerance"
    else:
        status = "raw_cell_stationarity_uncertain"
    return RelationalStationarityAssessment(
        status=status,
        cell_comparisons=tuple(comparisons),
        change_witness=None if not changes else changes[0],
    )
