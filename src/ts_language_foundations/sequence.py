"""Typed categorical sequence construction with verifier-authorised repair.

The v0.7 fixture is deliberately not a binary-language-model toy.  The raw
state is a length-three word over a ternary alphabet.  A typed grammar proposes
symbol-valued and predicate-valued features; a simultaneous total-variation
confidence contract is the only authority allowed to accept a partition.

The primary predictive law depends on whether the oldest and newest symbols
agree.  That two-block statistic predicts the next categorical outcome but is
not an autonomous state.  Exact controlled partition refinement repairs it to
an eighteen-block state while leaving one raw distinction unrepresented.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, permutations, product
from math import exp, log, sqrt
from random import Random
from typing import Iterable, Mapping, Sequence

from .active import CompiledControlledInterface
from .repair import Partition, common_refinement, normalize_partition

Action = str
Context = tuple[int, ...]


def categorical_contexts(
    history_length: int = 3, alphabet_size: int = 3
) -> tuple[Context, ...]:
    if history_length < 1 or alphabet_size < 2:
        raise ValueError("history_length must be positive and alphabet_size at least two")
    return tuple(product(range(alphabet_size), repeat=history_length))


_TYPE_PRIORITY = {"symbol": 0, "bool": 1}
_OPERATOR_PRIORITY = {
    "lag": 0,
    "symbol_const": 1,
    "eq": 2,
    "neq": 3,
    "contains": 4,
    "count_ge": 5,
    "all_equal": 6,
    "all_distinct": 7,
    "not": 8,
    "and": 9,
    "or": 10,
    "xor": 11,
}


@dataclass(frozen=True)
class TypedExpression:
    """Small, explicitly typed expression over a finite symbol history."""

    operator: str
    arguments: tuple["TypedExpression", ...] = ()
    value: int | tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.operator not in _OPERATOR_PRIORITY:
            raise ValueError(f"unknown typed operator: {self.operator}")
        if self.operator in {"lag", "symbol_const"}:
            if self.arguments or not isinstance(self.value, int) or self.value < 0:
                raise ValueError("symbol terminals require one non-negative integer")
            return
        if self.operator in {"eq", "neq"}:
            if len(self.arguments) != 2 or self.value is not None:
                raise ValueError("equality predicates require two symbol arguments")
            if any(argument.output_type != "symbol" for argument in self.arguments):
                raise TypeError("equality predicates accept only symbol arguments")
            return
        if self.operator in {"contains", "count_ge"}:
            if self.arguments:
                raise ValueError("sequence predicates take encoded integer parameters")
            if self.operator == "contains" and not isinstance(self.value, int):
                raise ValueError("contains requires one symbol")
            if self.operator == "count_ge" and (
                not isinstance(self.value, tuple)
                or len(self.value) != 2
                or any(not isinstance(item, int) for item in self.value)
            ):
                raise ValueError("count_ge requires (symbol, threshold)")
            return
        if self.operator in {"all_equal", "all_distinct"}:
            if self.arguments or self.value is not None:
                raise ValueError("global sequence predicates take no arguments")
            return
        if self.operator == "not":
            if len(self.arguments) != 1 or self.arguments[0].output_type != "bool":
                raise TypeError("not requires one Boolean argument")
            return
        if len(self.arguments) != 2 or any(
            argument.output_type != "bool" for argument in self.arguments
        ):
            raise TypeError("Boolean connectives require two Boolean arguments")

    @property
    def output_type(self) -> str:
        return "symbol" if self.operator in {"lag", "symbol_const"} else "bool"

    @property
    def complexity(self) -> int:
        return 1 + sum(argument.complexity for argument in self.arguments)

    @property
    def dependencies(self) -> tuple[int, ...]:
        if self.operator == "lag":
            assert isinstance(self.value, int)
            return (self.value,)
        if self.operator in {"contains", "count_ge", "all_equal", "all_distinct"}:
            return (-1,)  # all supplied history coordinates
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
        if self.operator == "lag":
            return f"lag{self.value}"
        if self.operator == "symbol_const":
            return f"sym({self.value})"
        if self.operator in {"eq", "neq", "and", "or", "xor"}:
            return f"{self.operator}({self.arguments[0].render()},{self.arguments[1].render()})"
        if self.operator == "not":
            return f"not({self.arguments[0].render()})"
        if self.operator == "contains":
            return f"contains({self.value})"
        if self.operator == "count_ge":
            assert isinstance(self.value, tuple)
            return f"count_ge({self.value[0]},{self.value[1]})"
        return self.operator

    def evaluate(self, context: Sequence[int], *, alphabet_size: int = 3) -> int:
        materialized = tuple(int(value) for value in context)
        if not materialized or any(not 0 <= value < alphabet_size for value in materialized):
            raise ValueError("context is outside the declared categorical alphabet")
        if self.operator == "lag":
            assert isinstance(self.value, int)
            if self.value >= len(materialized):
                raise ValueError("lag exceeds the supplied history")
            return materialized[-1 - self.value]
        if self.operator == "symbol_const":
            assert isinstance(self.value, int)
            if self.value >= alphabet_size:
                raise ValueError("symbol constant is outside the alphabet")
            return self.value
        if self.operator == "contains":
            assert isinstance(self.value, int)
            return int(self.value in materialized)
        if self.operator == "count_ge":
            assert isinstance(self.value, tuple)
            symbol, threshold = self.value
            return int(materialized.count(symbol) >= threshold)
        if self.operator == "all_equal":
            return int(len(set(materialized)) == 1)
        if self.operator == "all_distinct":
            return int(len(set(materialized)) == len(materialized))
        values = tuple(
            argument.evaluate(materialized, alphabet_size=alphabet_size)
            for argument in self.arguments
        )
        if self.operator == "eq":
            return int(values[0] == values[1])
        if self.operator == "neq":
            return int(values[0] != values[1])
        if self.operator == "not":
            return 1 - values[0]
        if self.operator == "and":
            return values[0] & values[1]
        if self.operator == "or":
            return values[0] | values[1]
        if self.operator == "xor":
            return values[0] ^ values[1]
        raise AssertionError("unreachable typed expression")

    def output_table(
        self, contexts: Sequence[Context], *, alphabet_size: int = 3
    ) -> tuple[int, ...]:
        return tuple(
            self.evaluate(context, alphabet_size=alphabet_size) for context in contexts
        )

    def to_dict(
        self,
        contexts: Sequence[Context] | None = None,
        *,
        alphabet_size: int = 3,
    ) -> dict[str, object]:
        record: dict[str, object] = {
            "expression": self.render(),
            "output_type": self.output_type,
            "operator": self.operator,
            "complexity": self.complexity,
            "dependencies": self.dependencies,
        }
        if contexts is not None:
            record["output_table"] = self.output_table(
                contexts, alphabet_size=alphabet_size
            )
        return record


def expression_key(expression: TypedExpression) -> tuple[object, ...]:
    return (
        expression.complexity,
        _TYPE_PRIORITY[expression.output_type],
        _OPERATOR_PRIORITY[expression.operator],
        expression.render(),
    )


def generate_typed_features(
    *,
    history_length: int = 3,
    alphabet_size: int = 3,
    allowed_lags: Sequence[int] | None = None,
    include_sequence_predicates: bool = True,
    include_boolean_compositions: bool = True,
) -> tuple[TypedExpression, ...]:
    """Generate a finite typed grammar, deduplicated by typed output table.

    Dedupe is semantic only on the declared finite domain.  The output type is
    part of the key, so a symbol taking values 0/1 is not silently identified
    with a Boolean predicate.
    """

    contexts = categorical_contexts(history_length, alphabet_size)
    lags = tuple(range(history_length)) if allowed_lags is None else tuple(allowed_lags)
    if not lags or len(set(lags)) != len(lags) or any(
        lag < 0 or lag >= history_length for lag in lags
    ):
        raise ValueError("allowed_lags must be unique valid history coordinates")
    symbol_features = tuple(TypedExpression("lag", value=lag) for lag in lags)
    candidates: list[TypedExpression] = list(symbol_features)
    equalities = []
    for left, right in combinations(symbol_features, 2):
        ordered = tuple(sorted((left, right), key=expression_key))
        equalities.append(TypedExpression("eq", ordered))
        candidates.append(TypedExpression("eq", ordered))
        candidates.append(TypedExpression("neq", ordered))
    if include_sequence_predicates:
        candidates.extend(
            TypedExpression("contains", value=symbol)
            for symbol in range(alphabet_size)
        )
        candidates.extend(
            TypedExpression("count_ge", value=(symbol, threshold))
            for symbol in range(alphabet_size)
            for threshold in range(2, history_length + 1)
        )
        candidates.extend(
            (TypedExpression("all_equal"), TypedExpression("all_distinct"))
        )
    if include_boolean_compositions:
        atomics = tuple(equalities)
        for left, right in combinations(atomics, 2):
            for operator in ("and", "or", "xor"):
                candidates.append(TypedExpression(operator, (left, right)))
    best: dict[tuple[str, tuple[int, ...]], TypedExpression] = {}
    for expression in candidates:
        key = (
            expression.output_type,
            expression.output_table(contexts, alphabet_size=alphabet_size),
        )
        incumbent = best.get(key)
        if incumbent is None or expression_key(expression) < expression_key(incumbent):
            best[key] = expression
    return tuple(sorted(best.values(), key=expression_key))


def partition_from_typed_features(
    contexts: Sequence[Context],
    features: Sequence[TypedExpression],
    *,
    alphabet_size: int = 3,
) -> Partition:
    groups: dict[tuple[tuple[str, int], ...], list[int]] = {}
    for index, context in enumerate(contexts):
        signature = tuple(
            (
                feature.output_type,
                feature.evaluate(context, alphabet_size=alphabet_size),
            )
            for feature in features
        )
        groups.setdefault(signature, []).append(index)
    return normalize_partition(groups.values())


@dataclass(frozen=True)
class TypedRepresentation:
    features: tuple[TypedExpression, ...]
    partition: Partition
    search_enabled: bool = True

    @property
    def name(self) -> str:
        return "one_state" if not self.features else " + ".join(
            feature.render() for feature in self.features
        )

    @property
    def block_count(self) -> int:
        return len(self.partition)

    @property
    def feature_count(self) -> int:
        return len(self.features)

    @property
    def total_complexity(self) -> int:
        return sum(feature.complexity for feature in self.features)

    def block_index(self, state: int) -> int:
        for index, block in enumerate(self.partition):
            if state in block:
                return index
        raise ValueError("state is outside the representation")

    def to_dict(
        self, contexts: Sequence[Context], *, alphabet_size: int = 3
    ) -> dict[str, object]:
        return {
            "name": self.name,
            "block_count": self.block_count,
            "feature_count": self.feature_count,
            "total_complexity": self.total_complexity,
            "features": tuple(
                feature.to_dict(contexts, alphabet_size=alphabet_size)
                for feature in self.features
            ),
            "partition": self.partition,
            "block_contexts": tuple(
                tuple(contexts[state] for state in block) for block in self.partition
            ),
            "search_enabled": self.search_enabled,
        }


def _representation_tie_key(
    representation: TypedRepresentation,
) -> tuple[object, ...]:
    return (
        representation.total_complexity,
        representation.feature_count,
        tuple(expression_key(feature) for feature in representation.features),
        representation.name,
    )


def representation_search_key(
    representation: TypedRepresentation,
) -> tuple[object, ...]:
    return (representation.block_count, *_representation_tie_key(representation))


def generate_typed_representations(
    *,
    history_length: int = 3,
    alphabet_size: int = 3,
    maximum_feature_count: int = 3,
    allowed_lags: Sequence[int] | None = None,
    include_sequence_predicates: bool = True,
    include_boolean_compositions: bool = True,
) -> tuple[TypedRepresentation, ...]:
    if maximum_feature_count < 0:
        raise ValueError("maximum_feature_count must be non-negative")
    contexts = categorical_contexts(history_length, alphabet_size)
    features = generate_typed_features(
        history_length=history_length,
        alphabet_size=alphabet_size,
        allowed_lags=allowed_lags,
        include_sequence_predicates=include_sequence_predicates,
        include_boolean_compositions=include_boolean_compositions,
    )
    best: dict[Partition, TypedRepresentation] = {}

    def register(items: tuple[TypedExpression, ...]) -> None:
        partition = partition_from_typed_features(
            contexts, items, alphabet_size=alphabet_size
        )
        candidate = TypedRepresentation(items, partition)
        incumbent = best.get(partition)
        if incumbent is None or _representation_tie_key(candidate) < _representation_tie_key(
            incumbent
        ):
            best[partition] = candidate

    register(())
    for count in range(1, maximum_feature_count + 1):
        for items in combinations(features, count):
            register(items)
    return tuple(sorted(best.values(), key=representation_search_key))


def raw_typed_identity(
    history_length: int = 3, alphabet_size: int = 3
) -> TypedRepresentation:
    contexts = categorical_contexts(history_length, alphabet_size)
    features = tuple(
        TypedExpression("lag", value=lag) for lag in range(history_length)
    )
    return TypedRepresentation(
        features,
        partition_from_typed_features(
            contexts, features, alphabet_size=alphabet_size
        ),
        search_enabled=False,
    )


def representation_by_name(
    representations: Sequence[TypedRepresentation], name: str
) -> TypedRepresentation:
    matches = tuple(item for item in representations if item.name == name)
    if len(matches) != 1:
        raise ValueError(f"expected one representation named {name!r}, found {len(matches)}")
    return matches[0]


@dataclass(frozen=True)
class CategoricalSequenceWorld:
    """Ternary controlled sequence world with configurable hidden relation."""

    relation: str = "endpoint_repeat"
    actions: tuple[Action, ...] = ("baseline", "probe")
    informative_action: Action = "probe"
    alphabet_size: int = 3
    dominant_probability: Fraction = Fraction(4, 5)
    other_probability: Fraction = Fraction(1, 10)
    endpoint_true_outcome: int = 0
    endpoint_false_outcome: int = 1
    drift_step: int | None = None

    def __post_init__(self) -> None:
        if self.relation not in {"endpoint_repeat", "middle_symbol", "null"}:
            raise ValueError("unknown categorical sequence relation")
        if self.alphabet_size != 3:
            raise ValueError("the v0.7 verifier currently declares a ternary alphabet")
        if not self.actions or len(set(self.actions)) != len(self.actions):
            raise ValueError("actions must be non-empty and unique")
        if self.informative_action not in self.actions:
            raise ValueError("informative action must be declared")
        if self.dominant_probability + 2 * self.other_probability != 1:
            raise ValueError("categorical fixture probabilities must sum to one")
        if self.dominant_probability <= self.other_probability:
            raise ValueError("dominant probability must exceed the other probability")
        if any(
            not 0 <= outcome < self.alphabet_size
            for outcome in (self.endpoint_true_outcome, self.endpoint_false_outcome)
        ):
            raise ValueError("endpoint outcomes must be alphabet values")
        if self.endpoint_true_outcome == self.endpoint_false_outcome:
            raise ValueError("endpoint outcomes must be distinct")
        if self.drift_step is not None and self.drift_step < 1:
            raise ValueError("drift_step must be positive")

    def relation_value(self, context: Sequence[int]) -> int:
        materialized = tuple(int(value) for value in context)
        if len(materialized) < 3 or any(
            not 0 <= value < self.alphabet_size for value in materialized
        ):
            raise ValueError("world requires a ternary history of length at least three")
        if self.relation == "endpoint_repeat":
            return int(materialized[0] == materialized[-1])
        if self.relation == "middle_symbol":
            return materialized[-2]
        return 0

    def law(self, context: Sequence[int], action: Action, step: int = 0) -> tuple[Fraction, ...]:
        if action not in self.actions:
            raise ValueError("action is outside the declared family")
        if action != self.informative_action or self.relation == "null":
            return (Fraction(1, self.alphabet_size),) * self.alphabet_size
        if self.relation == "middle_symbol":
            dominant = self.relation_value(context)
        else:
            true_outcome = self.endpoint_true_outcome
            false_outcome = self.endpoint_false_outcome
            if self.drift_step is not None and step >= self.drift_step:
                true_outcome, false_outcome = false_outcome, true_outcome
            dominant = true_outcome if self.relation_value(context) else false_outcome
        return tuple(
            self.dominant_probability if outcome == dominant else self.other_probability
            for outcome in range(self.alphabet_size)
        )

    def sample(
        self, context: Sequence[int], action: Action, step: int, rng: Random
    ) -> int:
        draw = rng.random()
        cumulative = 0.0
        for outcome, probability in enumerate(self.law(context, action, step)):
            cumulative += float(probability)
            if draw < cumulative:
                return outcome
        return self.alphabet_size - 1

    @classmethod
    def null(
        cls, *, actions: tuple[Action, ...] = ("baseline", "probe")
    ) -> "CategoricalSequenceWorld":
        return cls(relation="null", actions=actions, informative_action=actions[-1])


@dataclass(frozen=True)
class CategoricalInteraction:
    history: Context
    action: Action
    outcome: int
    cost: float
    episode_index: int
    step: int

    def __post_init__(self) -> None:
        if not self.history or any(value not in (0, 1, 2) for value in self.history):
            raise ValueError("categorical interaction requires a ternary history")
        if self.outcome not in (0, 1, 2) or self.cost <= 0.0:
            raise ValueError("invalid categorical interaction")

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
class CategoricalEstimate:
    action: Action
    context: Context
    counts: tuple[int, ...]
    empirical_law: tuple[float, ...]
    tv_radius: float

    @property
    def sample_count(self) -> int:
        return sum(self.counts)

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "context": self.context,
            "counts": self.counts,
            "sample_count": self.sample_count,
            "empirical_law": self.empirical_law,
            "tv_radius": self.tv_radius,
        }


def categorical_tv_radius(
    sample_count: int,
    *,
    delta: float,
    family_cell_budget: int,
    alphabet_size: int = 3,
) -> float:
    """Weissman-type simultaneous total-variation radius.

    The multinomial L1 inequality gives
    ``P(TV(phat,p) >= r) <= (2**k-2) exp(-2 n r**2)``.  Union bounding over
    the declared anytime cell budget yields this radius.
    """

    if sample_count < 1 or family_cell_budget < 1 or not 0.0 < delta < 1.0:
        raise ValueError("invalid categorical confidence arguments")
    factor = 2**alphabet_size - 2
    return min(
        1.0,
        sqrt(log(factor * family_cell_budget / delta) / (2.0 * sample_count)),
    )


@dataclass(frozen=True)
class CategoricalComparison:
    label: str
    decision: str
    empirical_tv: float | None
    lower_tv: float | None
    upper_tv: float | None
    left_samples: int
    right_samples: int

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "decision": self.decision,
            "empirical_tv": self.empirical_tv,
            "lower_tv": self.lower_tv,
            "upper_tv": self.upper_tv,
            "left_samples": self.left_samples,
            "right_samples": self.right_samples,
        }


def compare_categorical_estimates(
    left: CategoricalEstimate | None,
    right: CategoricalEstimate | None,
    *,
    label: str,
    merge_tolerance: float,
    split_tolerance: float,
    minimum_samples: int,
) -> CategoricalComparison:
    if not 0 <= merge_tolerance < split_tolerance <= 1:
        raise ValueError("invalid categorical split/merge hysteresis")
    if left is None or right is None:
        return CategoricalComparison(label, "insufficient_support", None, None, None, 0 if left is None else left.sample_count, 0 if right is None else right.sample_count)
    if left.sample_count < minimum_samples or right.sample_count < minimum_samples:
        return CategoricalComparison(label, "insufficient_support", None, None, None, left.sample_count, right.sample_count)
    empirical = 0.5 * sum(
        abs(lvalue - rvalue)
        for lvalue, rvalue in zip(left.empirical_law, right.empirical_law, strict=True)
    )
    lower = max(0.0, empirical - left.tv_radius - right.tv_radius)
    upper = min(1.0, empirical + left.tv_radius + right.tv_radius)
    if upper <= merge_tolerance:
        decision = "equivalent_within_tolerance"
    elif lower >= split_tolerance:
        decision = "separated"
    else:
        decision = "unresolved"
    return CategoricalComparison(
        label, decision, empirical, lower, upper, left.sample_count, right.sample_count
    )


class CategoricalCounts:
    def __init__(
        self,
        actions: Sequence[Action],
        *,
        history_length: int = 3,
        alphabet_size: int = 3,
    ) -> None:
        if not actions or len(set(actions)) != len(actions):
            raise ValueError("actions must be non-empty and unique")
        self.actions = tuple(actions)
        self.history_length = history_length
        self.alphabet_size = alphabet_size
        self.contexts = categorical_contexts(history_length, alphabet_size)
        self._counts: dict[tuple[Action, Context], list[int]] = defaultdict(
            lambda: [0] * alphabet_size
        )
        self.action_counts: Counter[Action] = Counter()
        self.total_records = 0

    def add(self, interaction: CategoricalInteraction) -> None:
        if interaction.action not in self.actions:
            raise ValueError("interaction action is undeclared")
        if len(interaction.history) < self.history_length:
            raise ValueError("interaction history is too short")
        context = tuple(interaction.history[-self.history_length :])
        self._counts[(interaction.action, context)][interaction.outcome] += 1
        self.action_counts[interaction.action] += 1
        self.total_records += 1

    def add_cell_counts(
        self, action: Action, context: Context, values: Sequence[int]
    ) -> None:
        """Add an already-aggregated machine-readable cell.

        This is used by fixture importers and unit tests; it applies the same
        validation and updates the same sufficient statistics as individual
        interaction records.
        """

        context = tuple(int(value) for value in context)
        counts = tuple(int(value) for value in values)
        if action not in self.actions or context not in self.contexts:
            raise ValueError("aggregated cell is outside the declared interface")
        if len(counts) != self.alphabet_size or any(value < 0 for value in counts):
            raise ValueError("aggregated categorical counts are malformed")
        for outcome, value in enumerate(counts):
            self._counts[(action, context)][outcome] += value
        total = sum(counts)
        self.action_counts[action] += total
        self.total_records += total

    def target_counts(self, action: Action, context: Context) -> tuple[int, ...]:
        return tuple(self._counts.get((action, context), [0] * self.alphabet_size))

    def sample_count(self, action: Action, context: Context) -> int:
        return sum(self.target_counts(action, context))

    def estimate(
        self,
        action: Action,
        context: Context,
        *,
        delta: float,
        family_cell_budget: int,
    ) -> CategoricalEstimate | None:
        counts = self.target_counts(action, context)
        total = sum(counts)
        if total == 0:
            return None
        return CategoricalEstimate(
            action=action,
            context=context,
            counts=counts,
            empirical_law=tuple(count / total for count in counts),
            tv_radius=categorical_tv_radius(
                total,
                delta=delta,
                family_cell_budget=family_cell_budget,
                alphabet_size=self.alphabet_size,
            ),
        )

    def estimates(
        self, *, delta: float, family_cell_budget: int
    ) -> dict[tuple[Action, Context], CategoricalEstimate]:
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
class CategoricalActionAssessment:
    action: Action
    status: str
    all_contexts_supported: bool
    within_block_pair_count: int
    decision_counts: tuple[tuple[str, int], ...]
    split_witness: CategoricalComparison | None

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "status": self.status,
            "all_contexts_supported": self.all_contexts_supported,
            "within_block_pair_count": self.within_block_pair_count,
            "decision_counts": dict(self.decision_counts),
            "split_witness": None if self.split_witness is None else self.split_witness.to_dict(),
        }


@dataclass(frozen=True)
class CategoricalRepresentationAssessment:
    representation: TypedRepresentation
    status: str
    action_assessments: tuple[CategoricalActionAssessment, ...]


@dataclass(frozen=True)
class CategoricalFamilyAssessment:
    actions: tuple[Action, ...]
    candidate_assessments: tuple[CategoricalRepresentationAssessment, ...]
    status: str
    selected_index: int | None
    unresolved_predecessor_indices: tuple[int, ...]

    @property
    def selected(self) -> TypedRepresentation | None:
        return None if self.selected_index is None else self.candidate_assessments[
            self.selected_index
        ].representation

    def to_dict(
        self, contexts: Sequence[Context], *, alphabet_size: int = 3
    ) -> dict[str, object]:
        return {
            "actions": self.actions,
            "status": self.status,
            "selected_index": self.selected_index,
            "selected_representation": None
            if self.selected is None
            else self.selected.to_dict(contexts, alphabet_size=alphabet_size),
            "unresolved_predecessor_indices": self.unresolved_predecessor_indices,
            "candidate_count": len(self.candidate_assessments),
            "candidate_assessments": tuple(
                {
                    "name": item.representation.name,
                    "block_count": item.representation.block_count,
                    "status": item.status,
                    "action_assessments": tuple(
                        action.to_dict() for action in item.action_assessments
                    ),
                }
                for item in self.candidate_assessments
            ),
        }


def _finalize_assessment(
    actions: tuple[Action, ...],
    assessments: Sequence[CategoricalRepresentationAssessment],
) -> CategoricalFamilyAssessment:
    materialized = tuple(assessments)
    selected_index = None
    unresolved: tuple[int, ...] = ()
    status = "no_candidate_certified"
    for index, assessment in enumerate(materialized):
        if assessment.status != "certified":
            continue
        unresolved = tuple(
            predecessor
            for predecessor in range(index)
            if materialized[predecessor].status != "rejected"
        )
        if unresolved:
            status = "simpler_candidate_unresolved"
        else:
            selected_index = index
            status = "least_supported_representation_certified"
        break
    return CategoricalFamilyAssessment(
        actions, materialized, status, selected_index, unresolved
    )


def assess_categorical_family(
    counts: CategoricalCounts,
    representations: Sequence[TypedRepresentation],
    *,
    actions: Sequence[Action],
    delta: float,
    family_cell_budget: int,
    merge_tolerance: float,
    split_tolerance: float,
    minimum_samples: int,
) -> CategoricalFamilyAssessment:
    candidates = tuple(representations)
    if not candidates or tuple(sorted(candidates, key=representation_search_key)) != candidates:
        raise ValueError("representations must use the fixed search order")
    selected_actions = tuple(actions)
    if not selected_actions or any(action not in counts.actions for action in selected_actions):
        raise ValueError("assessment actions must be declared")
    estimates = counts.estimates(delta=delta, family_cell_budget=family_cell_budget)
    pairs: dict[tuple[Action, int, int], CategoricalComparison] = {}
    for action in selected_actions:
        for left, right in combinations(range(len(counts.contexts)), 2):
            pairs[(action, left, right)] = compare_categorical_estimates(
                estimates.get((action, counts.contexts[left])),
                estimates.get((action, counts.contexts[right])),
                label=f"action={action}; {counts.contexts[left]} versus {counts.contexts[right]}",
                merge_tolerance=merge_tolerance,
                split_tolerance=split_tolerance,
                minimum_samples=minimum_samples,
            )
    records = []
    for representation in candidates:
        action_records = []
        for action in selected_actions:
            supported = all(
                (estimate := estimates.get((action, context))) is not None
                and estimate.sample_count >= minimum_samples
                for context in counts.contexts
            )
            comparisons_for_partition = tuple(
                pairs[(action, left, right)]
                for block in representation.partition
                for left, right in combinations(block, 2)
            )
            decisions = Counter(item.decision for item in comparisons_for_partition)
            witness = next(
                (item for item in comparisons_for_partition if item.decision == "separated"),
                None,
            )
            if not supported:
                candidate_status = "unresolved"
            elif witness is not None:
                candidate_status = "rejected"
            elif all(
                item.decision == "equivalent_within_tolerance"
                for item in comparisons_for_partition
            ):
                candidate_status = "certified"
            else:
                candidate_status = "unresolved"
            action_records.append(
                CategoricalActionAssessment(
                    action,
                    candidate_status,
                    supported,
                    len(comparisons_for_partition),
                    tuple(sorted(decisions.items())),
                    witness,
                )
            )
        statuses = tuple(item.status for item in action_records)
        if "rejected" in statuses:
            overall = "rejected"
        elif statuses and all(status == "certified" for status in statuses):
            overall = "certified"
        else:
            overall = "unresolved"
        records.append(
            CategoricalRepresentationAssessment(
                representation, overall, tuple(action_records)
            )
        )
    return _finalize_assessment(selected_actions, records)


def _categorical_entropy(law: Sequence[float]) -> float:
    return -sum(probability * log(probability) for probability in law if probability > 0)


@dataclass(frozen=True)
class CategoricalHypothesis:
    name: str
    informative_action: Action | None
    output_table: tuple[int, ...] | None
    value_to_outcome: tuple[tuple[int, int], ...] = ()
    alphabet_size: int = 3

    def law(self, context_index: int, action: Action) -> tuple[float, ...]:
        if self.informative_action != action or self.output_table is None:
            return (1.0 / self.alphabet_size,) * self.alphabet_size
        mapping = dict(self.value_to_outcome)
        dominant = mapping[self.output_table[context_index]]
        return tuple(0.8 if outcome == dominant else 0.1 for outcome in range(3))


class CategoricalBelief:
    """Proposal-only Bayesian action belief; never changes a partition."""

    def __init__(self, hypotheses: Sequence[CategoricalHypothesis]) -> None:
        if not hypotheses or len({item.name for item in hypotheses}) != len(hypotheses):
            raise ValueError("hypotheses must be non-empty and uniquely named")
        self.hypotheses = tuple(hypotheses)
        self._log_weights = [-log(len(self.hypotheses))] * len(self.hypotheses)

    def probabilities(self) -> tuple[float, ...]:
        maximum = max(self._log_weights)
        values = tuple(exp(value - maximum) for value in self._log_weights)
        total = sum(values)
        return tuple(value / total for value in values)

    def information_gain(self, context_index: int, action: Action) -> float:
        weights = self.probabilities()
        laws = tuple(
            hypothesis.law(context_index, action) for hypothesis in self.hypotheses
        )
        mixture = tuple(
            sum(weight * law[outcome] for weight, law in zip(weights, laws, strict=True))
            for outcome in range(3)
        )
        return max(
            0.0,
            _categorical_entropy(mixture)
            - sum(
                weight * _categorical_entropy(law)
                for weight, law in zip(weights, laws, strict=True)
            ),
        )

    def expected_heterogeneity(self, action: Action) -> float:
        """Posterior mean context contrast for proposal persistence."""

        weights = self.probabilities()
        values = tuple(
            0.7
            if hypothesis.informative_action == action
            and hypothesis.output_table is not None
            and len(set(hypothesis.output_table)) > 1
            else 0.0
            for hypothesis in self.hypotheses
        )
        return sum(
            weight * value for weight, value in zip(weights, values, strict=True)
        )

    def update(self, context_index: int, action: Action, outcome: int) -> None:
        for index, hypothesis in enumerate(self.hypotheses):
            likelihood = hypothesis.law(context_index, action)[outcome]
            self._log_weights[index] += log(max(likelihood, 1e-300))
        maximum = max(self._log_weights)
        self._log_weights = [value - maximum for value in self._log_weights]

    def to_dict(self, limit: int = 10) -> dict[str, object]:
        ranked = sorted(
            zip(self.probabilities(), self.hypotheses, strict=True),
            key=lambda item: (-item[0], item[1].name),
        )
        return {
            "hypothesis_count": len(ranked),
            "top_hypotheses": tuple(
                {"name": item.name, "posterior_probability": probability}
                for probability, item in ranked[:limit]
            ),
            "authority": "proposal only; verifier retains all selection authority",
        }


def build_categorical_hypotheses(
    features: Sequence[TypedExpression],
    contexts: Sequence[Context],
    actions: Sequence[Action],
    *,
    alphabet_size: int = 3,
) -> tuple[CategoricalHypothesis, ...]:
    hypotheses = [CategoricalHypothesis("null", None, None)]
    # Keep the proposal model deliberately smaller than the verifier grammar.
    proposal_features = tuple(
        feature
        for feature in features
        if feature.operator in {"lag", "eq"}
    )
    for feature in proposal_features:
        table = feature.output_table(contexts, alphabet_size=alphabet_size)
        values = tuple(sorted(set(table)))
        for outcomes in permutations(range(alphabet_size), len(values)):
            mapping = tuple(zip(values, outcomes, strict=True))
            mapping_name = ",".join(f"{value}->{outcome}" for value, outcome in mapping)
            for action in actions:
                hypotheses.append(
                    CategoricalHypothesis(
                        f"{feature.render()}[{mapping_name}]@{action}",
                        action,
                        table,
                        mapping,
                        alphabet_size,
                    )
                )
    return tuple(hypotheses)


@dataclass(frozen=True)
class SequenceRepairConfig:
    actions: tuple[Action, ...] = ("baseline", "probe")
    action_costs: tuple[tuple[Action, float], ...] = (
        ("baseline", 1.0),
        ("probe", 2.0),
    )
    certification_actions: tuple[Action, ...] = ("probe",)
    history_length: int = 3
    alphabet_size: int = 3
    maximum_feature_count: int = 3
    maximum_steps: int = 65536
    check_interval: int = 2048
    delta: float = 0.01
    merge_tolerance: float = 0.32
    split_tolerance: float = 0.45
    minimum_samples: int = 32
    heterogeneity_weight: float = 0.25
    exploration_weight: float = 0.005

    def __post_init__(self) -> None:
        if not self.actions or len(set(self.actions)) != len(self.actions):
            raise ValueError("actions must be unique")
        if not self.certification_actions or any(
            action not in self.actions for action in self.certification_actions
        ):
            raise ValueError("certification actions must be declared")
        costs = dict(self.action_costs)
        if set(costs) != set(self.actions) or any(value <= 0 for value in costs.values()):
            raise ValueError("every action requires one positive cost")
        if self.history_length != 3 or self.alphabet_size != 3:
            raise ValueError("v0.7 declares a length-three ternary interface")
        if self.maximum_feature_count < 1 or self.maximum_steps < 1:
            raise ValueError("invalid grammar or sampling budget")
        if self.check_interval < 1 or not 0 < self.delta < 1:
            raise ValueError("invalid confidence schedule")
        if not 0 <= self.merge_tolerance < self.split_tolerance <= 1:
            raise ValueError("invalid split/merge hysteresis")
        if self.heterogeneity_weight < 0 or self.exploration_weight < 0:
            raise ValueError("proposal weights must be non-negative")

    @property
    def cost_map(self) -> dict[Action, float]:
        return dict(self.action_costs)

    @property
    def anytime_family_cell_budget(self) -> int:
        return len(self.actions) * self.alphabet_size**self.history_length * self.maximum_steps

    def to_dict(self) -> dict[str, object]:
        return {
            "actions": self.actions,
            "action_costs": dict(self.action_costs),
            "certification_actions": self.certification_actions,
            "history_length": self.history_length,
            "alphabet_size": self.alphabet_size,
            "maximum_feature_count": self.maximum_feature_count,
            "maximum_steps": self.maximum_steps,
            "check_interval": self.check_interval,
            "delta": self.delta,
            "merge_tolerance": self.merge_tolerance,
            "split_tolerance": self.split_tolerance,
            "minimum_samples": self.minimum_samples,
            "heterogeneity_weight": self.heterogeneity_weight,
            "exploration_weight": self.exploration_weight,
            "anytime_family_cell_budget": self.anytime_family_cell_budget,
        }


@dataclass(frozen=True)
class SequenceRepairResult:
    policy: str
    status: str
    certified: bool
    steps: int
    total_cost: float
    action_counts: tuple[tuple[Action, int], ...]
    selected_representation: TypedRepresentation | None
    first_relational_witness_step: int | None
    selection_trace: tuple[dict[str, object], ...]
    final_assessment: CategoricalFamilyAssessment
    final_estimates: tuple[CategoricalEstimate, ...]
    posterior: dict[str, object]
    confidence_contract: dict[str, object]
    interactions: tuple[CategoricalInteraction, ...]

    @property
    def estimate_map(self) -> dict[tuple[Action, Context], CategoricalEstimate]:
        return {(item.action, item.context): item for item in self.final_estimates}

    def to_dict(self, contexts: Sequence[Context]) -> dict[str, object]:
        return {
            "policy": self.policy,
            "status": self.status,
            "certified": self.certified,
            "steps": self.steps,
            "total_cost": self.total_cost,
            "action_counts": dict(self.action_counts),
            "selected_representation": None
            if self.selected_representation is None
            else self.selected_representation.to_dict(contexts),
            "first_relational_witness_step": self.first_relational_witness_step,
            "selection_trace": self.selection_trace,
            "final_assessment": self.final_assessment.to_dict(contexts),
            "state_support": tuple(item.to_dict() for item in self.final_estimates),
            "posterior": self.posterior,
            "confidence_contract": self.confidence_contract,
            "interaction_count": len(self.interactions),
        }


def simulate_sequence_repair(
    *,
    world: CategoricalSequenceWorld,
    policy: str,
    seed: int,
    config: SequenceRepairConfig,
    representations: Sequence[TypedRepresentation] | None = None,
    features: Sequence[TypedExpression] | None = None,
) -> SequenceRepairResult:
    if policy not in {"active", "uniform"}:
        raise ValueError("policy must be active or uniform")
    contexts = categorical_contexts(config.history_length, config.alphabet_size)
    candidates = tuple(representations or generate_typed_representations(
        history_length=config.history_length,
        alphabet_size=config.alphabet_size,
        maximum_feature_count=config.maximum_feature_count,
    ))
    feature_family = tuple(features or generate_typed_features(
        history_length=config.history_length,
        alphabet_size=config.alphabet_size,
    ))
    belief = CategoricalBelief(
        build_categorical_hypotheses(
            feature_family, contexts, config.actions, alphabet_size=config.alphabet_size
        )
    )
    counts = CategoricalCounts(
        config.actions,
        history_length=config.history_length,
        alphabet_size=config.alphabet_size,
    )
    rng = Random(seed)
    records = []
    trace = []
    first_witness = None
    last_assessment: CategoricalFamilyAssessment | None = None
    costs = config.cost_map

    for step in range(config.maximum_steps):
        context_index = step % len(contexts)
        context = contexts[context_index]
        if policy == "uniform":
            panel = step // len(contexts)
            action = config.actions[panel % len(config.actions)]
        else:
            scores = []
            for action in config.actions:
                proposal = (
                    belief.information_gain(context_index, action)
                    + config.heterogeneity_weight
                    * belief.expected_heterogeneity(action)
                ) / costs[action]
                exploration = config.exploration_weight / sqrt(
                    1 + counts.sample_count(action, context)
                )
                # deterministic tie breaker uses declared action order
                scores.append((proposal + exploration, -config.actions.index(action), action))
            action = max(scores)[2]
        outcome = world.sample(context, action, step, rng)
        interaction = CategoricalInteraction(
            context, action, outcome, costs[action], step // len(contexts), step
        )
        records.append(interaction)
        counts.add(interaction)
        belief.update(context_index, action, outcome)

        should_check = (step + 1) % config.check_interval == 0 or step + 1 == config.maximum_steps
        if not should_check:
            continue
        last_assessment = assess_categorical_family(
            counts,
            candidates,
            actions=config.certification_actions,
            delta=config.delta,
            family_cell_budget=config.anytime_family_cell_budget,
            merge_tolerance=config.merge_tolerance,
            split_tolerance=config.split_tolerance,
            minimum_samples=config.minimum_samples,
        )
        one_state = last_assessment.candidate_assessments[0]
        if first_witness is None and one_state.status == "rejected":
            first_witness = step + 1
        trace.append(
            {
                "step": step + 1,
                "status": last_assessment.status,
                "selected": None
                if last_assessment.selected is None
                else last_assessment.selected.name,
                "one_state_status": one_state.status,
                "unresolved_predecessor_count": len(
                    last_assessment.unresolved_predecessor_indices
                ),
                "action_counts": dict(counts.action_counts),
            }
        )
        if last_assessment.selected is not None:
            break

    if last_assessment is None:
        raise AssertionError("simulation never evaluated its evidence")
    estimates = tuple(
        sorted(
            counts.estimates(
                delta=config.delta,
                family_cell_budget=config.anytime_family_cell_budget,
            ).values(),
            key=lambda item: (config.actions.index(item.action), item.context),
        )
    )
    selected = last_assessment.selected
    return SequenceRepairResult(
        policy,
        "certified" if selected is not None else last_assessment.status,
        selected is not None,
        len(records),
        sum(item.cost for item in records),
        tuple((action, counts.action_counts[action]) for action in config.actions),
        selected,
        first_witness,
        tuple(trace),
        last_assessment,
        estimates,
        belief.to_dict(),
        {
            "metric": "total variation",
            "concentration": "multinomial L1 (Weissman-type) plus declared union bound",
            "radius": "sqrt(log((2^k-2)*M/delta)/(2*n))",
            "family_cell_budget_M": config.anytime_family_cell_budget,
            "delta": config.delta,
            "merge_tolerance": config.merge_tolerance,
            "split_tolerance": config.split_tolerance,
            "selection_rule": "first certified candidate only after every simpler candidate is rejected",
            "proposal_authority": "none",
        },
        tuple(records),
    )


def generate_sequence_interactions(
    *,
    world: CategoricalSequenceWorld,
    action: Action,
    interaction_count: int,
    seed: int,
    cost: float = 1.0,
    initial_step: int = 0,
) -> tuple[CategoricalInteraction, ...]:
    if action not in world.actions or interaction_count < 1:
        raise ValueError("invalid heldout sequence request")
    contexts = categorical_contexts(3, world.alphabet_size)
    rng = Random(seed)
    records = []
    for offset in range(interaction_count):
        step = initial_step + offset
        context = contexts[offset % len(contexts)]
        records.append(
            CategoricalInteraction(
                context,
                action,
                world.sample(context, action, step, rng),
                cost,
                offset // len(contexts),
                step,
            )
        )
    return tuple(records)


def oracle_sequence_interface(
    world: CategoricalSequenceWorld,
) -> CompiledControlledInterface:
    if world.drift_step is not None:
        raise ValueError("a drifting world has no single stationary oracle interface")
    contexts = categorical_contexts(3, world.alphabet_size)
    indices = {context: index for index, context in enumerate(contexts)}
    all_rows = []
    all_laws = []
    for action in world.actions:
        rows = []
        laws = []
        for context in contexts:
            law = world.law(context, action, 0)
            laws.append(law)
            row = [Fraction(0)] * len(contexts)
            for outcome, probability in enumerate(law):
                row[indices[(*context[1:], outcome)]] += probability
            rows.append(tuple(row))
        all_rows.append(tuple(rows))
        all_laws.append(tuple(laws))
    return CompiledControlledInterface(
        world.actions,
        contexts,
        tuple(all_rows),
        tuple(all_laws),
        tuple(tuple(0 for _ in contexts) for _ in world.actions),
    )


def combine_oracle_interfaces(
    interfaces: Sequence[CompiledControlledInterface],
) -> CompiledControlledInterface:
    items = tuple(interfaces)
    if not items or any(item.contexts != items[0].contexts for item in items):
        raise ValueError("interfaces must share the same raw context state space")
    actions = tuple(action for item in items for action in item.actions)
    if len(set(actions)) != len(actions):
        raise ValueError("combined interface action names must be unique")
    return CompiledControlledInterface(
        actions,
        items[0].contexts,
        tuple(matrix for item in items for matrix in item.transition_rows),
        tuple(laws for item in items for laws in item.predictive_laws),
        tuple(support for item in items for support in item.support_counts),
    )


def predictive_partition_for_actions(
    interface: CompiledControlledInterface,
    actions: Sequence[Action] | None = None,
) -> Partition:
    selected = interface.actions if actions is None else tuple(actions)
    if not selected or any(action not in interface.actions for action in selected):
        raise ValueError("predictive partition actions must be declared")
    indices = tuple(interface.actions.index(action) for action in selected)
    groups: dict[tuple[tuple[Fraction, ...], ...], list[int]] = {}
    for state in range(len(interface.contexts)):
        signature = tuple(interface.predictive_laws[index][state] for index in indices)
        groups.setdefault(signature, []).append(state)
    return normalize_partition(groups.values())


def compile_empirical_sequence_interface(
    result: SequenceRepairResult,
    *,
    actions: Sequence[Action],
    pool_partition: Partition | None = None,
) -> CompiledControlledInterface:
    if not result.certified:
        raise ValueError("cannot compile an uncertified sequence result")
    contexts = categorical_contexts()
    indices = {context: index for index, context in enumerate(contexts)}
    actions = tuple(actions)
    estimates = result.estimate_map
    if pool_partition is None:
        pool_partition = raw_typed_identity().partition
    block_of = {
        state: block_index
        for block_index, block in enumerate(pool_partition)
        for state in block
    }
    all_rows = []
    all_laws = []
    all_support = []
    for action in actions:
        pooled = {}
        for block_index, block in enumerate(pool_partition):
            vector = [0, 0, 0]
            for state in block:
                estimate = estimates.get((action, contexts[state]))
                if estimate is None:
                    raise ValueError("empirical interface has an unsupported action/context")
                for outcome, count in enumerate(estimate.counts):
                    vector[outcome] += count
            pooled[block_index] = tuple(vector)
        rows = []
        laws = []
        support = []
        for state, context in enumerate(contexts):
            estimate = estimates[(action, context)]
            support.append(estimate.sample_count)
            vector = pooled[block_of[state]]
            total = sum(vector)
            law = tuple(Fraction(count, total) for count in vector)
            laws.append(law)
            row = [Fraction(0)] * len(contexts)
            for outcome, probability in enumerate(law):
                row[indices[(*context[1:], outcome)]] += probability
            rows.append(tuple(row))
        all_rows.append(tuple(rows))
        all_laws.append(tuple(laws))
        all_support.append(tuple(support))
    return CompiledControlledInterface(
        actions, contexts, tuple(all_rows), tuple(all_laws), tuple(all_support)
    )


def fit_categorical_table(
    interactions: Sequence[CategoricalInteraction],
    representation: TypedRepresentation,
    *,
    pseudocount: float = 0.5,
) -> dict[tuple[Action, int], tuple[float, ...]]:
    if pseudocount <= 0:
        raise ValueError("pseudocount must be positive")
    contexts = categorical_contexts()
    context_index = {context: index for index, context in enumerate(contexts)}
    counts: dict[tuple[Action, int], list[float]] = defaultdict(
        lambda: [pseudocount] * 3
    )
    for interaction in interactions:
        state = context_index[tuple(interaction.history[-3:])]
        counts[(interaction.action, representation.block_index(state))][
            interaction.outcome
        ] += 1
    return {
        key: tuple(value / sum(vector) for value in vector)
        for key, vector in counts.items()
    }


def evaluate_categorical_table(
    table: Mapping[tuple[Action, int], Sequence[float]],
    representation: TypedRepresentation,
    interactions: Sequence[CategoricalInteraction],
) -> dict[str, float | int]:
    contexts = categorical_contexts()
    context_index = {context: index for index, context in enumerate(contexts)}
    nll = 0.0
    correct = 0
    for interaction in interactions:
        state = context_index[tuple(interaction.history[-3:])]
        law = table[(interaction.action, representation.block_index(state))]
        nll -= log(max(float(law[interaction.outcome]), 1e-300))
        correct += int(max(range(3), key=lambda value: law[value]) == interaction.outcome)
    return {
        "interaction_count": len(interactions),
        "mean_negative_log_likelihood_nats": nll / len(interactions),
        "accuracy": correct / len(interactions),
        "parameter_cells": len(table),
        "representation_blocks": representation.block_count,
    }


def categorical_stationarity_assessment(
    interactions: Sequence[CategoricalInteraction],
    *,
    actions: Sequence[Action],
    delta: float,
    split_tolerance: float,
    minimum_samples: int,
) -> dict[str, object]:
    if len(interactions) < 2:
        raise ValueError("stationarity assessment requires a trace")
    midpoint = len(interactions) // 2
    halves = []
    for records in (interactions[:midpoint], interactions[midpoint:]):
        counts = CategoricalCounts(actions)
        for interaction in records:
            counts.add(interaction)
        halves.append(counts)
    budget = 2 * len(actions) * len(halves[0].contexts)
    witnesses = []
    unresolved = 0
    for action in actions:
        for context in halves[0].contexts:
            left = halves[0].estimate(
                action, context, delta=delta, family_cell_budget=budget
            )
            right = halves[1].estimate(
                action, context, delta=delta, family_cell_budget=budget
            )
            comparison = compare_categorical_estimates(
                left,
                right,
                label=f"stationarity:{action}:{context}",
                merge_tolerance=0.2,
                split_tolerance=split_tolerance,
                minimum_samples=minimum_samples,
            )
            if comparison.decision == "separated":
                witnesses.append(comparison.to_dict())
            elif comparison.decision != "equivalent_within_tolerance":
                unresolved += 1
    return {
        "status": "drift_detected" if witnesses else "no_drift_certificate",
        "drift_witness_count": len(witnesses),
        "unresolved_cell_count": unresolved,
        "witnesses": tuple(witnesses[:12]),
        "boundary": "absence of a witness is not proof of stationarity",
    }
