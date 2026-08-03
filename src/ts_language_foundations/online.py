"""Counterexample-guided finite interfaces learned from observations only.

The learner in this module never receives a hidden environment state.  Its
input is a finite stream of finite observation episodes.  It begins with a
short suffix interface and increases the represented history only when the
same current suffix has been observed with incompatible next symbols.

This is deliberately a small exact learner, not a neural model.  Its closest
standard relatives are variable-order Markov models, predictive-state
partitions, and deterministic automata inferred from distinguishing traces.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

Observation = int
Episode = tuple[Observation, ...]
Context = tuple[Observation, ...]


def normalize_episodes(episodes: Iterable[Iterable[int]]) -> tuple[Episode, ...]:
    """Materialise and validate an observation-only training payload."""

    result = tuple(tuple(int(symbol) for symbol in episode) for episode in episodes)
    if not result:
        raise ValueError("at least one observation episode is required")
    if any(len(episode) < 2 for episode in result):
        raise ValueError("every observation episode needs at least one transition")
    if any(symbol < 0 for episode in result for symbol in episode):
        raise ValueError("observation symbols must be non-negative integers")
    return result


def format_context(context: Context) -> str:
    """Return a stable, unambiguous receipt label for a context."""

    return "[" + ",".join(str(symbol) for symbol in context) + "]"


@dataclass(frozen=True)
class TransitionSample:
    episode_index: int
    time_index: int
    context: Context
    target: Observation
    observation_prefix: Episode

    def to_dict(self) -> dict[str, object]:
        return {
            "episode_index": self.episode_index,
            "time_index": self.time_index,
            "context": self.context,
            "target": self.target,
            "observation_prefix": self.observation_prefix,
        }


@dataclass(frozen=True)
class ConflictWitness:
    context: Context
    left: TransitionSample
    right: TransitionSample

    @property
    def passed(self) -> bool:
        return (
            self.left.context == self.right.context == self.context
            and self.left.target != self.right.target
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "shared_context": self.context,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
            "targets_disagree": self.left.target != self.right.target,
            "passed": self.passed,
        }


def transition_samples(
    episodes: Sequence[Episode],
    order: int,
) -> tuple[TransitionSample, ...]:
    if order < 1:
        raise ValueError("suffix order must be positive")
    samples: list[TransitionSample] = []
    for episode_index, episode in enumerate(episodes):
        for time_index in range(order - 1, len(episode) - 1):
            context = episode[time_index - order + 1 : time_index + 1]
            samples.append(
                TransitionSample(
                    episode_index=episode_index,
                    time_index=time_index,
                    context=context,
                    target=episode[time_index + 1],
                    observation_prefix=episode[: time_index + 1],
                )
            )
    return tuple(samples)


def first_conflict(
    episodes: Sequence[Episode],
    order: int,
) -> ConflictWitness | None:
    first_by_context: dict[Context, TransitionSample] = {}
    for sample in transition_samples(episodes, order):
        previous = first_by_context.get(sample.context)
        if previous is None:
            first_by_context[sample.context] = sample
        elif previous.target != sample.target:
            return ConflictWitness(sample.context, previous, sample)
    return None


def context_target_counts(
    episodes: Sequence[Episode],
    order: int,
) -> dict[Context, Counter[int]]:
    counts: dict[Context, Counter[int]] = {}
    for sample in transition_samples(episodes, order):
        counts.setdefault(sample.context, Counter())[sample.target] += 1
    return counts


@dataclass(frozen=True)
class SuffixLearningResult:
    initial_order: int
    final_order: int
    maximum_order: int
    minimum_context_support: int
    episodes: tuple[Episode, ...]
    repair_events: tuple[dict[str, object], ...]
    prediction_table: tuple[tuple[Context, Observation], ...]
    context_support: tuple[tuple[Context, int], ...]
    terminal_counterexample: ConflictWitness | None
    sparse_contexts: tuple[Context, ...]
    missing_successor_contexts: tuple[Context, ...]
    certified: bool
    status: str

    @property
    def prediction_map(self) -> dict[Context, Observation]:
        return dict(self.prediction_table)

    @property
    def support_map(self) -> dict[Context, int]:
        return dict(self.context_support)

    @property
    def contexts(self) -> tuple[Context, ...]:
        return tuple(context for context, _ in self.prediction_table)

    @property
    def observations_seen(self) -> int:
        return sum(len(episode) for episode in self.episodes)

    @property
    def transitions_considered(self) -> int:
        return len(transition_samples(self.episodes, self.final_order))

    def predict_next(self, history: Sequence[int]) -> int | None:
        """Fail closed unless the whole learned interface is certified."""

        if not self.certified or len(history) < self.final_order:
            return None
        context = tuple(int(value) for value in history[-self.final_order :])
        return self.prediction_map.get(context)

    def to_dict(self, *, include_episodes: bool = False) -> dict[str, object]:
        context_records = []
        prediction_map = self.prediction_map
        support_map = self.support_map
        for context in sorted(set(prediction_map) | set(support_map)):
            context_records.append(
                {
                    "context": context,
                    "context_label": format_context(context),
                    "support": support_map.get(context, 0),
                    "predicted_next_symbol": prediction_map.get(context),
                }
            )
        result: dict[str, object] = {
            "initial_order": self.initial_order,
            "final_order": self.final_order,
            "maximum_order": self.maximum_order,
            "minimum_context_support": self.minimum_context_support,
            "repair_trace": (
                self.initial_order,
                *(event["order_after"] for event in self.repair_events),
            ),
            "repair_events": self.repair_events,
            "repair_event_count": len(self.repair_events),
            "contexts": context_records,
            "context_count": len(self.prediction_table),
            "episodes_seen": len(self.episodes),
            "observations_seen": self.observations_seen,
            "transitions_considered": self.transitions_considered,
            "terminal_counterexample": (
                None
                if self.terminal_counterexample is None
                else self.terminal_counterexample.to_dict()
            ),
            "sparse_contexts": self.sparse_contexts,
            "missing_successor_contexts": self.missing_successor_contexts,
            "certified": self.certified,
            "status": self.status,
        }
        if include_episodes:
            result["observation_episodes"] = self.episodes
        return result


def learn_suffix_interface(
    episodes: Iterable[Iterable[int]],
    *,
    initial_order: int = 1,
    maximum_order: int = 8,
    minimum_context_support: int = 1,
) -> SuffixLearningResult:
    """Learn an exact suffix predictor by streaming counterexample repair.

    Episodes are admitted in request order.  After each admission the verifier
    searches the accumulated observation transitions for a shared suffix with
    incompatible next symbols.  Such a witness increases the suffix order by
    one, up to ``maximum_order``.  At the budget boundary the learner abstains.
    """

    materialized = normalize_episodes(episodes)
    if initial_order < 1:
        raise ValueError("initial_order must be positive")
    if maximum_order < initial_order:
        raise ValueError("maximum_order must not be below initial_order")
    if minimum_context_support < 1:
        raise ValueError("minimum_context_support must be positive")

    admitted: list[Episode] = []
    order = initial_order
    repair_events: list[dict[str, object]] = []
    terminal_counterexample: ConflictWitness | None = None

    for episode_index, episode in enumerate(materialized):
        admitted.append(episode)
        if terminal_counterexample is not None:
            continue
        while True:
            conflict = first_conflict(admitted, order)
            if conflict is None:
                break
            if order >= maximum_order:
                terminal_counterexample = conflict
                break
            repair_events.append(
                {
                    "event_index": len(repair_events),
                    "trigger_episode_index": episode_index,
                    "episodes_admitted": len(admitted),
                    "order_before": order,
                    "order_after": order + 1,
                    "repair": "register one additional observation-history coordinate",
                    "counterexample": conflict.to_dict(),
                }
            )
            order += 1

    final_conflict = first_conflict(materialized, order)
    if final_conflict is not None:
        terminal_counterexample = final_conflict
    counts = context_target_counts(materialized, order)
    unique_predictions = {
        context: next(iter(target_counts))
        for context, target_counts in counts.items()
        if len(target_counts) == 1
    }
    support = {
        context: sum(target_counts.values())
        for context, target_counts in counts.items()
    }
    sparse = tuple(
        sorted(
            context
            for context, count in support.items()
            if count < minimum_context_support
        )
    )
    missing_successors: tuple[Context, ...] = ()
    if terminal_counterexample is None:
        missing_successors = tuple(
            sorted(
                context
                for context, target in unique_predictions.items()
                if (*context[1:], target) not in unique_predictions
            )
        )

    if terminal_counterexample is not None:
        status = "memory_budget_exhausted"
    elif sparse:
        status = "insufficient_context_support"
    elif missing_successors:
        status = "transition_table_not_closed"
    elif not unique_predictions:
        status = "no_supported_contexts"
    else:
        status = "certified_exact_on_observed_support"
    certified = status == "certified_exact_on_observed_support"

    prediction_table = tuple(sorted(unique_predictions.items()))
    context_support = tuple(sorted(support.items()))
    return SuffixLearningResult(
        initial_order=initial_order,
        final_order=order,
        maximum_order=maximum_order,
        minimum_context_support=minimum_context_support,
        episodes=materialized,
        repair_events=tuple(repair_events),
        prediction_table=prediction_table,
        context_support=context_support,
        terminal_counterexample=terminal_counterexample,
        sparse_contexts=sparse,
        missing_successor_contexts=missing_successors,
        certified=certified,
        status=status,
    )


def evaluate_suffix_interface(
    result: SuffixLearningResult,
    episodes: Iterable[Iterable[int]],
) -> dict[str, object]:
    materialized = normalize_episodes(episodes)
    eligible = transition_samples(materialized, result.final_order)
    correct = 0
    wrong = 0
    abstained = 0
    errors = []
    for sample in eligible:
        prediction = result.predict_next(sample.observation_prefix)
        if prediction is None:
            abstained += 1
        elif prediction == sample.target:
            correct += 1
        else:
            wrong += 1
            errors.append(
                {
                    "episode_index": sample.episode_index,
                    "time_index": sample.time_index,
                    "context": sample.context,
                    "predicted": prediction,
                    "observed": sample.target,
                }
            )
    covered = correct + wrong
    return {
        "eligible_transitions": len(eligible),
        "correct": correct,
        "wrong": wrong,
        "abstained": abstained,
        "coverage": 0.0 if not eligible else covered / len(eligible),
        "accuracy_on_covered": None if covered == 0 else correct / covered,
        "exact_on_eligible_transitions": wrong == 0 and abstained == 0,
        "errors": errors,
    }


@dataclass(frozen=True)
class MajorityContextModel:
    order: int
    predictions: tuple[tuple[Context, Observation], ...]
    target_counts: tuple[tuple[Context, tuple[tuple[Observation, int], ...]], ...]

    @property
    def prediction_map(self) -> dict[Context, Observation]:
        return dict(self.predictions)

    def predict_next(self, history: Sequence[int]) -> int | None:
        if self.order == 0:
            context: Context = ()
        elif len(history) >= self.order:
            context = tuple(int(value) for value in history[-self.order :])
        else:
            return None
        return self.prediction_map.get(context)


def _baseline_samples(
    episodes: Sequence[Episode],
    order: int,
) -> tuple[TransitionSample, ...]:
    if order < 0:
        raise ValueError("baseline order must be non-negative")
    if order > 0:
        return transition_samples(episodes, order)
    samples = []
    for episode_index, episode in enumerate(episodes):
        for time_index in range(len(episode) - 1):
            samples.append(
                TransitionSample(
                    episode_index=episode_index,
                    time_index=time_index,
                    context=(),
                    target=episode[time_index + 1],
                    observation_prefix=episode[: time_index + 1],
                )
            )
    return tuple(samples)


def fit_majority_context_model(
    episodes: Iterable[Iterable[int]],
    order: int,
) -> MajorityContextModel:
    materialized = normalize_episodes(episodes)
    counts: dict[Context, Counter[int]] = {}
    for sample in _baseline_samples(materialized, order):
        counts.setdefault(sample.context, Counter())[sample.target] += 1
    predictions = []
    records = []
    for context, target_counts in sorted(counts.items()):
        # Stable tie break: largest support, then smallest symbol.
        prediction = min(target_counts, key=lambda value: (-target_counts[value], value))
        predictions.append((context, prediction))
        records.append((context, tuple(sorted(target_counts.items()))))
    return MajorityContextModel(order, tuple(predictions), tuple(records))


def evaluate_majority_context_model(
    model: MajorityContextModel,
    episodes: Iterable[Iterable[int]],
) -> dict[str, object]:
    materialized = normalize_episodes(episodes)
    samples = _baseline_samples(materialized, model.order)
    correct = 0
    wrong = 0
    abstained = 0
    errors = []
    for sample in samples:
        prediction = model.predict_next(sample.observation_prefix)
        if prediction is None:
            abstained += 1
        elif prediction == sample.target:
            correct += 1
        else:
            wrong += 1
            errors.append(
                {
                    "episode_index": sample.episode_index,
                    "time_index": sample.time_index,
                    "context": sample.context,
                    "predicted": prediction,
                    "observed": sample.target,
                }
            )
    covered = correct + wrong
    return {
        "order": model.order,
        "eligible_transitions": len(samples),
        "correct": correct,
        "wrong": wrong,
        "abstained": abstained,
        "coverage": 0.0 if not samples else covered / len(samples),
        "accuracy_on_covered": None if covered == 0 else correct / covered,
        "exact_on_eligible_transitions": wrong == 0 and abstained == 0,
        "errors": errors,
    }


def rotating_observation_episode(
    initial_a: int,
    initial_b: int,
    initial_counter: int,
    length: int,
) -> Episode:
    """Simulate the hidden infinite-fibre world and return observations only."""

    if initial_a not in (0, 1) or initial_b not in (0, 1):
        raise ValueError("rotating bits must be binary")
    if initial_counter < 0 or length < 2:
        raise ValueError("counter must be non-negative and length at least two")
    state = (initial_a, initial_b, initial_counter)
    observations = []
    for _ in range(length):
        a_value, b_value, n_value = state
        observations.append(a_value)
        state = (b_value, 1 - a_value, n_value + 1)
    return tuple(observations)


def countdown_observation_episode(
    initial_state: int,
    post_absorption_steps: int,
) -> Episode:
    """Simulate a hidden countdown and return its zero-detector observations."""

    if initial_state < 1 or post_absorption_steps < 2:
        raise ValueError("initial countdown must be positive and tail at least two")
    return (0,) * initial_state + (1,) * post_absorption_steps


@dataclass(frozen=True)
class CompiledSuffixInterface:
    contexts: tuple[Context, ...]
    symbols: tuple[str, ...]
    alphabet: tuple[Observation, ...]
    transition_rows: tuple[tuple[int, ...], ...]
    predictive_laws: tuple[tuple[int, ...], ...]
    successor_symbols: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "contexts": self.contexts,
            "symbols": self.symbols,
            "alphabet": self.alphabet,
            "transition_rows": self.transition_rows,
            "predictive_laws": self.predictive_laws,
            "successor_symbols": self.successor_symbols,
        }


def compile_suffix_interface(result: SuffixLearningResult) -> CompiledSuffixInterface:
    """Compile a certified suffix predictor into an autonomous finite machine."""

    if not result.certified:
        raise ValueError("cannot compile an uncertified suffix interface")
    contexts = tuple(sorted(result.prediction_map))
    indices = {context: index for index, context in enumerate(contexts)}
    symbols = tuple(f"z{index}" for index in range(len(contexts)))
    alphabet = tuple(sorted({target for target in result.prediction_map.values()}))
    outcome_indices = {value: index for index, value in enumerate(alphabet)}
    transition_rows = []
    laws = []
    successor_symbols = []
    for context in contexts:
        target = result.prediction_map[context]
        successor = (*context[1:], target)
        if successor not in indices:
            raise ValueError("learned transition leaves observed context support")
        row = [0] * len(contexts)
        row[indices[successor]] = 1
        transition_rows.append(tuple(row))
        law = [0] * len(alphabet)
        law[outcome_indices[target]] = 1
        laws.append(tuple(law))
        successor_symbols.append(symbols[indices[successor]])
    return CompiledSuffixInterface(
        contexts=contexts,
        symbols=symbols,
        alphabet=alphabet,
        transition_rows=tuple(transition_rows),
        predictive_laws=tuple(laws),
        successor_symbols=tuple(successor_symbols),
    )
