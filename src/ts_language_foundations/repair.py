"""Exact finite-state representation repair for the protocol fixture.

The implementation uses :class:`fractions.Fraction` for transition and
predictive laws.  It is intentionally small and independent of any neural or
language-model library.
"""

from __future__ import annotations

from fractions import Fraction
from math import log
from typing import Iterable, Sequence

Partition = tuple[tuple[int, ...], ...]
Matrix = tuple[tuple[Fraction, ...], ...]
Laws = tuple[tuple[Fraction, ...], ...]


def normalize_partition(partition: Iterable[Iterable[int]]) -> Partition:
    blocks = [tuple(sorted(int(state) for state in block)) for block in partition]
    blocks = [block for block in blocks if block]
    blocks.sort(key=lambda block: block[0])
    return tuple(blocks)


def validate_partition(partition: Partition, state_count: int) -> None:
    flat = [state for block in partition for state in block]
    if sorted(flat) != list(range(state_count)):
        raise ValueError(
            "partition must contain every state exactly once: "
            f"expected 0..{state_count - 1}, got {sorted(flat)}"
        )


def as_fraction_matrix(rows: Sequence[Sequence[int | Fraction]]) -> Matrix:
    matrix = tuple(tuple(Fraction(value) for value in row) for row in rows)
    state_count = len(matrix)
    if state_count == 0 or any(len(row) != state_count for row in matrix):
        raise ValueError("transition matrix must be non-empty and square")
    for row in matrix:
        if any(value < 0 for value in row) or sum(row) != 1:
            raise ValueError("transition rows must be non-negative and sum to one")
    return matrix


def as_fraction_laws(rows: Sequence[Sequence[int | Fraction]]) -> Laws:
    laws = tuple(tuple(Fraction(value) for value in row) for row in rows)
    if not laws or not laws[0]:
        raise ValueError("predictive laws must be non-empty")
    outcome_count = len(laws[0])
    for row in laws:
        if len(row) != outcome_count:
            raise ValueError("predictive-law rows must have equal length")
        if any(value < 0 for value in row) or sum(row) != 1:
            raise ValueError("predictive-law rows must be non-negative and sum to one")
    return laws


def partition_labels(partition: Partition, state_count: int) -> tuple[int, ...]:
    validate_partition(partition, state_count)
    labels = [-1] * state_count
    for block_index, block in enumerate(partition):
        for state in block:
            labels[state] = block_index
    return tuple(labels)


def refines(finer: Partition, coarser: Partition) -> bool:
    coarser_sets = [set(block) for block in coarser]
    return all(any(set(block) <= target for target in coarser_sets) for block in finer)


def common_refinement(left: Partition, right: Partition) -> Partition:
    intersections = []
    for left_block in left:
        for right_block in right:
            block = tuple(sorted(set(left_block) & set(right_block)))
            if block:
                intersections.append(block)
    return normalize_partition(intersections)


def predictive_partition(laws: Laws) -> Partition:
    groups: dict[tuple[Fraction, ...], list[int]] = {}
    for state, law in enumerate(laws):
        groups.setdefault(law, []).append(state)
    return normalize_partition(groups.values())


def predictive_repair(partition: Partition, task_partition: Partition) -> Partition:
    return common_refinement(partition, task_partition)


def projected_transition_rows(matrix: Matrix, partition: Partition) -> Matrix:
    state_count = len(matrix)
    labels = partition_labels(partition, state_count)
    rows: list[tuple[Fraction, ...]] = []
    for source in range(state_count):
        masses = [Fraction(0) for _ in partition]
        for target, probability in enumerate(matrix[source]):
            masses[labels[target]] += probability
        rows.append(tuple(masses))
    return tuple(rows)


def is_strongly_lumpable(matrix: Matrix, partition: Partition) -> bool:
    projected = projected_transition_rows(matrix, partition)
    return all(all(projected[state] == projected[block[0]] for state in block) for block in partition)


def dynamic_closure(
    matrix: Matrix,
    partition: Partition,
    *,
    return_trace: bool = False,
) -> Partition | tuple[Partition, tuple[Partition, ...]]:
    """Return the coarsest strongly lumpable refinement of ``partition``."""

    current = normalize_partition(partition)
    validate_partition(current, len(matrix))
    trace = [current]

    while True:
        labels = partition_labels(current, len(matrix))
        projected = projected_transition_rows(matrix, current)
        groups: dict[tuple[int, tuple[Fraction, ...]], list[int]] = {}
        for state in range(len(matrix)):
            signature = (labels[state], projected[state])
            groups.setdefault(signature, []).append(state)
        nxt = normalize_partition(groups.values())
        if nxt == current:
            if return_trace:
                return current, tuple(trace)
            return current
        current = nxt
        trace.append(current)


def joint_repair(
    matrix: Matrix,
    laws: Laws,
    partition: Partition,
    *,
    schedule: str = "predictive_then_dynamic",
    return_trace: bool = False,
) -> Partition | tuple[Partition, tuple[Partition, ...]]:
    task_partition = predictive_partition(laws)
    current = normalize_partition(partition)
    trace = [current]

    for _ in range(1 + len(matrix)):
        if schedule == "predictive_then_dynamic":
            repaired = predictive_repair(current, task_partition)
            nxt = dynamic_closure(matrix, repaired)
        elif schedule == "dynamic_then_predictive":
            closed = dynamic_closure(matrix, current)
            assert isinstance(closed, tuple)
            nxt = predictive_repair(closed, task_partition)
        else:
            raise ValueError("unknown repair schedule")
        assert isinstance(nxt, tuple)
        if nxt == current:
            if return_trace:
                return current, tuple(trace)
            return current
        current = nxt
        trace.append(current)
    raise RuntimeError("joint repair failed to converge on a finite partition lattice")


def _weights(state_count: int, weights: Sequence[Fraction] | None) -> tuple[Fraction, ...]:
    if weights is None:
        return tuple(Fraction(1, state_count) for _ in range(state_count))
    result = tuple(Fraction(value) for value in weights)
    if len(result) != state_count or any(value < 0 for value in result):
        raise ValueError("weights must be finite, non-negative, and state-aligned")
    total = sum(result)
    if total <= 0:
        raise ValueError("weights must have positive total mass")
    return tuple(value / total for value in result)


def _kl(probability: Sequence[Fraction], reference: Sequence[Fraction]) -> float:
    total = 0.0
    for p_value, r_value in zip(probability, reference):
        if p_value == 0:
            continue
        if r_value <= 0:
            return float("inf")
        p_float = float(p_value)
        total += p_float * log(p_float / float(r_value))
    return total


def predictive_residual(
    laws: Laws,
    partition: Partition,
    weights: Sequence[Fraction] | None = None,
) -> float:
    state_weights = _weights(len(laws), weights)
    validate_partition(partition, len(laws))
    total = 0.0
    for block in partition:
        mass = sum(state_weights[state] for state in block)
        if mass == 0:
            continue
        barycentre = tuple(
            sum(state_weights[state] * laws[state][outcome] for state in block) / mass
            for outcome in range(len(laws[0]))
        )
        total += sum(
            float(state_weights[state]) * _kl(laws[state], barycentre)
            for state in block
            if state_weights[state] > 0
        )
    return total


def lumpability_residual(
    matrix: Matrix,
    partition: Partition,
    weights: Sequence[Fraction] | None = None,
) -> float:
    state_weights = _weights(len(matrix), weights)
    projected = projected_transition_rows(matrix, partition)
    total = 0.0
    for block in partition:
        mass = sum(state_weights[state] for state in block)
        if mass == 0:
            continue
        barycentre = tuple(
            sum(state_weights[state] * projected[state][target] for state in block) / mass
            for target in range(len(partition))
        )
        total += sum(
            float(state_weights[state]) * _kl(projected[state], barycentre)
            for state in block
            if state_weights[state] > 0
        )
    return total


def partition_entropy(
    partition: Partition,
    weights: Sequence[Fraction] | None = None,
) -> float:
    state_count = sum(len(block) for block in partition)
    state_weights = _weights(state_count, weights)
    total = 0.0
    for block in partition:
        mass = float(sum(state_weights[state] for state in block))
        if mass > 0:
            total -= mass * log(mass)
    return total

