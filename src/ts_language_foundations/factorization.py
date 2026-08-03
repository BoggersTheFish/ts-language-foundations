"""Search for compact binary coordinates inside a repaired finite state."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import ceil, log2, prod

from .repair import Partition, common_refinement, normalize_partition, partition_labels


def balanced_binary_partitions(state_count: int) -> tuple[Partition, ...]:
    if state_count < 2 or state_count % 2:
        raise ValueError("balanced binary partitions require a positive even state count")
    half = state_count // 2
    states = set(range(state_count))
    partitions = []
    # Requiring state zero in the first block removes label-complement duplicates.
    for tail in combinations(range(1, state_count), half - 1):
        left = (0, *tail)
        right = tuple(sorted(states - set(left)))
        partitions.append(normalize_partition((left, right)))
    return tuple(partitions)


def coordinate_pairs(alpha: Partition, beta: Partition) -> tuple[tuple[int, int], ...]:
    state_count = sum(len(block) for block in alpha)
    if len(alpha) != 2 or len(beta) != 2:
        raise ValueError("coordinate factors must both be binary")
    alpha_labels = partition_labels(alpha, state_count)
    beta_labels = partition_labels(beta, state_count)
    return tuple(zip(alpha_labels, beta_labels))


def truth_tables(
    successor_indices: tuple[int, ...],
    alpha: Partition,
    beta: Partition,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    coordinates = coordinate_pairs(alpha, beta)
    if set(coordinates) != {(0, 0), (0, 1), (1, 0), (1, 1)}:
        raise ValueError("the factors do not form a complete binary product")
    lookup = {coordinate: state for state, coordinate in enumerate(coordinates)}
    outputs = [[], []]
    for coordinate in ((0, 0), (0, 1), (1, 0), (1, 1)):
        state = lookup[coordinate]
        next_coordinate = coordinates[successor_indices[state]]
        outputs[0].append(next_coordinate[0])
        outputs[1].append(next_coordinate[1])
    return tuple(outputs[0]), tuple(outputs[1])


def essential_inputs(table: tuple[int, ...]) -> tuple[int, ...]:
    if len(table) != 4 or any(value not in (0, 1) for value in table):
        raise ValueError("a binary two-input truth table must contain four bits")
    dependencies = []
    for variable in (0, 1):
        essential = False
        for a_value in (0, 1):
            for b_value in (0, 1):
                left = (a_value, b_value)
                right = list(left)
                right[variable] = 1 - right[variable]
                left_index = 2 * left[0] + left[1]
                right_index = 2 * right[0] + right[1]
                if table[left_index] != table[right_index]:
                    essential = True
        if essential:
            dependencies.append(variable)
    return tuple(dependencies)


def dependency_score(tables: tuple[tuple[int, ...], tuple[int, ...]]) -> int:
    return sum(len(essential_inputs(table)) for table in tables)


@dataclass(frozen=True)
class FactorCandidate:
    complement: Partition
    coordinate_pairs: tuple[tuple[int, int], ...]
    truth_tables: tuple[tuple[int, ...], tuple[int, ...]]
    essential_inputs: tuple[tuple[int, ...], tuple[int, ...]]
    dependency_score: int

    def to_dict(self) -> dict[str, object]:
        return {
            "complement_partition": self.complement,
            "state_coordinates": self.coordinate_pairs,
            "truth_tables": self.truth_tables,
            "essential_inputs": self.essential_inputs,
            "dependency_score": self.dependency_score,
        }


@dataclass(frozen=True)
class FactorSearchResult:
    task_factor: Partition
    candidates: tuple[FactorCandidate, ...]
    selected: FactorCandidate | None
    abstention_reason: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "task_factor": self.task_factor,
            "candidate_count": len(self.candidates),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "selected": None if self.selected is None else self.selected.to_dict(),
            "abstention_reason": self.abstention_reason,
            "selection_rule": (
                "unique minimum total essential-input count across the two "
                "coordinate update truth tables"
            ),
        }


def search_binary_complement(
    successor_indices: tuple[int, ...],
    task_factor: Partition,
    repaired_partition: Partition,
) -> FactorSearchResult:
    """Find a balanced factor complementing the registered task factor.

    The candidate family and the essential-input score are declared scaffolds,
    not universal consequences of representation repair.
    """

    state_count = len(successor_indices)
    candidates = []
    for complement in balanced_binary_partitions(state_count):
        if complement == task_factor:
            continue
        if common_refinement(task_factor, complement) != repaired_partition:
            continue
        tables = truth_tables(successor_indices, task_factor, complement)
        dependencies = tuple(essential_inputs(table) for table in tables)
        candidates.append(
            FactorCandidate(
                complement=complement,
                coordinate_pairs=coordinate_pairs(task_factor, complement),
                truth_tables=tables,
                essential_inputs=dependencies,
                dependency_score=dependency_score(tables),
            )
        )

    ordered = tuple(sorted(candidates, key=lambda item: (item.dependency_score, item.complement)))
    if not ordered:
        return FactorSearchResult(task_factor, ordered, None, "no product complement found")
    best_score = ordered[0].dependency_score
    best = [candidate for candidate in ordered if candidate.dependency_score == best_score]
    if len(best) != 1:
        return FactorSearchResult(
            task_factor,
            ordered,
            None,
            f"{len(best)} candidates tie at dependency score {best_score}",
        )
    return FactorSearchResult(task_factor, ordered, best[0], None)


def _factor_coordinates(factors: tuple[Partition, ...]) -> tuple[tuple[int, ...], ...]:
    state_count = sum(len(block) for block in factors[0])
    label_columns = [partition_labels(factor, state_count) for factor in factors]
    return tuple(
        tuple(labels[state] for labels in label_columns) for state in range(state_count)
    )


def _determining_subsets(
    coordinates: tuple[tuple[int, ...], ...],
    outputs: tuple[int, ...],
    radices: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    compatible = []
    factor_count = len(radices)
    for subset_size in range(factor_count + 1):
        for subset in combinations(range(factor_count), subset_size):
            table: dict[tuple[int, ...], int] = {}
            valid = True
            for coordinate, output in zip(coordinates, outputs):
                key = tuple(coordinate[index] for index in subset)
                if key in table and table[key] != output:
                    valid = False
                    break
                table[key] = output
            if valid:
                compatible.append(subset)
    if not compatible:
        raise AssertionError("the full coordinate system must determine every state function")
    minimum_cells = min(prod(radices[index] for index in subset) for subset in compatible)
    return tuple(
        subset
        for subset in compatible
        if prod(radices[index] for index in subset) == minimum_cells
    )


def _table_bits(radices: tuple[int, ...], subset: tuple[int, ...], output_radix: int) -> int:
    cells = prod(radices[index] for index in subset)
    bits_per_cell = max(1, ceil(log2(output_radix)))
    return cells * bits_per_cell


@dataclass(frozen=True)
class CoordinateSystemCandidate:
    factors: tuple[Partition, ...]
    radices: tuple[int, ...]
    state_coordinates: tuple[tuple[int, ...], ...]
    dynamics_input_subsets: tuple[tuple[int, ...], ...]
    dynamics_table_bits: int
    task_input_subset: tuple[int, ...]
    task_table_bits: int
    total_model_bits: int
    signal_inventory_size: int
    message_arity: int

    def to_dict(self) -> dict[str, object]:
        return {
            "factors": self.factors,
            "radices": self.radices,
            "state_coordinates": self.state_coordinates,
            "dynamics_input_subsets": self.dynamics_input_subsets,
            "dynamics_table_bits": self.dynamics_table_bits,
            "task_input_subset": self.task_input_subset,
            "task_table_bits": self.task_table_bits,
            "total_model_bits": self.total_model_bits,
            "signal_inventory_size": self.signal_inventory_size,
            "message_arity": self.message_arity,
            "is_product_code": len(self.factors) > 1,
        }


@dataclass(frozen=True)
class CoordinateSystemSearchResult:
    candidates: tuple[CoordinateSystemCandidate, ...]
    selected: CoordinateSystemCandidate | None
    abstention_reason: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_count": len(self.candidates),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "selected": None if self.selected is None else self.selected.to_dict(),
            "abstention_reason": self.abstention_reason,
            "selection_rule": (
                "unique minimum fixed-width lookup-table bits for registered-task "
                "and autonomous-dynamics laws"
            ),
            "scope": (
                "one four-valued holistic factor versus irredundant pairs of "
                "balanced binary factors"
            ),
        }


def search_coordinate_system(
    successor_indices: tuple[int, ...],
    task_outputs: tuple[int, ...],
    repaired_partition: Partition,
) -> CoordinateSystemSearchResult:
    """Choose a finite coordinate system without fixing its factor count.

    The score is a declared finite description-length proxy.  It counts the
    fixed-width lookup-table bits required for the registered task and every
    output-coordinate update.  It does not include transmission cost and is
    not claimed to be a universal MDL objective.
    """

    state_count = len(successor_indices)
    if repaired_partition != tuple((state,) for state in range(state_count)):
        return CoordinateSystemSearchResult((), None, "v0.1 requires singleton repaired states")
    holistic = (repaired_partition,)
    balanced = balanced_binary_partitions(state_count)
    factor_sets: list[tuple[Partition, ...]] = [holistic]
    for left, right in combinations(balanced, 2):
        if common_refinement(left, right) == repaired_partition:
            factor_sets.append((left, right))

    candidates = []
    task_radix = len(set(task_outputs))
    for factors in factor_sets:
        radices = tuple(len(factor) for factor in factors)
        coordinates = _factor_coordinates(factors)
        if len(set(coordinates)) != state_count:
            continue
        dynamics_subsets = []
        dynamics_bits = 0
        for output_coordinate, output_radix in enumerate(radices):
            outputs = tuple(
                coordinates[successor][output_coordinate]
                for successor in successor_indices
            )
            best_subsets = _determining_subsets(coordinates, outputs, radices)
            chosen = min(best_subsets)
            dynamics_subsets.append(chosen)
            dynamics_bits += _table_bits(radices, chosen, output_radix)
        task_subsets = _determining_subsets(coordinates, task_outputs, radices)
        task_subset = min(task_subsets)
        task_bits = _table_bits(radices, task_subset, task_radix)
        candidates.append(
            CoordinateSystemCandidate(
                factors=factors,
                radices=radices,
                state_coordinates=coordinates,
                dynamics_input_subsets=tuple(dynamics_subsets),
                dynamics_table_bits=dynamics_bits,
                task_input_subset=task_subset,
                task_table_bits=task_bits,
                total_model_bits=dynamics_bits + task_bits,
                signal_inventory_size=sum(radices),
                message_arity=len(factors),
            )
        )
    ordered = tuple(
        sorted(candidates, key=lambda item: (item.total_model_bits, item.factors))
    )
    if not ordered:
        return CoordinateSystemSearchResult(ordered, None, "no coordinate system found")
    best_score = ordered[0].total_model_bits
    best = [candidate for candidate in ordered if candidate.total_model_bits == best_score]
    if len(best) != 1:
        return CoordinateSystemSearchResult(
            ordered,
            None,
            f"{len(best)} coordinate systems tie at {best_score} model bits",
        )
    return CoordinateSystemSearchResult(ordered, best[0], None)
