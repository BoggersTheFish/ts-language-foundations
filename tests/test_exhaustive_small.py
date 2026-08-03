"""Exhaustive finite verification is a receipt, not a general proof."""

from itertools import product

from ts_language_foundations.repair import (
    as_fraction_matrix,
    dynamic_closure,
    is_strongly_lumpable,
    normalize_partition,
    refines,
)


def all_partitions(state_count: int):
    labels = [0] * state_count

    def visit(position: int, current_max: int):
        if position == state_count:
            blocks = [[] for _ in range(current_max + 1)]
            for state, label in enumerate(labels):
                blocks[label].append(state)
            yield normalize_partition(blocks)
            return
        for label in range(current_max + 2):
            labels[position] = label
            yield from visit(position + 1, max(current_max, label))

    # Restricted-growth strings start with label zero.
    yield from visit(1, 0)


def test_dynamic_closure_is_coarsest_lumpable_refinement_for_all_four_state_functions():
    partitions = tuple(all_partitions(4))
    assert len(partitions) == 15
    checked = 0
    for successors in product(range(4), repeat=4):
        matrix = as_fraction_matrix(
            [[int(target == successors[source]) for target in range(4)] for source in range(4)]
        )
        for partition in partitions:
            closed = dynamic_closure(matrix, partition)
            assert isinstance(closed, tuple)
            assert refines(closed, partition)
            assert is_strongly_lumpable(matrix, closed)
            for candidate in partitions:
                if refines(candidate, partition) and is_strongly_lumpable(matrix, candidate):
                    assert refines(candidate, closed)
            checked += 1
    assert checked == 256 * 15
