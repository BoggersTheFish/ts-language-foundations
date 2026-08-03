from ts_language_foundations.factorization import (
    balanced_binary_partitions,
    search_binary_complement,
    search_coordinate_system,
)
from ts_language_foundations.repair import dynamic_closure
from ts_language_foundations.world import RotatingBitWorld


def test_all_balanced_binary_partitions_are_enumerated_once():
    assert balanced_binary_partitions(4) == (
        ((0, 1), (2, 3)),
        ((0, 2), (1, 3)),
        ((0, 3), (1, 2)),
    )


def test_minimum_dependency_search_recovers_phase_coordinate_uniquely():
    world = RotatingBitWorld()
    repaired = dynamic_closure(world.transition_matrix, world.task_partition)
    assert isinstance(repaired, tuple)
    result = search_binary_complement(
        world.successor_indices,
        world.task_partition,
        repaired,
    )
    assert result.abstention_reason is None
    assert result.selected is not None
    assert result.selected.complement == ((0, 2), (1, 3))
    assert result.selected.dependency_score == 2
    assert [candidate.dependency_score for candidate in result.candidates] == [2, 3]
    assert result.selected.truth_tables == ((0, 1, 0, 1), (1, 1, 0, 0))


def test_factor_search_abstains_when_repaired_partition_is_incompatible():
    world = RotatingBitWorld()
    result = search_binary_complement(
        world.successor_indices,
        world.task_partition,
        world.task_partition,
    )
    assert result.selected is None
    assert result.abstention_reason == "no product complement found"


def test_factor_search_abstains_on_a_real_score_tie():
    world = RotatingBitWorld()
    result = search_binary_complement(
        (0, 1, 2, 3),  # identity dynamics makes both complements equally local
        world.task_partition,
        world.identity_partition,
    )
    assert result.selected is None
    assert result.abstention_reason == "2 candidates tie at dependency score 2"


def test_coordinate_system_search_selects_factor_count_and_true_product():
    world = RotatingBitWorld()
    result = search_coordinate_system(
        world.successor_indices,
        tuple(state[0] for state in world.states),
        world.identity_partition,
    )
    assert result.abstention_reason is None
    assert result.selected is not None
    assert result.selected.factors == (
        ((0, 1), (2, 3)),
        ((0, 2), (1, 3)),
    )
    assert result.selected.radices == (2, 2)
    assert [candidate.total_model_bits for candidate in result.candidates] == [6, 8, 10, 12]
    assert result.candidates[-1].message_arity == 1  # holistic four-valued code


def test_coordinate_system_search_abstains_on_description_length_tie():
    result = search_coordinate_system(
        (0, 0, 0, 0),
        (0, 0, 0, 1),
        ((0,), (1,), (2,), (3,)),
    )
    assert result.selected is None
    assert result.abstention_reason == "4 coordinate systems tie at 6 model bits"
