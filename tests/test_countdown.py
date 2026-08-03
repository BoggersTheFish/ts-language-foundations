from itertools import combinations

import pytest

from ts_language_foundations.infinite import (
    CountdownPartition,
    CountdownWorld,
    countdown_exact_closure_entropy_excess,
    countdown_lumpability_residual,
    countdown_partition_entropy,
    geometric_entropy,
    numerical_countdown_lumpability_residual,
)


@pytest.mark.parametrize("horizon", [1, 2, 4, 8, 16, 32])
def test_horizon_partition_is_exact_predictive_factorisation(horizon):
    world = CountdownWorld()
    partition = world.horizon_partition(horizon)
    words_by_label = {}
    for state in range(4 * horizon + 17):
        label = partition.label(state)
        word = world.horizon_word(state, horizon)
        assert word == world.word_from_partition_label(label, horizon)
        words_by_label.setdefault(label, word)
        assert words_by_label[label] == word
    assert len(set(words_by_label.values())) == horizon + 1
    assert partition.block_count == horizon + 1


def test_every_finite_tail_has_a_lumpability_counterexample():
    for threshold in range(1, 257):
        partition = CountdownPartition(threshold)
        for right_state in (threshold + 1, threshold + 2, 2 * threshold + 17):
            witness = partition.separation_witness(right_state)
            assert witness["shared_tail_label"] == threshold
            assert witness["left_successor_label"] == threshold - 1
            assert witness["right_successor_label"] == threshold
        assert not partition.strongly_lumpable


def test_refinement_trace_never_reaches_fixed_point_on_checked_prefix():
    world = CountdownWorld()
    trace = world.repair_trace(horizon=4, steps=512)
    assert len(trace) == 513
    assert [partition.threshold for partition in trace[:4]] == [4, 5, 6, 7]
    assert trace[-1].threshold == 516
    assert all(not partition.strongly_lumpable for partition in trace)
    certificate = world.induction_certificate(4, 512)
    assert certificate["all_checked_steps_passed"]


@pytest.mark.parametrize("probability", [0.1, 0.25, 0.5, 0.8])
@pytest.mark.parametrize("threshold", [1, 2, 4, 8, 16, 32])
def test_analytic_lumpability_residual_matches_independent_numeric_sum(
    probability,
    threshold,
):
    analytic = countdown_lumpability_residual(threshold, probability)
    numeric = numerical_countdown_lumpability_residual(
        threshold,
        probability,
        maximum_state=10_000,
    )
    assert abs(analytic - numeric) < 1e-10


@pytest.mark.parametrize("probability", [0.1, 0.25, 0.5, 0.8])
@pytest.mark.parametrize("threshold", [1, 4, 16, 64])
def test_entropy_excess_formula_matches_full_minus_interface_entropy(
    probability,
    threshold,
):
    observed = geometric_entropy(probability) - countdown_partition_entropy(
        threshold,
        probability,
    )
    expected = countdown_exact_closure_entropy_excess(threshold, probability)
    assert abs(observed - expected) < 1e-12


def test_residual_can_be_tiny_while_every_finite_partition_remains_nonclosed():
    threshold = 128
    probability = 0.25
    partition = CountdownPartition(threshold)
    assert countdown_lumpability_residual(threshold, probability) < 1e-15
    assert not partition.strongly_lumpable


def test_next_horizon_and_next_dynamic_repair_each_add_one_boundary_block():
    world = CountdownWorld()
    for horizon in range(1, 64):
        current = world.horizon_partition(horizon)
        next_horizon = world.horizon_partition(horizon + 1)
        dynamic_step = current.split_tail_once()
        assert next_horizon.finite_description == dynamic_step.finite_description
        assert next_horizon.block_count - current.block_count == 1


def test_adaptive_repair_splits_only_current_tail_using_one_longer_derivative():
    world = CountdownWorld()
    for horizon in range(1, 129):
        event = world.adaptive_repair_record(horizon)
        assert event["passed"]
        assert event["finite_block_count_after"] - event["finite_block_count_before"] == 1
        assert event["tail_fibre_cardinality_before"] == "countably_infinite"
        assert event["symbol_split"]["new_tail_fibre_cardinality"] == "countably_infinite"
        assert (
            event["verifier_counterexample"]["shared_current_signature"]
            == (0,) * horizon
        )
        assert event["symbol_split"]["boundary_child_signature"] == (
            (0,) * horizon + (1,)
        )


def test_finite_horizon_words_are_pairwise_distinct_before_the_tail():
    world = CountdownWorld()
    horizon = 32
    words = [world.horizon_word(state, horizon) for state in range(horizon + 1)]
    for left, right in combinations(words, 2):
        assert left != right
