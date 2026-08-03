from fractions import Fraction

import pytest

from ts_language_foundations.repair import (
    dynamic_closure,
    lumpability_residual,
    predictive_residual,
)
from ts_language_foundations.stochastic import (
    assess_episode_stationarity,
    compile_stochastic_interface,
    confidence_predictive_partition,
    evaluate_probability_table,
    fit_probability_table,
    generate_first_order_episodes,
    generate_second_order_episodes,
    hoeffding_radius,
    learn_distributional_suffix_interface,
    normalize_binary_episodes,
)


PRIMARY_KWARGS = {
    "probability_one_if_older_zero": 0.8,
    "probability_one_if_older_one": 0.2,
    "episode_count": 256,
    "episode_length": 128,
}


def learn_primary(seed=20260802):
    episodes = generate_second_order_episodes(seed=seed, **PRIMARY_KWARGS)
    result = learn_distributional_suffix_interface(
        episodes,
        batch_schedule=(16, 32, 64, 128, 256),
        minimum_samples_per_child=64,
    )
    return episodes, result


def test_hoeffding_radius_shrinks_and_validates_inputs():
    assert hoeffding_radius(1000, 0.01, 64) < hoeffding_radius(100, 0.01, 64)
    with pytest.raises(ValueError):
        hoeffding_radius(100, 0.0, 64)
    with pytest.raises(ValueError):
        normalize_binary_episodes(((0, 2, 1),))
    with pytest.raises((TypeError, ValueError)):
        normalize_binary_episodes((((0, 1, 9), (1, 0, 10)),))


def test_distributional_learner_repairs_to_order_two_and_certifies():
    _, result = learn_primary()
    assert result.certified
    assert result.final_order == 2
    assert len(result.repair_events) == 1
    assert result.repair_events[0]["order_before"] == 1
    assert result.repair_events[0]["order_after"] == 2
    assert result.final_assessment.status == "certified_within_tolerance"
    assert set(result.estimate_map) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    for context, estimate in result.estimate_map.items():
        truth = 0.8 if context[0] == 0 else 0.2
        assert estimate.lower_probability_one <= truth <= estimate.upper_probability_one


def test_confidence_partition_finds_two_predictive_blocks_then_dynamic_four():
    _, result = learn_primary()
    partition = confidence_predictive_partition(
        result.estimate_map,
        merge_tolerance=result.merge_tolerance,
        split_tolerance=result.split_tolerance,
        minimum_samples=result.minimum_samples_per_child,
    )
    assert partition.certified
    assert partition.blocks == ((0, 1), (2, 3))

    compiled = compile_stochastic_interface(result)
    weights = tuple(Fraction(value, sum(compiled.support_counts)) for value in compiled.support_counts)
    assert predictive_residual(compiled.predictive_laws, partition.blocks, weights) < 1e-4
    assert lumpability_residual(compiled.transition_rows, partition.blocks, weights) > 0.68
    assert dynamic_closure(compiled.transition_rows, partition.blocks) == (
        (0,),
        (1,),
        (2,),
        (3,),
    )


def test_first_order_null_certifies_without_false_split():
    episodes = generate_first_order_episodes(
        probability_one_if_current_zero=0.25,
        probability_one_if_current_one=0.75,
        episode_count=256,
        episode_length=128,
        seed=99,
    )
    result = learn_distributional_suffix_interface(
        episodes,
        batch_schedule=(16, 32, 64, 128, 256),
        minimum_samples_per_child=64,
    )
    assert result.certified
    assert result.final_order == 1
    assert result.repair_events == ()


def test_weak_signal_remains_uncertain_instead_of_forcing_a_split():
    episodes = generate_second_order_episodes(
        probability_one_if_older_zero=0.6,
        probability_one_if_older_one=0.4,
        episode_count=256,
        episode_length=128,
        seed=20260802,
    )
    result = learn_distributional_suffix_interface(
        episodes,
        batch_schedule=(16, 32, 64, 128, 256),
        minimum_samples_per_child=64,
    )
    assert not result.certified
    assert result.status == "sample_budget_exhausted"
    assert result.final_order == 1
    assert result.repair_events == ()


def test_regime_drift_is_detected():
    first = generate_second_order_episodes(
        probability_one_if_older_zero=0.8,
        probability_one_if_older_one=0.2,
        episode_count=128,
        episode_length=128,
        seed=1,
    )
    second = generate_second_order_episodes(
        probability_one_if_older_zero=0.2,
        probability_one_if_older_one=0.8,
        episode_count=128,
        episode_length=128,
        seed=2,
    )
    assessment = assess_episode_stationarity(
        first + second,
        2,
        delta=0.01,
        family_cell_budget=64,
        merge_tolerance=0.12,
        split_tolerance=0.25,
        minimum_samples_per_half=64,
    )
    assert assessment.status == "distribution_shift_detected"
    assert assessment.change_witness is not None
    assert assessment.change_witness.lower_difference > 0.5


def test_order_two_probability_model_beats_shorter_baselines_on_log_loss():
    train = generate_second_order_episodes(seed=20260802, **PRIMARY_KWARGS)
    test = generate_second_order_episodes(
        probability_one_if_older_zero=0.8,
        probability_one_if_older_one=0.2,
        episode_count=64,
        episode_length=128,
        seed=20260803,
        initial_counter=10**9,
    )
    evaluations = {
        order: evaluate_probability_table(
            fit_probability_table(train, order), order, test
        )
        for order in (0, 1, 2)
    }
    assert evaluations[2]["accuracy"] > 0.78
    assert evaluations[2]["mean_negative_log_likelihood_nats"] < 0.52
    assert evaluations[2]["mean_negative_log_likelihood_nats"] < evaluations[1][
        "mean_negative_log_likelihood_nats"
    ]
    assert evaluations[2]["mean_brier_score"] < evaluations[1]["mean_brier_score"]


def test_ten_seed_recovery_and_null_false_split_control():
    recovered = 0
    false_splits = 0
    for index in range(10):
        episodes = generate_second_order_episodes(seed=1000 + index, **PRIMARY_KWARGS)
        learned = learn_distributional_suffix_interface(
            episodes,
            batch_schedule=(16, 32, 64, 128, 256),
            minimum_samples_per_child=64,
        )
        recovered += int(learned.certified and learned.final_order == 2)

        null = generate_first_order_episodes(
            probability_one_if_current_zero=0.25,
            probability_one_if_current_one=0.75,
            episode_count=256,
            episode_length=128,
            seed=2000 + index,
        )
        null_result = learn_distributional_suffix_interface(
            null,
            batch_schedule=(16, 32, 64, 128, 256),
            minimum_samples_per_child=64,
        )
        false_splits += int(bool(null_result.repair_events))
    assert recovered == 10
    assert false_splits == 0
