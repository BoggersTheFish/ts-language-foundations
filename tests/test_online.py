import pytest

from ts_language_foundations.online import (
    compile_suffix_interface,
    countdown_observation_episode,
    evaluate_majority_context_model,
    evaluate_suffix_interface,
    fit_majority_context_model,
    learn_suffix_interface,
    normalize_episodes,
    rotating_observation_episode,
)


def stable_training_episodes():
    return tuple(
        rotating_observation_episode(a_value, b_value, counter, 32)
        for counter in (0, 3, 17)
        for a_value in (0, 1)
        for b_value in (0, 1)
    )


def stable_test_episodes():
    return tuple(
        rotating_observation_episode(a_value, b_value, 10**6, 64)
        for a_value in (0, 1)
        for b_value in (0, 1)
    )


def test_observation_payload_rejects_nested_hidden_states():
    with pytest.raises((TypeError, ValueError)):
        normalize_episodes((((0, 1, 7), (1, 1, 8)),))


def test_stable_world_repairs_one_bit_memory_to_two_and_certifies():
    result = learn_suffix_interface(
        stable_training_episodes(),
        initial_order=1,
        maximum_order=6,
        minimum_context_support=8,
    )
    assert result.certified
    assert result.final_order == 2
    assert len(result.repair_events) == 1
    assert result.repair_events[0]["order_before"] == 1
    assert result.repair_events[0]["order_after"] == 2
    assert set(result.contexts) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert result.prediction_map == {
        (0, 0): 1,
        (0, 1): 1,
        (1, 0): 0,
        (1, 1): 0,
    }


def test_compiled_observation_interface_is_the_four_cycle():
    result = learn_suffix_interface(
        stable_training_episodes(), maximum_order=2, minimum_context_support=8
    )
    compiled = compile_suffix_interface(result)
    assert compiled.contexts == ((0, 0), (0, 1), (1, 0), (1, 1))
    successor_by_context = dict(zip(compiled.contexts, compiled.successor_symbols))
    symbol_by_context = dict(zip(compiled.contexts, compiled.symbols))
    assert successor_by_context == {
        (0, 0): symbol_by_context[(0, 1)],
        (0, 1): symbol_by_context[(1, 1)],
        (1, 0): symbol_by_context[(0, 0)],
        (1, 1): symbol_by_context[(1, 0)],
    }


def test_stable_interface_transfers_to_unseen_infinite_fibre_indices():
    result = learn_suffix_interface(
        stable_training_episodes(), maximum_order=6, minimum_context_support=8
    )
    evaluation = evaluate_suffix_interface(result, stable_test_episodes())
    assert evaluation["coverage"] == 1.0
    assert evaluation["accuracy_on_covered"] == 1.0
    assert evaluation["exact_on_eligible_transitions"]


def test_fixed_baselines_expose_the_memory_threshold():
    train = stable_training_episodes()
    test = stable_test_episodes()
    global_result = evaluate_majority_context_model(
        fit_majority_context_model(train, 0), test
    )
    current_result = evaluate_majority_context_model(
        fit_majority_context_model(train, 1), test
    )
    order_two_result = evaluate_majority_context_model(
        fit_majority_context_model(train, 2), test
    )
    assert global_result["accuracy_on_covered"] == 0.5
    assert current_result["accuracy_on_covered"] == 0.5
    assert order_two_result["accuracy_on_covered"] == 1.0
    assert order_two_result["exact_on_eligible_transitions"]


def test_countdown_exhausts_every_finite_suffix_budget_and_abstains():
    episodes = tuple(countdown_observation_episode(state, 12) for state in range(1, 11))
    for maximum_order in (1, 2, 4, 8):
        result = learn_suffix_interface(
            episodes,
            maximum_order=maximum_order,
            minimum_context_support=1,
        )
        assert not result.certified
        assert result.status == "memory_budget_exhausted"
        assert result.final_order == maximum_order
        witness = result.terminal_counterexample
        assert witness is not None and witness.passed
        assert witness.context == (0,) * maximum_order
        evaluation = evaluate_suffix_interface(result, episodes)
        assert evaluation["coverage"] == 0.0
        assert evaluation["wrong"] == 0


def test_high_accuracy_countdown_baseline_is_not_exact():
    train = tuple(countdown_observation_episode(state, 16) for state in range(1, 13))
    test = tuple(countdown_observation_episode(state, 16) for state in range(13, 21))
    current = evaluate_majority_context_model(
        fit_majority_context_model(train, 1), test
    )
    order_eight = evaluate_majority_context_model(
        fit_majority_context_model(train, 8), test
    )
    assert current["accuracy_on_covered"] > 0.9
    assert order_eight["accuracy_on_covered"] > 0.9
    assert current["wrong"] == 8
    assert order_eight["wrong"] == 8
    assert not current["exact_on_eligible_transitions"]
    assert not order_eight["exact_on_eligible_transitions"]


def test_directly_contradictory_observations_fail_closed():
    result = learn_suffix_interface(
        ((0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1)),
        maximum_order=4,
    )
    assert not result.certified
    assert result.status == "memory_budget_exhausted"
    assert result.terminal_counterexample is not None
    with pytest.raises(ValueError):
        compile_suffix_interface(result)
