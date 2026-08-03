import pytest

from ts_language_foundations.experiment import (
    run_protocol_trial,
    run_unordered_protocol_trial,
)
from ts_language_foundations.factorization import (
    search_binary_complement,
    search_coordinate_system,
)
from ts_language_foundations.protocol import (
    learn_decoders,
    learn_factorized_dynamics,
    learn_unordered_decoders,
    make_codebook,
    make_unordered_codebook,
)
from ts_language_foundations.repair import dynamic_closure
from ts_language_foundations.world import RotatingBitWorld


def _factor_search():
    world = RotatingBitWorld()
    repaired = dynamic_closure(world.transition_matrix, world.task_partition)
    assert isinstance(repaired, tuple)
    return world, search_binary_complement(
        world.successor_indices,
        world.task_partition,
        repaired,
    )


def test_every_signal_renaming_and_holdout_passes():
    world, factors = _factor_search()
    trials = [
        run_protocol_trial(world, factors, seed, holdout)
        for seed in range(24)
        for holdout in range(4)
    ]
    assert len(trials) == 96
    assert all(trial.passed for trial in trials)


def test_unordered_grammar_is_inferred_for_every_renaming_and_holdout():
    world, _ = _factor_search()
    coordinate_search = search_coordinate_system(
        world.successor_indices,
        tuple(state[0] for state in world.states),
        world.identity_partition,
    )
    trials = [
        run_unordered_protocol_trial(world, coordinate_search, seed, holdout)
        for seed in range(24)
        for holdout in range(4)
    ]
    assert len(trials) == 96
    assert all(trial.passed for trial in trials)


def test_decoder_abstains_on_insufficient_signal_coverage():
    codebook = make_codebook(5)
    with pytest.raises(ValueError, match="exactly two observed signals"):
        learn_decoders(((codebook.encode((0, 0)), (0, 0)),))


def test_dynamics_learner_abstains_on_ambiguous_single_observation():
    assert learn_factorized_dynamics((((0, 0), (0, 1)),)) is None


def test_signal_renaming_changes_surface_not_verified_consequences():
    left = make_codebook(1)
    right = make_codebook(2)
    messages_left = tuple(left.encode(coordinate) for coordinate in ((0, 0), (0, 1), (1, 0), (1, 1)))
    messages_right = tuple(right.encode(coordinate) for coordinate in ((0, 0), (0, 1), (1, 0), (1, 1)))
    assert messages_left != messages_right
    assert len(set(messages_left)) == len(set(messages_right)) == 4


def test_all_four_signal_renamings_are_enumerated_once():
    ordered = {str(make_codebook(index).signal_for_value) for index in range(24)}
    unordered = {
        str(make_unordered_codebook(index, (2, 2)).signal_for_value)
        for index in range(24)
    }
    assert len(ordered) == len(unordered) == 24
    with pytest.raises(ValueError, match="renaming index"):
        make_codebook(24)


def test_unordered_grammar_learner_requires_grounding_for_every_atom():
    codebook = make_unordered_codebook(7, (2, 2))
    with pytest.raises(ValueError, match="all atomic signals"):
        learn_unordered_decoders(((codebook.encode((0, 0)), (0, 0)),), (2, 2))
