from fractions import Fraction

import pytest

from ts_language_foundations.active import (
    ActionHypothesis,
    BayesianBelief,
    ControlledCounts,
    ControlledRepairConfig,
    ControlledWorld,
    assess_action_stationarity,
    compile_controlled_interface,
    confidence_action_partition,
    controlled_dynamic_closure,
    controlled_lumpability_residual,
    controlled_predictive_residual,
    evaluate_action_probability_table,
    fit_action_probability_table,
    generate_controlled_interactions,
    simulate_controlled_repair,
)


@pytest.fixture(scope="module")
def primary_config():
    return ControlledRepairConfig(maximum_steps=65536)


@pytest.fixture(scope="module")
def primary_active(primary_config):
    return simulate_controlled_repair(
        world=ControlledWorld(),
        policy="active",
        seed=20260802,
        config=primary_config,
    )


def test_anytime_family_budget_covers_every_cell_prefix(primary_config):
    assert primary_config.declared_intrinsic_cell_count == 24
    assert primary_config.anytime_family_cell_budget == 24 * 65536


def test_information_gain_proposes_the_revealing_action():
    belief = BayesianBelief(
        (
            ActionHypothesis("null", None),
            ActionHypothesis("probe-memory", "probe"),
        )
    )
    assert belief.expected_information_gain(0, "baseline") == pytest.approx(0.0)
    assert belief.expected_information_gain(0, "probe") > 0.0
    assert belief.expected_extension_gap("probe") > 0.0


def test_active_discovers_repair_before_uniform():
    config = ControlledRepairConfig(maximum_steps=8192)
    world = ControlledWorld()
    active = simulate_controlled_repair(
        world=world,
        policy="active",
        seed=20260802,
        config=config,
        stop_after_first_repair=True,
    ).result
    uniform = simulate_controlled_repair(
        world=world,
        policy="uniform",
        seed=20260802,
        config=config,
        stop_after_first_repair=True,
    ).result
    assert active.repair_discovered and uniform.repair_discovered
    assert active.first_repair_step < uniform.first_repair_step
    assert active.first_repair_cost < uniform.first_repair_cost


def test_primary_reaches_action_family_fixed_point(primary_active):
    result = primary_active.result
    assert result.certified
    assert result.final_order == 2
    assert result.status == "certified_action_family_memory_within_tolerance"
    assert result.first_repair_step == 1344


def test_passive_certificate_is_policy_relative():
    config = ControlledRepairConfig(maximum_steps=32768)
    simulation = simulate_controlled_repair(
        world=ControlledWorld(),
        policy="passive",
        seed=20260802,
        config=config,
    )
    counts = ControlledCounts(config.actions, config.maximum_order)
    for interaction in simulation.interactions:
        counts.add(interaction)
    baseline = counts.assess_memory(
        1,
        delta=config.delta,
        family_cell_budget=config.anytime_family_cell_budget,
        merge_tolerance=config.merge_tolerance,
        split_tolerance=config.split_tolerance,
        minimum_samples=config.minimum_samples,
        actions=("baseline",),
    )
    assert baseline.status == "certified_within_tolerance"
    assert simulation.result.final_assessment.status == "insufficient_evidence"
    assert not simulation.result.certified


def test_null_action_family_does_not_invent_memory():
    config = ControlledRepairConfig(maximum_steps=32768)
    result = simulate_controlled_repair(
        world=ControlledWorld.null(),
        policy="active",
        seed=20260802,
        config=config,
    ).result
    assert result.certified
    assert result.final_order == 1
    assert not result.repair_discovered


def test_wrong_greedy_proposal_abstains_instead_of_certifying():
    config = ControlledRepairConfig(
        maximum_steps=8192,
        exploration_weight=0.0,
        split_evidence_weight=0.0,
    )
    result = simulate_controlled_repair(
        world=ControlledWorld(),
        policy="greedy_misspecified",
        seed=20260802,
        config=config,
    ).result
    assert result.status == "sample_budget_exhausted"
    assert not result.certified
    assert not result.repair_discovered
    assert dict(result.action_counts)["probe"] == 0


def test_action_predictive_two_blocks_need_four_autonomous_states(
    primary_active, primary_config
):
    result = primary_active.result
    partition = confidence_action_partition(
        result,
        actions=primary_config.actions,
        merge_tolerance=primary_config.merge_tolerance,
        split_tolerance=primary_config.split_tolerance,
        minimum_samples=primary_config.minimum_samples,
    )
    assert partition.certified
    assert partition.blocks == ((0, 1), (2, 3))
    interface = compile_controlled_interface(result, actions=primary_config.actions)
    weights = tuple(Fraction(1, 4) for _ in range(4))
    assert controlled_predictive_residual(interface, partition.blocks, weights) < 0.005
    assert controlled_lumpability_residual(interface, partition.blocks, weights) > 0.68
    assert controlled_dynamic_closure(interface.transition_rows, partition.blocks) == (
        (0,),
        (1,),
        (2,),
        (3,),
    )


def test_controlled_dynamic_closure_requires_every_action():
    deterministic_identity = (
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    deterministic_swap = (
        (Fraction(0), Fraction(1)),
        (Fraction(1), Fraction(0)),
    )
    assert controlled_dynamic_closure(
        (deterministic_identity, deterministic_swap), ((0, 1),)
    ) == ((0, 1),)
    assert controlled_dynamic_closure(
        (deterministic_identity, deterministic_swap), ((0,), (1,))
    ) == ((0,), (1,))


def test_repaired_table_beats_current_observation_on_balanced_actions(
    primary_active, primary_config
):
    heldout = generate_controlled_interactions(
        world=ControlledWorld(),
        actions=primary_config.actions,
        action_costs=primary_config.cost_map,
        episode_count=16,
        episode_length=128,
        history_length=3,
        seed=20260803,
        initial_step=10**9,
    )
    order_one = evaluate_action_probability_table(
        fit_action_probability_table(primary_active.interactions, 1), 1, heldout
    )
    order_two = evaluate_action_probability_table(
        fit_action_probability_table(primary_active.interactions, 2), 2, heldout
    )
    assert order_two["mean_negative_log_likelihood_nats"] < order_one[
        "mean_negative_log_likelihood_nats"
    ]


def test_action_conditioned_drift_is_rejected():
    action_costs = {"baseline": 1.0, "probe": 2.0}
    interaction_count = 128 * (64 - 3)
    interactions = generate_controlled_interactions(
        world=ControlledWorld(drift_step=interaction_count // 2),
        actions=("baseline", "probe"),
        action_costs=action_costs,
        episode_count=128,
        episode_length=64,
        history_length=3,
        seed=20260802,
    )
    result = assess_action_stationarity(
        interactions,
        2,
        actions=("baseline", "probe"),
        delta=0.01,
        merge_tolerance=0.2,
        split_tolerance=0.35,
        minimum_samples=32,
    )
    assert result.status == "action_conditioned_distribution_shift_detected"
    assert result.change_witness[0] == "probe"


def test_controlled_configuration_rejects_missing_costs():
    with pytest.raises(ValueError, match="every action"):
        ControlledRepairConfig(action_costs=(("baseline", 1.0),))

