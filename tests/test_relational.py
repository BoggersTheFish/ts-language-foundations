from fractions import Fraction

import pytest

from ts_language_foundations.active import (
    controlled_dynamic_closure,
    controlled_lumpability_residual,
    controlled_predictive_residual,
)
from ts_language_foundations.relational import (
    BooleanExpression,
    RelationalRepairConfig,
    RelationalWorld,
    assess_relational_stationarity,
    binary_contexts,
    compile_relational_interface,
    controlled_joint_repair,
    evaluate_representation_probability_table,
    fit_representation_probability_table,
    generate_boolean_features,
    generate_feature_representations,
    generate_relational_interactions,
    oracle_relational_interface,
    partition_from_features,
    raw_identity_representation,
    representation_by_name,
    representation_search_key,
    simulate_relational_repair,
)


@pytest.fixture(scope="module")
def representations():
    return generate_feature_representations()


@pytest.fixture(scope="module")
def primary_config():
    return RelationalRepairConfig(maximum_steps=32768)


@pytest.fixture(scope="module")
def primary_active(primary_config):
    return simulate_relational_repair(
        world=RelationalWorld(),
        policy="active",
        seed=20260802,
        config=primary_config,
    )


@pytest.fixture(scope="module")
def primary_uniform(primary_config):
    return simulate_relational_repair(
        world=RelationalWorld(),
        policy="uniform",
        seed=20260802,
        config=primary_config,
    )


def test_typed_boolean_expressions_evaluate_raw_histories():
    lag0 = BooleanExpression("lag", value=0)
    lag1 = BooleanExpression("lag", value=1)
    xor = BooleanExpression("xor", (lag0, lag1))
    assert xor.type_name == "bool"
    assert xor.complexity == 3
    assert xor.dependencies == (0, 1)
    assert xor.evaluate((1, 0, 1)) == 1
    assert xor.evaluate((0, 1, 1)) == 0


def test_grammar_is_truth_table_deduplicated_and_contains_xor():
    contexts = binary_contexts(3)
    features = generate_boolean_features()
    truth_tables = [feature.truth_table(contexts) for feature in features]
    assert len(features) == 20
    assert len(set(truth_tables)) == len(truth_tables)
    assert "xor(lag0,lag1)" in {feature.render() for feature in features}


def test_representation_family_is_fixed_order_and_partition_deduplicated(
    representations,
):
    assert tuple(sorted(representations, key=representation_search_key)) == representations
    assert len({item.partition for item in representations}) == len(representations)
    assert representations[0].name == "one_state"
    assert representations[0].block_count == 1


def test_xor_is_canonical_but_eq_has_the_same_unlabelled_partition(
    representations,
):
    contexts = binary_contexts(3)
    features = {feature.render(): feature for feature in generate_boolean_features()}
    xor_partition = partition_from_features(contexts, (features["xor(lag0,lag1)"],))
    eq_partition = partition_from_features(contexts, (features["eq(lag0,lag1)"],))
    assert xor_partition == eq_partition
    assert representation_by_name(representations, "xor(lag0,lag1)").partition == xor_partition
    assert "eq(lag0,lag1)" not in {item.name for item in representations}


def test_anytime_budget_covers_all_raw_action_cells_and_prefixes(primary_config):
    assert primary_config.declared_intrinsic_cell_count == 16
    assert primary_config.anytime_family_cell_budget == 16 * 32768


def test_active_constructs_xor_and_beats_uniform_to_the_relation(
    primary_active, primary_uniform
):
    active = primary_active.result
    uniform = primary_uniform.result
    assert active.certified and uniform.certified
    assert active.selected_representation.name == "xor(lag0,lag1)"
    assert uniform.selected_representation.name == "xor(lag0,lag1)"
    assert active.first_relational_witness_step <= 10000
    assert active.first_relational_witness_step < uniform.first_relational_witness_step


def test_selected_candidate_has_a_fail_closed_minimality_certificate(primary_active):
    assessment = primary_active.result.final_assessment
    assert assessment.status == "least_supported_representation_certified"
    assert assessment.selected.name == "xor(lag0,lag1)"
    assert all(
        item.status == "rejected"
        for item in assessment.candidate_assessments[: assessment.selected_index]
    )


def test_predictive_xor_repairs_to_four_state_suffix(primary_active, primary_config):
    selected = primary_active.result.selected_representation
    interface = compile_relational_interface(
        primary_active.result,
        actions=primary_config.actions,
        history_length=3,
        pool_partition=selected.partition,
    )
    weights = tuple(Fraction(1, 8) for _ in range(8))
    closure = controlled_dynamic_closure(interface.transition_rows, selected.partition)
    assert closure == ((0, 4), (1, 5), (2, 6), (3, 7))
    assert controlled_predictive_residual(interface, selected.partition, weights) == 0.0
    assert controlled_lumpability_residual(interface, selected.partition, weights) > 0.13


def test_empirical_and_oracle_dynamic_repairs_agree(primary_active, primary_config):
    selected = primary_active.result.selected_representation
    empirical = compile_relational_interface(
        primary_active.result,
        actions=primary_config.actions,
        history_length=3,
        pool_partition=selected.partition,
    )
    oracle = oracle_relational_interface(RelationalWorld())
    assert controlled_dynamic_closure(
        empirical.transition_rows, selected.partition
    ) == controlled_dynamic_closure(oracle.transition_rows, selected.partition)


def test_noncommuting_partial_repairs_share_a_completed_fixed_point(
    primary_active,
):
    selected = primary_active.result.selected_representation
    oracle = oracle_relational_interface(RelationalWorld())
    one_state = ((0, 1, 2, 3, 4, 5, 6, 7),)
    predictive_then_dynamic = controlled_joint_repair(
        oracle.transition_rows,
        selected.partition,
        one_state,
        schedule="predictive_then_dynamic",
    )
    dynamic_then_predictive = controlled_joint_repair(
        oracle.transition_rows,
        selected.partition,
        one_state,
        schedule="dynamic_then_predictive",
    )
    assert predictive_then_dynamic == dynamic_then_predictive
    assert len(predictive_then_dynamic) == 4


def test_autonomous_repair_is_expressible_as_two_recent_raw_features(
    primary_active, representations
):
    selected = primary_active.result.selected_representation
    interface = compile_relational_interface(
        primary_active.result,
        actions=("baseline", "probe"),
        history_length=3,
        pool_partition=selected.partition,
    )
    closure = controlled_dynamic_closure(interface.transition_rows, selected.partition)
    repaired = representation_by_name(representations, "lag0 + lag1")
    assert repaired.partition == closure


def test_selected_relation_beats_primitive_and_matches_raw_identity_on_holdout(
    primary_active, primary_config, representations
):
    heldout = generate_relational_interactions(
        world=RelationalWorld(),
        actions=primary_config.actions,
        action_costs=primary_config.cost_map,
        interaction_count=8192,
        history_length=3,
        seed=20260803,
        initial_step=10**9,
    )
    selected = primary_active.result.selected_representation
    lag1 = representation_by_name(representations, "lag1")
    identity = raw_identity_representation()

    def evaluate(representation):
        table = fit_representation_probability_table(
            primary_active.interactions,
            representation,
            contexts=primary_active.contexts,
        )
        return evaluate_representation_probability_table(
            table,
            representation,
            heldout,
            contexts=primary_active.contexts,
        )

    selected_score = evaluate(selected)
    primitive_score = evaluate(lag1)
    identity_score = evaluate(identity)
    assert selected_score["mean_negative_log_likelihood_nats"] < primitive_score[
        "mean_negative_log_likelihood_nats"
    ]
    assert abs(
        selected_score["mean_negative_log_likelihood_nats"]
        - identity_score["mean_negative_log_likelihood_nats"]
    ) < 0.005


def test_null_world_selects_one_state_without_a_false_relation():
    simulation = simulate_relational_repair(
        world=RelationalWorld(relation="null"),
        policy="active",
        seed=20260802,
        config=RelationalRepairConfig(maximum_steps=32768),
    )
    assert simulation.result.certified
    assert simulation.result.selected_representation.name == "one_state"
    assert simulation.result.first_relational_witness_step is None


def test_primitive_world_selects_primitive_instead_of_fancy_relation():
    simulation = simulate_relational_repair(
        world=RelationalWorld(relation="lag1"),
        policy="active",
        seed=20260802,
        config=RelationalRepairConfig(maximum_steps=32768),
    )
    assert simulation.result.certified
    assert simulation.result.selected_representation.name == "lag1"
    assert not simulation.result.selected_representation.contains_relation


def test_omitted_grammar_abstains_instead_of_claiming_a_primitive_solution():
    primitive_only = generate_feature_representations(
        maximum_complexity=1,
        maximum_feature_count=1,
        operators=(),
    )
    simulation = simulate_relational_repair(
        world=RelationalWorld(),
        policy="active",
        seed=20260802,
        config=RelationalRepairConfig(maximum_steps=16384),
        representations=primitive_only,
    )
    assert not simulation.result.certified
    assert simulation.result.selected_representation is None
    assert "abstention" in simulation.result.status


def test_passive_scope_certifies_baseline_only_but_not_declared_action_family():
    simulation = simulate_relational_repair(
        world=RelationalWorld(),
        policy="passive",
        seed=20260802,
        config=RelationalRepairConfig(maximum_steps=16384),
    )
    by_action = dict(simulation.result.final_action_assessments)
    assert by_action["baseline"].selected.name == "one_state"
    assert by_action["probe"].selected is None
    assert not simulation.result.certified


def test_relational_drift_is_detected_per_raw_context_and_action():
    config = RelationalRepairConfig()
    interaction_count = 32768
    interactions = generate_relational_interactions(
        world=RelationalWorld(drift_step=interaction_count // 2),
        actions=config.actions,
        action_costs=config.cost_map,
        interaction_count=interaction_count,
        history_length=3,
        seed=20260802,
    )
    assessment = assess_relational_stationarity(
        interactions,
        actions=config.actions,
        history_length=3,
        delta=0.01,
        merge_tolerance=0.2,
        split_tolerance=0.35,
        minimum_samples_per_half=32,
    )
    assert assessment.status == "raw_cell_distribution_shift_detected"
    assert assessment.change_witness[0] == "probe"


def test_configuration_rejects_incomplete_action_costs():
    with pytest.raises(ValueError, match="every action"):
        RelationalRepairConfig(action_costs=(("baseline", 1.0),))
