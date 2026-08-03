from dataclasses import replace
from fractions import Fraction

import pytest

from ts_language_foundations.active import (
    controlled_dynamic_closure,
    controlled_lumpability_residual,
    controlled_predictive_residual,
)
from ts_language_foundations.relational import controlled_joint_repair
from ts_language_foundations.repair import common_refinement, refines
from ts_language_foundations.sequence import (
    CategoricalCounts,
    CategoricalInteraction,
    CategoricalSequenceWorld,
    SequenceRepairConfig,
    TypedExpression,
    assess_categorical_family,
    categorical_contexts,
    categorical_stationarity_assessment,
    categorical_tv_radius,
    combine_oracle_interfaces,
    compare_categorical_estimates,
    evaluate_categorical_table,
    fit_categorical_table,
    generate_sequence_interactions,
    generate_typed_features,
    generate_typed_representations,
    oracle_sequence_interface,
    partition_from_typed_features,
    predictive_partition_for_actions,
    raw_typed_identity,
    representation_by_name,
    simulate_sequence_repair,
)


def test_typed_expression_rejects_invalid_arguments():
    lag = TypedExpression("lag", value=0)
    eq = TypedExpression("eq", (lag, TypedExpression("lag", value=2)))
    assert lag.output_type == "symbol"
    assert eq.output_type == "bool"
    with pytest.raises(TypeError):
        TypedExpression("and", (lag, lag))
    with pytest.raises(TypeError):
        TypedExpression("eq", (eq, eq))


def test_typed_expression_evaluates_sequence_predicates():
    context = (2, 1, 2)
    lag0 = TypedExpression("lag", value=0)
    lag2 = TypedExpression("lag", value=2)
    assert TypedExpression("eq", (lag0, lag2)).evaluate(context) == 1
    assert TypedExpression("contains", value=1).evaluate(context) == 1
    assert TypedExpression("count_ge", value=(2, 2)).evaluate(context) == 1
    assert TypedExpression("all_equal").evaluate(context) == 0
    assert TypedExpression("all_distinct").evaluate((0, 1, 2)) == 1


def test_feature_deduplication_preserves_types():
    features = generate_typed_features()
    keys = {
        (feature.output_type, feature.output_table(categorical_contexts()))
        for feature in features
    }
    assert len(keys) == len(features)
    assert {feature.output_type for feature in features} == {"symbol", "bool"}


def test_fixed_candidate_family_contains_key_partitions():
    representations = generate_typed_representations()
    assert len(representations) == 1504
    assert representation_by_name(representations, "one_state").block_count == 1
    assert representation_by_name(representations, "eq(lag0,lag2)").block_count == 2
    assert (
        representation_by_name(
            representations, "lag0 + lag1 + eq(lag0,lag2)"
        ).block_count
        == 18
    )
    assert raw_typed_identity().block_count == 27


def test_eq_and_neq_have_same_unlabelled_partition():
    contexts = categorical_contexts()
    lag0 = TypedExpression("lag", value=0)
    lag2 = TypedExpression("lag", value=2)
    assert partition_from_typed_features(
        contexts, (TypedExpression("eq", (lag0, lag2)),)
    ) == partition_from_typed_features(
        contexts, (TypedExpression("neq", (lag0, lag2)),)
    )


def test_categorical_world_exact_laws():
    world = CategoricalSequenceWorld()
    assert world.law((0, 1, 0), "baseline") == (Fraction(1, 3),) * 3
    assert world.law((0, 1, 0), "probe") == (
        Fraction(4, 5),
        Fraction(1, 10),
        Fraction(1, 10),
    )
    assert world.law((0, 1, 2), "probe") == (
        Fraction(1, 10),
        Fraction(4, 5),
        Fraction(1, 10),
    )


def test_tv_radius_shrinks_and_is_anytime_budget_sensitive():
    small = categorical_tv_radius(100, delta=0.01, family_cell_budget=1000)
    large = categorical_tv_radius(1000, delta=0.01, family_cell_budget=1000)
    larger_family = categorical_tv_radius(1000, delta=0.01, family_cell_budget=10000)
    assert large < small
    assert larger_family > large


def test_aggregated_counts_certify_endpoint_partition():
    contexts = categorical_contexts()
    counts = CategoricalCounts(("probe",))
    for context in contexts:
        values = (800, 100, 100) if context[0] == context[2] else (100, 800, 100)
        counts.add_cell_counts("probe", context, values)
    representations = generate_typed_representations()
    assessment = assess_categorical_family(
        counts,
        representations,
        actions=("probe",),
        delta=0.01,
        family_cell_budget=27 * 65536,
        merge_tolerance=0.32,
        split_tolerance=0.45,
        minimum_samples=32,
    )
    assert assessment.selected is not None
    assert assessment.selected.name == "eq(lag0,lag2)"


def test_aggregated_counts_reject_one_state():
    contexts = categorical_contexts()
    counts = CategoricalCounts(("probe",))
    for context in contexts:
        values = (800, 100, 100) if context[0] == context[2] else (100, 800, 100)
        counts.add_cell_counts("probe", context, values)
    reps = generate_typed_representations()
    assessment = assess_categorical_family(
        counts,
        reps,
        actions=("probe",),
        delta=0.01,
        family_cell_budget=27 * 65536,
        merge_tolerance=0.32,
        split_tolerance=0.45,
        minimum_samples=32,
    )
    assert assessment.candidate_assessments[0].status == "rejected"


def test_exact_predictive_state_is_not_lumpable():
    interface = oracle_sequence_interface(CategoricalSequenceWorld())
    reps = generate_typed_representations()
    selected = representation_by_name(reps, "eq(lag0,lag2)")
    weights = (Fraction(1, 27),) * 27
    assert controlled_predictive_residual(interface, selected.partition, weights) == pytest.approx(0)
    assert controlled_lumpability_residual(interface, selected.partition, weights) > 0


def test_exact_dynamic_closure_is_eighteen_blocks():
    interface = oracle_sequence_interface(CategoricalSequenceWorld())
    selected = representation_by_name(
        generate_typed_representations(), "eq(lag0,lag2)"
    )
    closure, trace = controlled_dynamic_closure(
        interface.transition_rows, selected.partition, return_trace=True
    )
    assert tuple(map(len, trace)) == (2, 4, 12, 18)
    expected = representation_by_name(
        generate_typed_representations(), "lag0 + lag1 + eq(lag0,lag2)"
    )
    assert closure == expected.partition


def test_partial_repairs_do_not_commute_but_joint_repairs_agree():
    interface = oracle_sequence_interface(CategoricalSequenceWorld())
    reps = generate_typed_representations()
    one = representation_by_name(reps, "one_state").partition
    predictive = predictive_partition_for_actions(interface, ("probe",))
    cp = controlled_dynamic_closure(
        interface.transition_rows, common_refinement(one, predictive)
    )
    pc = common_refinement(
        controlled_dynamic_closure(interface.transition_rows, one), predictive
    )
    assert len(cp) == 18
    assert len(pc) == 2
    assert cp != pc
    pd = controlled_joint_repair(
        interface.transition_rows, predictive, one, schedule="predictive_then_dynamic"
    )
    dp = controlled_joint_repair(
        interface.transition_rows, predictive, one, schedule="dynamic_then_predictive"
    )
    assert pd == dp == cp


def test_compatible_and_incompatible_transfer_partitions():
    reps = generate_typed_representations()
    source = representation_by_name(reps, "eq(lag0,lag2)").partition
    compatible = oracle_sequence_interface(
        CategoricalSequenceWorld(
            actions=("challenge",),
            informative_action="challenge",
            endpoint_true_outcome=2,
            endpoint_false_outcome=0,
        )
    )
    middle = oracle_sequence_interface(
        CategoricalSequenceWorld(
            relation="middle_symbol",
            actions=("target",),
            informative_action="target",
        )
    )
    weights = (Fraction(1, 27),) * 27
    assert predictive_partition_for_actions(compatible) == source
    assert controlled_predictive_residual(compatible, source, weights) == pytest.approx(0)
    assert controlled_predictive_residual(middle, source, weights) > 0


def test_task_family_predictive_repair_has_six_blocks():
    source = oracle_sequence_interface(
        CategoricalSequenceWorld(
            actions=("source",), informative_action="source"
        )
    )
    target = oracle_sequence_interface(
        CategoricalSequenceWorld(
            relation="middle_symbol",
            actions=("target",),
            informative_action="target",
        )
    )
    combined = combine_oracle_interfaces((source, target))
    family = predictive_partition_for_actions(combined)
    assert len(family) == 6
    assert refines(family, predictive_partition_for_actions(source))
    assert refines(family, predictive_partition_for_actions(target))
    assert len(controlled_dynamic_closure(combined.transition_rows, family)) == 18


def test_categorical_probability_table_transfer():
    world = CategoricalSequenceWorld(
        actions=("challenge",), informative_action="challenge"
    )
    training = generate_sequence_interactions(
        world=world, action="challenge", interaction_count=2700, seed=3
    )
    test = generate_sequence_interactions(
        world=world, action="challenge", interaction_count=5400, seed=4
    )
    reps = generate_typed_representations()
    selected = representation_by_name(reps, "eq(lag0,lag2)")
    one = representation_by_name(reps, "one_state")
    selected_score = evaluate_categorical_table(
        fit_categorical_table(training, selected), selected, test
    )
    one_score = evaluate_categorical_table(
        fit_categorical_table(training, one), one, test
    )
    assert selected_score["mean_negative_log_likelihood_nats"] < one_score[
        "mean_negative_log_likelihood_nats"
    ]


def test_stationarity_and_drift_controls():
    count = 30000
    stable = CategoricalSequenceWorld(actions=("probe",), informative_action="probe")
    stable_trace = generate_sequence_interactions(
        world=stable, action="probe", interaction_count=count, seed=8
    )
    drift = CategoricalSequenceWorld(
        actions=("probe",), informative_action="probe", drift_step=count // 2
    )
    drift_trace = generate_sequence_interactions(
        world=drift, action="probe", interaction_count=count, seed=8
    )
    stable_result = categorical_stationarity_assessment(
        stable_trace,
        actions=("probe",),
        delta=0.01,
        split_tolerance=0.45,
        minimum_samples=32,
    )
    drift_result = categorical_stationarity_assessment(
        drift_trace,
        actions=("probe",),
        delta=0.01,
        split_tolerance=0.45,
        minimum_samples=32,
    )
    assert stable_result["drift_witness_count"] == 0
    assert drift_result["status"] == "drift_detected"


def test_grammar_omission_abstains():
    representations = generate_typed_representations(
        allowed_lags=(0, 1),
        include_sequence_predicates=False,
        include_boolean_compositions=False,
    )
    result = simulate_sequence_repair(
        world=CategoricalSequenceWorld(),
        policy="active",
        seed=12,
        config=replace(SequenceRepairConfig(), maximum_steps=32768),
        representations=representations,
    )
    assert not result.certified
    assert result.selected_representation is None


def test_interaction_validation_is_fail_closed():
    with pytest.raises(ValueError):
        CategoricalInteraction((0, 1, 3), "probe", 0, 1.0, 0, 0)
    with pytest.raises(ValueError):
        CategoricalInteraction((0, 1, 2), "probe", 3, 1.0, 0, 0)

