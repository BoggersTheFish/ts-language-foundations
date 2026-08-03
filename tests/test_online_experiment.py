from ts_language_foundations.online_experiment import run_online_experiment


def test_online_experiment_passes_without_hidden_state_supervision(tmp_path):
    receipt = run_online_experiment(
        maximum_order=4,
        stable_episode_length=24,
        stable_heldout_length=40,
        countdown_post_absorption_steps=8,
        minimum_context_support=4,
        project_root=tmp_path,
    )
    assert receipt["verification"]["passed"]
    assert receipt["verification"]["passed_count"] == receipt["verification"][
        "check_count"
    ]

    stable = receipt["stable_infinite_fibre_case"]
    assert not stable["learner_input_boundary"]["hidden_state_received"]
    assert stable["learned_interface"]["final_order"] == 2
    assert stable["learned_interface"]["repair_event_count"] == 1
    assert stable["result"]["learned_state_count"] == 4
    assert stable["heldout_evaluation"]["accuracy_on_covered"] == 1.0

    predictive = stable["candidate_abstractions"][
        "next_symbol_predictive_partition"
    ]
    assert predictive["predictive_residual_nats"] == 0.0
    assert predictive["lumpability_residual_nats"] > 0.0
    assert len(predictive["joint_repair"]) == 4

    countdown = receipt["unbounded_countdown_case"]
    assert not countdown["learner_input_boundary"]["hidden_state_received"]
    assert countdown["result"]["all_finite_budgets_abstained"]
    assert countdown["result"]["exact_baselines"] == []
    assert countdown["heldout_evaluation"]["coverage"] == 0.0


def test_online_experiment_is_deterministic_apart_from_provenance(tmp_path):
    left = run_online_experiment(
        maximum_order=4,
        stable_episode_length=20,
        stable_heldout_length=32,
        countdown_post_absorption_steps=8,
        minimum_context_support=2,
        project_root=tmp_path,
    )
    right = run_online_experiment(
        maximum_order=4,
        stable_episode_length=20,
        stable_heldout_length=32,
        countdown_post_absorption_steps=8,
        minimum_context_support=2,
        project_root=tmp_path,
    )
    for section in (
        "algorithm",
        "stable_infinite_fibre_case",
        "unbounded_countdown_case",
        "negative_control",
        "verification",
        "conclusion",
    ):
        assert left[section] == right[section]
