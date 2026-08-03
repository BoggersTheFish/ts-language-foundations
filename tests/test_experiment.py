from ts_language_foundations.experiment import run_experiment


def test_full_receipt_has_expected_verified_result(tmp_path):
    receipt = run_experiment(renaming_count=24, project_root=tmp_path)
    repair = receipt["representation_repair"]
    summary = receipt["protocol"]["summary"]
    assert repair["task_partition_predictive_residual_nats"] == 0.0
    assert repair["task_partition_lumpability_residual_nats"] > 0.0
    assert repair["dynamic_closure"] == [[0], [1], [2], [3]]
    assert receipt["run"]["renaming_coverage_exhaustive"] is True
    assert summary["trial_count"] == 96
    assert summary["all_checks_passed_count"] == 96
    assert summary["held_out_compositional_decode_accuracy"] == 1.0
    assert summary["four_step_rollout_accuracy"] == 1.0
    assert summary["holistic_baseline_held_out_coverage"] == 0.0
    assert summary["coarse_task_message_best_next_task_accuracy"] == 0.5
    assert receipt["coordinate_system_search"]["selected"]["total_model_bits"] == 6
    assert summary["unordered_trial_count"] == 96
    assert summary["unordered_all_checks_passed_count"] == 96
    assert summary["unordered_held_out_decode_accuracy"] == 1.0
    assert summary["unordered_four_step_rollout_accuracy"] == 1.0
