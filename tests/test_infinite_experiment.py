from ts_language_foundations.infinite_experiment import run_infinite_experiment


def test_infinite_experiment_receipt_passes_and_separates_block_from_entropy_excess(
    tmp_path,
):
    receipt = run_infinite_experiment(
        horizons=(1, 2, 4, 8),
        probabilities=(0.1, 0.25, 0.5),
        primary_horizon=4,
        primary_probability=0.25,
        trace_steps=8,
        induction_depth=64,
        states_checked=512,
        project_root=tmp_path,
    )
    assert receipt["verification"]["passed"]
    countdown = receipt["unbounded_countdown_case"]
    assert countdown["verification_summary"]["predictive_state_checks"] == 4 * 512
    assert countdown["adaptive_interface_trace"]["all_events_passed"]
    assert countdown["adaptive_interface_trace"]["events_checked"] == 8
    for record in countdown["horizon_records"]:
        assert record["predictive_residual_nats"] == 0.0
        assert record["exact_closure_excess_blocks"] == "countably_infinite"
        assert all(
            sensitivity["exact_closure_entropy_excess_nats"] < float("inf")
            for sensitivity in record["geometric_weight_sensitivity"]
        )
