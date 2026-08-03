from ts_language_foundations.stochastic_experiment import (
    render_report,
    render_svg,
    run_stochastic_experiment,
)


def test_stochastic_experiment_passes_all_primary_and_hostile_checks(tmp_path):
    receipt = run_stochastic_experiment(monte_carlo_runs=5, project_root=tmp_path)
    assert receipt["verification"]["passed"]
    assert receipt["verification"]["passed_count"] == receipt["verification"][
        "check_count"
    ]
    primary = receipt["primary_stochastic_infinite_fibre_case"]
    assert not primary["hidden_environment"]["learner_hidden_state_access"]
    assert primary["result"]["memory_repair_trace"] == [1, 2]
    assert primary["result"]["predictive_blocks"] == 2
    assert primary["result"]["autonomous_blocks"] == 4
    assert primary["confidence_contract"]["cell_budget_respected"]
    assert primary["result"]["heldout_nll_nats"] < 0.52

    controls = receipt["controls"]
    assert controls["first_order_null"]["passed"]
    assert controls["iid_collapse_control"]["passed"]
    assert controls["weak_signal_uncertainty_control"]["passed"]
    assert controls["regime_drift_control"]["passed"]

    assert receipt["monte_carlo"]["primary_recovery"]["successes"] == 5
    assert receipt["monte_carlo"]["first_order_false_splits"]["events"] == 0


def test_stochastic_report_and_svg_are_generated_from_receipt(tmp_path):
    receipt = run_stochastic_experiment(monte_carlo_runs=2, project_root=tmp_path)
    report = tmp_path / "report.md"
    figure = tmp_path / "figure.svg"
    render_report(receipt, report)
    render_svg(receipt, figure)
    assert "## Statistical contract" in report.read_text(encoding="utf-8")
    assert "## Verifier results" in report.read_text(encoding="utf-8")
    assert figure.read_text(encoding="utf-8").startswith("<svg")
