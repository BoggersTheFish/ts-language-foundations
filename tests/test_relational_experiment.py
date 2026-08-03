from pathlib import Path

from ts_language_foundations.relational_experiment import (
    run_relational_experiment,
    write_relational_artifacts,
)


def test_relational_experiment_emits_verified_machine_readable_artifacts(
    tmp_path: Path,
):
    receipt = run_relational_experiment(
        heldout_interactions=8192,
        monte_carlo_runs=3,
        project_root=Path.cwd(),
    )
    assert receipt["verification"]["passed"]
    assert receipt["verification"]["passed_count"] == 30
    primary = receipt["primary_relational_case"]
    assert primary["result"]["constructed_feature"] == "xor(lag0,lag1)"
    assert primary["result"]["predictive_compression"] == [8, 2]
    assert primary["result"]["autonomous_repair"] == [2, 4]

    output = tmp_path / "receipt.json"
    report = tmp_path / "report.md"
    figure = tmp_path / "figure.svg"
    write_relational_artifacts(
        receipt,
        output=output,
        report=report,
        figure=figure,
    )
    assert output.read_text(encoding="utf-8").startswith("{")
    assert "## Verifier results" in report.read_text(encoding="utf-8")
    assert figure.read_text(encoding="utf-8").startswith("<svg")

