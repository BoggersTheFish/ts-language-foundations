from pathlib import Path

from ts_language_foundations.active_experiment import (
    run_active_experiment,
    write_active_artifacts,
)


def test_active_experiment_emits_verified_machine_readable_artifacts(tmp_path: Path):
    receipt = run_active_experiment(
        monte_carlo_runs=3,
        project_root=Path.cwd(),
    )
    assert receipt["verification"]["passed"]
    assert receipt["verification"]["passed_count"] == 22
    assert receipt["primary_controlled_case"]["result"]["predictive_blocks"] == 2
    assert receipt["primary_controlled_case"]["result"]["autonomous_blocks"] == 4

    output = tmp_path / "receipt.json"
    report = tmp_path / "report.md"
    figure = tmp_path / "figure.svg"
    write_active_artifacts(
        receipt,
        output=output,
        report=report,
        figure=figure,
    )
    assert output.read_text(encoding="utf-8").startswith("{")
    assert "## Verifier results" in report.read_text(encoding="utf-8")
    assert figure.read_text(encoding="utf-8").startswith("<svg")
