from pathlib import Path

from ts_language_foundations.sequence_experiment import (
    run_sequence_experiment,
    write_sequence_artifacts,
)


def test_sequence_experiment_smoke(tmp_path: Path):
    receipt = run_sequence_experiment(
        seed=1701,
        monte_carlo_runs=2,
        project_root=Path(__file__).resolve().parents[1],
    )
    assert receipt["verification"]["passed"]
    assert receipt["verification"]["passed_count"] == 29
    assert receipt["primary_sequence_case"]["result"]["predictive_compression"] == [27, 2]
    assert receipt["primary_sequence_case"]["result"]["autonomous_repair"] == [2, 18]
    output = tmp_path / "receipt.json"
    report = tmp_path / "report.md"
    figure = tmp_path / "figure.svg"
    write_sequence_artifacts(
        receipt, output=output, report=report, figure=figure
    )
    assert output.read_text().startswith("{")
    assert "## Verifier results" in report.read_text()
    assert figure.read_text().startswith("<svg")
