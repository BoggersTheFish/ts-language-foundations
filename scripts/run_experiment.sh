#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

python -m ts_language_foundations.cli \
  --output artifacts/run_receipt.json \
  --figure figures/finite_protocol.svg \
  --report docs/TECHNICAL_REPORT.md \
  --renaming-count 24 \
  --reference-kernel-archive-sha256 \
  1489110536881c1a63f5ad32261521cf150f3b5ae3442d22000fb81546d703c8

python -m ts_language_foundations.infinite_cli \
  --output artifacts/infinite_interface_receipt.json \
  --figure figures/infinite_interface.svg \
  --report docs/INFINITE_INTERFACE_REPORT.md \
  --trace-steps 24 \
  --induction-depth 512 \
  --states-checked 4096

python -m ts_language_foundations.online_cli \
  --output artifacts/online_interface_receipt.json \
  --figure figures/online_interface_learning.svg \
  --report docs/ONLINE_INTERFACE_REPORT.md \
  --maximum-order 8 \
  --stable-episode-length 64 \
  --stable-heldout-length 128 \
  --countdown-tail 16 \
  --minimum-context-support 8

python -m ts_language_foundations.stochastic_cli \
  --output artifacts/stochastic_interface_receipt.json \
  --figure figures/stochastic_interface_learning.svg \
  --report docs/STOCHASTIC_INTERFACE_REPORT.md \
  --seed 20260802 \
  --episode-count 256 \
  --episode-length 128 \
  --heldout-episodes 64 \
  --maximum-order 3 \
  --delta 0.01 \
  --family-cell-budget 256 \
  --merge-tolerance 0.12 \
  --split-tolerance 0.25 \
  --minimum-child-samples 64 \
  --monte-carlo-runs 100

python -m ts_language_foundations.active_cli \
  --output artifacts/active_repair_receipt.json \
  --figure figures/active_repair.svg \
  --report docs/ACTIVE_REPAIR_REPORT.md \
  --seed 20260802 \
  --maximum-steps 65536 \
  --heldout-episodes 64 \
  --delta 0.01 \
  --merge-tolerance 0.2 \
  --split-tolerance 0.35 \
  --minimum-samples 32 \
  --probe-cost 2.0 \
  --monte-carlo-runs 100

python -m ts_language_foundations.relational_cli \
  --output artifacts/relational_repair_receipt.json \
  --figure figures/relational_repair.svg \
  --report docs/RELATIONAL_REPAIR_REPORT.md \
  --seed 20260802 \
  --maximum-steps 32768 \
  --heldout-interactions 65536 \
  --delta 0.01 \
  --merge-tolerance 0.2 \
  --split-tolerance 0.35 \
  --minimum-samples 32 \
  --probe-cost 2.0 \
  --monte-carlo-runs 50

python -m ts_language_foundations.sequence_cli \
  --output artifacts/sequence_repair_receipt.json \
  --figure figures/sequence_repair.svg \
  --report docs/SEQUENCE_REPAIR_REPORT.md \
  --seed 20260802 \
  --maximum-steps 65536 \
  --heldout-interactions 27000 \
  --delta 0.01 \
  --merge-tolerance 0.32 \
  --split-tolerance 0.45 \
  --minimum-samples 32 \
  --probe-cost 2.0 \
  --monte-carlo-runs 20
