# Changelog

## 0.7.0

- Generalised the construction domain from binary histories and Bernoulli
  outcomes to length-three ternary histories and categorical next-event laws.
- Added an explicitly typed grammar with symbol terminals, equality and
  inequality, containment, counting, all-equal/all-distinct predicates, and
  Boolean composition; semantic deduplication retains the output type.
- Added simultaneous multinomial total-variation confidence radii with a
  declared anytime action/context/prefix family and fixed split/merge
  hysteresis.
- Constructed the hidden endpoint-equality relation from 27 raw states without
  receiving its label, producing a two-block predictive representation.
- Computed the exact controlled closure trace `2 -> 4 -> 12 -> 18`, showing
  that autonomous repair restores necessary history without restoring all 27
  raw identities.
- Added an explicit noncommuting partial-repair example with order-independent
  least-common-fixed-point completion.
- Added frozen compatible transfer, explicit incompatible transfer, and
  monotone task-family repair from two and three task states to six predictive
  blocks and eighteen autonomous blocks.
- Added categorical held-out likelihood evaluation, transition entropy,
  persistence, occupancy, support, repair traces, exact residuals, and source
  and interaction hashes.
- Added null, primitive, grammar-omission, stationarity, drift, and EQ/NEQ
  polarity controls plus paired Monte Carlo and bootstrap sensitivity.
- Added a CLI, technical report, code-generated figure, release verifier
  integration, and regression tests while preserving v0.1–v0.6 behavior.

## 0.6.0

- Replaced the hand-supplied suffix-depth repair coordinate with a typed
  Boolean feature grammar over raw histories.
- Added exhaustive bounded-complexity construction for constants, lags, NOT,
  XOR, equality, conjunction, and disjunction, with truth-table and partition
  deduplication.
- Added a frozen candidate order and fail-closed minimality certificate:
  selection requires every earlier candidate to be rejected and every raw
  context/action cell to be supported.
- Added an XOR-hidden controlled fixture in which the learner receives no true
  feature label or hidden state.
- Recovered a two-block relational predictive quotient from eight raw
  histories, then repaired it to the four-block last-two-bit autonomous state
  while discarding an irrelevant older coordinate.
- Added controlled joint repair and a concrete noncommuting partial-repair
  receipt whose fair completions agree.
- Separated raw empirical transition estimates from confidence-pooled dynamic
  compilation and cross-checked the pooled closure against the exact fixture
  oracle.
- Added held-out relational tables, occupancy, persistence, transition entropy,
  all-candidate residual/repair metrics, and code-generated state/efficiency
  figures.
- Added null, primitive, omitted-grammar, passive-policy, misspecified-proposal,
  stationarity, drift, and XOR/EQ polarity controls.
- Added paired Monte Carlo recovery, active-versus-uniform construction cost,
  bootstrap intervals, a CLI, a technical report, release verification, and
  regression tests.

## 0.5.0

- Added a controlled stochastic infinite-fibre world with passive and revealing
  intervention channels.
- Separated proposal from authority: Bayesian information and uncertainty
  scores choose actions, while only simultaneous confidence certificates may
  repair or certify state.
- Added an anytime finite-family Hoeffding contract covering every declared
  action/context cell and within-cell sample prefix under predictable action
  choice and adaptive stopping.
- Demonstrated that baseline-policy sufficiency cannot be promoted to the full
  action family without probe evidence.
- Repaired memory order one to order two, compressed four contexts to two
  action-predictive blocks, and restored four states under controlled dynamic
  closure.
- Added active-versus-uniform sample and intervention-cost comparisons,
  balanced-action heldout evaluation, and a fixed-order-two honest null.
- Added null-action, passive-scope, misspecified-proposal, intervention-cost,
  and action-conditioned drift controls.
- Added paired-seed Monte Carlo recovery and bootstrap efficiency intervals,
  false-split checks, safe-abstention checks, and drift-detection receipts.
- Added a technical report, code-generated active-repair figure, CLI, release
  verifier integration, and regression tests.

## 0.4.0

- Replaced exact next-symbol contradictions with simultaneous Hoeffding
  confidence bounds for stochastic Bernoulli context laws.
- Added three-way verifier decisions: merge, split, or wait for more evidence.
- Added a fixed hysteresis band and fail-closed sample-budget exhaustion.
- Recovered a two-symbol stochastic interface over an infinite nuisance fibre,
  then separated its two predictive blocks from its four-state dynamic closure.
- Added a chronological stationarity gate that rejects regime drift.
- Added first-order, IID-collapse, weak-signal, and drift controls.
- Added held-out accuracy, negative log likelihood, Brier score, parameter
  calibration, sample-budget sensitivity, and signal-gap sensitivity.
- Added seeded Monte Carlo recovery, confidence-coverage, false-split, correct
  null-certification, and drift-detection receipts with Wilson intervals.
- Added a technical report and code-generated statistical-repair figure.

## 0.3.0

- Added a counterexample-guided suffix-interface learner whose public input is
  observation episodes rather than simulator state.
- Recovered the exact four-state autonomous quotient of the infinite-fibre
  fixture from two-symbol observation histories.
- Separated the learned two-block next-symbol predictor from its four-block
  strong-lumpability repair.
- Added memory-budget sensitivity and fail-closed abstention on the countdown
  fixture.
- Added one-state, current-observation, and fixed-order sequence baselines.
- Added a contradictory-observation negative control, held-out infinite-fibre
  transfer checks, a machine-readable receipt, a technical report, and a
  code-generated figure.

## 0.2.0

- Replaced the implicit finite-world interpretation with finite-index
  interfaces over countably infinite state spaces.
- Added an infinite-fibre product fixture with an exact four-state autonomous
  quotient.
- Added the absorbing countdown fixture: every finite horizon has an exact
  finite predictive quotient, but its coarsest strongly lumpable refinement is
  the countably infinite identity.
- Proved the countdown nontermination result by an explicit boundary-splitting
  induction.
- Derived exact geometric-weight formulas for lumpability residual and entropy
  closure excess.
- Added the regression that arbitrarily small residual does not imply a finite
  exact repair.
- Added an adaptive interface trace in which one verifier witness revises the
  current infinite-tail symbol while preserving a finite interface at every
  version.
- Added code-generated infinite-interface receipts, report, and figure.
- Preserved the v0.1 finite opaque-signal protocol and reference-kernel checks.

## 0.1.0

- Added the four-state repaired-coordinate protocol fixture.
- Added exhaustive opaque-signal renaming and leave-one-state-out verification.
- Added unordered signal-type inference without fixed message slots.
