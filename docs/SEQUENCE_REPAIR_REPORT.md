# Typed categorical sequence construction and repair — v0.7.0

## Question

Can a verifier construct a useful relation over non-binary sequence events, determine whether that relation is an autonomous dynamical state, and decide—without semantic hand-waving—when the state transfers or must be repaired?

## Result boundary

This is a finite controlled fixture with an unbounded nuisance counter. It is a real executable test of representation construction and repair, not a language model and not evidence about consciousness. The typed grammar is declared by hand; the endpoint-equality label is not supplied to the learner.

## World and evidence

The observable state is a ternary history `(oldest, middle, newest)` in `{0,1,2}^3`, giving 27 raw states. `baseline` produces a uniform next symbol. Under `probe`, outcome 0 has probability 0.8 when oldest equals newest; otherwise outcome 1 has probability 0.8. The hidden counter increments forever but changes no registered law.

The proposal layer is Bayesian and chooses actions. It has no authority to alter a representation. The verifier uses simultaneous multinomial total-variation radii

\[
r(n)=\sqrt{\frac{\log((2^k-2)M/\delta)}{2n}}
\]

with a fixed candidate order and split/merge hysteresis. A candidate is selected only after it is certified and every simpler candidate has been rejected.

## Constructed state

The grammar generated 26 typed features and 1504 distinct candidate partitions. It selected `eq(lag0,lag2)` independently in active and uniform runs.

| Quantity | Value |
|---|---:|
| Raw histories | 27 |
| Predictive blocks | 2 |
| Autonomous blocks | 18 |
| Identity blocks | 27 |
| Dynamic closure excess | 16 |
| Selected predictive residual | 0 nats |
| Selected lumpability residual | 0.126495689094 nats |

The exact closure trace is `2 -> 4 -> 12 -> 18`. The autonomous representation is `lag0 + lag1 + eq(lag0,lag2)`. It remembers the newest symbol, middle symbol, and whether oldest equals newest. When oldest differs from newest, it still discards which of the two alternatives the oldest symbol was. Thus repair restores necessary hidden state without collapsing to raw identity.

## Noncommuting partial repairs

At the one-state start, predictive repair followed by dynamic closure has 18 blocks, while dynamic closure followed by predictive repair has 2 blocks. The partial operators therefore do not commute. Fair completion is nevertheless order-independent and reaches the same 18-block common fixed point.

## Exhaustive declared-family audit

Every one of the 1504 grammar-generated partitions was checked for extensivity, dynamic-closure idempotence, and agreement of the two fair joint-repair schedules. The generated subposet contained 20307 comparable ordered pairs; no closure-monotonicity violation occurred. Enlarging the controlled action family also produced no monotonicity violation on any candidate.

Only 3 candidates were dynamically closed: `one_state, lag0 + lag1 + eq(lag0,lag2), lag0 + lag1 + lag2`. Only 2 were common predictive/dynamic fixed points: `lag0 + lag1 + eq(lag0,lag2), lag0 + lag1 + lag2`. Partial noncommutation occurred at 8 candidates, and the candidate family flowed into 42 distinct joint completions. These are exhaustive computations over the declared grammar family, not a proof by enumeration over the full partition lattice of 27 states.

## Active intervention

| Policy | Steps | Cost | Probe observations |
|---|---:|---:|---:|
| Active | 18432 | 36860 | 18428 |
| Uniform | 34816 | 52217 | 17401 |

The active policy used 0.529 times the observations and 0.706 times the cost of uniform allocation in the primary run. This is an algorithmic fixture result, not a universal sample-complexity claim.

## Frozen transfer and task-family repair

The compatible held-out task changed which categorical outcomes mark the endpoint relation. Its exact residual through the frozen state was 0; the two-block state transferred after fitting only a new predictive head.

The incompatible held-out task depended on the middle symbol. The frozen endpoint state had exact residual 0.459580 nats and was rejected. Combining both tasks produced the common predictive refinement `lag1 + eq(lag0,lag2)` with 6 blocks. Its autonomous closure again had 18 blocks.

This supplies an explicit finite counterexample to unrestricted frozen transfer and verifies the monotonicity rule: adding task laws can preserve or refine the predictive partition, never coarsen it.

## Controls

| Control | Outcome | Passed |
|---|---|---:|
| Null world | selected `one_state` | True |
| Primitive middle-symbol world | selected `lag1` | True |
| Grammar omits oldest coordinate | `no_candidate_certified` | True |
| Stationary trace | no_drift_certificate | True |
| Drifting trace | drift_detected | True |
| EQ/NEQ polarity | same unlabelled partition | True |

## Monte Carlo sensitivity

Across 20 paired seeds, active recovery was 20/20 and uniform recovery was 20/20. Median steps were 18432 active versus 35840 uniform. The paired median step ratio was 0.5000, with algorithmic bootstrap interval (0.4737, 0.5294).

The 10 repeated control seeds produced null/primitive/omission/drift failure counts of 0, 0, 0, and 0.

## Verifier results

29/29 scientific checks passed. The JSON receipt records raw partitions, every repair trace, exact residuals, source hashes, interaction hashes, support counts, policy traces, held-out scores, controls, and Monte Carlo seeds.

## What this establishes

For this fixture, the engine can construct a typed categorical relation, certify it from interventions, distinguish prediction from autonomous evolution, stop short of raw identity, reuse the state on a compatible task, reject it on an incompatible task, and compute the least task-family repair.

It does not establish grammar completeness, open-ended concept learning, natural-language competence, neural implementation, or any metaphysical claim. The next falsifiable step is variable-length compositional event structure with frozen cross-task evaluation.
