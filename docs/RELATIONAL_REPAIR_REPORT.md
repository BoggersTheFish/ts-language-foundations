# Relational State Construction — Technical Report

## Question

Can a learner construct the missing coordinate from raw histories, rather than being told to append a specific older bit, while keeping every representation change under an independent finite-sample verifier?

## Current result

Yes, in the declared controlled fixture. The generated grammar contains 20 distinct Boolean truth functions and induces 65 unique candidate partitions. From those candidates, the verifier selected `xor(lag0,lag1)`. This compresses eight raw histories to 2 predictive blocks. That quotient is not autonomous: controlled dynamic closure repairs it to 4 blocks, represented canonically as `lag0 + lag1`.

This is a fixture-specific computational result, not a claim that arbitrary relations or language have been learned.

## Source and integrity boundary

- Environment: `(history,n) in {0,1}^3 x N_0`; `n` is unbounded, increments, and never enters a registered law.
- Learner input: raw three-bit history, action, action cost, and binary outcome.
- Hidden from learner: `n`, the true XOR label, and the world probability function.
- Episode resets cycle all eight histories; action panels are scheduled separately so action is not confounded with context.
- No corpus, tokenizer, neural component, or network access is used.

## Typed feature grammar

The grammar has Boolean terminals `0`, `1`, `lag0`, `lag1`, `lag2`; unary `not`; and binary `xor`, `eq`, `and`, `or`. Expressions are exhaustively enumerated through node complexity 3. Exact truth-table duplicates are removed. Feature tuples contain at most 2 features.

Candidate order is frozen before observations: block count, total expression complexity, feature count, fixed operator priority, then name. A candidate may be selected only when it is certified and every earlier candidate has been rejected. Unresolved predecessors force abstention.

`xor(lag0,lag1)` and `eq(lag0,lag1)` induce the same unlabelled partition. The XOR name is a deterministic canonical representative, not an empirically identified semantic polarity.

## Confidence contract

For every raw history/action cell and every within-cell sample prefix up to the declared budget, the verifier uses

\[
r(n)=\sqrt{\frac{\log(2M/\delta)}{2n}},
\qquad M=|A|2^hT.
\]

All eight histories must have at least 32 observations under every assessed action. A candidate block is rejected when a within-block pair has lower confidence difference above 0.35; it is certified only when every within-block upper difference is at most 0.2.

The proposal belief chooses actions but has no split, merge, or stopping authority.

## Primary construction

| Quantity | Active | Uniform |
|---|---:|---:|
| First relational witness step | 9,472 | 20,736 |
| First relational witness cost | 18,944 | 31,104 |
| Full-family fixed-point step | 20,224 | 20,736 |
| Full-family fixed-point cost | 29,696 | 31,104 |

The active policy first resolves the informative probe and then fills the unresolved baseline cells. Uniform sampling rotates complete context panels through both actions.

## Predictive and autonomous state

The selected predictive partition is `[[0, 3, 4, 7], [1, 2, 5, 6]]`. Its confidence-pooled predictive residual is 0.000000 nats; the unpooled empirical residual is 0.000235 nats.

Its controlled lumpability residual is 0.137065 nats, and dynamic closure is `[[0, 4], [1, 5], [2, 6], [3, 7]]`. The exact fixture oracle gives the same closure. The repaired state keeps `lag0` and `lag1` and discards irrelevant `lag2`.

At the one-state candidate, partial repairs do not commute: `C(P(pi))` has four blocks while `P(C(pi))` initially has two. Fair completion reaches the same four-block common fixed point.

## Candidate receipts

| Candidate | Blocks | Verifier | Oracle R | Oracle L | ΔK_C | ΔK_S |
|---|---:|---|---:|---:|---:|---:|
| `one_state` | 1 | rejected | 0.135219 | 0.000000 | 0 | 3 |
| `lag0` | 2 | rejected | 0.135219 | 0.135219 | 2 | 2 |
| `lag1` | 2 | rejected | 0.135219 | 0.693147 | 2 | 2 |
| `lag2` | 2 | rejected | 0.135219 | 0.693147 | 6 | 6 |
| `xor(lag0,lag1)` | 2 | certified | 0.000000 | 0.135219 | 2 | 2 |
| `lag0 + lag1` | 4 | certified | 0.000000 | 0.000000 | 0 | 0 |
| `lag2 + lag1 + lag0` | 8 | control | 0.000000 | 0.000000 | 0 | 0 |

The machine-readable receipt contains metrics and repair traces for all 66 candidate/control partitions.

## Heldout prediction

| Model | States | Accuracy | NLL (nats) | Brier |
|---|---:|---:|---:|---:|
| one state | 1 | 0.4997 | 0.693233 | 0.250043 |
| lag0 | 2 | 0.4997 | 0.693243 | 0.250048 |
| lag1 | 2 | 0.4967 | 0.693299 | 0.250076 |
| selected XOR | 2 | 0.6752 | 0.560884 | 0.189906 |
| autonomous suffix | 4 | 0.6722 | 0.560971 | 0.189944 |
| raw identity | 8 | 0.6757 | 0.561068 | 0.189980 |

Honest null: the two-state selected quotient matches rather than beats the eight-state identity control within finite-sample variation. Its gain is compression and explicit sufficiency, not extra oracle information.

## Monte Carlo sensitivity

Across 50 paired seeds, active recovery was 50/50 and uniform recovery was 50/50. Median relational-witness steps were 9,728 active versus 19,456 uniform. The paired median step ratio was 0.5064, with algorithmic bootstrap interval [0.4875, 0.530316526351089].

Hostile controls used 25 seeds: null false constructions 0; primitive-world wrong selections 0; omitted-grammar unsafe outputs 0; drift detections 25/25.

## What failed or remains uncertain

- The active primary trace itself leaves two raw-cell stationarity comparisons unresolved at its stopping time. A separate balanced 65,536-interaction stationary audit certifies all cells; the drift fixture is rejected. This distinction is retained in the receipt.
- Exact dynamic refinement of noisy, unpooled empirical probabilities would split all eight contexts. Dynamic closure is therefore applied to response laws pooled only inside confidence-certified predictive blocks and cross-checked against the exact fixture oracle.
- The grammar is finite and hand-declared. Omit XOR/EQ and feature tuples, and the verifier abstains.
- XOR versus EQ polarity is not identified by an unlabelled partition.
- This does not establish language, planning, general causal discovery, neural mechanisms, consciousness, or a general new AI architecture.

## Verifier results

| Check | Result |
|---|---|
| `generated_relation_is_xor` | PASS |
| `selected_predictive_state_has_two_blocks` | PASS |
| `autonomous_repair_has_four_blocks` | PASS |
| `dynamic_closure_excess_is_two` | PASS |
| `pooled_predictive_residual_is_zero` | PASS |
| `raw_empirical_predictive_residual_is_small` | PASS |
| `selected_state_is_not_dynamically_closed` | PASS |
| `oracle_and_empirical_closure_agree` | PASS |
| `partial_repairs_do_not_commute_at_one_state` | PASS |
| `active_primary_finds_relation_before_uniform` | PASS |
| `selected_nll_beats_lag0` | PASS |
| `selected_nll_beats_lag1` | PASS |
| `selected_matches_raw_identity_within_point_zero_one_nats` | PASS |
| `null_control_selects_one_state` | PASS |
| `primitive_control_selects_lag1` | PASS |
| `omitted_grammar_abstains` | PASS |
| `passive_full_family_abstains` | PASS |
| `misspecified_proposal_abstains` | PASS |
| `stationary_audit_certifies` | PASS |
| `drift_audit_detects_shift` | PASS |
| `polarity_ambiguity_recorded` | PASS |
| `monte_carlo_active_recovery_at_least_95_percent` | PASS |
| `monte_carlo_uniform_recovery_at_least_95_percent` | PASS |
| `monte_carlo_active_median_relation_witness_is_earlier` | PASS |
| `monte_carlo_active_median_fixed_point_step_is_earlier` | PASS |
| `monte_carlo_null_false_constructions_zero` | PASS |
| `monte_carlo_primitive_wrong_selections_zero` | PASS |
| `monte_carlo_grammar_omission_unsafe_outputs_zero` | PASS |
| `monte_carlo_drift_detection_complete` | PASS |
| `anytime_budget_matches_all_raw_cells_and_prefixes` | PASS |

Overall: **30/30 checks passed**.

## Next falsifiable experiment

Replace Boolean truth-table features with typed event and sequence relations. Train the representation on one controlled family, freeze it, and test whether the same repaired state predicts and evolves autonomously on held-out task families without further repair.
