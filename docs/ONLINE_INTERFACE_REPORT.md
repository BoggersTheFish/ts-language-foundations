# Observation-Only Adaptive Interface Learning — v0.3

## Question

Can a learner that never receives the hidden state discover enough finite memory to predict and evolve autonomously? Can it also refuse a false finite closure when every available memory budget still contains a verifier counterexample?

## Input and algorithm

The learner receives only finite binary observation episodes. Environment states are retained outside the learner for simulation and post-hoc evaluation. Its initial representation is the last observed symbol. Whenever one represented suffix has been followed by two different symbols, the verifier emits the two observation prefixes as a counterexample and increases suffix order by one.

This is an exact, deterministic variable-memory learner. Suffix growth itself is standard and is closely related to variable-order Markov models and automata inference. The experiment-specific connection is that its learned context machine is then subjected to predictive repair and strong-lumpability repair, with fail-closed abstention at a declared resource boundary.

## Positive fixture — the hidden state is never supplied

The simulator evolves

```text
(a,b,n) -> (b,1-a,n+1),    observation = a,
```

but the learner receives only bit strings. At suffix order one, the same observed bit has incompatible next symbols. One verifier event changes the represented history length from `1` to `2`.

The four supported observation contexts are `00`, `01`, `10`, and `11`. They compile to the exact autonomous cycle

```text
00 -> 01 -> 11 -> 10 -> 00.
```

The learned machine is exact on `504` held-out transitions generated at the unseen counter value `1000000`. Post-hoc evaluation finds a bijection between the four learned contexts and the four hidden `(a,b)` macrostates. That alignment was not used for learning.

## Prediction is still weaker than autonomous state

Grouping the four contexts only by their next-symbol law yields two blocks. Its predictive residual is `0.000000000000` nats, while its lumpability residual is `0.693147180560` nats. Dynamic repair splits it into `4` blocks.

Thus the observation-only run independently reconstructs the original distinction:

```text
enough state for the next answer != enough state for autonomous continuation.
```

## Baselines

On the infinite-fibre holdout:

| Model | Held-out accuracy | Exact? | Selects memory? |
|---|---:|---:|---:|
| One-state majority | 0.500 | no | no |
| Current observation | 0.500 | no | no |
| Fixed order two | 1.000 | yes | no |
| Counterexample-guided interface | 1.000 | yes | yes |

The fixed order-two baseline shows that raw prediction does not require a new algorithm here. The additional result is automatic memory selection plus a separately verified autonomous quotient.

## Negative fixture — finite memory cannot identify the countdown boundary

For observation strings `0^n 1^...`, every tested suffix budget contains the context `0^K` followed by both `0` and `1`. Budgets `1, 2, 4, 8` all terminate with valid counterexamples. The exact learner therefore abstains.

This contrasts accuracy with certification. The current-observation majority baseline scores `0.968` but makes `8` boundary errors. The fixed order-8 baseline scores `0.959` and still makes `8` boundary errors. Neither is exact.

The computation verifies the declared finite budgets. The universal no-finite-suffix conclusion is inherited from the symbolic countdown proof in the v0.2 report, not inferred from finite testing alone.

## Verifier results

- Checks passed: `13/13`.
- Observation-only learned order: `2`.
- Learned autonomous states: `4`.
- Stable holdout errors: `0`.
- Countdown strict coverage after failed closure: `0.000`.
- Contradictory-control abstention: `True`.

## Interpretation boundary

This establishes a small deterministic mechanism: observable verifier failures can revise memory, and predictive state can be repaired into autonomous state without hidden-state supervision. It does not establish robust learning under noise, continuous state discovery, scalable active experimentation, language learning, or superiority to neural sequence models.

The next falsifiable experiment replaces exact equality by confidence-bounded comparison of next-symbol distributions under controlled noise, with explicit penalties for memory, samples, abstention, and mistaken merges.
