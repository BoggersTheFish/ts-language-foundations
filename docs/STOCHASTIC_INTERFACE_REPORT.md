# Confidence-Bounded Stochastic Representation Repair — v0.4

## Question

When observations are stochastic, a differing next symbol is not itself a contradiction. When is the evidence strong enough to change the represented state, when is a merge statistically defensible, and when must the machine wait or abstain?

## Statistical contract

For at most `M=256` predeclared Bernoulli cell-stage queries and family error `delta=0.01`, each empirical probability receives radius

```text
r(n) = sqrt(log(2M/delta)/(2n)).
```

Hoeffding's inequality plus a finite union bound gives simultaneous coverage at least `1-delta`, under the fixture assumption that each context visit uses an independent Bernoulli innovation with a stationary context law.

For two cells `i,j`, the verifier computes

```text
lower = max(0, |p_hat_i-p_hat_j|-r_i-r_j),
upper = min(1, |p_hat_i-p_hat_j|+r_i+r_j).
```

It merges only when `upper <= 0.12`, splits only when `lower > 0.25`, and otherwise requests the next declared sample batch. This hysteresis band prevents a noisy near-threshold estimate from forcing either operation.

The primary run used `56` confidence cells. Its precomputed worst-case query family contains `164` cells, within the declared budget `256`.

## Hidden infinite-fibre process

The evaluator simulates

```text
(u,v,n) -> (v,W,n+1),
P(W=1|u=0)=0.8,
P(W=1|u=1)=0.2,
```

but supplies only binary observation episodes. The unbounded coordinate `n` is never passed to the learner and is irrelevant to the registered transition laws.

At order one, conditioning on one older observation exposes a certified difference, so memory repairs from `1` to `2`. At order two, one further historical coordinate is certified irrelevant within tolerance.

The four observed contexts have estimated next-one probabilities:

| Context | Estimate | Simultaneous interval | Evaluator truth |
|---|---:|---:|---:|
| `00` | 0.7961 | [0.7702, 0.8220] | 0.8 |
| `01` | 0.8002 | [0.7743, 0.8262] | 0.8 |
| `10` | 0.2006 | [0.1747, 0.2266] | 0.2 |
| `11` | 0.2043 | [0.1784, 0.2302] | 0.2 |


All four selected intervals contain their evaluator truths. Parameter RMSE is `0.002936`.

## Predictive repair versus autonomous repair

Confidence-bounded predictive equivalence groups

```text
{00,01} and {10,11}.
```

This two-block representation has empirical predictive residual `0.00001197` nats, but lumpability residual `0.69314425` nats. Dynamic repair restores four context states. Thus stochastic sampling preserves the central distinction: approximate sufficiency for the selected next-symbol law does not imply autonomous macro-dynamics.

Raw empirical probabilities are almost never exactly equal, so exact equality would spuriously return four predictive blocks. The confidence relation is explicitly tolerance-dependent and carries a finite-sample certificate.

## Held-out prediction

| Model | Accuracy | NLL (nats) | Brier |
|---|---:|---:|---:|
| One state | 0.4982 | 0.6931 | 0.2500 |
| Current observation | 0.4978 | 0.6932 | 0.2500 |
| Fixed order two | 0.8074 | 0.4903 | 0.1556 |
| Adaptive repair | 0.8074 | 0.4903 | 0.1556 |

Adaptive repair selects the correct order and matches the correctly specified fixed-order-two model. It does not beat that oracle-capacity baseline.

## Hostile controls

- First-order null: certified order one with no false split — `True`.
- IID collapse: the two observed symbols merge into one predictive block — `True`.
- Weak signal `0.6 versus 0.4`: remained unresolved at the sample budget rather than forcing a decision — `True`.
- Regime reversal: chronological stationarity gate detected the change and rejected a stationary quotient — `True`.

## Monte Carlo receipt

- Primary recovery: `99/100`; Wilson 95% interval `[0.9455138038212946, 0.9982325679358593]`.
- Primary nonrecoveries: `1` safe abstentions and `0` unsafe certifications. Full seeds and statuses are retained in the machine receipt.
- First-order false splits: `0/100`; Wilson 95% interval `[0.0, 0.03699349820698568]`.
- First-order correct certifications: `100/100`.
- Drift detections: `50/50`.
- Selected-interval empirical coverage: `100/100`.

The Monte Carlo study is a deterministic seeded implementation stress test. The probability guarantee comes from the stated inequality and assumptions, not from observing a favourable finite run.

## Verifier results

- Checks passed: `19/19`.
- Episodes admitted: `256`.
- Observations admitted: `32768`.
- Predictive blocks: `2`.
- Autonomous blocks: `4`.
- Held-out nuisance counter: `1000000000`.

## Boundaries and next experiment

This release establishes a confidence-bounded stochastic repair mechanism for a finite family of binary stationary Markov fixtures. It does not solve arbitrary hidden-state inference, dependence beyond the declared process, continuous observations, language acquisition, or neural scaling.

The next falsifiable extension is active repair: introduce actions and let the verifier select interventions that maximise expected reduction in representation uncertainty per unit cost.
