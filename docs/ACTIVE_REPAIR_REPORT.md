# Active Intervention Repair — Technical Report

## Question

Can a representation-repair learner choose an intervention that exposes missing state more efficiently than passive or uniform data collection, while keeping representation changes under an independent statistical verifier?

## Controlled world and source boundary

The hidden environment is

\[
(u,v,n)\xrightarrow{a}(v,W,n+1),\qquad (u,v)\in\{0,1}^2,\ n\in\mathbb N_0.
\]

For the `baseline` action, `P(W=1|u,baseline)=0.5`. For the `probe` action,

\[
P(W=1|u=0,probe)=0.85,\qquad P(W=1|u=1,probe)=0.15.
\]

The learner receives visible binary history, its own chosen action, the declared action cost, and the resulting bit. It never receives `n`, a hidden-state identifier, or the correct memory order. No corpus, network service, tokenizer, neural component, or external model is used.

## Action-family mathematics

For action family `A`, define `x ~_A y` when `q_x^a=q_y^a` for every `a in A`. If `A0` is a subset of `A`, then `Q_A` refines `Q_A0`: equality under every action implies equality under the restricted family. Expanding the interventions can therefore reveal distinctions invisible to a passive policy.

Controlled autonomous closure is stronger still. A partition must be strongly lumpable for every action kernel, not merely predictive for each immediate response.

## Proposal versus authority

The action policy scores interventions using posterior expected information gain, expected older-bit contrast, and an uncertainty-reduction proxy per unit cost. These quantities have no authority to change the representation.

For each predeclared action/context cell and every within-cell sample prefix `n<=T`, the verifier uses

\[
r(n)=\sqrt{\frac{\log(2M/\delta)}{2n}}.
\]

The finite family contains `24` intrinsic cells and `1572864` cell-prefix events. The union bound therefore remains valid under predictable action choice and adaptive stopping, assuming stationary conditionally independent Bernoulli innovations within each action/context cell.

## Primary repair

| Quantity | Active | Uniform |
|---|---:|---:|
| First certified repair step | 1344 | 2240 |
| First certified repair cost | 2688 | 3360 |
| Full fixed-point step | 22400 | 32256 |
| Full fixed-point cost | 32391 | 48384 |

The active learner repaired memory order `1 -> 2`. It then certified that one still-older coordinate was unnecessary across both declared actions.

## Predictive compression and autonomous closure

The confidence partition is

\[
\{00,01\},\qquad \{10,11\}.
\]

Its action-averaged predictive residual is `0.000380204` nats. Its action-averaged lumpability residual is `0.693147181` nats. Controlled dynamic repair restores all four suffix states, giving closure excess `2`.

The reason is structural: the current response vector depends on the older bit `u`, while the next predictive block is determined by the current bit `v`. Merging `00` with `01` answers the current action query but destroys autonomous macrostate evolution.

## Heldout controlled prediction

Heldout evaluation uses `8000` balanced action-conditioned transitions at evaluator-only nuisance counter `10^9`.

| Model | Accuracy | NLL (nats) | Brier |
|---|---:|---:|---:|
| One state per action | 0.5046 | 0.6931 | 0.2500 |
| Current bit per action | 0.5055 | 0.6931 | 0.2500 |
| Adaptive repaired order two | 0.6774 | 0.5556 | 0.1878 |
| Fixed order two | 0.6774 | 0.5556 | 0.1878 |

The adaptive and fixed-order-two results are identical. This is a state-discovery result, not predictive superiority over an oracle supplied with the correct order.

## Policy-scope and hostile controls

- Passive baseline data: restricted extension status `certified_within_tolerance`; full-family status `insufficient_evidence`.
- Null action family: final order `1`, split invented `False`.
- Misspecified greedy proposal: `sample_budget_exhausted` with no unsafe certificate.
- Reversing probe law: `action_conditioned_distribution_shift_detected`.
- Probe-cost sensitivity: every declared cost recovered the same repair sample requirement; paid cost scaled because the revealing intervention was necessary.

## Monte Carlo

Across `100` paired discovery seeds:

- active recovery `100/100`, median `1088` steps and `2176` cost;
- uniform recovery `100/100`, median `2112` steps and `3168` cost;
- null false splits `0/50`;
- misspecified unsafe outputs `0/20`;
- drift detections `50/50`.

The paired bootstrap median active/uniform step ratio is `0.514` with 95% percentile interval `[0.48484848484848486, 0.5303308823529411]`. This interval describes deterministic fixture sensitivity across seeds, not a population parameter.

## Verifier results

`22/22` declared checks passed.

## What failed or remains uncertain

- Passive sufficiency is policy-relative and cannot be promoted to untried actions.
- A misspecified greedy proposal without forced exploration can completely miss the useful action. The verifier protects correctness by abstaining, but cannot manufacture data.
- Active selection cannot optimise away the cost of an intervention that is uniquely informative.
- The action semantics and suffix constructor family are hand-declared.
- The coverage proof assumes stationary conditionally independent cell innovations. The drift gate detects the supplied reversal but does not solve general nonstationary control.
- This is not general causal discovery, continuous control, language learning, planning, or evidence of neural-model superiority.

## Next falsifiable experiment

Replace the hand-declared suffix family with typed relational constructors. Require the learner to choose both a candidate feature and an intervention, then certify whether the constructed relation predicts and evolves autonomously without hidden-state labels.
