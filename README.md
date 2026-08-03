# TS Language Foundations v0.7.0

v0.7 moves the construction engine beyond binary truth tables. It observes raw
length-three words over a ternary alphabet, proposes both symbol-valued and
predicate-valued expressions through a typed grammar, chooses interventions,
and lets a simultaneous categorical total-variation verifier select the least
supported partition.

In the primary fixture, the hidden probe law depends on whether the oldest and
newest symbols agree. The learner receives no equality label. It constructs
`eq(lag0,lag2)`, compressing 27 raw histories to two predictive states. Exact
controlled partition refinement then forces the repair

```text
2 -> 4 -> 12 -> 18 blocks.
```

The 18-state autonomous representation records the newest symbol, middle
symbol, and endpoint-equality relation. It does not restore raw identity: when
the endpoints differ, which alternative old symbol occurred remains
irrelevant.

The release also freezes the constructed state across new tasks. A compatible
task changes the categorical outcome mapping but preserves the endpoint factor
and transfers exactly. An incompatible middle-symbol task has a strictly
positive predictive residual; combining it with the source task repairs the
predictive state from two to six blocks, whose autonomous closure is again 18.

The partial repairs concretely fail to commute at one state:

```text
C(P(pi)) has 18 blocks; P(C(pi)) has 2 blocks.
```

Fair completion remains order-independent. Null, primitive, grammar-omission,
stationarity, drift, semantic-polarity, held-out transfer, and paired Monte
Carlo controls are included with machine-readable receipts and a code-generated
figure.

This release makes the central correction forced by the mathematics:

> The world is not assumed finite. The finite object is the observer-facing
> derivative or interface. Each finite label may denote an infinite fibre.

Language remains downstream. It serializes the distinctions currently present
in the repaired interface; it neither contains nor exhausts the interior
represented by those distinctions.

v0.3 turns that static mathematics into a first observation-only adaptive
machine. It begins with a finite memory interface, accepts verifier
counterexamples, registers additional history only when required, compiles an
autonomous quotient when possible, and abstains at a declared resource limit.

v0.4 removes the deterministic contradiction assumption. It uses simultaneous
confidence bounds to distinguish stochastic variation from missing state,
supports merge/split/wait decisions, rejects time-varying laws, and records the
sample cost of every certification.

v0.5 makes data acquisition endogenous. A proposal policy chooses actions by
expected information and uncertainty reduction per cost, while an independent
anytime confidence verifier retains exclusive authority to repair or certify
the representation. The release distinguishes passive-policy sufficiency from
sufficiency across a declared intervention family.

v0.6 removes the hand-supplied repair coordinate. A typed Boolean grammar
constructs candidate relational features from raw three-bit histories. In the
primary fixture the verifier selects an XOR partition without receiving the
XOR label, rejects every earlier candidate, and then repairs the two-state
predictive quotient into the four-state autonomous suffix representation.

No LLM, neural network, pretrained weights, tokenizer, corpus, embeddings, or
external data participate in the experiments.

## Mathematical object

Let `X` be an infinite state or history space and let

```text
phi : X -> Z,     |Z| < infinity.
```

The finite representation is the finite-index quotient `Z`. A label `z` denotes
the fibre `phi^-1(z)`, which may be infinite. A repair fixed point means that
the registered predictive laws and macro-dynamics factor through `phi`. It does
not mean the fibre is completely modelled.

The retained v0.2 mathematics supplies two countably infinite fixtures that
separate finite-interface behaviours.

## A. Stable finite quotient with infinite fibres

```text
X = {0,1} x {0,1} x N_0
T(a,b,n) = (b,1-a,n+1)
q(a,b,n) = a.
```

The immediate-task interface `{a}` has zero predictive residual but
lumpability residual `ln(2)`. Dependency repair gives

```text
{a} -> {a,b}.
```

The repaired update `(a,b) -> (b,1-a)` is independent of `n`. It therefore
defines an exact four-state autonomous quotient, while each of the four fibres

```text
{(a,b,n) : n in N_0}
```

remains countably infinite.

This demonstrates that finite exact closure does not require a finite interior.

## B. Finite prediction with no finite autonomous closure

```text
X = N_0
T(n) = max(n-1,0)
Y(n) = 1[n=0].
```

For finite horizon `H`, define

```text
q_H(n) = (Y(n),Y(Tn),...,Y(T^(H-1)n)).
```

The exact predictive partition is

```text
pi_H = {0}|{1}|...|{H-1}|{n>=H}.
```

It has only `H+1` blocks and predictive residual zero. It is never dynamically
closed. At any finite tail threshold `m`, state `m` transitions to the singleton
`m-1`, whereas every state `r>m` transitions to a state still in the remaining
tail. Strong lumpability therefore forces `m` to split. Induction forces every
natural number to a singleton.

Consequently:

```text
C_T(pi_H) = identity on N_0,
Delta K_C(pi_H) = countably infinite.
```

No finite exact autonomous refinement exists.

The interface can still evolve finitely. At version `H`, its unresolved tail
symbol `tau_H` denotes the infinite fibre `{n>=H}`. A newly registered
one-step-longer derivative performs one local symbol revision:

```text
tau_H -> {0^H 1, tau_(H+1)}.
```

The first child resolves the newly exposed boundary state; the second remains
an infinite fibre. Thus every operational interface version is finite without
pretending that the interior has become finite or completely enumerated.

## Small residual does not imply finite repair

Under geometric state weights

```text
mu_p(n) = p(1-p)^n,
```

the exact formulas are

```text
L_lump(pi_m)   = (1-p)^m h_2(p),
Delta H_C(pi_m)= (1-p)^m H(Geometric(p)).
```

Both weighted quantities approach zero as the finite boundary moves outward,
but every finite partition still has a lumpability counterexample and the exact
block closure excess remains infinite.

This is the principal v0.2 warning:

> An arbitrarily small empirical closure residual does not imply that a finite
> exact autonomous state exists.

## v0.3 — learn the interface from observations only

The learner's input is a tuple of observation episodes. It never receives the
hidden `(a,b,n)` state, the countdown value, or state IDs. At suffix order `k`,
its verifier searches for one represented context followed by two distinct
next symbols. A witness repairs the interface by registering one more history
coordinate:

```text
suffix order k -> suffix order k+1.
```

On the stable infinite-fibre process, one-symbol memory fails and two-symbol
memory closes:

```text
00 -> 01 -> 11 -> 10 -> 00.
```

The equal-next-symbol predictive partition has two blocks and predictive
residual zero, but lumpability residual `ln(2)`. Dynamic repair restores the
four learned suffix states. Held-out trajectories at hidden counter value
`1,000,000` are predicted exactly.

On the countdown process, every tested finite suffix budget contains a context
`0^k` followed by both `0` and `1`. The learner therefore abstains. Fixed
sequence baselines can exceed 95% accuracy while still failing at every tested
countdown boundary.

This is not claimed as a novel automata-learning algorithm. Suffix refinement
has standard connections to variable-order Markov models, automata inference,
predictive equivalence, and causal-state construction. The tested connection is
its integration with explicit predictive repair, dynamic closure, infinite
fibres, resource bounds, and fail-closed receipts.

## v0.4 — confidence-bounded stochastic repair

For a predeclared family of at most `M` Bernoulli context laws and family error
`delta`, each empirical probability receives the simultaneous Hoeffding radius

```text
r(n) = sqrt(log(2M/delta)/(2n)).
```

For two context laws, the verifier computes lower and upper confidence bounds
on their absolute difference. With fixed thresholds
`epsilon_merge < epsilon_split`, it performs exactly one of three operations:

```text
upper <= epsilon_merge  -> certify approximate merge
lower > epsilon_split   -> certify split
otherwise               -> acquire data or abstain
```

The primary stochastic world is

```text
(u,v,n) -> (v,W,n+1)
P(W=1|u=0)=0.8
P(W=1|u=1)=0.2.
```

Only the binary trajectories reach the learner. Memory repairs from order one
to order two. Confidence-bounded predictive repair groups `{00,01}` and
`{10,11}`; strong-lumpability repair then restores four autonomous states.
The held-out nuisance counter is `1,000,000,000`.

The hostile controls include:

- a true first-order process that must not split;
- an IID process whose predictive interface should collapse to one block;
- a weak signal inside the uncertainty band, which must remain unresolved;
- a reversed second-half transition law, which the stationarity gate must reject;
- seeded Monte Carlo recovery, false-split, interval-coverage, and drift tests.

The fixed order-two probability model remains an oracle-capacity baseline and
matches adaptive prediction. The tested advance is evidence-controlled state
selection and closure, not a claim of superior raw prediction.

## v0.5 — active intervention repair

The controlled stochastic world is

```text
(u,v,n) --a--> (v,W,n+1)
P(W=1|u,baseline)=0.5
P(W=1|u=0,probe)=0.85
P(W=1|u=1,probe)=0.15.
```

Passive baseline data cannot expose the older-bit dependence. Under the full
action family, however, the probe supplies a confidence-certified witness that
repairs memory order one to order two.

For action family `A`, define `x ~_A y` when the response laws agree for every
`a in A`. If `A0` is a subset of `A`, then `Q_A` refines `Q_A0`. Expanding the
available interventions can reveal distinctions invisible under a restricted
policy. Autonomous controlled dynamics require strong lumpability for every
action kernel.

Because action choice and stopping are adaptive, v0.5 declares every
action/context cell and every possible within-cell sample prefix up to the
resource bound. A finite union bound over this enlarged family supports the
same Hoeffding radius under predictable action selection. Proposal scores do
not certify anything.

The primary run finds the first memory repair with fewer interactions and lower
cost than uniform action selection, then reaches the full action-family fixed
point. The confidence action-predictive quotient has two blocks; controlled
dynamic closure restores four states. Controls cover passive scope, a null
action family, a misspecified greedy proposal, action-conditioned drift, probe
cost, and paired Monte Carlo comparisons.

## v0.6 — construct the relational coordinate

The v0.5 learner was still told the form of repair: register one additional
history coordinate. v0.6 replaces that scaffold with a finite typed grammar:

```text
bool := 0 | 1 | lag0 | lag1 | lag2
      | not(bool)
      | xor(bool,bool) | eq(bool,bool)
      | and(bool,bool) | or(bool,bool).
```

Expressions are exhaustively enumerated through a declared node-complexity
budget, deduplicated by their truth tables on the eight raw histories, and
combined into feature tuples. Candidate representations are ordered before
any outcome is observed by block count, expression complexity, feature count,
fixed grammar priority, and name.

The primary controlled world is

```text
(x[t-2],x[t-1],x[t],n) --a--> (x[t-1],x[t],W,n+1)
P(W=1|history,baseline) = 0.5
P(W=1|xor(x[t-1],x[t])=0,probe) = 0.85
P(W=1|xor(x[t-1],x[t])=1,probe) = 0.15.
```

The learner receives raw histories, actions, costs, and outcomes. It does not
receive the hidden counter, the true relation, or a feature label. The fixed
grammar generates 20 distinct Boolean truth functions and 65 unique candidate
partitions. A candidate is selected only if every raw context/action cell is
supported, every within-block law is confidence-equivalent, and every earlier
candidate is rejected. Unresolved predecessors force abstention.

The verifier selects

```text
xor(lag0,lag1): 8 raw histories -> 2 predictive blocks.
```

That quotient is not autonomous. Controlled dynamic closure gives

```text
{xor(lag0,lag1)} -> {lag0,lag1}: 2 blocks -> 4 blocks,
```

while correctly discarding irrelevant `lag2`. The confidence-pooled empirical
interface and the exact fixture oracle give the same repair. The unpooled
empirical model is retained separately because exact equality tests would turn
finite-sample noise into eight spurious singleton states.

At the one-state abstraction the partial repairs do not commute, but both fair
orders reach the same four-block common fixed point. Held-out likelihood of the
two-block XOR quotient matches the raw eight-state identity control while
beating all one-coordinate candidates.

Hostile controls require one state in a null world, select primitive `lag1` in
a primitive world, abstain when the relational grammar is removed, abstain on
passive or deliberately misspecified action policies, and reject a reversed
response law. XOR and EQ induce the same unlabelled partition, so the XOR name
is explicitly treated as canonical syntax rather than identified semantics.

## Preserved v0.1 protocol result

The original finite protocol remains as a kernel fixture. It repairs a coarse
task label into two binary coordinates, selects the reusable product code over
a holistic state code, and grounds an unordered opaque-signal protocol.

- All `4! = 24` signal renamings are enumerated exactly once.
- Four held-out states per renaming give `96` trials.
- Ordered protocol: `96/96`.
- Unordered, slot-free signal-type inference: `96/96`.
- Holistic held-out coverage: `0/96`.

## Run

```bash
python -m pip install -e ".[dev]"

ts-language-foundations \
  --output artifacts/run_receipt.json \
  --figure figures/finite_protocol.svg \
  --report docs/TECHNICAL_REPORT.md \
  --renaming-count 24

ts-infinite-interface \
  --output artifacts/infinite_interface_receipt.json \
  --figure figures/infinite_interface.svg \
  --report docs/INFINITE_INTERFACE_REPORT.md

ts-online-interface \
  --output artifacts/online_interface_receipt.json \
  --figure figures/online_interface_learning.svg \
  --report docs/ONLINE_INTERFACE_REPORT.md

ts-stochastic-interface \
  --output artifacts/stochastic_interface_receipt.json \
  --figure figures/stochastic_interface_learning.svg \
  --report docs/STOCHASTIC_INTERFACE_REPORT.md

ts-active-repair \
  --output artifacts/active_repair_receipt.json \
  --figure figures/active_repair.svg \
  --report docs/ACTIVE_REPAIR_REPORT.md

ts-relational-repair \
  --output artifacts/relational_repair_receipt.json \
  --figure figures/relational_repair.svg \
  --report docs/RELATIONAL_REPAIR_REPORT.md

pytest -q
```

Or run all experiments:

```bash
./scripts/run_experiment.sh
```

After generating the release manifest, verify every hash and recompute all
scientific receipts:

```bash
PYTHONPATH=src python scripts/verify_release.py
```

## Verification boundary

The countdown nontermination statement is proved by induction in
`docs/INFINITE_INTERFACE_REPORT.md`. The code separately checks a long finite
prefix of the induction and verifies the analytic residual and entropy formulas
against numerical summation. The finite computation is a receipt for the
implementation, not a substitute for the universal proof.

The v0.1–v0.3 fixtures are deterministic and finite or countably infinite. The
exact online learner is exact only on its observed support.
The v0.4 learner adds a finite-family statistical guarantee under an explicitly
stationary conditional-Bernoulli assumption. v0.5 enlarges that finite family
over within-cell sample prefixes so that predictable actions and adaptive
stopping are covered. The action semantics, hypothesis family, and suffix
constructors remain hand-declared. v0.6 replaces the selected suffix coordinate
with a generated relation, but its Boolean grammar, complexity limit, feature
tuple limit, actions, and reset protocol are still declared by the experimenter.
Nothing here establishes results for arbitrary dependent stochastic processes,
general causal discovery, continuous control, language learning, planning, or
superiority to neural or reinforcement learning systems.

## Artifacts

- `artifacts/infinite_interface_receipt.json`: infinite-interface result and proof receipts.
- `docs/INFINITE_INTERFACE_REPORT.md`: definitions, theorem, proof, and limitations.
- `figures/infinite_interface.svg`: code-generated infinite-interface figure.
- `artifacts/online_interface_receipt.json`: observation-only learning, baselines, repair traces, and abstention receipt.
- `docs/ONLINE_INTERFACE_REPORT.md`: method, findings, standard connections, and limitations.
- `figures/online_interface_learning.svg`: code-generated online-repair figure.
- `artifacts/stochastic_interface_receipt.json`: confidence cells, repair decisions, controls, sensitivities, Monte Carlo results, and statistical boundaries.
- `docs/STOCHASTIC_INTERFACE_REPORT.md`: confidence theorem, stochastic findings, controls, and limitations.
- `figures/stochastic_interface_learning.svg`: code-generated statistical-repair figure.
- `artifacts/active_repair_receipt.json`: action traces, anytime confidence contract, policy comparisons, controlled closure, hostile controls, and Monte Carlo receipt.
- `docs/ACTIVE_REPAIR_REPORT.md`: action-family mathematics, active result, controls, and limitations.
- `figures/active_repair.svg`: code-generated active-repair figure.
- `artifacts/relational_repair_receipt.json`: generated feature family, all-candidate metrics, confidence evidence, transition models, hostile controls, and Monte Carlo receipt.
- `docs/RELATIONAL_REPAIR_REPORT.md`: relational construction result, mathematical contract, failures, and limitations.
- `figures/relational_repair.svg`: code-generated relational-state and policy-efficiency figure.
- `artifacts/run_receipt.json`: preserved finite protocol receipt.
- `artifacts/reference_cross_check.json`: five comparisons with repair kernel v0.3.0.
- `artifacts/release_manifest.json`: fail-closed SHA-256 inventory.
