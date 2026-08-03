# Finite Interfaces over Infinite Interiors — v0.2

## Question

The world is not assumed finite. Only the observer-facing derivative is finite. When does that derivative admit a finite autonomous state, and when must exact repair continue without bound?

We model an infinite state or history space `X` through a finite-index representation `phi : X -> Z`. Each label `z in Z` denotes the fibre `phi^-1(z)`, which may be infinite. A repair fixed point means that the currently registered task and macro-dynamics factor through `Z`; it does not mean the fibre has been exhausted.

## Result A — finite exact closure with infinite fibres

Let

```text
X = {0,1} x {0,1} x N_0
T(a,b,n) = (b,1-a,n+1)
q(a,b,n) = a.
```

The task interface `a` is predictively exact for the immediate task but not dynamically closed. Its predictive residual is `0.000000000000` nats and its lumpability residual is `0.693147180560` nats. The dependency repair is

```text
{a} -> {a,b}.
```

The repaired macro-update is `(a,b) -> (b,1-a)`, independent of `n`. Therefore four macrostates form an exact autonomous quotient, while every macrostate still contains the countably infinite fibre `{(a,b,n): n in N_0}`.

This is an exact symbolic reduction. The accompanying `4100` concrete state checks support the implementation but are not substituted for the symbolic argument.

## Result B — finite prediction without finite autonomous closure

Let

```text
X = N_0
T(n) = max(n-1,0)
Y(n) = 1[n=0].
```

For horizon `H`, define the finite derivative

```text
q_H(n) = (Y(n),Y(Tn),...,Y(T^(H-1)n)).
```

Then

```text
q_H(n) = 0^n 1^(H-n),  n < H,
q_H(n) = 0^H,          n >= H.
```

Hence the exact predictive partition is

```text
pi_H = {0}|{1}|...|{H-1}|{n>=H},
```

with `H+1` blocks.

### Theorem — no finite strongly lumpable refinement

In the tail block of `pi_H`, compare state `H` with any `r>H`. Their successors are `H-1` and `r-1>=H`. The first lies in the singleton block `{H-1}`; the second remains in the tail. Strong lumpability therefore forces `H` to split from the entire tail.

After that split, compare `H+1` with any larger state and repeat. Inductively, every natural number must become a singleton. Therefore the coarsest strongly lumpable refinement of `pi_H` is the identity partition on `N_0`, which has countably infinitely many blocks. No finite exact autonomous refinement exists.

The receipt checks `512` concrete induction instances. The paragraph above is the universal mathematical proof; the finite check is only its implementation receipt.

## Weighted residuals do not settle exact closure

Give state `n` geometric weight

```text
mu_p(n) = p(1-p)^n.
```

For the singleton-prefix partition with threshold `m`, the only defect is its infinite tail. Conditional on that tail, state `m` has mass `p` and the deeper tail has mass `1-p`. Therefore

```text
L_lump(pi_m) = (1-p)^m h_2(p).
```

Likewise,

```text
Delta H_C(pi_m) = (1-p)^m H(Geometric(p)),
Delta K_C(pi_m) = countably infinite.
```

For the primary run `H=4`, `p=0.25`, the computed residual falls from `0.177926354352` to `0.000178529752` nats over `24` finite refinements, but no finite step is a dynamic fixed point.

Thus an empirical lumpability residual can be arbitrarily small while exact repair still requires infinitely many blocks. Weighted entropy excess can remain finite even when block-count excess is infinite.

## Adaptive finite interface and symbol revision

The observer need not name the hidden countdown state. At interface version `H`, the two verifier witnesses `H` and `H+1` expose the same finite derivative `0^H`, yet their successor derivatives differ. Registering one additional future-observation coordinate changes the interface from `q_H` to `q_(H+1)` and performs the finite symbol split

```text
tau_H -> {0^H 1, tau_(H+1)}.
```

The boundary child is newly resolved; `tau_(H+1)` still denotes the countably infinite fibre `{n>=H+1}`. The receipt checks `32` consecutive repair events. This is temporary stabilization made explicit: each version is finite and task-exact for its registered horizon, while later verifier evidence can refine one symbol without claiming to enumerate the interior.

## Verification

- Predictive word checks: `24576`.
- Induction instances: `512`.
- Adaptive interface repairs: `32`.
- Analytic/numerical residual comparisons: `18`.
- Maximum residual error: `5.551e-17`.
- Maximum entropy-decomposition error: `1.152e-15`.
- Overall verifier: `True`.

## Interpretation

The finite object is the interface, not the interior. In the stable fixture, four finite labels are enough for exact autonomous continuation despite infinite fibres. In the countdown fixture, every finite horizon has an exact finite predictive summary, but autonomous exactness forces unbounded refinement.

Language therefore serializes the current quotient. A symbol points into an equivalence class whose interior may remain infinite, and new horizons or verifiers may force that class to split.

## Boundaries

These are countably infinite deterministic fixtures with exact symbolic partitions. They do not establish results for arbitrary measurable stochastic processes, continuous state spaces, learned observers, or noisy empirical repair. The next experiment should infer and revise a finite interface online from sampled trajectories without access to the hidden countdown state.
