"""Receipts, report, and figure for finite interfaces over infinite worlds."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from math import log10
from pathlib import Path
from typing import Any, Iterable

from .experiment import source_hashes
from .infinite import (
    CountdownWorld,
    InfiniteFibreRotatingWorld,
    countdown_exact_closure_entropy_excess,
    countdown_lumpability_residual,
    countdown_partition_entropy,
    geometric_entropy,
    numerical_countdown_lumpability_residual,
)
from .repair import dynamic_closure, lumpability_residual, predictive_residual
from .world import RotatingBitWorld


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _stable_case() -> dict[str, object]:
    infinite_world = InfiniteFibreRotatingWorld()
    finite_reduction = RotatingBitWorld()
    task_partition = finite_reduction.task_partition
    closed = dynamic_closure(finite_reduction.transition_matrix, task_partition)
    sample_counter_values = (*range(1024), 10**6)
    failures = []
    checked = 0
    for a_value in (0, 1):
        for b_value in (0, 1):
            for n_value in sample_counter_values:
                state = (a_value, b_value, n_value)
                observed = infinite_world.repaired_label(infinite_world.successor(state))
                expected = infinite_world.repaired_macro_successor(
                    infinite_world.repaired_label(state)
                )
                checked += 1
                if observed != expected:
                    failures.append(
                        {"state": state, "observed": observed, "expected": expected}
                    )
    return {
        "name": "stable finite quotient with infinite fibres",
        "exact_symbolic_certificate": infinite_world.exact_certificate(),
        "finite_reduction": {
            "reason": (
                "all registered task and repaired-interface laws are independent "
                "of n, so summing over any probability law on n cancels exactly"
            ),
            "task_partition": task_partition,
            "predictive_residual_nats": predictive_residual(
                finite_reduction.task_laws,
                task_partition,
            ),
            "lumpability_residual_nats": lumpability_residual(
                finite_reduction.transition_matrix,
                task_partition,
            ),
            "dynamic_closure": closed,
            "closure_excess_blocks": len(closed) - len(task_partition),
        },
        "computed_invariance_check": {
            "states_checked": checked,
            "counter_values_checked": len(sample_counter_values),
            "largest_counter_checked": max(sample_counter_values),
            "failures": failures,
            "passed": not failures,
            "status": (
                "finite computation supporting the symbolic dependency certificate; "
                "not the universal proof by itself"
            ),
        },
        "mathematical_result": (
            "the repaired four-state interface is an exact autonomous quotient, "
            "while every one of its four fibres remains countably infinite"
        ),
    }


def _horizon_record(
    world: CountdownWorld,
    horizon: int,
    probabilities: tuple[float, ...],
    states_checked: int,
) -> dict[str, object]:
    partition = world.horizon_partition(horizon)
    predictive_check = world.verify_predictive_factorisation(
        horizon,
        states_checked,
    )
    sensitivity = []
    for probability in probabilities:
        partition_entropy = countdown_partition_entropy(horizon, probability)
        full_entropy = geometric_entropy(probability)
        entropy_excess = countdown_exact_closure_entropy_excess(
            horizon,
            probability,
        )
        analytic_residual = countdown_lumpability_residual(horizon, probability)
        numerical_residual = numerical_countdown_lumpability_residual(
            horizon,
            probability,
        )
        sensitivity.append(
            {
                "geometric_probability": probability,
                "lumpability_residual_nats": analytic_residual,
                "numerical_residual_nats": numerical_residual,
                "analytic_numeric_absolute_error": abs(
                    analytic_residual - numerical_residual
                ),
                "interface_entropy_nats": partition_entropy,
                "identity_entropy_nats": full_entropy,
                "exact_closure_entropy_excess_nats": entropy_excess,
                "entropy_decomposition_absolute_error": abs(
                    (full_entropy - partition_entropy) - entropy_excess
                ),
            }
        )
    return {
        "horizon": horizon,
        "predictive_partition": partition.finite_description,
        "predictive_partition_block_count": partition.block_count,
        "predictive_outcome_word_count": horizon + 1,
        "predictive_residual_nats": 0.0,
        "predictive_factorisation_check": predictive_check,
        "dynamic_fixed": False,
        "nonclosure_witness": partition.nonclosure_witness,
        "one_step_dynamic_refinement": partition.split_tail_once().finite_description,
        "one_step_repair_excess_blocks": 1,
        "next_horizon_predictive_repair": world.horizon_partition(
            horizon + 1
        ).finite_description,
        "next_horizon_predictive_excess_blocks": 1,
        "exact_dynamic_closure": "identity partition on N_0",
        "exact_dynamic_closure_block_count": "countably_infinite",
        "exact_closure_excess_blocks": "countably_infinite",
        "geometric_weight_sensitivity": sensitivity,
    }


def _countdown_case(
    horizons: tuple[int, ...],
    probabilities: tuple[float, ...],
    primary_horizon: int,
    primary_probability: float,
    trace_steps: int,
    induction_depth: int,
    states_checked: int,
) -> dict[str, object]:
    world = CountdownWorld()
    horizon_records = tuple(
        _horizon_record(world, horizon, probabilities, states_checked)
        for horizon in horizons
    )
    induction = world.induction_certificate(primary_horizon, induction_depth)
    trace = []
    for step, partition in enumerate(world.repair_trace(primary_horizon, trace_steps)):
        trace.append(
            {
                "repair_step": step,
                "threshold": partition.threshold,
                "finite_block_count": partition.block_count,
                "tail_block": f"n>={partition.threshold}",
                "lumpability_residual_nats": countdown_lumpability_residual(
                    partition.threshold,
                    primary_probability,
                ),
                "remaining_exact_closure_entropy_excess_nats": (
                    countdown_exact_closure_entropy_excess(
                        partition.threshold,
                        primary_probability,
                    )
                ),
                "dynamic_fixed": False,
                "counterexample": partition.nonclosure_witness,
            }
        )
    analytic_checks = [
        sensitivity
        for record in horizon_records
        for sensitivity in record["geometric_weight_sensitivity"]
    ]
    adaptive_trace = tuple(
        world.adaptive_repair_record(horizon)
        for horizon in range(1, max(horizons) + 1)
    )
    return {
        "name": "finite predictive derivatives with no finite autonomous closure",
        "state_space": "N_0",
        "state_space_cardinality": "countably_infinite",
        "micro_update": "T(n)=max(n-1,0)",
        "observation": "Y(n)=1[n=0]",
        "finite_derivative": (
            "q_H(n)=(Y(n),Y(Tn),...,Y(T^{H-1}n)) in {0,1}^H"
        ),
        "predictive_partition_formula": (
            "pi_H={0}|{1}|...|{H-1}|{n>=H}, equivalently label min(n,H)"
        ),
        "horizon_records": horizon_records,
        "primary_repair_trace": {
            "horizon": primary_horizon,
            "geometric_probability": primary_probability,
            "steps": trace,
            "terminated": False,
        },
        "adaptive_interface_trace": {
            "initial_derivative_length": 1,
            "final_derivative_length": max(horizons) + 1,
            "repair_events": adaptive_trace,
            "events_checked": len(adaptive_trace),
            "all_events_passed": all(event["passed"] for event in adaptive_trace),
            "interpretation": (
                "each verifier failure splits only the current infinite tail symbol "
                "by registering one more finite derivative coordinate; the new "
                "tail remains infinite and revisable"
            ),
        },
        "nontermination_certificate": induction,
        "theorems": {
            "finite_horizon_sufficiency": (
                "q_H(n)=0^n 1^(H-n) for n<H and 0^H for n>=H, so q_H "
                "factors exactly through pi_H and all H+1 blocks are distinct"
            ),
            "no_finite_dynamic_closure": (
                "at threshold m, compare m with any r>m: their successors lie in "
                "the singleton m-1 and the remaining tail; hence m is forced to a "
                "singleton. Induction forces every n to split, so the identity is "
                "the coarsest strongly lumpable refinement"
            ),
            "residual_formula": (
                "under mu_p(n)=p(1-p)^n, L_lump(pi_m)=(1-p)^m h_2(p)"
            ),
            "entropy_excess_formula": (
                "Delta H_C(pi_m)=(1-p)^m H(Geometric(p)), finite even though "
                "Delta K_C(pi_m) is countably infinite"
            ),
            "small_residual_warning": (
                "L_lump(pi_m) tends to zero as m tends to infinity, but no finite "
                "m is dynamically closed; arbitrarily small residual does not "
                "imply finite exact repair"
            ),
        },
        "verification_summary": {
            "horizons_checked": len(horizons),
            "predictive_state_checks": sum(
                record["predictive_factorisation_check"]["checks_passed"]
                for record in horizon_records
            ),
            "predictive_checks_all_passed": all(
                record["predictive_factorisation_check"]["passed"]
                for record in horizon_records
            ),
            "induction_instances_checked": induction_depth,
            "induction_instances_all_passed": induction[
                "all_checked_steps_passed"
            ],
            "adaptive_repair_events_checked": len(adaptive_trace),
            "adaptive_repair_events_all_passed": all(
                event["passed"] for event in adaptive_trace
            ),
            "analytic_numeric_comparisons": len(analytic_checks),
            "maximum_residual_absolute_error": max(
                check["analytic_numeric_absolute_error"] for check in analytic_checks
            ),
            "maximum_entropy_decomposition_error": max(
                check["entropy_decomposition_absolute_error"]
                for check in analytic_checks
            ),
        },
    }


def run_infinite_experiment(
    *,
    horizons: Iterable[int] = (1, 2, 4, 8, 16, 32),
    probabilities: Iterable[float] = (0.1, 0.25, 0.5),
    primary_horizon: int = 4,
    primary_probability: float = 0.25,
    trace_steps: int = 24,
    induction_depth: int = 512,
    states_checked: int = 4096,
    project_root: Path | None = None,
) -> dict[str, object]:
    horizons = tuple(sorted(set(int(horizon) for horizon in horizons)))
    probabilities = tuple(float(probability) for probability in probabilities)
    if not horizons or any(horizon < 1 for horizon in horizons):
        raise ValueError("horizons must be positive")
    if primary_horizon not in horizons:
        raise ValueError("primary_horizon must be included in horizons")
    if primary_probability not in probabilities:
        raise ValueError("primary_probability must be included in probabilities")
    if states_checked <= max(horizons):
        raise ValueError("states_checked must cross every horizon boundary")
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]

    stable = _stable_case()
    countdown = _countdown_case(
        horizons,
        probabilities,
        primary_horizon,
        primary_probability,
        trace_steps,
        induction_depth,
        states_checked,
    )
    verification_passed = all(
        (
            stable["computed_invariance_check"]["passed"],
            countdown["verification_summary"]["predictive_checks_all_passed"],
            countdown["verification_summary"]["induction_instances_all_passed"],
            countdown["verification_summary"]["adaptive_repair_events_all_passed"],
            countdown["verification_summary"]["maximum_residual_absolute_error"]
            < 1e-10,
            countdown["verification_summary"]["maximum_entropy_decomposition_error"]
            < 1e-12,
        )
    )
    receipt = {
        "schema_version": "ts-language-foundations-infinite-interface-v0.2",
        "experiment": {
            "name": "finite adaptive interfaces over countably infinite interiors",
            "version": "0.2.0",
            "question": (
                "When does a finite derivative admit a finite autonomous quotient, "
                "and when must exact repair grow without bound?"
            ),
        },
        "run": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "horizons": horizons,
            "geometric_probabilities": probabilities,
            "primary_horizon": primary_horizon,
            "primary_probability": primary_probability,
            "trace_steps": trace_steps,
            "induction_depth": induction_depth,
            "states_checked_per_horizon": states_checked,
            "python": sys.version,
            "platform": platform.platform(),
            "neural_components": [],
            "external_corpus": None,
            "tokenizer": None,
            "network_required": False,
        },
        "provenance": {"source_hashes": source_hashes(project_root)},
        "stable_infinite_fibre_case": stable,
        "unbounded_countdown_case": countdown,
        "joint_conclusion": {
            "finite_world_assumption_rejected": True,
            "retained_formulation": (
                "repair acts on a finite-index quotient of an infinite history or "
                "state space, not on a claim that the interior is finite"
            ),
            "three_possible_behaviours": [
                "finite exact closure with infinite fibres",
                "finite horizon sufficiency with continually revisable quotients",
                "no finite exact autonomous closure",
            ],
            "language_role": (
                "symbols serialize the current finite quotient; they neither contain "
                "nor exhaust the infinite fibre denoted by each symbol"
            ),
        },
        "verification": {
            "passed": verification_passed,
            "boundary": (
                "the universal countdown nontermination result is proved by induction "
                "in the report; finite computation checks instances and formulas but "
                "is not presented as the proof"
            ),
        },
    }
    return _jsonable(receipt)


def write_json(receipt: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_report(receipt: dict[str, object], path: Path) -> None:
    stable = receipt["stable_infinite_fibre_case"]
    countdown = receipt["unbounded_countdown_case"]
    stable_reduction = stable["finite_reduction"]
    verification = countdown["verification_summary"]
    primary = countdown["primary_repair_trace"]
    first_step = primary["steps"][0]
    last_step = primary["steps"][-1]
    text = f"""# Finite Interfaces over Infinite Interiors — v0.2

## Question

The world is not assumed finite. Only the observer-facing derivative is finite. When does that derivative admit a finite autonomous state, and when must exact repair continue without bound?

We model an infinite state or history space `X` through a finite-index representation `phi : X -> Z`. Each label `z in Z` denotes the fibre `phi^-1(z)`, which may be infinite. A repair fixed point means that the currently registered task and macro-dynamics factor through `Z`; it does not mean the fibre has been exhausted.

## Result A — finite exact closure with infinite fibres

Let

```text
X = {{0,1}} x {{0,1}} x N_0
T(a,b,n) = (b,1-a,n+1)
q(a,b,n) = a.
```

The task interface `a` is predictively exact for the immediate task but not dynamically closed. Its predictive residual is `{stable_reduction['predictive_residual_nats']:.12f}` nats and its lumpability residual is `{stable_reduction['lumpability_residual_nats']:.12f}` nats. The dependency repair is

```text
{{a}} -> {{a,b}}.
```

The repaired macro-update is `(a,b) -> (b,1-a)`, independent of `n`. Therefore four macrostates form an exact autonomous quotient, while every macrostate still contains the countably infinite fibre `{{(a,b,n): n in N_0}}`.

This is an exact symbolic reduction. The accompanying `{stable['computed_invariance_check']['states_checked']}` concrete state checks support the implementation but are not substituted for the symbolic argument.

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
pi_H = {{0}}|{{1}}|...|{{H-1}}|{{n>=H}},
```

with `H+1` blocks.

### Theorem — no finite strongly lumpable refinement

In the tail block of `pi_H`, compare state `H` with any `r>H`. Their successors are `H-1` and `r-1>=H`. The first lies in the singleton block `{{H-1}}`; the second remains in the tail. Strong lumpability therefore forces `H` to split from the entire tail.

After that split, compare `H+1` with any larger state and repeat. Inductively, every natural number must become a singleton. Therefore the coarsest strongly lumpable refinement of `pi_H` is the identity partition on `N_0`, which has countably infinitely many blocks. No finite exact autonomous refinement exists.

The receipt checks `{verification['induction_instances_checked']}` concrete induction instances. The paragraph above is the universal mathematical proof; the finite check is only its implementation receipt.

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

For the primary run `H={primary['horizon']}`, `p={primary['geometric_probability']}`, the computed residual falls from `{first_step['lumpability_residual_nats']:.12f}` to `{last_step['lumpability_residual_nats']:.12f}` nats over `{len(primary['steps']) - 1}` finite refinements, but no finite step is a dynamic fixed point.

Thus an empirical lumpability residual can be arbitrarily small while exact repair still requires infinitely many blocks. Weighted entropy excess can remain finite even when block-count excess is infinite.

## Adaptive finite interface and symbol revision

The observer need not name the hidden countdown state. At interface version `H`, the two verifier witnesses `H` and `H+1` expose the same finite derivative `0^H`, yet their successor derivatives differ. Registering one additional future-observation coordinate changes the interface from `q_H` to `q_(H+1)` and performs the finite symbol split

```text
tau_H -> {{0^H 1, tau_(H+1)}}.
```

The boundary child is newly resolved; `tau_(H+1)` still denotes the countably infinite fibre `{{n>=H+1}}`. The receipt checks `{countdown['adaptive_interface_trace']['events_checked']}` consecutive repair events. This is temporary stabilization made explicit: each version is finite and task-exact for its registered horizon, while later verifier evidence can refine one symbol without claiming to enumerate the interior.

## Verification

- Predictive word checks: `{verification['predictive_state_checks']}`.
- Induction instances: `{verification['induction_instances_checked']}`.
- Adaptive interface repairs: `{verification['adaptive_repair_events_checked']}`.
- Analytic/numerical residual comparisons: `{verification['analytic_numeric_comparisons']}`.
- Maximum residual error: `{verification['maximum_residual_absolute_error']:.3e}`.
- Maximum entropy-decomposition error: `{verification['maximum_entropy_decomposition_error']:.3e}`.
- Overall verifier: `{receipt['verification']['passed']}`.

## Interpretation

The finite object is the interface, not the interior. In the stable fixture, four finite labels are enough for exact autonomous continuation despite infinite fibres. In the countdown fixture, every finite horizon has an exact finite predictive summary, but autonomous exactness forces unbounded refinement.

Language therefore serializes the current quotient. A symbol points into an equivalence class whose interior may remain infinite, and new horizons or verifiers may force that class to split.

## Boundaries

These are countably infinite deterministic fixtures with exact symbolic partitions. They do not establish results for arbitrary measurable stochastic processes, continuous state spaces, learned observers, or noisy empirical repair. The next experiment should infer and revise a finite interface online from sampled trajectories without access to the hidden countdown state.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def render_svg(receipt: dict[str, object], path: Path) -> None:
    countdown = receipt["unbounded_countdown_case"]
    primary = countdown["primary_repair_trace"]
    steps = primary["steps"]
    residuals = [max(step["lumpability_residual_nats"], 1e-16) for step in steps]
    logs = [log10(value) for value in residuals]
    minimum_log = min(logs)
    maximum_log = max(logs)
    chart_left, chart_top, chart_width, chart_height = 760, 555, 560, 210
    points = []
    for index, value in enumerate(logs):
        x_value = chart_left + chart_width * index / max(1, len(logs) - 1)
        scale = (value - minimum_log) / max(1e-12, maximum_log - minimum_log)
        y_value = chart_top + chart_height * (1.0 - scale)
        points.append(f"{x_value:.2f},{y_value:.2f}")
    point_string = " ".join(points)
    first_residual = residuals[0]
    last_residual = residuals[-1]
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="900" viewBox="0 0 1400 900">
<defs><style>
.bg{{fill:#07111f}} .panel{{fill:#0d1d31;stroke:#294767;stroke-width:2}} .h{{fill:#7ee7ff;font:bold 34px system-ui}}
.t{{fill:#e8f1ff;font:21px system-ui}} .s{{fill:#9bb3ce;font:16px system-ui}} .tiny{{fill:#9bb3ce;font:14px system-ui}}
.node{{fill:#173a55;stroke:#7ee7ff;stroke-width:3}} .tail{{fill:#3b2953;stroke:#c499ff;stroke-width:3}}
.good{{fill:#75f0a7}} .warn{{fill:#ffcf70}} .line{{stroke:#75f0a7;stroke-width:4;fill:none}}
.arrow{{stroke:#9bb3ce;stroke-width:3;fill:none;marker-end:url(#m)}}
</style><marker id="m" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#9bb3ce"/></marker></defs>
<rect class="bg" width="1400" height="900"/><text class="h" x="50" y="58">Finite interface, infinite interior — exact v0.2 receipts</text>
<rect class="panel" x="40" y="90" width="650" height="390" rx="18"/><text class="t" x="70" y="130">A. Stable finite quotient with infinite fibres</text>
<text class="s" x="70" y="163">(a,b,n) → (b,1−a,n+1),   n ∈ N₀</text>
<path class="arrow" d="M190 235 C280 180,450 180,540 235"/><path class="arrow" d="M565 255 C615 310,615 340,565 395"/>
<path class="arrow" d="M520 415 C430 465,300 465,210 415"/><path class="arrow" d="M165 395 C115 340,115 310,165 255"/>
<circle class="node" cx="180" cy="235" r="48"/><circle class="node" cx="550" cy="235" r="48"/><circle class="node" cx="550" cy="405" r="48"/><circle class="node" cx="180" cy="405" r="48"/>
<text class="t" x="139" y="242">(0,0,*)</text><text class="t" x="509" y="242">(0,1,*)</text><text class="t" x="509" y="412">(1,1,*)</text><text class="t" x="139" y="412">(1,0,*)</text>
<text class="good t" x="270" y="330">4 exact macrostates</text><text class="s" x="252" y="358">each * is a countably infinite fibre</text>
<rect class="panel" x="710" y="90" width="650" height="390" rx="18"/><text class="t" x="740" y="130">B. Predictively finite; dynamically unbounded</text>
<text class="s" x="740" y="163">T(n)=max(n−1,0),   q₄ is a four-step zero detector trace</text>
<circle class="node" cx="775" cy="245" r="34"/><circle class="node" cx="875" cy="245" r="34"/><circle class="node" cx="975" cy="245" r="34"/><circle class="node" cx="1075" cy="245" r="34"/><rect class="tail" x="1150" y="211" width="165" height="68" rx="25"/>
<text class="t" x="768" y="252">0</text><text class="t" x="868" y="252">1</text><text class="t" x="968" y="252">2</text><text class="t" x="1068" y="252">3</text><text class="t" x="1176" y="252">n ≥ 4</text>
<path class="arrow" d="M1147 300 C1130 340,1130 360,1147 390"/><text class="s" x="1162" y="345">split 4</text>
<circle class="node" cx="775" cy="410" r="28"/><circle class="node" cx="850" cy="410" r="28"/><circle class="node" cx="925" cy="410" r="28"/><circle class="node" cx="1000" cy="410" r="28"/><circle class="node" cx="1075" cy="410" r="28"/><rect class="tail" x="1130" y="382" width="185" height="56" rx="23"/>
<text class="s" x="769" y="416">0</text><text class="s" x="844" y="416">1</text><text class="s" x="919" y="416">2</text><text class="s" x="994" y="416">3</text><text class="s" x="1069" y="416">4</text><text class="s" x="1162" y="416">n ≥ 5</text>
<text class="warn t" x="780" y="330">Every finite tail contains the next counterexample.</text>
<rect class="panel" x="40" y="510" width="650" height="330" rx="18"/><text class="t" x="70" y="550">Exact distinction</text>
<text class="s" x="70" y="595">Finite horizon H</text><text class="good t" x="330" y="595">H+1 predictive blocks</text>
<text class="s" x="70" y="640">Exact dynamic closure</text><text class="warn t" x="330" y="640">countably infinite blocks</text>
<text class="s" x="70" y="685">Block closure excess</text><text class="warn t" x="330" y="685">∞</text>
<text class="s" x="70" y="730">Entropy closure excess</text><text class="good t" x="330" y="730">finite under geometric weights</text>
<text class="s" x="70" y="775">Meaning</text><text class="t" x="330" y="775">small residual ≠ finite exact repair</text>
<rect class="panel" x="710" y="510" width="650" height="330" rx="18"/><text class="t" x="740" y="550">Residual during finite repair prefix (log₁₀ scale)</text>
<line x1="{chart_left}" y1="{chart_top + chart_height}" x2="{chart_left + chart_width}" y2="{chart_top + chart_height}" stroke="#5e7894" stroke-width="2"/>
<line x1="{chart_left}" y1="{chart_top}" x2="{chart_left}" y2="{chart_top + chart_height}" stroke="#5e7894" stroke-width="2"/>
<polyline class="line" points="{point_string}"/><text class="tiny" x="{chart_left}" y="{chart_top + chart_height + 28}">repair 0</text><text class="tiny" x="{chart_left + chart_width - 70}" y="{chart_top + chart_height + 28}">repair {len(steps)-1}</text>
<text class="tiny" x="{chart_left + 12}" y="{chart_top + 20}">{first_residual:.3e}</text><text class="tiny" x="{chart_left + chart_width - 90}" y="{chart_top + chart_height - 12}">{last_residual:.3e}</text>
<text class="warn s" x="760" y="815">Residual falls every step; exact closure is never reached finitely.</text>
<text class="s" x="50" y="875">Symbols serialize finite quotient labels. They do not contain or exhaust the infinite fibres those labels denote.</text>
</svg>'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
