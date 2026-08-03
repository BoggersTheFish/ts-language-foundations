"""Finite interfaces over countably infinite deterministic worlds.

Two fixtures separate facts that a wholly finite model conflates:

* ``InfiniteFibreRotatingWorld`` has a four-state autonomous quotient, but
  every quotient state contains countably infinitely many interior states.
* ``CountdownWorld`` has an exact finite predictive quotient at every finite
  horizon, but no finite strongly lumpable refinement of any such quotient.

The code uses symbolic formulas rather than truncating the state space and
pretending the truncation is the world.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Iterable

InfiniteProductState = tuple[int, int, int]


@dataclass(frozen=True)
class DependencyClosure:
    initial_coordinates: tuple[str, ...]
    closed_coordinates: tuple[str, ...]
    trace: tuple[tuple[str, ...], ...]
    omitted_infinite_coordinates: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "initial_coordinates": self.initial_coordinates,
            "closed_coordinates": self.closed_coordinates,
            "trace": self.trace,
            "omitted_infinite_coordinates": self.omitted_infinite_coordinates,
        }


@dataclass(frozen=True)
class InfiniteFibreRotatingWorld:
    """The v0.1 rotating-bit dynamics crossed with an unbounded counter."""

    finite_domains: tuple[tuple[str, tuple[int, ...]], ...] = (
        ("a", (0, 1)),
        ("b", (0, 1)),
    )
    infinite_coordinates: tuple[str, ...] = ("n",)

    @staticmethod
    def validate_state(state: InfiniteProductState) -> None:
        a_value, b_value, n_value = state
        if a_value not in (0, 1) or b_value not in (0, 1) or n_value < 0:
            raise ValueError("state must lie in {0,1} x {0,1} x N_0")

    def successor(self, state: InfiniteProductState) -> InfiniteProductState:
        self.validate_state(state)
        a_value, b_value, n_value = state
        return b_value, 1 - a_value, n_value + 1

    def task(self, state: InfiniteProductState) -> int:
        self.validate_state(state)
        return state[0]

    def coarse_label(self, state: InfiniteProductState) -> tuple[int]:
        return (self.task(state),)

    def repaired_label(self, state: InfiniteProductState) -> tuple[int, int]:
        self.validate_state(state)
        return state[0], state[1]

    @staticmethod
    def repaired_macro_successor(label: tuple[int, int]) -> tuple[int, int]:
        a_value, b_value = label
        if a_value not in (0, 1) or b_value not in (0, 1):
            raise ValueError("macro label must be binary")
        return b_value, 1 - a_value

    @property
    def coordinate_update_dependencies(self) -> dict[str, tuple[str, ...]]:
        # Exact dependencies read directly from a'=b, b'=1-a, n'=n+1.
        return {"a": ("b",), "b": ("a",), "n": ("n",)}

    def dependency_closure(
        self,
        initial_coordinates: Iterable[str] = ("a",),
    ) -> DependencyClosure:
        initial = tuple(initial_coordinates)
        represented = set(initial)
        if not represented <= {"a", "b", "n"}:
            raise ValueError("unknown coordinate in initial interface")
        trace = [tuple(sorted(represented))]
        while True:
            nxt = set(represented)
            for coordinate in represented:
                nxt.update(self.coordinate_update_dependencies[coordinate])
            if nxt == represented:
                break
            represented = nxt
            trace.append(tuple(sorted(represented)))
        omitted = tuple(
            coordinate for coordinate in self.infinite_coordinates if coordinate not in represented
        )
        return DependencyClosure(
            initial_coordinates=initial,
            closed_coordinates=tuple(sorted(represented)),
            trace=tuple(trace),
            omitted_infinite_coordinates=omitted,
        )

    def coarse_nonclosure_witnesses(
        self,
    ) -> tuple[dict[str, object], ...]:
        witnesses = []
        for a_value in (0, 1):
            left = (a_value, 0, 0)
            right = (a_value, 1, 0)
            witnesses.append(
                {
                    "left_state": left,
                    "right_state": right,
                    "same_current_coarse_label": self.coarse_label(left),
                    "left_next_coarse_label": self.coarse_label(self.successor(left)),
                    "right_next_coarse_label": self.coarse_label(self.successor(right)),
                }
            )
        return tuple(witnesses)

    def exact_certificate(self) -> dict[str, object]:
        closure = self.dependency_closure(("a",))
        repaired_updates = {
            str((a_value, b_value)): self.repaired_macro_successor(
                (a_value, b_value)
            )
            for a_value in (0, 1)
            for b_value in (0, 1)
        }
        return {
            "state_space": "{0,1} x {0,1} x N_0",
            "state_space_cardinality": "countably_infinite",
            "micro_update": "(a,b,n) -> (b,1-a,n+1)",
            "registered_task": "a",
            "coarse_interface": ("a",),
            "coarse_block_count": 2,
            "coarse_fibre_cardinality": "countably_infinite",
            "coarse_nonclosure_witnesses": self.coarse_nonclosure_witnesses(),
            "dependency_closure": closure.to_dict(),
            "repaired_interface": closure.closed_coordinates,
            "repaired_block_count": 4,
            "repaired_fibre_cardinality": "countably_infinite",
            "repaired_macro_updates": repaired_updates,
            "repaired_update_independent_of_n": True,
            "repaired_strongly_lumpable": True,
        }


@dataclass(frozen=True)
class CountdownPartition:
    """Singleton prefix plus one infinite tail block.

    ``threshold=m`` denotes

    ``{0}|{1}|...|{m-1}|{n >= m}``.
    """

    threshold: int

    def __post_init__(self) -> None:
        if self.threshold < 1:
            raise ValueError("threshold must be positive")

    @property
    def block_count(self) -> int:
        return self.threshold + 1

    def label(self, state: int) -> int:
        if state < 0:
            raise ValueError("countdown state must be non-negative")
        return min(state, self.threshold)

    @property
    def finite_description(self) -> tuple[object, ...]:
        return (*tuple((state,) for state in range(self.threshold)), (f"n>={self.threshold}",))

    def successor_labels(self, block_label: int) -> tuple[int, ...]:
        if not 0 <= block_label <= self.threshold:
            raise ValueError("block label out of range")
        if block_label == 0:
            return (0,)
        if block_label < self.threshold:
            return (block_label - 1,)
        return (self.threshold - 1, self.threshold)

    @property
    def strongly_lumpable(self) -> bool:
        return False

    @property
    def nonclosure_witness(self) -> dict[str, object]:
        return self.separation_witness(self.threshold + 1)

    def separation_witness(self, right_state: int) -> dict[str, object]:
        """Compare the boundary state with any deeper member of its tail."""

        left = self.threshold
        right = int(right_state)
        if right <= left:
            raise ValueError("right_state must lie strictly inside the deeper tail")
        return {
            "left_state": left,
            "right_state": right,
            "shared_tail_label": self.label(left),
            "left_successor": max(0, left - 1),
            "right_successor": max(0, right - 1),
            "left_successor_label": self.label(max(0, left - 1)),
            "right_successor_label": self.label(max(0, right - 1)),
        }

    def split_tail_once(self) -> "CountdownPartition":
        return CountdownPartition(self.threshold + 1)


@dataclass(frozen=True)
class CountdownWorld:
    """An absorbing countdown on N_0 with a binary zero detector."""

    @staticmethod
    def successor(state: int) -> int:
        if state < 0:
            raise ValueError("countdown state must be non-negative")
        return max(0, state - 1)

    @staticmethod
    def observation(state: int) -> int:
        if state < 0:
            raise ValueError("countdown state must be non-negative")
        return int(state == 0)

    def horizon_word(self, state: int, horizon: int) -> tuple[int, ...]:
        if state < 0 or horizon < 1:
            raise ValueError("state must be non-negative and horizon positive")
        return tuple(int(step >= state) for step in range(horizon))

    @staticmethod
    def horizon_partition(horizon: int) -> CountdownPartition:
        return CountdownPartition(horizon)

    @staticmethod
    def word_from_partition_label(label: int, horizon: int) -> tuple[int, ...]:
        if horizon < 1 or not 0 <= label <= horizon:
            raise ValueError("invalid horizon label")
        if label == horizon:
            return (0,) * horizon
        return (0,) * label + (1,) * (horizon - label)

    def verify_predictive_factorisation(
        self,
        horizon: int,
        states_checked: int,
    ) -> dict[str, object]:
        if states_checked < horizon + 1:
            raise ValueError("verification prefix must include the tail boundary")
        partition = self.horizon_partition(horizon)
        mismatches = []
        for state in range(states_checked):
            actual = self.horizon_word(state, horizon)
            factored = self.word_from_partition_label(partition.label(state), horizon)
            if actual != factored:
                mismatches.append({"state": state, "actual": actual, "factored": factored})
        return {
            "horizon": horizon,
            "states_checked": states_checked,
            "checks_passed": states_checked - len(mismatches),
            "mismatches": mismatches,
            "passed": not mismatches,
        }

    def repair_trace(self, horizon: int, steps: int) -> tuple[CountdownPartition, ...]:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        current = self.horizon_partition(horizon)
        trace = [current]
        for _ in range(steps):
            current = current.split_tail_once()
            trace.append(current)
        return tuple(trace)

    def induction_certificate(self, horizon: int, depth: int) -> dict[str, object]:
        """Emit the first ``depth`` instances of the nontermination induction.

        The accompanying report proves the universal induction.  This finite
        certificate checks concrete instances and must not be confused with
        the proof itself.
        """

        trace = self.repair_trace(horizon, depth)
        steps = []
        for index, partition in enumerate(trace[:-1]):
            witness = partition.nonclosure_witness
            passed = (
                witness["shared_tail_label"] == partition.threshold
                and witness["left_successor_label"] == partition.threshold - 1
                and witness["right_successor_label"] == partition.threshold
            )
            steps.append(
                {
                    "step": index,
                    "threshold": partition.threshold,
                    "witness": witness,
                    "forces_next_singleton": partition.threshold,
                    "next_threshold": partition.threshold + 1,
                    "passed": passed,
                }
            )
        return {
            "horizon": horizon,
            "depth": depth,
            "base_singletons": tuple(range(horizon)),
            "induction_rule": (
                "if 0..m-1 are singleton blocks, compare m with any r>m: their "
                "successors are m-1 and r-1>=m, which lie in distinct prefix and "
                "tail blocks; therefore m must split from the entire remaining tail"
            ),
            "steps": steps,
            "all_checked_steps_passed": all(step["passed"] for step in steps),
            "mathematical_conclusion": (
                "by induction every n is forced to a singleton; the coarsest "
                "strongly lumpable refinement is the countably infinite identity"
            ),
        }

    def adaptive_repair_record(self, horizon: int) -> dict[str, object]:
        """One verifier-driven interface revision using only finite derivatives.

        The current tail states ``H`` and ``H+1`` share the same length-H word
        but have different successor words.  Registering one additional
        derivative coordinate separates them without claiming access to the
        rest of the infinite interior.
        """

        if horizon < 1:
            raise ValueError("horizon must be positive")
        left_state = horizon
        right_state = horizon + 1
        current_left = self.horizon_word(left_state, horizon)
        current_right = self.horizon_word(right_state, horizon)
        left_successor_word = self.horizon_word(
            self.successor(left_state),
            horizon,
        )
        right_successor_word = self.horizon_word(
            self.successor(right_state),
            horizon,
        )
        extended_left = self.horizon_word(left_state, horizon + 1)
        extended_right = self.horizon_word(right_state, horizon + 1)
        passed = (
            current_left == current_right
            and left_successor_word != right_successor_word
            and extended_left != extended_right
        )
        return {
            "interface_version": horizon,
            "registered_derivative_length": horizon,
            "finite_block_count_before": horizon + 1,
            "tail_symbol_before": f"tau_{horizon}",
            "tail_signature_before": current_left,
            "tail_fibre_before": f"{{n>={horizon}}}",
            "tail_fibre_cardinality_before": "countably_infinite",
            "verifier_counterexample": {
                "left_state": left_state,
                "right_state": right_state,
                "shared_current_signature": current_left,
                "left_successor_signature": left_successor_word,
                "right_successor_signature": right_successor_word,
            },
            "repair": "register one additional future-observation coordinate",
            "registered_derivative_length_after": horizon + 1,
            "finite_block_count_after": horizon + 2,
            "symbol_split": {
                "parent": f"tau_{horizon}",
                "boundary_child_signature": extended_left,
                "boundary_child_fibre": (horizon,),
                "new_tail_child": f"tau_{horizon + 1}",
                "new_tail_signature": extended_right,
                "new_tail_fibre": f"{{n>={horizon + 1}}}",
                "new_tail_fibre_cardinality": "countably_infinite",
            },
            "passed": passed,
        }


def validate_geometric_parameter(probability: float) -> float:
    probability = float(probability)
    if not 0.0 < probability < 1.0:
        raise ValueError("geometric probability must lie strictly between zero and one")
    return probability


def binary_entropy(probability: float) -> float:
    probability = validate_geometric_parameter(probability)
    remainder = 1.0 - probability
    return -probability * log(probability) - remainder * log(remainder)


def geometric_entropy(probability: float) -> float:
    probability = validate_geometric_parameter(probability)
    remainder = 1.0 - probability
    return -log(probability) - (remainder / probability) * log(remainder)


def countdown_lumpability_residual(threshold: int, probability: float) -> float:
    if threshold < 1:
        raise ValueError("threshold must be positive")
    probability = validate_geometric_parameter(probability)
    return (1.0 - probability) ** threshold * binary_entropy(probability)


def countdown_partition_entropy(threshold: int, probability: float) -> float:
    if threshold < 1:
        raise ValueError("threshold must be positive")
    probability = validate_geometric_parameter(probability)
    remainder = 1.0 - probability
    masses = [probability * remainder**state for state in range(threshold)]
    masses.append(remainder**threshold)
    return -sum(mass * log(mass) for mass in masses if mass > 0.0)


def countdown_exact_closure_entropy_excess(
    threshold: int,
    probability: float,
) -> float:
    if threshold < 1:
        raise ValueError("threshold must be positive")
    probability = validate_geometric_parameter(probability)
    return (1.0 - probability) ** threshold * geometric_entropy(probability)


def numerical_countdown_lumpability_residual(
    threshold: int,
    probability: float,
    maximum_state: int = 10_000,
) -> float:
    """Independent finite-prefix approximation for the analytic residual."""

    if maximum_state <= threshold:
        raise ValueError("maximum_state must extend beyond the tail boundary")
    probability = validate_geometric_parameter(probability)
    remainder = 1.0 - probability
    weights = [probability * remainder**state for state in range(maximum_state + 1)]
    normalizer = sum(weights)
    weights = [weight / normalizer for weight in weights]
    tail_mass = sum(weights[threshold:])
    boundary_probability = weights[threshold] / tail_mass
    inner_probability = sum(weights[threshold + 1 :]) / tail_mass
    total = 0.0
    if weights[threshold] > 0.0:
        total += weights[threshold] * -log(boundary_probability)
    if inner_probability > 0.0:
        total += sum(weights[threshold + 1 :]) * -log(inner_probability)
    return total
