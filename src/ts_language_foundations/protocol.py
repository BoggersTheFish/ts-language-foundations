"""Opaque-signal protocol construction and verifier-gated learning."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice, permutations, product
from math import factorial
from typing import Iterable

Coordinate = tuple[int, int]
Message = tuple[str, str]

SIGNAL_POOL = ("ka", "zu", "mi", "te")


def _permutation_at(items: tuple[str, ...], index: int) -> tuple[str, ...]:
    if index < 0 or index >= factorial(len(items)):
        raise ValueError(
            f"renaming index must be in 0..{factorial(len(items)) - 1}"
        )
    return next(islice(permutations(items), index, index + 1))


@dataclass(frozen=True)
class Codebook:
    signal_for_value: tuple[dict[int, str], dict[int, str]]

    def encode(self, coordinate: Coordinate) -> Message:
        return tuple(
            self.signal_for_value[slot][coordinate[slot]] for slot in (0, 1)
        )  # type: ignore[return-value]

    def to_dict(self) -> dict[str, object]:
        return {
            "slot_0": dict(self.signal_for_value[0]),
            "slot_1": dict(self.signal_for_value[1]),
        }


def make_codebook(renaming_index: int) -> Codebook:
    signals = _permutation_at(SIGNAL_POOL, renaming_index)
    return Codebook(
        (
            {0: signals[0], 1: signals[1]},
            {0: signals[2], 1: signals[3]},
        )
    )


@dataclass(frozen=True)
class Decoder:
    coordinate_for_slot: tuple[int, int]
    bit_for_signal: tuple[dict[str, int], dict[str, int]]

    def decode(self, message: Message) -> Coordinate:
        output = [-1, -1]
        for slot in (0, 1):
            signal = message[slot]
            if signal not in self.bit_for_signal[slot]:
                raise KeyError(f"ungrounded signal {signal!r} in slot {slot}")
            coordinate = self.coordinate_for_slot[slot]
            output[coordinate] = self.bit_for_signal[slot][signal]
        return output[0], output[1]

    def to_dict(self) -> dict[str, object]:
        return {
            "coordinate_for_slot": self.coordinate_for_slot,
            "bit_for_signal": self.bit_for_signal,
        }


def learn_decoders(examples: Iterable[tuple[Message, Coordinate]]) -> tuple[Decoder, ...]:
    examples = tuple(examples)
    if not examples:
        raise ValueError("at least one verifier-labelled example is required")
    observed = [sorted({message[slot] for message, _ in examples}) for slot in (0, 1)]
    if any(len(symbols) != 2 for symbols in observed):
        raise ValueError("this v0.1 learner requires exactly two observed signals per slot")

    decoders = []
    for coordinate_for_slot in ((0, 1), (1, 0)):
        for slot_0_bits in permutations((0, 1)):
            for slot_1_bits in permutations((0, 1)):
                decoder = Decoder(
                    coordinate_for_slot=coordinate_for_slot,
                    bit_for_signal=(
                        dict(zip(observed[0], slot_0_bits)),
                        dict(zip(observed[1], slot_1_bits)),
                    ),
                )
                if all(decoder.decode(message) == target for message, target in examples):
                    decoders.append(decoder)
    return tuple(decoders)


def _table_value(table: tuple[int, ...], coordinate: Coordinate) -> int:
    return table[2 * coordinate[0] + coordinate[1]]


def essential_input_count(table: tuple[int, ...]) -> int:
    count = 0
    for variable in (0, 1):
        if any(
            _table_value(table, coordinate)
            != _table_value(
                table,
                tuple(
                    1 - value if index == variable else value
                    for index, value in enumerate(coordinate)
                ),  # type: ignore[arg-type]
            )
            for coordinate in product((0, 1), repeat=2)
        ):
            count += 1
    return count


@dataclass(frozen=True)
class LearnedDynamics:
    truth_tables: tuple[tuple[int, ...], tuple[int, ...]]
    candidate_counts_at_best_score: tuple[int, int]

    def successor(self, coordinate: Coordinate) -> Coordinate:
        return tuple(
            _table_value(table, coordinate) for table in self.truth_tables
        )  # type: ignore[return-value]

    def rollout(self, coordinate: Coordinate, steps: int) -> tuple[Coordinate, ...]:
        trajectory = [coordinate]
        for _ in range(steps):
            trajectory.append(self.successor(trajectory[-1]))
        return tuple(trajectory)

    def to_dict(self) -> dict[str, object]:
        return {
            "truth_tables": self.truth_tables,
            "candidate_counts_at_best_score": self.candidate_counts_at_best_score,
        }


def learn_factorized_dynamics(
    observations: Iterable[tuple[Coordinate, Coordinate]],
) -> LearnedDynamics | None:
    observations = tuple(observations)
    # Fail closed on sparse evidence.  With one or two of four input rows, a
    # minimum-complexity rule can spuriously prefer a constant extrapolation.
    # v0.1 therefore requires three distinct input combinations before it is
    # allowed to identify a Boolean update law.
    if len({source for source, _ in observations}) < 3:
        return None
    selected_tables = []
    multiplicities = []
    for output_coordinate in (0, 1):
        compatible = []
        for bits in product((0, 1), repeat=4):
            table = tuple(bits)
            if all(
                _table_value(table, source) == target[output_coordinate]
                for source, target in observations
            ):
                compatible.append(table)
        if not compatible:
            return None
        best_score = min(essential_input_count(table) for table in compatible)
        best = [table for table in compatible if essential_input_count(table) == best_score]
        if len(best) != 1:
            return None
        selected_tables.append(best[0])
        multiplicities.append(len(best))
    return LearnedDynamics(
        truth_tables=(selected_tables[0], selected_tables[1]),
        candidate_counts_at_best_score=(multiplicities[0], multiplicities[1]),
    )


@dataclass(frozen=True)
class ProtocolTrial:
    renaming_index: int
    held_out_state: int
    decoder_count: int
    decoded_held_out_correctly: bool
    dynamics_identified: bool
    held_out_successor_correct: bool
    four_step_rollout_correct: bool
    holistic_baseline_abstained: bool

    @property
    def passed(self) -> bool:
        return all(
            (
                self.decoder_count == 1,
                self.decoded_held_out_correctly,
                self.dynamics_identified,
                self.held_out_successor_correct,
                self.four_step_rollout_correct,
                self.holistic_baseline_abstained,
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "renaming_index": self.renaming_index,
            "held_out_state": self.held_out_state,
            "decoder_count": self.decoder_count,
            "decoded_held_out_correctly": self.decoded_held_out_correctly,
            "dynamics_identified": self.dynamics_identified,
            "held_out_successor_correct": self.held_out_successor_correct,
            "four_step_rollout_correct": self.four_step_rollout_correct,
            "holistic_baseline_abstained": self.holistic_baseline_abstained,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class UnorderedCodebook:
    signal_for_value: tuple[dict[int, str], ...]

    def encode(self, coordinate: tuple[int, ...]) -> tuple[str, ...]:
        if len(coordinate) != len(self.signal_for_value):
            raise ValueError("coordinate arity does not match the codebook")
        return tuple(
            sorted(
                self.signal_for_value[factor][value]
                for factor, value in enumerate(coordinate)
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            str(factor): dict(mapping)
            for factor, mapping in enumerate(self.signal_for_value)
        }


def make_unordered_codebook(
    renaming_index: int,
    radices: tuple[int, ...],
) -> UnorderedCodebook:
    signal_pool = ["ka", "zu", "mi", "te", "lo", "pa", "fi", "ne"]
    required = sum(radices)
    if required > len(signal_pool):
        raise ValueError("v0.1 opaque signal pool is too small for these radices")
    selected_pool = tuple(signal_pool[:required])
    signals = _permutation_at(selected_pool, renaming_index)
    cursor = 0
    mappings = []
    for radix in radices:
        mappings.append(
            {value: signals[cursor + value] for value in range(radix)}
        )
        cursor += radix
    return UnorderedCodebook(tuple(mappings))


@dataclass(frozen=True)
class UnorderedDecoder:
    meaning_for_signal: dict[str, tuple[int, int]]
    radices: tuple[int, ...]

    def decode(self, message: tuple[str, ...]) -> tuple[int, ...]:
        output: list[int | None] = [None] * len(self.radices)
        for signal in message:
            if signal not in self.meaning_for_signal:
                raise KeyError(f"ungrounded signal {signal!r}")
            factor, value = self.meaning_for_signal[signal]
            if output[factor] is not None:
                raise ValueError("message contains two signals of the same inferred type")
            output[factor] = value
        if any(value is None for value in output):
            raise ValueError("message omits an inferred factor type")
        return tuple(int(value) for value in output)

    def to_dict(self) -> dict[str, object]:
        return {
            "meaning_for_signal": self.meaning_for_signal,
            "radices": self.radices,
        }


def learn_unordered_decoders(
    examples: Iterable[tuple[tuple[str, ...], tuple[int, ...]]],
    radices: tuple[int, ...],
) -> tuple[UnorderedDecoder, ...]:
    """Infer signal types and values without message positions.

    This is an exhaustive finite constraint solver.  It assigns the observed
    opaque signals bijectively to all candidate ``(factor, value)`` meanings
    and retains only assignments consistent with verifier-labelled examples.
    """

    examples = tuple(examples)
    if not examples:
        raise ValueError("at least one verifier-labelled example is required")
    observed = sorted({signal for message, _ in examples for signal in message})
    meanings = [
        (factor, value)
        for factor, radix in enumerate(radices)
        for value in range(radix)
    ]
    if len(observed) != len(meanings):
        raise ValueError(
            "all atomic signals must have real grounding evidence before grammar induction"
        )
    decoders = []
    for assignment in permutations(meanings):
        decoder = UnorderedDecoder(dict(zip(observed, assignment)), radices)
        valid = True
        for message, target in examples:
            try:
                if decoder.decode(message) != target:
                    valid = False
            except (KeyError, ValueError):
                valid = False
        if valid:
            decoders.append(decoder)
    return tuple(decoders)


@dataclass(frozen=True)
class UnorderedProtocolTrial:
    renaming_index: int
    held_out_state: int
    decoder_count: int
    decoded_held_out_correctly: bool
    inferred_types_without_slots: bool
    held_out_successor_correct: bool
    four_step_rollout_correct: bool

    @property
    def passed(self) -> bool:
        return all(
            (
                self.decoder_count == 1,
                self.decoded_held_out_correctly,
                self.inferred_types_without_slots,
                self.held_out_successor_correct,
                self.four_step_rollout_correct,
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "renaming_index": self.renaming_index,
            "held_out_state": self.held_out_state,
            "decoder_count": self.decoder_count,
            "decoded_held_out_correctly": self.decoded_held_out_correctly,
            "inferred_types_without_slots": self.inferred_types_without_slots,
            "held_out_successor_correct": self.held_out_successor_correct,
            "four_step_rollout_correct": self.four_step_rollout_correct,
            "passed": self.passed,
        }
