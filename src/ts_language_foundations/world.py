"""The deliberately tiny verifier-grounded world used by v0.1."""

from __future__ import annotations

from dataclasses import dataclass

from .repair import Laws, Matrix, Partition, as_fraction_laws, as_fraction_matrix

State = tuple[int, int]


@dataclass(frozen=True)
class RotatingBitWorld:
    """A four-state deterministic world with a task-visible and hidden bit.

    The state is ``(a, b)`` and the autonomous update is

    ``(a, b) -> (b, 1-a)``.

    The registered immediate task asks only for ``a``.  Consequently the
    task partition is sufficient now but not sufficient for autonomous
    continuation: the next value of ``a`` depends on ``b``.
    """

    states: tuple[State, ...] = ((0, 0), (0, 1), (1, 0), (1, 1))

    @property
    def state_to_index(self) -> dict[State, int]:
        return {state: index for index, state in enumerate(self.states)}

    def successor(self, state: State) -> State:
        a_value, b_value = state
        return b_value, 1 - a_value

    @property
    def successor_indices(self) -> tuple[int, ...]:
        lookup = self.state_to_index
        return tuple(lookup[self.successor(state)] for state in self.states)

    @property
    def transition_matrix(self) -> Matrix:
        rows = []
        for successor in self.successor_indices:
            rows.append([int(target == successor) for target in range(len(self.states))])
        return as_fraction_matrix(rows)

    @property
    def task_laws(self) -> Laws:
        return as_fraction_laws([(1 - state[0], state[0]) for state in self.states])

    @property
    def task_partition(self) -> Partition:
        return ((0, 1), (2, 3))

    @property
    def identity_partition(self) -> Partition:
        return tuple((index,) for index in range(len(self.states)))

    def task_trace(self, state: State, horizon: int) -> tuple[int, ...]:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        values = []
        current = state
        for _ in range(horizon):
            values.append(current[0])
            current = self.successor(current)
        return tuple(values)

