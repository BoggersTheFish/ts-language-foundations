from math import log

from ts_language_foundations.repair import (
    dynamic_closure,
    is_strongly_lumpable,
    joint_repair,
    lumpability_residual,
    predictive_partition,
    predictive_repair,
    predictive_residual,
)
from ts_language_foundations.world import RotatingBitWorld


def test_task_state_is_exact_now_but_not_autonomous():
    world = RotatingBitWorld()
    partition = world.task_partition
    assert predictive_residual(world.task_laws, partition) == 0.0
    assert abs(lumpability_residual(world.transition_matrix, partition) - log(2)) < 1e-12
    assert not is_strongly_lumpable(world.transition_matrix, partition)


def test_dynamic_closure_restores_the_omitted_bit():
    world = RotatingBitWorld()
    closed, trace = dynamic_closure(
        world.transition_matrix,
        world.task_partition,
        return_trace=True,
    )
    assert closed == world.identity_partition
    assert trace == (world.task_partition, world.identity_partition)
    assert is_strongly_lumpable(world.transition_matrix, closed)
    assert dynamic_closure(world.transition_matrix, closed) == closed


def test_joint_repair_is_schedule_independent():
    world = RotatingBitWorld()
    left = joint_repair(
        world.transition_matrix,
        world.task_laws,
        ((0, 1, 2, 3),),
        schedule="predictive_then_dynamic",
    )
    right = joint_repair(
        world.transition_matrix,
        world.task_laws,
        ((0, 1, 2, 3),),
        schedule="dynamic_then_predictive",
    )
    assert left == right == world.identity_partition


def test_negative_controls_separate_prediction_from_closure():
    world = RotatingBitWorld()
    one_state = ((0, 1, 2, 3),)
    parity = ((0, 3), (1, 2))
    assert predictive_residual(world.task_laws, one_state) > 0
    assert lumpability_residual(world.transition_matrix, one_state) == 0
    assert predictive_residual(world.task_laws, parity) > 0
    assert lumpability_residual(world.transition_matrix, parity) == 0
    assert predictive_repair(parity, predictive_partition(world.task_laws)) == world.identity_partition

