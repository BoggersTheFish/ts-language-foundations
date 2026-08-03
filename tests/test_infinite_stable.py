from math import log

from ts_language_foundations.infinite import InfiniteFibreRotatingWorld
from ts_language_foundations.infinite_experiment import run_infinite_experiment


def test_dependency_repair_closes_finite_interface_without_absorbing_infinite_counter():
    world = InfiniteFibreRotatingWorld()
    closure = world.dependency_closure(iter(("a",)))
    assert closure.initial_coordinates == ("a",)
    assert closure.trace == (("a",), ("a", "b"))
    assert closure.closed_coordinates == ("a", "b")
    assert closure.omitted_infinite_coordinates == ("n",)


def test_repaired_macro_update_is_counter_independent():
    world = InfiniteFibreRotatingWorld()
    for n_value in (0, 1, 2, 10, 10**6, 10**30):
        for a_value in (0, 1):
            for b_value in (0, 1):
                state = (a_value, b_value, n_value)
                assert world.repaired_label(world.successor(state)) == (
                    world.repaired_macro_successor(world.repaired_label(state))
                )


def test_coarse_interface_has_explicit_nonclosure_witnesses():
    world = InfiniteFibreRotatingWorld()
    witnesses = world.coarse_nonclosure_witnesses()
    assert len(witnesses) == 2
    for witness in witnesses:
        assert witness["left_next_coarse_label"] != witness["right_next_coarse_label"]
        assert witness["same_current_coarse_label"] == world.coarse_label(
            witness["left_state"]
        )


def test_stable_infinite_fibre_receipt_retains_finite_kernel_residuals(tmp_path):
    receipt = run_infinite_experiment(
        horizons=(1, 2, 4),
        probabilities=(0.25,),
        primary_horizon=4,
        primary_probability=0.25,
        trace_steps=4,
        induction_depth=16,
        states_checked=128,
        project_root=tmp_path,
    )
    stable = receipt["stable_infinite_fibre_case"]
    assert stable["finite_reduction"]["predictive_residual_nats"] == 0.0
    assert abs(stable["finite_reduction"]["lumpability_residual_nats"] - log(2)) < 1e-12
    assert stable["exact_symbolic_certificate"]["repaired_block_count"] == 4
    assert stable["exact_symbolic_certificate"]["repaired_fibre_cardinality"] == "countably_infinite"

