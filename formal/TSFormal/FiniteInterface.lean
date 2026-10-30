/-!
# A precise theorem for the published finite-interface / infinite-fibre fixture

Model from docs/INFINITE_INTERFACE_REPORT.md:
  T(a,b,n) = (b, not a, n+1)
  q(a,b,n) = (a,b)
  M(a,b)   = (b, not a)

This formalizes one mathematical fixture, not the empirical learning pipeline.
-/

namespace TSFormal

structure InfiniteWorld where
  a : Bool
  b : Bool
  nuisance : Nat
deriving Repr, DecidableEq

def worldStep (w : InfiniteWorld) : InfiniteWorld :=
  { a := w.b, b := !w.a, nuisance := w.nuisance + 1 }

def quotient (w : InfiniteWorld) : Bool × Bool := (w.a, w.b)

def macroStep (z : Bool × Bool) : Bool × Bool := (z.2, !z.1)

/-- The four-state macro dynamics commute with all concrete world transitions. -/
theorem quotient_commutes (w : InfiniteWorld) :
    quotient (worldStep w) = macroStep (quotient w) := by
  rfl

/-- Every macro label has witnesses with arbitrarily high nuisance counters. -/
theorem every_fibre_is_unbounded (a b : Bool) (lowerBound : Nat) :
    ∃ w : InfiniteWorld, quotient w = (a, b) ∧ lowerBound < w.nuisance := by
  refine ⟨⟨a, b, lowerBound + 1⟩, rfl, ?_⟩
  exact Nat.lt_succ_self lowerBound

def immediateTask (w : InfiniteWorld) : Bool := w.a

private def coarseWitness0 : InfiniteWorld := ⟨false, false, 0⟩
private def coarseWitness1 : InfiniteWorld := ⟨false, true, 0⟩

/-- Coarse current-task labels cannot determine the next task label. -/
theorem coarse_task_is_not_dynamically_closed :
    immediateTask coarseWitness0 = immediateTask coarseWitness1 ∧
    immediateTask (worldStep coarseWitness0) ≠
      immediateTask (worldStep coarseWitness1) := by
  decide

end TSFormal
