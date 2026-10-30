/-!
# Abstract fail-closed gate for versioned governing specifications

The *trusted* validity function below is deliberately supplied as a parameter.
Proving this gate's behaviour does not certify that an arbitrary Python verifier
implements the validity function, or that its specification is scientifically apt.
-/

namespace TSFormal

structure Governed (α : Type) where
  value : α
  specVersion : Nat

structure Proposal (α : Type) where
  value : α
  attestedVersion : Nat

/-- Only a version-matching, currently compliant candidate may replace state. -/
def gate {α : Type} (valid : Nat → α → Bool)
    (current : Governed α) (candidate : Proposal α) : Governed α :=
  if candidate.attestedVersion = current.specVersion ∧
      valid current.specVersion candidate.value = true
  then { current with value := candidate.value }
  else current

/-- Old-version evidence does not authorise promotion under a newer version. -/
theorem stale_version_rejected {α : Type} (valid : Nat → α → Bool)
    (current : Governed α) (candidate : Proposal α)
    (h : candidate.attestedVersion ≠ current.specVersion) :
    gate valid current candidate = current := by
  have hReject :
      ¬ (candidate.attestedVersion = current.specVersion ∧
        valid current.specVersion candidate.value = true) := by
    intro hAccept
    exact h hAccept.1
  simp [gate, hReject]

/-- If trusted state is compliant, gating preserves compliance. -/
theorem gate_preserves_compliance {α : Type} (valid : Nat → α → Bool)
    (current : Governed α) (candidate : Proposal α)
    (hCurrent : valid current.specVersion current.value = true) :
    valid (gate valid current candidate).specVersion
      (gate valid current candidate).value = true := by
  by_cases hAccept :
      candidate.attestedVersion = current.specVersion ∧
      valid current.specVersion candidate.value = true
  · simpa [gate, hAccept] using hAccept.2
  · simpa [gate, hAccept] using hCurrent

/-- A rejected candidate leaves the trusted state exactly unchanged. -/
theorem rejected_candidate_unchanged {α : Type} (valid : Nat → α → Bool)
    (current : Governed α) (candidate : Proposal α)
    (hReject : ¬ (candidate.attestedVersion = current.specVersion ∧
      valid current.specVersion candidate.value = true)) :
    gate valid current candidate = current := by
  simp [gate, hReject]

end TSFormal
