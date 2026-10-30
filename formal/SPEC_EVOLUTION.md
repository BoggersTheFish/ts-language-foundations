# Specification evolution after model retraining (research design)

**Status:** Proposed protocol, not an implemented end-to-end guarantee.

## Response to Sherwin's scalability question

A model update and a governing specification update are different events.

- **Model change:** retrain/fine-tune or replace the proposer. The underlying acceptance specification is not automatically rewritten. New candidate outputs must still pass the existing verifier.
- **Specification change:** a separately authorised governance operation. The current model or an automated proof searcher may *propose* a revision; neither is allowed to approve its own proposal.
- **Evidence change:** new observations may invalidate the empirical justification for an old specification. This triggers review or fail-closed abstention rather than silent rule replacement.

A scalable architecture automates **proposal generation, counterexample search,
proof search, regression, and compatibility assessment** while retaining an
explicitly authorised policy for changing the specification itself.

## Transaction protocol

1. **Freeze current authority.** Produce immutable `SpecID` for a canonical
   typed specification, its assumptions, trusted verifier implementation,
   imported declarations, build toolchain, approved axioms and dependencies.
   A `ModelID` is tracked separately.
2. **Detect reason for change.** Record a falsifiable counterexample,
   drift alarm, new capability or explicitly authorised requirement change.
   Retain source evidence and its confidence/provenance.
3. **Propose `SpecCandidate`.** The model or synthesis engine supplies candidate
   definitions, transition rules, migrations, safety claims, and a set of
   potential Lean proof terms or proof scripts.
4. **Check the exact statement.** An independent, pinned Lean kernel (and, for
   untrusted proof source, `lake check` / comparator and independent checkers)
   checks proof objects against **reviewed theorem statements**. Formal validity
   does not prove that the statement adequately represents real-world safety.
5. **Prove the safety envelope.** Before a revision may become authoritative,
   verify that `NewSpec(s)` implies the separately fixed `SafetyInvariant(s)`
   within the declared model; establish migration/precondition obligations.
   If the new rules cannot establish these, reject or require governance review.
6. **Run empirical and hostile controls.** Existing Python behavioural tests,
   counterexamples, null/drift tests, fuzzed malformed terms and tampered
   receipts; reject incomplete evidence. Record failures, not only successes.
7. **Commit atomically.** Authorise the new specification ID through a distinct
   governance capability. Old version-bound admissions are not automatically
   valid under the new version. Recheck or quarantine each affected state,
   action, proof, or receipt before promotion.
8. **Seal an auditable trace.** Persist old/new hashes, migration evidence,
   theorem statement digest, toolchain and approved-axiom ledger, model ID,
   replay manifest, pass/reject decision and rollback/quarantine capability.

## Formal/engineering separation

The current Lean prototype proves a version-matched abstract gate. It does NOT
yet prove that the Python engine's actual mutation path refines that gate, that
policy versions are hashed canonically, or that the statistical confidence
process satisfies its theoretical assumptions.

Next bridge: a versioned serialisation format with a deterministic reference
parser and independent checker. Validate that a concrete transaction emitted by
Python corresponds to the exact Lean-modeled transition before allowing
mutation. Do not use proof output text or a model's self-reported certainty as
proof authority.

## Upgrade semantics

| Event | Existing spec | Required evidence |
|---|---|---|
| Fine-tune proposer | unchanged | rerun tests, pin new ModelID |
| New candidate action | unchanged | version match + current proof/certificate |
| New observed counterexample | flagged | recompute evidence; abstain/quarantine pending review |
| Stronger specification | new SpecID | new compliance proofs; migrate/reject affected states |
| Weaker or incomparable specification | new SpecID | safety-envelope proof **and** governance approval |
| Changed assumptions / Lean imports | new proof environment | recheck affected theorem objects and receipts |

Do not conflate Lean type-checking with a proof that an empirical system is safe
outside the encoded assumptions. Stochastic confidence guarantees are separate
mathematical/probabilistic obligations.

## Initial adversarial tests to automate

- Replace the theorem statement with an easier claim, keeping a valid proof.
- Submit `sorry` / custom axioms / unsafe proof-check bypasses.
- Present a proof from version N under a materially altered version N+1.
- Alter JSON serialisation, spec hashes, proof hashes or input receipts.
- Construct a candidate that passes model scoring but fails explicit policy.
- Change the observation process while preserving superficially similar data.
- Ensure rejection has **no persistent mutation**, including after replay.

## Publication lineage

- TS Adaptive State Abstraction v0.2.0:
  https://doi.org/10.5281/zenodo.21954123
- PRIME v1.0.0:
  https://doi.org/10.5281/zenodo.22058441

A formal-methods companion can link to both as *related works* but must not
retroactively imply that either archived release was checked by Lean.
