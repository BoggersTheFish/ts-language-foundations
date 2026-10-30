# Zenodo release candidate — NOT YET PUBLISHED

## Proposed record metadata (pending independent proof/build review)

**Title:** Finite Interfaces and Versioned Specification Gates: A Lean 4 Formal-Verification Companion to TS

**Creator:** Michalek, Ben — Independent Research

**Resource type:** Software

**Version:** 0.1.0 (proposed; must match first frozen release tag)

**Keywords:** Lean 4; formal verification; theorem proving; finite state abstraction; infinite fibres; specification versioning; verifier-first systems; proof-carrying state; reproducibility

**Publication date:** set to the actual publication date, not the draft date.

**Description:**

> This research-software companion supplies machine-checkable Lean 4 statements
> for a deliberately restricted subset of the Thinking System programme.
> A deterministic countably-infinite world with a four-state observer-facing
> quotient is modelled directly, including the commuting macro-transition
> identity, arbitrarily large hidden-nuisance witnesses in each quotient
> fibre, and a counterexample showing that an immediate-task label is not
> dynamically closed. A separate abstract versioned admission gate prohibits
> stale-specification proposals from modifying trusted state, leaves rejected
> proposals unchanged, and preserves a supplied compliance predicate.
>
> Formal verification is restricted to these definitions and proof obligations.
> The package does not verify the underlying statistical evidence pipeline,
> adaptive representation-learning algorithm, Python runtime refinement,
> neural policy, PRIME architecture in general, or the appropriateness of
> external specifications. Formal proofs cannot independently validate the
> modelling assumptions. The release includes pinned Lean tooling, source
> declarations and a fail-closed check path.

## Related work

- TS Adaptive State Abstraction v0.2.0:
  https://doi.org/10.5281/zenodo.21954123
- PRIME v1.0.0:
  https://doi.org/10.5281/zenodo.22058441
- Source fixture: https://github.com/BoggersTheFish/ts-language-foundations/tree/v0.7.0
- Formal companion source: [replace with immutable Git commit URL after merging and tagging]

Use **a new Zenodo record** for a distinct companion artifact, cross-linked to
prior work. Do not create a new version of TSASA or PRIME unless updating those
same deposited artifacts; their historical claims must remain unchanged.

## Publication blockers

- [ ] Lean CI: `lake build` and `lake check` succeed with exact pinned toolchain.
- [ ] Verify no admitted `sorry`, home-rolled axioms, `native_decide` bypass,
      or unreviewed theorem statement substitutions.
- [ ] Independently review mathematical statements against the published fixture.
- [ ] Record a frozen source commit or release tag and every file's SHA-256.
- [ ] Run existing Python tests; specify exact commands, environment and results.
- [ ] Test malicious/invalid proofs and stale-spec submissions against a real
      execution bridge; otherwise label the bridge **not implemented**.
- [ ] Specify the explicit software license. Current ts-language-foundations
      repository root does not advertise a LICENSE file.
- [ ] Build a deterministic archive with README, all Lean sources, Lake files,
      this metadata, verification transcript, checksums and limitations.
- [ ] Confirm the author and Zenodo account permissions before irreversible publish.
- [ ] Never invent a DOI; reserve it in an authenticated Zenodo draft or let
      Zenodo allocate it on publication.

## Critical distinction

A kernel-accepted theorem about an abstract transition is not a certification
of Python behavior. The concrete bridge remains a separate engineering claim.
`lake comparator` can help defend fixed challenge statements against
untrusted generated proof submissions (Lean 4.35.0), but using that feature
for actual generated certificates is not yet implemented here.
