# Finite Repaired-State Compositional Protocol v0.1

## Question

Can a communication protocol be built from verifier-grounded state repair rather than from a language model or corpus?

## Construction

The controlled world has four states `(a,b)` and deterministic dynamics `(a,b) -> (b,1-a)`. The registered immediate task reports `a`. Its coarse partition is therefore predictively exact for the current task, but it discards `b`, which determines the next task value.

No LLM, neural network, tokenizer, corpus, embedding model, or external data source participates in the experiment.

## Representation-repair result

- Predictive residual of the task partition: `0.000000000000` nats.
- Lumpability residual of the task partition: `0.693147180560` nats.
- Dynamic closure: `[[0], [1], [2], [3]]`.
- Joint repair: `[[0], [1], [2], [3]]`.

Thus the current-task label is sufficient for the immediate verifier but not an autonomous Markov state. Closure restores two block distinctions, yielding the four-state identity partition.

## Factor and protocol scaffold

The first diagnostic searches balanced binary complements of the task factor. It selects the unique candidate minimising the total number of essential inputs in the two coordinate-update truth tables. The selected complement is `[[0, 2], [1, 3]]` with score `2`.

A second search does not preselect one versus two factors. It compares a four-valued holistic coordinate with all irredundant pairs of balanced binary coordinates. Its fixed-width task-plus-dynamics table score uniquely selects `[[[0, 1], [2, 3]], [[0, 2], [1, 3]]]` at `6` bits; the other candidates require `[8, 10, 12]` bits.

Each coordinate value receives an arbitrary opaque signal. A message is an **unordered set** containing one signal from each selected factor. A receiver is shown three of four states. Exact verifier receipts expose only the current and next registered task consequences. The receiver exhaustively assigns four signals to candidate factor/value meanings, so both signal type and value must be inferred without message positions. It then learns factorised Boolean dynamics with a declared minimal-dependency rule.

The learner is required to abstain with fewer than three distinct transition inputs; this blocks the simplest compatible constant law from being promoted on one-example evidence.

## Verifier results

- Trials: `96` (24 symbol renamings × four held-out states).
- Unique decoder and all checks passed: `96/96`.
- Held-out compositional decoding: `1.000`.
- Held-out successor prediction: `1.000`.
- Four-step autonomous rollout: `1.000`.
- Unordered signal-type inference and held-out decoding: `1.000`.
- Unordered held-out successor prediction: `1.000`.
- Unordered four-step autonomous rollout: `1.000`.
- Holistic one-signal-per-state held-out coverage: `0.000`.
- Coarse current-task message accuracy: `1.000`.
- Best coarse next-task accuracy: `0.500`.

## Retained claim

In this fixture, language-like symbols can be introduced after representation repair as a serialization of verified distinctions. A bounded description-length search prefers the reusable product coordinates over the holistic state code. The resulting unordered factor code infers symbol types from co-occurrence plus verifier consequences, generalises to unseen combinations, and carries enough state for autonomous continuation; the coarse task label and ungrounded holistic code do not.

## What this does not establish

This is not a theory of human language, a result about consciousness, or evidence that factor discovery will scale. The world, exact feedback, bounded one-or-two-factor candidate family, one-signal-per-factor message form, and simplicity score are explicit scaffolds. The next falsifiable experiment must infer factor arities and message structure in a larger noisy world while preserving exact receipts and abstention on ambiguity.
