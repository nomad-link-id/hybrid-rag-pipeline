# ADR 0002: Similarity threshold default of 0.60

**Status:** accepted

## Context

The dense retriever (`match_documents`) takes a `match_threshold` — a cosine
similarity floor below which chunks are discarded before fusion. A common
default in RAG tutorials is ~0.20, which is permissive: it returns many chunks,
most only loosely related to the query.

Loosely-related chunks are not harmless. When they reach a downstream LLM as
"context," they invite confabulation — the model treats retrieved-but-irrelevant
text as grounds to answer.

## Decision

Default `similarityThreshold` to 0.60. Set in the `HybridRAG` constructor.

## Consequences

- **Higher precision, lower raw recall on the dense side.** The floor drops
  weakly-related chunks. BM25 compensates on the exact-term queries where dense
  retrieval is weakest, so hybrid recall stays high (that complementarity is
  the whole point — see the query-type table in the README).
- **Corpus-dependent.** 0.60 is a default, not a law. The right floor depends
  on the embedding model and corpus. The value is exposed as config precisely
  so it can be tuned.
- TODO: Document the evaluation methodology used during threshold calibration
  (query set, metric, the 0.20 vs 0.60 comparison) in experiments/EXP-001.
