# ADR 0001: Reciprocal Rank Fusion over cross-encoder reranking

**Status:** accepted

## Context

Combining sparse and dense retrieval needs a way to merge two ranked lists into
one. Two families of approaches:

1. **Score-based fusion** — normalize and combine the raw scores. Requires the
   two scoring scales (cosine similarity vs. BM25/ts_rank) to be made
   comparable, which is fragile and corpus-dependent.
2. **Rank-based fusion (RRF)** — ignore raw scores, use only the position of a
   document in each list.
3. **Learned reranking** — feed candidates to a cross-encoder that scores
   query-document pairs directly.

## Decision

Use Reciprocal Rank Fusion. Each document gets `1 / (k + rank)` from each list
it appears in, summed across lists (`rrf_k` defaults to 60). Implemented in
`reciprocalRankFusion()` in `src/index.ts`.

## Consequences

- **No score normalization.** RRF never compares cosine to ts_rank directly —
  it only compares positions. This removes the most fragile part of hybrid
  retrieval.
- **No second model to serve.** A cross-encoder would likely improve ranking
  quality, but it adds a model to host, GPU/latency budget per query, and an
  operational surface. The binding constraint was serving cost, so RRF wins:
  it is arithmetic over two retrievers already running.
- **Trade-off accepted:** RRF cannot express "this document is *much* more
  relevant" — only "this document ranked higher." For coarse retrieval feeding
  a downstream LLM, position is enough. If fine-grained ranking ever becomes
  the bottleneck, a reranker is the documented next step.
- The constant `k=60` follows the original RRF paper (Cormack et al.). TODO:
  document whether k was tuned on this corpus or left at the paper default.
