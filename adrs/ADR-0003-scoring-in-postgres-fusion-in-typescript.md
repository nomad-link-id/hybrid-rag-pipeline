# ADR 0003: Retrieval scoring in Postgres, fusion in TypeScript

**Status:** accepted

## Context

The pipeline has two kinds of computation: (1) retrieval scoring — cosine
similarity over vectors, ts_rank over tsvectors — and (2) fusion + boosting —
combining ranks, applying authority and recency multipliers. These could live
together (all in SQL, or all in application code) or be split.

## Decision

Keep retrieval scoring in Postgres (the `match_documents` and `bm25_search`
RPCs) and fusion + boosting in TypeScript (`reciprocalRankFusion`,
`applyAuthorityBoosting`, `applyRecencyWeighting`).

## Consequences

- **Each layer does what it is good at.** Postgres does HNSW ANN search and
  full-text ranking natively and fast; it never leaves the database. The TS
  layer only ever handles already-ranked rows — never raw vectors or tsvectors.
- **Fusion logic is testable in isolation.** RRF is pure functions over arrays,
  unit-testable without a database.
- **Boundary invariant:** all ranking-relevant SQL lives in the two RPCs. If
  ranking behavior changes, there are exactly two places to look.
- **Trade-off:** two retrievers mean two round-trips to Postgres. They run
  concurrently (`Promise.all`), so latency is the max of the two, not the sum —
  but it is still two queries, not one.
