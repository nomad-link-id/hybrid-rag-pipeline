# Build journal

First-person notes on building this pipeline. Rough by design — the tidy
version is in ARCHITECTURE.md and the ADRs.

## Starting point

Semantic-only retrieval handled conceptual queries well but missed exact terms
— identifiers, named scores, specific drug names would land in the wrong part
of embedding space and simply not surface. Adding BM25 back was the obvious fix;
the real question was how to combine the two rankings.

TODO: Expand with the actual chronology — what was tried first, what broke, in
roughly the order it happened.

## Fusion: why RRF

Score-based fusion meant reconciling cosine similarity with BM25/ts_rank on a
common scale, which is fragile. RRF sidesteps it entirely by using rank position
only. Decided against a cross-encoder reranker because it adds a model to serve
for a coarse-retrieval step — see ADR-0001.

## Threshold: the thing that mattered most

The single highest-impact change was the dense similarity floor. TODO: write up
the moment this became clear — the before/after on confabulation once the floor
went up. See EXP-001 / ADR-0002.

## Known debt

- Constructor still takes `supabaseUrl`/`supabaseKey`; README documents a
  generic `connectionString`. Needs reconciling.
- No fallback when one retriever is down — a thrown RPC aborts the whole search.
- TODO: anything else worth flagging for the next person (including future-me).
