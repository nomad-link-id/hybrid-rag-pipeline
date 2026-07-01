# Start here

This repo answers one question: **how do you get high-recall retrieval over a
mixed corpus where queries range from conceptual to exact-term — without a
reranker model in the hot path?**

The answer: run sparse (BM25) and dense (vector) retrieval in parallel and fuse
them with Reciprocal Rank Fusion. No cross-encoder, no score normalization, no
second model to serve.

## Run it

```bash
npm install
npm test          # unit tests
```

Requires PostgreSQL 15+ with the pgvector extension. See `setup.sql` for the
schema (documents table, HNSW index, tsvector GIN index).

## If you have 15 minutes, in order

1. **README.md** — what it is, the query-type table, the headline idea (2 min)
2. **engineering/ARCHITECTURE.md** — how a query flows through the system (4 min)
3. **adrs/** — the three decisions that shaped it: RRF over reranking, the
   0.60 threshold, dense+sparse in parallel (5 min)
4. **experiments/** — the threshold calibration that drove ADR-0002 (2 min)
5. **src/index.ts** — the `search()` method is the whole story in ~30 lines (2 min)

## Where things live

| Looking for | Go to |
|---|---|
| How the pipeline works | `engineering/ARCHITECTURE.md` |
| Why a decision was made | `adrs/` |
| Evidence behind a decision | `experiments/` |
| Performance numbers | `benchmarks/` |
| The build story | `engineering/JOURNAL.md` |
| The code | `src/index.ts` |

## The one thing to take away

The binding constraint was **serving cost**: a cross-encoder reranker would
lift quality but adds a model to serve and latency to every query. RRF gets
most of the benefit from two retrievers you already run, fused with arithmetic.
Every decision here falls out of preferring arithmetic over a second model.
