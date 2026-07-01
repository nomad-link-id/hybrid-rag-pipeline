# Architecture

## Bird's Eye View

`hybrid-rag-pipeline` takes a text query and returns a ranked list of document
chunks. It runs two independent retrievers over the same corpus — one sparse,
one dense — and fuses their rankings into a single ordering, then applies two
optional post-fusion weights (source authority, document recency).

The entire flow lives in one public method, `HybridRAG.search()`:

```
query
  → enrichQuery()          (optional: fold in conversation context)
  → Promise.all([
        semanticSearch()   → pgvector: match_documents RPC, cosine, HNSW
        bm25Search()       → postgres FTS: bm25_search RPC, ts_rank
    ])
  → reciprocalRankFusion() (merge the two ranked lists)
  → applyAuthorityBoosting()
  → applyRecencyWeighting()
  → sort by score, take topK
```

The two retrievers run concurrently (`Promise.all`), so total latency is the
slower of the two, not their sum.

## Code Map

`src/index.ts` is the whole library. Sections, in order:

- **Types** — `HybridRAGConfig`, `SearchResult`, `IngestOptions`, and the two
  internal raw-result shapes (`RawSemanticResult`, `RawBM25Result`).
- **`HybridRAG.search()`** — the orchestrator. Read this first; it names every
  stage in sequence.
  *Invariant:* each stage takes and returns `SearchResult[]`, so stages compose
  and reorder freely.
- **`HybridRAG.ingest()`** — chunk → embed → insert. Chunking defaults to 512
  tokens / 64 overlap, headers preserved.
- **`semanticSearch()` / `bm25Search()`** — thin wrappers over two Postgres
  RPCs (`match_documents`, `bm25_search`). Both fetch `topK * 2` so fusion has
  headroom to reorder.
  *Boundary:* all ranking-relevant SQL lives in these two RPCs, not in TS. The
  TS layer never sees raw vectors or tsvectors — only ranked rows.
- **`reciprocalRankFusion()`** — the core. Assigns each result `1/(k + rank)`,
  sums the contributions when a document appears in both lists, keyed by id.
- **`applyAuthorityBoosting()` / `applyRecencyWeighting()`** — multiplicative
  post-fusion adjustments. Both off by default (identity weights).

## Cross-Cutting Concerns

- **Where ranking lives.** Retrieval scoring is in Postgres (the two RPCs);
  fusion and boosting are in TypeScript. This split is deliberate — see
  ADR-0003.
- **Failure mode.** Either RPC throwing aborts the search (no partial results).
  TODO: document the fallback behavior when one retriever is down — currently
  there is none, and that is a known gap.
- **Config discrepancy (known).** The code constructor takes `supabaseUrl` /
  `supabaseKey`; the README documents a generic `connectionString`. The README
  reflects the intended public API; the code has not been refactored yet.
  TODO: reconcile the constructor signature with the documented config.
