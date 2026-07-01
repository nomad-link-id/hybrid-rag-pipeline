# hybrid-rag-pipeline

Retrieval-Augmented Generation combining **BM25 + semantic search + Reciprocal Rank Fusion**.

Most RAG implementations rely on semantic search alone. This pipeline runs sparse (BM25) and dense (vector) retrieval together and fuses them with RRF, with optional authority and recency weighting — reaching higher recall than either method by itself.

`CI` · `TypeScript` · `MIT`

> Companion implementation to the paper *BM25 and Dense Retrieval Are Complementary for Portuguese Clinical Text* (Zenodo, CC BY). The paper reports the study; this repository shows the working technique on public/example data.

---

## Why hybrid

Semantic-only retrieval misses exact terms (identifiers, named scores, drug names); BM25-only misses conceptual matches and synonyms. Running both and fusing the rankings covers both failure modes:

| Query type | Semantic only | BM25 only | Hybrid |
|---|---|---|---|
| Conceptual ("heart failure management") | good | misses synonyms | best |
| Exact terms ("CHA2DS2-VASc score") | lost in embedding space | good | best |
| Mixed ("warfarin dosing for AF") | partial | partial | best |

A calibrated similarity floor matters more than most prompt changes: a low threshold returns noisy chunks that induce confabulation, while a higher floor keeps only genuinely relevant results. The right value is corpus-dependent — tune it against your own data.

---

## Architecture

```
User Query
  |
  |--> [Embedding Model] --> dense vector
  |         '--> pgvector HNSW cosine similarity --> Top K
  |
  |--> [BM25 Tokenizer] --> sparse representation
  |         '--> PostgreSQL full-text search (tsvector) --> Top K
  |
  '--> [Reciprocal Rank Fusion]
            |-- combine both result sets
            |-- optional authority weighting (source-level)
            |-- optional recency weighting (age-based)
            '-- return final ranked results
```

---

## How RRF works

Reciprocal Rank Fusion merges multiple ranked lists into one:

```
RRF_score(d) = sum over lists of  1 / (k + rank_i(d))
```

where `k` is a constant (default 60) and `rank_i(d)` is the rank of document `d` in the i-th list. It needs no score normalization between retrieval methods, which is what makes combining BM25 and dense scores clean.

---

## Features

- **Hybrid search** — BM25 (exact keyword) + semantic (meaning-based) in parallel
- **RRF fusion** — combines rankings without cross-method score normalization
- **Authority weighting** — optional, configurable source-level multipliers
- **Recency weighting** — optional, configurable age-based boost/decay
- **Calibrated threshold** — configurable similarity floor to drop low-relevance noise
- **pgvector HNSW** — approximate nearest-neighbor search on PostgreSQL native vectors
- **Streaming-ready** — results as async iterables for SSE
- **TypeScript strict mode** — full type safety across the pipeline

---

## Quick start

**Prerequisites:** Node.js 18+ · PostgreSQL 15+ with pgvector · an embedding API (OpenAI, Cohere, or local)

```bash
npm install hybrid-rag-pipeline
```

### Database

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  content TEXT NOT NULL,
  embedding vector(1536),
  source TEXT,
  source_authority FLOAT DEFAULT 1.0,
  published_at TIMESTAMPTZ,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW index for similarity search
CREATE INDEX ON documents
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- Full-text search
ALTER TABLE documents ADD COLUMN tsv tsvector
  GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;
CREATE INDEX ON documents USING gin(tsv);
```

### Usage

```ts
import { HybridRAG } from 'hybrid-rag-pipeline';

const rag = new HybridRAG({
  connectionString: process.env.DATABASE_URL!,
  embeddingProvider: 'openai',
  embeddingModel: 'text-embedding-3-small',

  similarityThreshold: 0.6,   // tune against your corpus
  topK: 10,
  rrf_k: 60,

  // optional — off by default
  authorityWeights: { /* source -> multiplier */ },
  recency: { /* age-based boost/decay */ },
});

const results = await rag.search('management of heart failure with reduced ejection fraction');
```

Each result carries its final fused score alongside the raw semantic and BM25 scores, so ranking decisions are inspectable.

### Ingestion

```ts
await rag.ingest({
  content: documentText,
  source: 'example_source',
  publishedAt: new Date('2024-06-15'),
  chunkSize: 512,
  chunkOverlap: 64,
  preserveHeaders: true,
});
```

---

## Configuration reference

```ts
interface HybridRAGConfig {
  connectionString: string;
  tableName?: string;              // default: 'documents'

  embeddingProvider: 'openai' | 'cohere' | 'custom';
  embeddingModel?: string;         // default: 'text-embedding-3-small'
  embeddingDimensions?: number;    // default: 1536

  similarityThreshold?: number;    // default: 0.6 — tune to your corpus
  topK?: number;                   // default: 10
  rrf_k?: number;                  // default: 60

  authorityWeights?: Record<string, number>;  // optional
  recency?: RecencyConfig;                     // optional
  searchTimeout?: number;          // ms, default: 5000
}
```

---

## License

MIT

## Author

Igor Eduardo — igoreduardo.com · ORCID 0009-0005-6288-1135
