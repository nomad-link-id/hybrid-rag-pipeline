import { describe, it, expect } from 'vitest';
import {
  reciprocalRankFusion,
  applyAuthorityBoosting,
  applyRecencyWeighting,
  enrichQuery,
  chunkText,
} from '../index.js';
import type { RawSemanticResult, RawBM25Result, SearchResult, RecencyConfig } from '../index.js';

// ─── Test Helpers ─────────────────────────────────────────────────────────────

function makeSemanticResult(overrides: Partial<RawSemanticResult> = {}): RawSemanticResult {
  return {
    id: 'sem-1',
    content: 'Heart failure management guidelines',
    similarity: 0.85,
    source: 'official_guidelines',
    source_authority: 1.5,
    published_at: '2024-01-15T00:00:00Z',
    metadata: {},
    ...overrides,
  };
}

function makeBM25Result(overrides: Partial<RawBM25Result> = {}): RawBM25Result {
  return {
    id: 'bm25-1',
    content: 'Heart failure treatment protocols',
    rank: 5.2,
    source: 'peer_reviewed',
    source_authority: 1.1,
    published_at: '2023-06-01T00:00:00Z',
    metadata: {},
    ...overrides,
  };
}

function makeSearchResult(overrides: Partial<SearchResult> = {}): SearchResult {
  return {
    id: 'res-1',
    content: 'Test content',
    score: 0.5,
    semanticScore: 0.8,
    bm25Score: 4.0,
    source: 'official_guidelines',
    authorityBoost: 1.0,
    recencyBoost: 1.0,
    metadata: {},
    ...overrides,
  };
}

// ─── RRF Fusion ───────────────────────────────────────────────────────────────

describe('reciprocalRankFusion', () => {
  it('combines semantic and BM25 results with RRF scoring', () => {
    const semantic = [makeSemanticResult({ id: 'doc-1' })];
    const bm25 = [makeBM25Result({ id: 'doc-2' })];

    const results = reciprocalRankFusion(semantic, bm25, 60);

    expect(results).toHaveLength(2);
    const doc1 = results.find(r => r.id === 'doc-1')!;
    const doc2 = results.find(r => r.id === 'doc-2')!;

    // RRF score for rank 0 with k=60: 1/(60+1) = 0.01639...
    expect(doc1.score).toBeCloseTo(1 / 61, 5);
    expect(doc2.score).toBeCloseTo(1 / 61, 5);
    expect(doc1.semanticScore).toBe(0.85);
    expect(doc2.bm25Score).toBe(5.2);
  });

  it('boosts documents found in both result sets', () => {
    const sharedId = 'shared-doc';
    const semantic = [makeSemanticResult({ id: sharedId })];
    const bm25 = [makeBM25Result({ id: sharedId })];

    const results = reciprocalRankFusion(semantic, bm25, 60);

    expect(results).toHaveLength(1);
    // Score should be sum of both RRF contributions: 1/61 + 1/61
    expect(results[0].score).toBeCloseTo(2 / 61, 5);
  });

  it('preserves ranking order across multiple results', () => {
    const semantic = [
      makeSemanticResult({ id: 'a', similarity: 0.95 }),
      makeSemanticResult({ id: 'b', similarity: 0.80 }),
      makeSemanticResult({ id: 'c', similarity: 0.70 }),
    ];

    const results = reciprocalRankFusion(semantic, [], 60);

    // First result gets highest RRF score
    expect(results[0].score).toBeGreaterThan(results[1].score);
    expect(results[1].score).toBeGreaterThan(results[2].score);
  });
});

// ─── Threshold Filtering ──────────────────────────────────────────────────────

describe('threshold filtering', () => {
  it('filters results below similarity threshold', () => {
    const semantic = [
      makeSemanticResult({ id: 'high', similarity: 0.85 }),
      makeSemanticResult({ id: 'low', similarity: 0.30 }),
    ];

    const results = reciprocalRankFusion(semantic, [], 60);
    const threshold = 0.60;
    const filtered = results.filter(r => r.semanticScore >= threshold);

    expect(filtered).toHaveLength(1);
    expect(filtered[0].id).toBe('high');
  });
});

// ─── Authority Boosting ───────────────────────────────────────────────────────

describe('applyAuthorityBoosting', () => {
  it('multiplies score by authority weight for known sources', () => {
    const results = [
      makeSearchResult({ source: 'official_guidelines', score: 1.0 }),
      makeSearchResult({ id: 'res-2', source: 'blog_post', score: 1.0 }),
    ];
    const weights = { official_guidelines: 1.5 };

    const boosted = applyAuthorityBoosting(results, weights);

    expect(boosted[0].score).toBe(1.5);
    expect(boosted[0].authorityBoost).toBe(1.5);
    // Unknown source gets default weight of 1.0
    expect(boosted[1].score).toBe(1.0);
    expect(boosted[1].authorityBoost).toBe(1.0);
  });
});

// ─── Recency Weighting ───────────────────────────────────────────────────────

describe('applyRecencyWeighting', () => {
  const config: RecencyConfig = {
    recentBoost: 1.2,
    recentWindowYears: 3,
    decayPenalty: 0.8,
    decayAfterYears: 10,
    landmarkExempt: true,
  };

  const now = new Date('2025-01-01T00:00:00Z');

  it('boosts recent documents', () => {
    const results = [
      makeSearchResult({
        score: 1.0,
        metadata: { published_at: '2024-06-01T00:00:00Z' }, // ~0.5 years ago
      }),
    ];

    const weighted = applyRecencyWeighting(results, config, now);
    expect(weighted[0].score).toBe(1.2);
    expect(weighted[0].recencyBoost).toBe(1.2);
  });

  it('penalizes old documents', () => {
    const results = [
      makeSearchResult({
        score: 1.0,
        metadata: { published_at: '2010-01-01T00:00:00Z' }, // 15 years ago
      }),
    ];

    const weighted = applyRecencyWeighting(results, config, now);
    expect(weighted[0].score).toBe(0.8);
    expect(weighted[0].recencyBoost).toBe(0.8);
  });

  it('exempts landmark documents from decay', () => {
    const results = [
      makeSearchResult({
        score: 1.0,
        metadata: {
          published_at: '2005-01-01T00:00:00Z', // 20 years old
          is_landmark: true,
        },
      }),
    ];

    const weighted = applyRecencyWeighting(results, config, now);
    expect(weighted[0].score).toBe(1.0); // No decay applied
  });
});

// ─── Empty Query Guard ────────────────────────────────────────────────────────

describe('empty query handling', () => {
  it('returns empty results for empty string in enrichQuery', () => {
    expect(enrichQuery('')).toBe('');
    expect(enrichQuery('  ')).toBe('  ');
  });
});

// ─── Query Enrichment ─────────────────────────────────────────────────────────

describe('enrichQuery', () => {
  it('returns original query when no context provided', () => {
    expect(enrichQuery('treatment options')).toBe('treatment options');
    expect(enrichQuery('treatment options', [])).toBe('treatment options');
  });

  it('prepends recent context to the query', () => {
    const result = enrichQuery('What is the treatment?', [
      'What is heart failure?',
      'How is it diagnosed?',
    ]);
    expect(result).toBe('What is heart failure? How is it diagnosed? What is the treatment?');
  });

  it('respects context window limit', () => {
    const result = enrichQuery('dosing?', ['msg1', 'msg2', 'msg3'], 1);
    expect(result).toBe('msg3 dosing?');
  });
});

// ─── Text Chunking ────────────────────────────────────────────────────────────

describe('chunkText', () => {
  it('splits text into chunks with overlap', () => {
    // Create text with 100 words
    const words = Array.from({ length: 100 }, (_, i) => `word${i}`);
    const text = words.join(' ');

    const chunks = chunkText(text, 50, 10, false);

    expect(chunks.length).toBeGreaterThanOrEqual(2);
    // Each chunk should have at most 50 words
    for (const chunk of chunks) {
      expect(chunk.split(/\s+/).length).toBeLessThanOrEqual(50);
    }
  });

  it('skips tiny trailing chunks', () => {
    // 60 words with chunk size 50, overlap 10 -> second chunk would be 20 words (>=20) so it keeps it
    // 55 words with chunk size 50, overlap 10 -> second chunk starts at 40, has 15 words (<20), skipped
    const words = Array.from({ length: 55 }, (_, i) => `word${i}`);
    const text = words.join(' ');

    const chunks = chunkText(text, 50, 10, false);
    expect(chunks).toHaveLength(1);
  });
});
