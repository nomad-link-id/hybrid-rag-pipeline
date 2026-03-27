/**
 * hybrid-rag-pipeline
 * Production-grade RAG combining BM25 + Semantic Search + Reciprocal Rank Fusion
 *
 * @author Igor Eduardo
 * @license MIT
 */

import { createClient, SupabaseClient } from '@supabase/supabase-js';

// ─── Types ───────────────────────────────────────────────────────────────────

export interface HybridRAGConfig {
  supabaseUrl: string;
  supabaseKey: string;
  tableName?: string;

  embeddingProvider: 'openai' | 'cohere' | 'custom';
  embeddingModel?: string;
  embeddingDimensions?: number;
  openaiApiKey?: string;
  customEmbedFn?: (text: string) => Promise<number[]>;

  similarityThreshold?: number;
  topK?: number;
  rrf_k?: number;

  authorityWeights?: Record<string, number>;
  recency?: RecencyConfig;

  searchTimeout?: number;
}

export interface RecencyConfig {
  recentBoost: number;
  recentWindowYears: number;
  decayPenalty: number;
  decayAfterYears: number;
  landmarkExempt?: boolean;
}

export interface SearchResult {
  id: string;
  content: string;
  score: number;
  semanticScore: number;
  bm25Score: number;
  source: string;
  authorityBoost: number;
  recencyBoost: number;
  metadata: Record<string, unknown>;
}

export interface IngestOptions {
  content: string;
  source?: string;
  sourceAuthority?: number;
  publishedAt?: Date;
  metadata?: Record<string, unknown>;
  chunkSize?: number;
  chunkOverlap?: number;
  preserveHeaders?: boolean;
  isLandmark?: boolean;
}

interface RawSemanticResult {
  id: string;
  content: string;
  similarity: number;
  source: string;
  source_authority: number;
  published_at: string | null;
  metadata: Record<string, unknown>;
}

interface RawBM25Result {
  id: string;
  content: string;
  rank: number;
  source: string;
  source_authority: number;
  published_at: string | null;
  metadata: Record<string, unknown>;
}

// ─── Core ────────────────────────────────────────────────────────────────────

export class HybridRAG {
  private supabase: SupabaseClient;
  private config: Required<
    Pick<HybridRAGConfig, 'tableName' | 'similarityThreshold' | 'topK' | 'rrf_k' | 'searchTimeout'>
  > & HybridRAGConfig;

  constructor(config: HybridRAGConfig) {
    this.supabase = createClient(config.supabaseUrl, config.supabaseKey);
    this.config = {
      tableName: 'documents',
      similarityThreshold: 0.60,
      topK: 10,
      rrf_k: 60,
      searchTimeout: 5000,
      ...config,
    };
  }

  /**
   * Hybrid search: BM25 + Semantic + RRF fusion with authority and recency boosting
   */
  async search(
    query: string,
    options?: { conversationContext?: string[]; contextWindow?: number }
  ): Promise<SearchResult[]> {
    if (!query || query.trim().length === 0) return [];

    // Contextual query enrichment for follow-ups
    const enrichedQuery = this.enrichQuery(query, options?.conversationContext, options?.contextWindow);

    // Run both searches in parallel
    const [semanticResults, bm25Results] = await Promise.all([
      this.semanticSearch(enrichedQuery),
      this.bm25Search(enrichedQuery),
    ]);

    // Fuse with RRF
    const fused = this.reciprocalRankFusion(semanticResults, bm25Results);

    // Apply authority boosting
    const boosted = this.applyAuthorityBoosting(fused);

    // Apply recency weighting
    const weighted = this.applyRecencyWeighting(boosted);

    // Sort by final score, take topK
    return weighted
      .sort((a, b) => b.score - a.score)
      .slice(0, this.config.topK);
  }

  /**
   * Ingest a document with chunking and embedding generation
   */
  async ingest(options: IngestOptions): Promise<{ chunksCreated: number }> {
    const chunks = this.chunkText(
      options.content,
      options.chunkSize ?? 512,
      options.chunkOverlap ?? 64,
      options.preserveHeaders ?? true
    );

    let created = 0;
    for (const chunk of chunks) {
      const embedding = await this.generateEmbedding(chunk);

      const { error } = await this.supabase.from(this.config.tableName).insert({
        content: chunk,
        embedding: embedding,
        source: options.source ?? 'unknown',
        source_authority: options.sourceAuthority ?? 1.0,
        published_at: options.publishedAt?.toISOString() ?? null,
        metadata: {
          ...options.metadata,
          is_landmark: options.isLandmark ?? false,
        },
      });

      if (!error) created++;
    }

    return { chunksCreated: created };
  }

  // ─── Private: Search Methods ─────────────────────────────────────────────

  private async semanticSearch(query: string): Promise<RawSemanticResult[]> {
    const embedding = await this.generateEmbedding(query);

    const { data, error } = await this.supabase.rpc('match_documents', {
      query_embedding: embedding,
      match_threshold: this.config.similarityThreshold,
      match_count: this.config.topK * 2, // Fetch extra for fusion
    });

    if (error) throw new Error(`Semantic search failed: ${error.message}`);
    return data ?? [];
  }

  private async bm25Search(query: string): Promise<RawBM25Result[]> {
    // PostgreSQL full-text search with ts_rank
    const tsQuery = query
      .split(/\s+/)
      .filter(w => w.length > 2)
      .join(' & ');

    const { data, error } = await this.supabase.rpc('bm25_search', {
      search_query: tsQuery,
      result_limit: this.config.topK * 2,
    });

    if (error) throw new Error(`BM25 search failed: ${error.message}`);
    return data ?? [];
  }

  // ─── Private: Fusion ─────────────────────────────────────────────────────

  private reciprocalRankFusion(
    semanticResults: RawSemanticResult[],
    bm25Results: RawBM25Result[]
  ): SearchResult[] {
    const k = this.config.rrf_k;
    const scoreMap = new Map<string, SearchResult>();

    // Score semantic results
    semanticResults.forEach((result, index) => {
      const rrfScore = 1 / (k + index + 1);
      scoreMap.set(result.id, {
        id: result.id,
        content: result.content,
        score: rrfScore,
        semanticScore: result.similarity,
        bm25Score: 0,
        source: result.source,
        authorityBoost: 1.0,
        recencyBoost: 1.0,
        metadata: {
          ...result.metadata,
          published_at: result.published_at,
          source_authority: result.source_authority,
        },
      });
    });

    // Add BM25 scores
    bm25Results.forEach((result, index) => {
      const rrfScore = 1 / (k + index + 1);
      const existing = scoreMap.get(result.id);

      if (existing) {
        // Document found in both — combine scores
        existing.score += rrfScore;
        existing.bm25Score = result.rank;
      } else {
        scoreMap.set(result.id, {
          id: result.id,
          content: result.content,
          score: rrfScore,
          semanticScore: 0,
          bm25Score: result.rank,
          source: result.source,
          authorityBoost: 1.0,
          recencyBoost: 1.0,
          metadata: {
            ...result.metadata,
            published_at: result.published_at,
            source_authority: result.source_authority,
          },
        });
      }
    });

    return Array.from(scoreMap.values());
  }

  // ─── Private: Boosting ───────────────────────────────────────────────────

  private applyAuthorityBoosting(results: SearchResult[]): SearchResult[] {
    if (!this.config.authorityWeights) return results;

    return results.map(result => {
      const weight = this.config.authorityWeights?.[result.source] ?? 1.0;
      return {
        ...result,
        score: result.score * weight,
        authorityBoost: weight,
      };
    });
  }

  private applyRecencyWeighting(results: SearchResult[]): SearchResult[] {
    if (!this.config.recency) return results;

    const { recentBoost, recentWindowYears, decayPenalty, decayAfterYears, landmarkExempt } =
      this.config.recency;
    const now = new Date();

    return results.map(result => {
      const publishedAt = result.metadata.published_at
        ? new Date(result.metadata.published_at as string)
        : null;

      if (!publishedAt) return result;

      const ageYears = (now.getTime() - publishedAt.getTime()) / (365.25 * 24 * 60 * 60 * 1000);

      // Landmark documents are exempt from decay
      if (landmarkExempt && result.metadata.is_landmark) {
        return result;
      }

      let boost = 1.0;
      if (ageYears <= recentWindowYears) {
        boost = recentBoost;
      } else if (ageYears > decayAfterYears) {
        boost = decayPenalty;
      }

      return {
        ...result,
        score: result.score * boost,
        recencyBoost: boost,
      };
    });
  }

  // ─── Private: Query Enrichment ───────────────────────────────────────────

  private enrichQuery(
    query: string,
    context?: string[],
    window: number = 2
  ): string {
    if (!context || context.length === 0) return query;

    const recentContext = context.slice(-window).join(' ');
    return `${recentContext} ${query}`;
  }

  // ─── Private: Embedding ──────────────────────────────────────────────────

  private async generateEmbedding(text: string): Promise<number[]> {
    if (this.config.customEmbedFn) {
      return this.config.customEmbedFn(text);
    }

    if (this.config.embeddingProvider === 'openai') {
      const response = await fetch('https://api.openai.com/v1/embeddings', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${this.config.openaiApiKey}`,
        },
        body: JSON.stringify({
          model: this.config.embeddingModel ?? 'text-embedding-3-small',
          input: text,
        }),
      });

      const data = await response.json();
      return data.data[0].embedding;
    }

    throw new Error(`Unsupported embedding provider: ${this.config.embeddingProvider}`);
  }

  // ─── Private: Chunking ───────────────────────────────────────────────────

  private chunkText(
    text: string,
    chunkSize: number,
    overlap: number,
    preserveHeaders: boolean
  ): string[] {
    // Simple word-based chunking with overlap
    const words = text.split(/\s+/);
    const chunks: string[] = [];
    let currentHeader = '';

    if (preserveHeaders) {
      // Extract section headers for context preservation
      const lines = text.split('\n');
      for (const line of lines) {
        if (line.startsWith('#') || line.match(/^[A-Z][A-Z\s]{3,}$/)) {
          currentHeader = line.replace(/^#+\s*/, '').trim();
        }
      }
    }

    for (let i = 0; i < words.length; i += chunkSize - overlap) {
      const chunkWords = words.slice(i, i + chunkSize);
      if (chunkWords.length < 20) break; // Skip tiny trailing chunks

      const chunkText = chunkWords.join(' ');
      const finalChunk = preserveHeaders && currentHeader
        ? `[${currentHeader}] ${chunkText}`
        : chunkText;

      chunks.push(finalChunk);
    }

    return chunks;
  }
}

// ─── Exported Pure Functions (for testing and advanced usage) ──────────────

/**
 * Reciprocal Rank Fusion: combines two ranked result lists into one.
 */
export function reciprocalRankFusion(
  semanticResults: RawSemanticResult[],
  bm25Results: RawBM25Result[],
  k: number = 60
): SearchResult[] {
  const scoreMap = new Map<string, SearchResult>();

  semanticResults.forEach((result, index) => {
    const rrfScore = 1 / (k + index + 1);
    scoreMap.set(result.id, {
      id: result.id,
      content: result.content,
      score: rrfScore,
      semanticScore: result.similarity,
      bm25Score: 0,
      source: result.source,
      authorityBoost: 1.0,
      recencyBoost: 1.0,
      metadata: {
        ...result.metadata,
        published_at: result.published_at,
        source_authority: result.source_authority,
      },
    });
  });

  bm25Results.forEach((result, index) => {
    const rrfScore = 1 / (k + index + 1);
    const existing = scoreMap.get(result.id);

    if (existing) {
      existing.score += rrfScore;
      existing.bm25Score = result.rank;
    } else {
      scoreMap.set(result.id, {
        id: result.id,
        content: result.content,
        score: rrfScore,
        semanticScore: 0,
        bm25Score: result.rank,
        source: result.source,
        authorityBoost: 1.0,
        recencyBoost: 1.0,
        metadata: {
          ...result.metadata,
          published_at: result.published_at,
          source_authority: result.source_authority,
        },
      });
    }
  });

  return Array.from(scoreMap.values());
}

/**
 * Apply authority boosting to search results based on source weights.
 */
export function applyAuthorityBoosting(
  results: SearchResult[],
  weights: Record<string, number>
): SearchResult[] {
  return results.map(result => {
    const weight = weights[result.source] ?? 1.0;
    return {
      ...result,
      score: result.score * weight,
      authorityBoost: weight,
    };
  });
}

/**
 * Apply recency weighting to search results.
 */
export function applyRecencyWeighting(
  results: SearchResult[],
  config: RecencyConfig,
  now: Date = new Date()
): SearchResult[] {
  const { recentBoost, recentWindowYears, decayPenalty, decayAfterYears, landmarkExempt } = config;

  return results.map(result => {
    const publishedAt = result.metadata.published_at
      ? new Date(result.metadata.published_at as string)
      : null;

    if (!publishedAt) return result;

    const ageYears = (now.getTime() - publishedAt.getTime()) / (365.25 * 24 * 60 * 60 * 1000);

    if (landmarkExempt && result.metadata.is_landmark) {
      return result;
    }

    let boost = 1.0;
    if (ageYears <= recentWindowYears) {
      boost = recentBoost;
    } else if (ageYears > decayAfterYears) {
      boost = decayPenalty;
    }

    return {
      ...result,
      score: result.score * boost,
      recencyBoost: boost,
    };
  });
}

/**
 * Enrich a query with conversation context for follow-up questions.
 */
export function enrichQuery(
  query: string,
  context?: string[],
  window: number = 2
): string {
  if (!context || context.length === 0) return query;
  const recentContext = context.slice(-window).join(' ');
  return `${recentContext} ${query}`;
}

/**
 * Chunk text into overlapping segments with optional header preservation.
 */
export function chunkText(
  text: string,
  chunkSize: number = 512,
  overlap: number = 64,
  preserveHeaders: boolean = true
): string[] {
  const words = text.split(/\s+/);
  const chunks: string[] = [];
  let currentHeader = '';

  if (preserveHeaders) {
    const lines = text.split('\n');
    for (const line of lines) {
      if (line.startsWith('#') || line.match(/^[A-Z][A-Z\s]{3,}$/)) {
        currentHeader = line.replace(/^#+\s*/, '').trim();
      }
    }
  }

  for (let i = 0; i < words.length; i += chunkSize - overlap) {
    const chunkWords = words.slice(i, i + chunkSize);
    if (chunkWords.length < 20) break;

    const chunkText = chunkWords.join(' ');
    const finalChunk = preserveHeaders && currentHeader
      ? `[${currentHeader}] ${chunkText}`
      : chunkText;

    chunks.push(finalChunk);
  }

  return chunks;
}

// Also export types needed for the standalone functions
export type { RawSemanticResult, RawBM25Result };
