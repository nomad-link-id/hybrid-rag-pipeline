-- Semantic search function
CREATE OR REPLACE FUNCTION match_documents(
  query_embedding vector(1536),
  match_threshold float,
  match_count int
)
RETURNS TABLE (
  id uuid,
  content text,
  similarity float,
  source text,
  source_authority float,
  published_at timestamptz,
  metadata jsonb
)
LANGUAGE sql STABLE
AS $$
  SELECT
    id,
    content,
    1 - (embedding <=> query_embedding) AS similarity,
    source,
    source_authority,
    published_at,
    metadata
  FROM documents
  WHERE 1 - (embedding <=> query_embedding) > match_threshold
  ORDER BY embedding <=> query_embedding
  LIMIT match_count;
$$;

-- BM25 search function
CREATE OR REPLACE FUNCTION bm25_search(
  search_query text,
  result_limit int
)
RETURNS TABLE (
  id uuid,
  content text,
  rank float,
  source text,
  source_authority float,
  published_at timestamptz,
  metadata jsonb
)
LANGUAGE sql STABLE
AS $$
  SELECT
    id,
    content,
    ts_rank(tsv, to_tsquery('english', search_query)) AS rank,
    source,
    source_authority,
    published_at,
    metadata
  FROM documents
  WHERE tsv @@ to_tsquery('english', search_query)
  ORDER BY rank DESC
  LIMIT result_limit;
$$;
