-- 1. Enable pgvector extension (run once per project)
create extension if not exists vector;

-- 2. KB chunks table
-- Voyage AI's "voyage-3" model outputs 1024-dim embeddings.
-- If you use a different Voyage model, change the dimension below to match.
create table if not exists shopease_kb_chunks (
  id uuid primary key default gen_random_uuid(),
  chunk_id text unique not null,
  doc_id text not null,
  title text not null,
  category text not null,
  chunk_text text not null,
  embedding vector(1024),
  created_at timestamptz default now()
);

-- 3. Index for fast similarity search
create index if not exists shopease_kb_chunks_embedding_idx
  on shopease_kb_chunks
  using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);

-- 4. Similarity search function (call this from your backend for retrieval)
create or replace function match_shopease_chunks (
  query_embedding vector(1024),
  match_count int default 5
)
returns table (
  chunk_id text,
  doc_id text,
  title text,
  category text,
  chunk_text text,
  similarity float
)
language sql stable
as $$
  select
    chunk_id,
    doc_id,
    title,
    category,
    chunk_text,
    1 - (embedding <=> query_embedding) as similarity
  from shopease_kb_chunks
  order by embedding <=> query_embedding
  limit match_count;
$$;
