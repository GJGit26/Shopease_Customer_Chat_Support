"""
One-time script: reads shopease_kb_chunks.csv, generates embeddings
via Voyage AI, and upserts everything into Supabase (pgvector table).

Setup:
    pip install python-dotenv supabase requests

Run:
    python embed_and_push.py
"""

import os
import csv
import time
from dotenv import load_dotenv
import requests
from supabase import create_client

load_dotenv()

VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")  # service role key, not anon key
CSV_PATH = os.environ.get("KB_CHUNKS_CSV", "shopease_kb_chunks.csv")

VOYAGE_MODEL = "voyage-3"  # 1024-dim. Change here + in supabase_setup.sql if you switch models.
BATCH_SIZE = 20  # Voyage AI accepts batched inputs per request — keeps calls fast & cheap
TABLE_NAME = "shopease_kb_chunks"

# Voyage AI free tier (no payment method added) = 3 requests/minute.
# 21s gap keeps you safely under that. Once you add a payment method on
# https://dashboard.voyageai.com/ billing page, you can drop this to ~1.
DELAY_BETWEEN_REQUESTS_SEC = 21
MAX_RETRIES = 3

if not all([VOYAGE_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY]):
    raise SystemExit(
        "Missing env vars. Check your .env file (VOYAGE_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY)."
    )

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


# --- Step 1: Load chunks from CSV ---
def load_chunks():
    with open(CSV_PATH, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)  # each row: {doc_id, chunk_id, title, category, chunk_text}


# --- Step 2: Call Voyage AI embeddings endpoint for a batch of texts ---
# Retries on 429 (rate limit) with growing backoff instead of giving up.
def embed_batch(texts, attempt=1):
    res = requests.post(
        "https://api.voyageai.com/v1/embeddings",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {VOYAGE_API_KEY}",
        },
        json={
            "input": texts,
            "model": VOYAGE_MODEL,
            "input_type": "document",  # "document" for indexing; use "query" at search time
        },
    )

    if res.status_code == 429:
        if attempt > MAX_RETRIES:
            raise RuntimeError("Voyage AI error 429: rate limited after max retries")
        wait_sec = DELAY_BETWEEN_REQUESTS_SEC * attempt
        print(f"  Rate limited, waiting {wait_sec}s before retry {attempt}/{MAX_RETRIES}...")
        time.sleep(wait_sec)
        return embed_batch(texts, attempt + 1)

    if not res.ok:
        raise RuntimeError(f"Voyage AI error {res.status_code}: {res.text}")

    data = res.json()
    return [item["embedding"] for item in data["data"]]


# --- Step 2b: Find which chunk_ids are already in Supabase, so a re-run
# doesn't re-embed (and re-pay for) chunks that already succeeded. ---
def get_already_pushed_ids():
    result = supabase.table(TABLE_NAME).select("chunk_id").execute()
    return {row["chunk_id"] for row in result.data}


# --- Step 3: Upsert a batch of rows (with embeddings) into Supabase ---
def push_batch(rows):
    supabase.table(TABLE_NAME).upsert(rows, on_conflict="chunk_id").execute()


def main():
    all_chunks = load_chunks()
    print(f"Loaded {len(all_chunks)} chunks from {CSV_PATH}")

    already_pushed = get_already_pushed_ids()
    chunks = [c for c in all_chunks if c["chunk_id"] not in already_pushed]
    print(f"{len(already_pushed)} already in Supabase, skipping those. {len(chunks)} left to embed.")

    processed = 0
    failed_batches = []

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c["chunk_text"] for c in batch]

        try:
            embeddings = embed_batch(texts)

            rows = [
                {
                    "chunk_id": c["chunk_id"],
                    "doc_id": c["doc_id"],
                    "title": c["title"],
                    "category": c["category"],
                    "chunk_text": c["chunk_text"],
                    "embedding": emb,
                }
                for c, emb in zip(batch, embeddings)
            ]

            push_batch(rows)
            processed += len(batch)
            print(f"Embedded + pushed {processed}/{len(chunks)}")
        except Exception as err:
            print(f"Failed on batch starting at index {i}: {err}")
            failed_batches.append(i)
            # Continue with next batch instead of killing the whole run

        # Stay under Voyage AI's free-tier rate limit between requests
        if i + BATCH_SIZE < len(chunks):
            time.sleep(DELAY_BETWEEN_REQUESTS_SEC)

    if failed_batches:
        print(
            f"\n{len(failed_batches)} batch(es) failed permanently. "
            "Just re-run the script — it will skip already-pushed chunks and retry the rest."
        )
    else:
        print("\nDone. All chunks processed successfully.")


if __name__ == "__main__":
    main()
