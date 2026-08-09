"""
Retrieval function: given a user query, embeds it with Voyage AI
(input_type: "query" — different from "document" used at index time)
and fetches the top-k most similar KB chunks from Supabase.

Setup: same env vars as embed_and_push.py (VOYAGE_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY)

Usage:
    from retrieve import retrieve_chunks
    chunks = retrieve_chunks("mera order kab aayega?")

Or run directly:
    python retrieve.py "mera order kab aayega?"
"""

import os
import sys
from dotenv import load_dotenv
import requests
from supabase import create_client

load_dotenv()

VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
VOYAGE_MODEL = "voyage-3"  # must match the model used in embed_and_push.py

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


# --- Embed the user's query ---
def embed_query(query):
    res = requests.post(
        "https://api.voyageai.com/v1/embeddings",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {VOYAGE_API_KEY}",
        },
        json={
            "input": [query],
            "model": VOYAGE_MODEL,
            "input_type": "query",  # "query" here, was "document" when we embedded the KB chunks
        },
    )

    if not res.ok:
        raise RuntimeError(f"Voyage AI error {res.status_code}: {res.text}")

    data = res.json()
    return data["data"][0]["embedding"]


# --- Retrieve top-k similar chunks using the Supabase RPC function
#     (match_shopease_chunks, created in supabase_setup.sql) ---
def retrieve_chunks(query, top_k=5):
    query_embedding = embed_query(query)

    result = supabase.rpc(
        "match_shopease_chunks",
        {"query_embedding": query_embedding, "match_count": top_k},
    ).execute()

    return result.data  # list of {chunk_id, doc_id, title, category, chunk_text, similarity}


if __name__ == "__main__":
    test_query = sys.argv[1] if len(sys.argv) > 1 else "mera order kab tak deliver hoga?"
    results = retrieve_chunks(test_query)

    print(f'\nQuery: "{test_query}"\n')
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r['category']}] {r['title']} (similarity: {r['similarity']:.3f})")
        print(f"   {r['chunk_text'][:150]}...\n")
