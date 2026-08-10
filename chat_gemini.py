"""
Full customer support chat pipeline (Gemini version):
1. Retrieve relevant KB chunks (retrieve.py — still uses Voyage AI embeddings,
   that part doesn't change, only the response-generation model changes)
2. Pass them + user query to Gemini
3. Return a grounded, support-toned response

Setup: same env vars as before + GEMINI_API_KEY
    Get a free key (no credit card): https://aistudio.google.com/apikey

Usage:
    python chat_gemini.py "mera refund kab tak aa jayega?"
"""

import os
import sys
from dotenv import load_dotenv
import requests
from retrieve import retrieve_chunks

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
# Free tier as of Aug 2026 = Flash / Flash-Lite models only (Pro is paid-only).
# gemini-3.5-flash = best quality on free tier. Switch to gemini-2.5-flash if
# 3.5 isn't enabled for your project, or gemini-3.1-flash-lite for higher RPM/day.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")

SYSTEM_PROMPT = """You are ShopEase's customer support assistant.
Answer ONLY using the context provided below. If the context doesn't cover
the question, say you don't have that information and suggest contacting
human support — do not make up policies or details.
Match the customer's language/tone (Hindi, English, or Hinglish — mirror whatever they used).
Keep responses concise and friendly, like a helpful support agent, not a wall of text."""


def generate_response(user_query, chat_history=None):
    chat_history = chat_history or []  # list of {"role": "user"|"assistant", "content": "..."}

    # Step 1: Retrieve relevant KB context (unchanged — Voyage AI embeddings)
    chunks = retrieve_chunks(user_query, top_k=5)

    context = "\n\n".join(
        f"[{i+1}] ({c['category']} — {c['title']})\n{c['chunk_text']}"
        for i, c in enumerate(chunks)
    )

    # Step 2: Build Gemini request
    # Gemini's REST format: contents[] alternate user/model turns, systemInstruction is separate.
    contents = [
        {
            "role": "model" if m["role"] == "assistant" else "user",
            "parts": [{"text": m["content"]}],
        }
        for m in chat_history
    ]
    contents.append({"role": "user", "parts": [{"text": user_query}]})

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

    res = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json={
            "systemInstruction": {
                "parts": [{"text": f"{SYSTEM_PROMPT}\n\n--- CONTEXT ---\n{context}"}]
            },
            "contents": contents,
            "generationConfig": {"maxOutputTokens": 10000, "temperature": 0.4},
        },
    )

    if not res.ok:
        raise RuntimeError(f"Gemini API error {res.status_code}: {res.text}")

    data = res.json()
    parts = data["candidates"][0]["content"]["parts"]
    answer = "".join(p.get("text", "") for p in parts)

    sources = [
        {"title": c["title"], "category": c["category"], "similarity": c["similarity"]}
        for c in chunks
    ]

    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    test_query = sys.argv[1] if len(sys.argv) > 1 else "mera order kab tak deliver hoga?"
    result = generate_response(test_query)

    print(f"\nUser: {test_query}\n")
    print(f"Assistant: {result['answer']}\n")
    print("Sources used:", ", ".join(s["title"] for s in result["sources"]))
