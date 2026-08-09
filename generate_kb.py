"""
ShopEase Knowledge Base Generator
Generates FAQ/policy documents for RAG pipeline.

Uses Google Gemini 2.5 Flash (free tier).
"""

from google import genai
import json
import os
import time
from dotenv import load_dotenv

load_dotenv()  # reads .env file in the same folder and loads GEMINI_API_KEY

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

CATEGORIES = [
    "Order Tracking & Delivery",
    "Returns & Refunds",
    "Payment Issues (UPI, Cards, COD, Wallets)",
    "Account Management",
    "Product Warranty & Damage Claims",
    "Order Cancellation",
    "Coupons & Discounts",
    "Shipping & Delivery Charges",
    "Product Availability & Stock",
    "Customer Support Escalation Policy",
]

SYSTEM_PROMPT = """You are creating a knowledge base for "ShopEase", a fictional 
Indian e-commerce company (similar to Flipkart/Meesho in tone and policies). 
Generate realistic, detailed FAQ/policy documents. Output ONLY valid JSON, 
no markdown fences, no preamble."""

OUTPUT_PATH = "data/raw/knowledge_base/shopease_kb.json"


def generate_docs_for_category(category: str, num_docs: int = 3, max_retries: int = 5):
    user_prompt = f"""
Generate {num_docs} detailed knowledge base documents for ShopEase 
in the category: "{category}".

Each document should be 200-400 words, written as an internal support 
policy/FAQ reference (not a conversation). Include specific realistic 
details (timelines, e.g. "7 days", amounts, e.g. "₹499", processes).

Return ONLY a JSON array, each item with fields:
- "title": short doc title
- "category": "{category}"
- "content": the full document text

No markdown formatting, no code fences, just raw JSON array.
"""

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model="gemini-flash-latest",
                contents=user_prompt,
                config={
                    "system_instruction": SYSTEM_PROMPT,
                    "temperature": 0.8,
                },
            )
            text = response.text.strip()
            if text.startswith("```"):
                text = text.strip("`")
                text = text.replace("json\n", "", 1)

            return json.loads(text)

        except Exception as e:
            wait = min(60, 5 * attempt)
            print(f"  ⚠️ Attempt {attempt}/{max_retries} failed ({e}). "
                  f"Retrying in {wait}s...")
            time.sleep(wait)

    print(f"  ❌ Giving up on category '{category}' after {max_retries} attempts.")
    return []


def load_existing():
    if not os.path.exists(OUTPUT_PATH):
        return [], set()

    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        try:
            existing = json.load(f)
        except json.JSONDecodeError:
            return [], set()

    done_categories = {item.get("category") for item in existing if item.get("category")}
    return existing, done_categories


def main():
    os.makedirs("data/raw/knowledge_base", exist_ok=True)

    all_docs, done_categories = load_existing()
    if done_categories:
        print(f"Resuming. Already have docs for: {sorted(done_categories)}")

    for category in CATEGORIES:
        if category in done_categories:
            print(f"Skipping (already done): {category}")
            continue

        print(f"Generating docs for: {category}")
        docs = generate_docs_for_category(category, num_docs=3)
        all_docs.extend(docs)

        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(all_docs, f, ensure_ascii=False, indent=2)

        time.sleep(4)

    print(f"\n✅ Generated {len(all_docs)} total documents -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()