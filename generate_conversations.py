"""
ShopEase Conversational Data Generator
Generates labeled customer support conversations (Hindi/English/Hinglish)
for intent classification training.

Uses Google Gemini 2.5 Flash (free tier).
"""

from google import genai
import json
import os
import time
from dotenv import load_dotenv

load_dotenv()  # reads .env file in the same folder and loads GEMINI_API_KEY

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

INTENTS = [
    "order_status",
    "refund_request",
    "cancellation",
    "payment_issue",
    "delivery_delay",
    "product_query",
    "complaint",
    "account_issue",
    "coupon_discount",
    "escalation_request",
]

SYSTEM_PROMPT = """You are generating synthetic training data for an Indian 
e-commerce ("ShopEase") customer support NLP system. Output ONLY valid JSON, 
no markdown fences, no preamble, no explanation."""

OUTPUT_PATH = "data/raw/conversations/shopease_conversations.json"


def generate_batch(intent: str, num_samples: int = 15, max_retries: int = 5):
    user_prompt = f"""
Generate {num_samples} realistic customer support query-response pairs 
for ShopEase, all with intent: "{intent}".

Language mix (spread across the {num_samples} samples):
- ~30% pure Hindi (Devanagari script)
- ~30% pure English
- ~40% Hinglish (Roman script, code-mixed, like real Indian users type 
  e.g. "Mera order kaha hai, 5 din ho gaye")

Make queries sound natural and varied (different phrasing, some short, 
some detailed, some slightly frustrated/urgent where appropriate).

Return ONLY a JSON array, each item with fields:
- "query": the customer message
- "response": a realistic ShopEase support agent reply
- "intent": "{intent}"
- "language": "hindi" | "english" | "hinglish"

No markdown formatting, no code fences, just raw JSON array.
"""

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_prompt,
                config={
                    "system_instruction": SYSTEM_PROMPT,
                    "temperature": 0.9,
                },
            )
            text = response.text.strip()
            if text.startswith("```"):
                text = text.strip("`")
                text = text.replace("json\n", "", 1)

            return json.loads(text)

        except Exception as e:
            wait = min(60, 5 * attempt)  # exponential-ish backoff, capped at 60s
            print(f"  ⚠️ Attempt {attempt}/{max_retries} failed ({e}). "
                  f"Retrying in {wait}s...")
            time.sleep(wait)

    print(f"  ❌ Giving up on intent '{intent}' after {max_retries} attempts.")
    return []


def load_existing():
    """Resume support: load already-generated samples and figure out which
    intents are already done, so we don't waste calls regenerating them."""
    if not os.path.exists(OUTPUT_PATH):
        return [], set()

    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        try:
            existing = json.load(f)
        except json.JSONDecodeError:
            return [], set()

    done_intents = {item.get("intent") for item in existing if item.get("intent")}
    return existing, done_intents


def main():
    os.makedirs("data/raw/conversations", exist_ok=True)

    all_samples, done_intents = load_existing()
    if done_intents:
        print(f"Resuming. Already have data for: {sorted(done_intents)}")

    for intent in INTENTS:
        if intent in done_intents:
            print(f"Skipping (already done): {intent}")
            continue

        print(f"Generating samples for intent: {intent}")
        samples = generate_batch(intent, num_samples=15)
        all_samples.extend(samples)

        # save after every intent so progress is never lost
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(all_samples, f, ensure_ascii=False, indent=2)

        time.sleep(4)  # stay comfortably within free tier RPM limit

    print(f"\n✅ Generated {len(all_samples)} total samples -> {OUTPUT_PATH}")

 
if __name__ == "__main__":
    main()