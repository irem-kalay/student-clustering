import os
import time
from dotenv import load_dotenv
from supabase import create_client
from google import genai

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
assert SUPABASE_URL and SUPABASE_KEY and GEMINI_API_KEY, "Missing env vars in .env"

sb = create_client(SUPABASE_URL, SUPABASE_KEY)
client = genai.Client(api_key=GEMINI_API_KEY)

EMBED_MODEL = "models/gemini-embedding-001"
EMBED_DIM = 1536

FETCH_LIMIT = 120     # daha çok çek
BATCH = 20            # gemini request başına 20 içerik
SLEEP_OK = 0.2        # hızlı
SLEEP_429 = 30        # 429 yerse bekle

def fetch_unembedded(limit=FETCH_LIMIT):
    res = (
        sb.table("course_chunks")
        .select("id, chunk_text")
        .is_("embedding", "null")
        .limit(limit)
        .execute()
    )
    return res.data or []

def update_embedding(row_id, emb):
    sb.table("course_chunks").update({"embedding": emb}).eq("id", row_id).execute()

def embed_texts(texts):
    resp = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config={"output_dimensionality": EMBED_DIM},
    )
    return [e.values for e in resp.embeddings]

def main():
    total_done = 0
    sleep_429 = SLEEP_429

    while True:
        rows = fetch_unembedded()
        if not rows:
            print("✅ All chunks embedded.")
            break

        for i in range(0, len(rows), BATCH):
            batch = rows[i:i+BATCH]
            texts = [r["chunk_text"] for r in batch]

            try:
                embs = embed_texts(texts)

                # tek tek update (ama batch büyük + sleep küçük => pratikte hızlı)
                for r, emb in zip(batch, embs):
                    update_embedding(r["id"], emb)

                total_done += len(batch)
                print(f"✅ Embedded+saved {len(batch)} (total this run: {total_done})")

                sleep_429 = max(SLEEP_429, sleep_429 * 0.9)
                time.sleep(SLEEP_OK)

            except Exception as e:
                msg = str(e)
                print("❌ Error:", msg)

                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    print(f"⏳ 429 hit. Sleeping {int(sleep_429)}s then retrying...")
                    time.sleep(sleep_429)
                    sleep_429 = min(180, sleep_429 * 1.5)
                    continue
                else:
                    raise

    print("Done.")

if __name__ == "__main__":
    main()