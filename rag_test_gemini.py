# rag_test_gemini.py
# Improved for:
# - mixed course_code formats ("/", "-", "E" suffix, etc.)
# - less timeouts (avoid broad search when filter exists)
# - pedagogical / slide-design questions => infer_mode automatically true
# - if context is weak, still produce analyst-style pedagogical guidance (without pretending catalog facts exist)

import os
import re
from typing import Optional, Tuple, List, Dict, Any

from dotenv import load_dotenv
from supabase import create_client
from google import genai

# -------------------------
# ENV / CLIENTS
# -------------------------
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not (SUPABASE_URL and SUPABASE_KEY and GEMINI_API_KEY):
    raise RuntimeError("Missing env vars. Ensure SUPABASE_URL, SUPABASE_KEY, GEMINI_API_KEY are set in .env")

sb = create_client(SUPABASE_URL, SUPABASE_KEY)
client = genai.Client(api_key=GEMINI_API_KEY)

EMBED_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "models/gemini-2.5-flash"
EMBED_DIM = 1536

# -------------------------
# COURSE TOKEN PARSING
# -------------------------

# Examples caught: BLG222, BLG222E, MAT104, FIZ101, etc.
COURSE_TOKEN_RE = re.compile(r"\b([A-ZÇĞİÖŞÜ]{3})\s?(\d{3})(E)?\b", re.IGNORECASE)

# Some codes have 1 extra suffix letter (A/L) e.g., ING201A, FIZ101L
COURSE_TOKEN_EXTRA_RE = re.compile(r"\b([A-ZÇĞİÖŞÜ]{3})\s?(\d{3})([A-Z])\b", re.IGNORECASE)

# Lightweight aliasing from common names -> code tokens
ALIASES = {
    "COMP ORG": "BLG222",
    "COMPUTER ORGANIZATION": "BLG222",
    "COMPUTER ORGANISATION": "BLG222",
    "BİLGİSAYAR ORGANİZASYONU": "BLG222",
    "BILGISAYAR ORGANIZASYONU": "BLG222",

    "OPERATING SYSTEMS": "BLG312",
    "İŞLETİM SİSTEMLERİ": "BLG312",
    "ISLETIM SISTEMLERI": "BLG312",

    "COMPUTER VISION": "BLG453",
    "BİLGİSAYARLA GÖRÜ": "BLG453",
    "BILGISAYARLA GORU": "BLG453",
}


def normalize_course_tokens(q: str) -> List[str]:
    """
    Extract course-like tokens from text.
    Returns list like ["BLG222", "BLG222E"] or ["ING201A"] etc.
    """
    q_up = q.upper()
    tokens: List[str] = []

    # Main pattern: ABC123 or ABC123E
    for m in COURSE_TOKEN_RE.finditer(q_up):
        prefix, num, e = m.group(1), m.group(2), m.group(3)
        base = f"{prefix}{num}"
        if e:
            tokens.append(f"{base}E")
        else:
            tokens.append(base)

    # Extra suffix letter: ABC123A / ABC123L etc.
    for m in COURSE_TOKEN_EXTRA_RE.finditer(q_up):
        prefix, num, suff = m.group(1), m.group(2), m.group(3)
        t = f"{prefix}{num}{suff}"
        tokens.append(t)

    # Unique preserve order
    seen = set()
    out: List[str] = []
    for t in tokens:
        t = t.replace(" ", "")
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def apply_aliases(q: str, tokens: List[str]) -> List[str]:
    q_up = q.upper()
    for k, v in ALIASES.items():
        if k in q_up:
            # Put alias token first (higher priority)
            if v not in tokens:
                return [v] + tokens
            # If already inside, keep it first
            return [v] + [t for t in tokens if t != v]
    return tokens


def guess_course_code_filter(tokens: List[str]) -> Optional[str]:
    """
    Return a filter token that should match course_catalogs.course_code via ILIKE '%token%'.
    Prefer the most specific token first (endswith E or suffix).
    """
    if not tokens:
        return None

    # Prefer explicit E or suffix (more specific)
    specific = [t for t in tokens if t.endswith("E") or re.match(r".*\d{3}[A-Z]$", t)]
    return specific[0] if specific else tokens[0]


def parse_week(q: str) -> Optional[int]:
    q_up = q.upper()
    m1 = re.search(r"\b(?:HAFTA|WEEK)\s*(\d{1,2})\b", q_up)
    if m1:
        return int(m1.group(1))
    m2 = re.search(r"\b(\d{1,2})\s*\.?\s*(?:HAFTA|WEEK)\b", q_up)
    if m2:
        return int(m2.group(1))
    return None


def is_infer_question(q: str) -> bool:
    ql = q.lower()
    return any(
        phrase in ql
        for phrase in [
            "yorum", "yorumla", "sence", "tahmin",
            "karşılaştır", "hangisi daha", "öner",
            "en yoğun", "en ağır", "en zor",
            "most intensive", "hardest", "most demanding",
        ]
    )


def is_pedagogical(q: str) -> bool:
    ql = q.lower()
    keys = [
        "slayt", "sunum", "ders anlat", "anlatım", "öğrenci profili", "ogrenci profili",
        "kitle", "kitlesine", "nasıl hazırlamalıyım", "nasil hazirlamaliyim",
        "nasıl anlatmalıyım", "nasil anlatmaliyim", "öğretim", "ogretim",
        "pedagoji", "teaching", "lecture", "slides",
    ]
    return any(k in ql for k in keys)


def parse_question(q: str) -> Tuple[Optional[str], Optional[int], bool, List[str]]:
    tokens = normalize_course_tokens(q)
    tokens = apply_aliases(q, tokens)
    course_filter = guess_course_code_filter(tokens)
    week_no = parse_week(q)
    infer_mode = is_infer_question(q) or is_pedagogical(q)
    return course_filter, week_no, infer_mode, tokens


# -------------------------
# EMBEDDING + RETRIEVAL
# -------------------------

def embed_query(text: str) -> List[float]:
    resp = client.models.embed_content(
        model=EMBED_MODEL,
        contents=[text],
        config={"output_dimensionality": EMBED_DIM},
    )
    return resp.embeddings[0].values


def rpc_match(query_embedding: List[float], match_count: int, course_code_filter: Optional[str], keyword_filter: Optional[str]) -> List[Dict[str, Any]]:
    payload = {
        "query_embedding": query_embedding,
        "match_count": match_count,
        "course_code_filter": course_code_filter,
        "keyword_filter": keyword_filter,
    }
    res = sb.rpc("match_course_chunks_smart", payload).execute()
    return res.data or []


def retrieve_smart(query_embedding: List[float], course_filter: Optional[str], week_no: Optional[int], tokens: List[str]) -> List[Dict[str, Any]]:
    """
    Safer retrieval to reduce statement timeouts:
    - If course_filter exists => DO NOT do broad search (fast + clean).
    - Else do small broad search.
    - Keyword filter in DB can slow down; keep it off for now.
    """
    keyword = None  # keep DB keyword off to reduce latency/timeouts

    if course_filter:
        k = 18 if week_no is not None else 12
        data = rpc_match(query_embedding, k, course_code_filter=course_filter, keyword_filter=keyword)

        # If nothing, try next tokens (sometimes filter token mismatch)
        if not data and tokens:
            for t in tokens[1:3]:
                data = rpc_match(query_embedding, k, course_code_filter=t, keyword_filter=keyword)
                if data:
                    break
        return data

    # No filter => small broad
    k = 12 if week_no is not None else 8
    return rpc_match(query_embedding, k, course_code_filter=None, keyword_filter=keyword)


def build_context(chunks: List[Dict[str, Any]]) -> str:
    parts = []
    for i, ch in enumerate(chunks, 1):
        parts.append(
            f"[Chunk {i}] (course={ch.get('course_code')}, idx={ch.get('chunk_index')}, score={float(ch.get('score', 0)):.3f})\n"
            f"{ch.get('chunk_text')}"
        )
    return "\n\n---\n\n".join(parts)


def print_retrieved(chunks: List[Dict[str, Any]]) -> None:
    print("\n--- RETRIEVED CHUNKS ---")
    if not chunks:
        print("(none)")
        return
    for i, ch in enumerate(chunks, 1):
        print(
            f"{i}) score={float(ch.get('score', 0)):.3f} "
            f"course={ch.get('course_code')} "
            f"course_id={ch.get('course_id')} "
            f"chunk_id={ch.get('chunk_id')} "
            f"chunk_index={ch.get('chunk_index')}"
        )


# -------------------------
# ANSWERING (Pedagogical analyst mode)
# -------------------------

def ask_gemini(question: str, context: str, infer_mode: bool) -> str:
    """
    Policy:
    - Use context as evidence when present.
    - If context weak/missing:
        * If question asks catalog-hard facts => say not visible in catalog.
        * If question is pedagogical/slide advice => give general best practices, and clearly say it's general.
    - Never claim "catalog says X" unless it appears in context.
    """
    prompt = f"""
Sen bir "ders tasarımı & müfredat analisti" asistansın.
Katalog chunk'ları (Context) senin için kanıt/veri kaynağıdır; ama cevapların kopya gibi değil, eğitmen odaklı ve pedagojik olmalı.

HEDEF:
Kullanıcının sorusuna, mümkünse katalog içeriğine dayanarak; değilse de iyi öğretim tasarımı pratiklerine dayanarak,
somut, uygulanabilir ve analitik öneriler ver.

KURALLAR:
1) Context varsa: dersin içerik/haftalık plan/öğrenme çıktıları/önşart/ölçme gibi alanlarından yararlanıp önerileri o derse göre özelleştir.
2) Context yoksa veya çok zayıfsa:
   - Kredi/önşart/hafta konusu gibi "katalogdan net bilgi" isteyen sorularda: "Katalogda bu detay görünmüyor" de.
   - Slayt/sunum/anlatım/pedagoji sorularında: katalogdan bağımsız GENEL iyi pratiklerle analist gibi öneri ver.
   - Bu durumda "katalogdan değil, genel öğretim tasarımı iyi pratiklerinden" konuştuğunu açıkça belirt.
3) Asla katalogda geçmeyen spesifik bir şeyi "katalogda varmış gibi" söyleme.
4) infer_mode={infer_mode}. infer_mode true ise yorum/öneri/karşılaştırma bekleniyor.

ÇIKTI FORMATI:
- (A) Slayt stratejisi (madde madde; hedef kitleye göre)
- (B) 10–12 slaytlık örnek akış (başlık + 1 cümle amaç)
- (C) 3 mini-aktivite / quiz fikri (derste uygulanabilir)
- (D) Eğer Context'ten dayandığın yer varsa: "Katalog dayanağı" (1-5 madde)

SORU:
{question}

CONTEXT:
{context}
"""
    resp = client.models.generate_content(model=CHAT_MODEL, contents=prompt)
    return resp.text or ""


# -------------------------
# MAIN LOOP
# -------------------------
if __name__ == "__main__":
    while True:
        q = input("\nSoru yaz (çıkmak için boş bırak): ").strip()
        if not q:
            break

        course_filter, week_no, infer_mode, tokens = parse_question(q)

        # Strengthen embedding query text
        q2 = q
        if tokens:
            q2 += "\nDetected course tokens: " + ", ".join(tokens)
        if week_no is not None:
            # Your YAML-ish chunk style: include these hints
            q2 += f"\nWEEKLY_COURSE_PLAN week: {week_no}\nweek: {week_no}\nHafta {week_no}"

        # Embed + Retrieve
        q_emb = embed_query(q2)
        chunks = retrieve_smart(q_emb, course_filter=course_filter, week_no=week_no, tokens=tokens)

        # Build context and answer
        context = build_context(chunks)
        print_retrieved(chunks)

        print("\n--- ANSWER ---")
        print(ask_gemini(q, context, infer_mode=infer_mode))