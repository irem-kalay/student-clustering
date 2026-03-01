import os
import re
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# Paragraf bazlı küçük parçaları maksimum bu uzunlukta tutacağız
MAX_CHARS = 1400

FIELD_ORDER = [
    "COURSE_CODE",
    "COURSE_NAME_TR",
    "COURSE_NAME_EN",
    "DEPARTMENT_TR",
    "DEPARTMENT_EN",
    "COURSE_LANGUAGE",
    "LOCAL_CREDITS",
    "ECTS",
    "HOURS_PER_WEEK",
    "PREREQUISITES",
    "COURSE_DESCRIPTION_TR",
    "COURSE_DESCRIPTION_EN",
    "COURSE_OBJECTIVES_TR",
    "COURSE_OBJECTIVES_EN",
    "COURSE_LEARNING_OUTCOMES_BULLETS_TR",
    "COURSE_LEARNING_OUTCOMES_BULLETS_EN",
    "WEEKLY_COURSE_PLAN_TR",
    "WEEKLY_COURSE_PLAN_EN",
    "TEXTBOOK",
    "OTHER_REFERENCES",
    "HOMEWORK_AND_PROJECTS_TR",
    "HOMEWORK_AND_PROJECTS_EN",
    "COMPUTER_USAGE_TR",
    "COMPUTER_USAGE_EN",
    "PROGRAM_OUTCOMES",
]

def looks_like_yaml_format(text: str) -> bool:
    return "COURSE_CODE:" in text and "COURSE_DESCRIPTION" in text

def split_old_header_format(text: str):
    # Eski format: [COURSE CATALOG] / [COURSE DESCRIPTION] gibi başlıklar
    # Başlıklara göre böl
    parts = re.split(r"\n\[(.*?)\]\n", text)
    chunks = []
    if len(parts) == 1:
        return [text.strip()]

    # parts: [before, title1, body1, title2, body2...]
    before = parts[0].strip()
    if before:
        chunks.append(before)

    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        body = parts[i+1].strip() if i+1 < len(parts) else ""
        block = f"[{title}]\n{body}".strip()
        if block:
            chunks.extend(pack_by_size(block))

    return chunks

def parse_yaml_blocks(text: str):
    # Basit blok parser: "FIELD:" ile başlayan satırlar = yeni blok
    lines = text.splitlines()
    blocks = []
    cur_key = None
    cur_lines = []

    def flush():
        nonlocal cur_key, cur_lines
        if cur_key is not None:
            blocks.append((cur_key, "\n".join(cur_lines).strip()))
        cur_key = None
        cur_lines = []

    field_re = re.compile(r"^([A-Z0-9_]+):\s*(.*)$")

    for ln in lines:
        m = field_re.match(ln)
        if m and m.group(1) in FIELD_ORDER:
            flush()
            cur_key = m.group(1)
            cur_lines = [ln]
        else:
            if cur_key is None:
                # başta boş/saçma satırlar
                continue
            cur_lines.append(ln)

    flush()
    return blocks

def chunk_weekly_plan(block_text: str):
    # WEEKLY_COURSE_PLAN_* içindeki her "- { week: X, topics: ... }" satırını ayrı chunk yap
    weeks = []
    for ln in block_text.splitlines():
        ln2 = ln.strip()
        if ln2.startswith("- {") and "week:" in ln2:
            weeks.append(ln2)
    if not weeks:
        return pack_by_size(block_text)

    out = []
    header = block_text.splitlines()[0].strip()  # "WEEKLY_COURSE_PLAN_TR:" gibi
    for w in weeks:
        out.append(f"{header}\n{w}")
    return out

def pack_by_size(text: str):
    # Uzun blokları paragraf bazlı MAX_CHARS altında paketle
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out = []
    buf = ""
    for p in paras:
        if not buf:
            buf = p
        elif len(buf) + 2 + len(p) <= MAX_CHARS:
            buf += "\n\n" + p
        else:
            out.append(buf)
            buf = p
    if buf:
        out.append(buf)
    return out

def build_chunks(raw_text: str):
    raw_text = raw_text.strip()

    # 1) Eski başlıklı format
    if "[COURSE CATALOG]" in raw_text or "\n[COURSE DESCRIPTION]\n" in raw_text:
        return split_old_header_format(raw_text)

    # 2) YAML/field format
    if looks_like_yaml_format(raw_text):
        blocks = parse_yaml_blocks(raw_text)

        chunks = []
        for key, block in blocks:
            if key.startswith("WEEKLY_COURSE_PLAN_"):
                chunks.extend(chunk_weekly_plan(block))
            else:
                chunks.extend(pack_by_size(block))

        return chunks

    # 3) fallback: düz paragraf paketleme
    return pack_by_size(raw_text)

def fetch_catalogs(limit=2000):
    res = sb.table("course_catalogs").select("id, raw_text").limit(limit).execute()
    return res.data or []

def insert_chunks(course_id: int, chunks):
    rows = []
    for idx, txt in enumerate(chunks):
        rows.append({
            "course_id": course_id,
            "chunk_index": idx,
            "chunk_text": txt,
            "embedding": None
        })
    B = 200
    for i in range(0, len(rows), B):
        sb.table("course_chunks").insert(rows[i:i+B]).execute()

def main():
    catalogs = fetch_catalogs()
    print(f"Found {len(catalogs)} catalogs")

    total = 0
    for c in catalogs:
        cid = c["id"]
        raw = c["raw_text"]
        chunks = build_chunks(raw)
        insert_chunks(cid, chunks)
        total += len(chunks)
        print(f"✅ course_id={cid} -> {len(chunks)} chunks")

    print(f"Done. Total chunks inserted: {total}")

if __name__ == "__main__":
    main()