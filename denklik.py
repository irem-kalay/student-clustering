import pandas as pd
import numpy as np
import re

# -----------------------------
# FILE PATHS (BURAYI DOLDUR)
# -----------------------------
VECTORS_FILE = "feature-vectors/student_feature_vectors2.csv"
DENKLIK_FILE = "DersDenklikleri/dersdenklikleri.csv"
OUTPUT_FILE  = "feature-vectors/student_feature_vectors2_collapsed.csv"

ID_COL = "Student_ID"
MISSING_VALUE = -1.0
MERGE_MODE = "max"   # "max" veya "mean"

# -----------------------------
# Helpers
# -----------------------------
def normalize_course(code: str) -> str:
    """
    '...E' ile biten ve sondan 1 önce digit olan kodlarda E'yi at.
    Örn: BLG 101E -> BLG 101
    """
    code = str(code).strip()
    if code.endswith('E') and len(code) > 1 and code[-2].isdigit():
        return code[:-1]
    return code

def variants(code: str) -> set:
    """
    E'li/E'siz varyantları üret.
    Örn: BLG 101 <-> BLG 101E
    """
    c = normalize_course(code)
    return {c, c + "E"}

def load_equivalency_map(denk_csv_path: str) -> dict:
    """
    dersdenklikleri.csv beklenen format:
      Ders Kodu, Ders Denklikleri
      FIZ 101E, "FIZ 111, GUV 103, ..."
    Çıktı:
      variant_code -> canonical_code (canonical = normalize edilmiş)
    """
    df = pd.read_csv(denk_csv_path, encoding="cp1254")
    df.columns = [c.strip() for c in df.columns]

    required = {"Ders Kodu", "Ders Denklikleri"}
    if not required.issubset(df.columns):
        raise ValueError(f"Denklik dosyasında şu kolonlar olmalı: {required}")

    m = {}

    for _, row in df.iterrows():
        canonical_raw = str(row["Ders Kodu"]).strip()
        canonical = normalize_course(canonical_raw)

        # canonical'ın kendisi (E'li/E'siz)
        for v in variants(canonical_raw):
            m[v] = canonical

        denks = row["Ders Denklikleri"]
        if pd.isna(denks):
            continue

        parts = re.split(r"[;,]", str(denks))
        for p in parts:
            code = p.strip().strip('"').strip("'")
            if not code:
                continue
            for v in variants(code):
                m[v] = canonical

    return m

def map_col_to_canonical(col: str, eq_map: dict):
    """
    Bir sütun adı ders koduysa:
      - eşleşirse canonical döndür
      - yoksa None
    """
    c = str(col).strip()

    if c in eq_map:
        return eq_map[c]

    c_noe = normalize_course(c)
    if c_noe in eq_map:
        return eq_map[c_noe]

    c_e = c_noe + "E"
    if c_e in eq_map:
        return eq_map[c_e]

    return None

# -----------------------------
# Main
# -----------------------------
def run():
    eq_map = load_equivalency_map(DENKLIK_FILE)

    df = pd.read_csv(VECTORS_FILE, encoding="cp1254")
    if ID_COL not in df.columns:
        raise ValueError(f"ID column '{ID_COL}' not found in vectors CSV.")

    id_series = df[ID_COL]
    course_cols = [c for c in df.columns if c != ID_COL]

    # course col -> canonical (or None)
    col_to_canon = {c: map_col_to_canonical(c, eq_map) for c in course_cols}

    # keep only mapped columns
    mapped_cols = [c for c in course_cols if col_to_canon[c] is not None]
    if not mapped_cols:
        raise ValueError("No columns matched the equivalency list. Check denklik file or column naming.")

    work = df[mapped_cols].copy()

    # make sure numbers
    for c in mapped_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    # group original columns by canonical
    canon_to_cols = {}
    for c in mapped_cols:
        canon = col_to_canon[c]
        canon_to_cols.setdefault(canon, []).append(c)

    # build new collapsed dataframe
    out_df = pd.DataFrame({ID_COL: id_series})

    for canon, cols in sorted(canon_to_cols.items()):
        sub = work[cols].replace(MISSING_VALUE, np.nan)

        if MERGE_MODE == "max":
            agg = sub.max(axis=1, skipna=True)
        elif MERGE_MODE == "mean":
            agg = sub.mean(axis=1, skipna=True)
        else:
            raise ValueError("MERGE_MODE must be 'max' or 'mean'.")

        out_df[canon] = agg.fillna(MISSING_VALUE)

    out_df.to_csv(OUTPUT_FILE, index=False)

    print("-" * 40)
    print("DONE")
    print(f"Input shape:  {df.shape}")
    print(f"Output shape: {out_df.shape}")
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Kept {len(out_df.columns)-1} canonical course columns.")
    print("-" * 40)

if __name__ == "__main__":
    run()
