import pandas as pd
import numpy as np
import re

# -----------------------------
# FILE PATHS (EDIT HERE)
# -----------------------------
VECTORS_FILE = "feature-vectors/vector-1400.csv"
DENKLIK_FILE = "DersDenklikleri/dersdenklikleri.csv"
OUTPUT_FILE  = "feature-vectors/vector-1400-equivalent.csv"

ID_COL = "Student_ID"
MISSING_VALUE = -1.0
MERGE_MODE = "max"   # "max" or "mean"

# -----------------------------
# Helpers
# -----------------------------
def normalize_course(code: str) -> str:
    """
    Remove trailing 'E' if the course code ends with 'E'
    and the character before it is a digit.
    Example: BLG 101E -> BLG 101
    """
    code = str(code).strip()
    if code.endswith('E') and len(code) > 1 and code[-2].isdigit():
        return code[:-1]
    return code

def variants(code: str) -> set:
    """
    Generate both E and non-E variants of a course code.
    Example: BLG 101 <-> BLG 101E
    """
    c = normalize_course(code)
    return {c, c + "E"}

def load_equivalency_map(denk_csv_path: str) -> dict:
    """
    Expected format of dersdenklikleri.csv:
      Ders Kodu, Ders Denklikleri
      FIZ 101E, "FIZ 111, GUV 103, ..."

    Output:
      variant_code -> canonical_code (canonical is normalized)
    """
    df = pd.read_csv(denk_csv_path, encoding="cp1254")
    df.columns = [c.strip() for c in df.columns]

    required = {"Ders Kodu", "Ders Denklikleri"}
    if not required.issubset(df.columns):
        raise ValueError(f"Equivalency file must contain columns: {required}")

    equivalency_map = {}

    for _, row in df.iterrows():
        canonical_raw = str(row["Ders Kodu"]).strip()
        canonical = normalize_course(canonical_raw)

        # Add canonical course itself (E / non-E variants)
        for v in variants(canonical_raw):
            equivalency_map[v] = canonical

        equivalents = row["Ders Denklikleri"]
        if pd.isna(equivalents):
            continue

        parts = re.split(r"[;,]", str(equivalents))
        for p in parts:
            code = p.strip().strip('"').strip("'")
            if not code:
                continue
            for v in variants(code):
                equivalency_map[v] = canonical

    return equivalency_map

def map_col_to_canonical(col: str, eq_map: dict):
    """
    Map a feature column (course code) to its canonical course.
    Returns:
      canonical course code if matched
      None if not found in equivalency list
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

    # Map each course column to its canonical course
    col_to_canon = {c: map_col_to_canonical(c, eq_map) for c in course_cols}

    # Keep only columns that exist in equivalency list
    mapped_cols = [c for c in course_cols if col_to_canon[c] is not None]
    if not mapped_cols:
        raise ValueError("No columns matched the equivalency list.")

    work = df[mapped_cols].copy()

    # Ensure numeric values
    for c in mapped_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    # Group original columns by canonical course
    canon_to_cols = {}
    for c in mapped_cols:
        canon = col_to_canon[c]
        canon_to_cols.setdefault(canon, []).append(c)

    # Build collapsed feature matrix
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
    print("PROCESS COMPLETED")
    print(f"Input shape:  {df.shape}")
    print(f"Output shape: {out_df.shape}")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"Kept {len(out_df.columns) - 1} canonical course columns.")
    print("-" * 40)

if __name__ == "__main__":
    run()
