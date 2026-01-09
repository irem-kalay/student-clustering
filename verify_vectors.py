import pandas as pd
import os
import numpy as np
import unicodedata

# -------------------------------------------------
# PORTABLE PATHS (works on every machine)
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VECTOR_FILE = os.path.join(BASE_DIR, "student_feature_vectors2.csv")
EXCEL_FOLDER = os.path.join(BASE_DIR, "data")

TARGET_STUDENT_ID = "Öğrenci Sınıf Listesi (97)"  # must match the .xlsx filename (without extension)

# -------------------------------------------------
# GRADE MAP (must match your vector creation logic)
# -------------------------------------------------
GRADE_MAP = {
    "AA": 4.0, "BA+": 3.75, "BA": 3.5, "BB+": 3.25, "BB": 3.0,
    "CB+": 2.75, "CB": 2.5, "CC+": 2.25, "CC": 2.0,
    "DC+": 1.75, "DC": 1.5, "DD+": 1.25, "DD": 1.0,
    "FF": 0.0, "VF": 0.0, "F": 0.0,
    "BL": -1,
}

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def normalize_unicode(s: str) -> str:
    return unicodedata.normalize("NFC", str(s)).strip()

def normalize_course(code: str) -> str:
    code = normalize_unicode(code)
    if code.endswith("E") and len(code) > 1 and code[-2].isdigit():
        return code[:-1]
    return code

def clean_grade(grade_str):
    if pd.isna(grade_str):
        return np.nan
    if not isinstance(grade_str, str):
        return grade_str

    grade_text = grade_str.split("/")[0].strip()
    return GRADE_MAP.get(grade_text, np.nan)

def shrink_factor(attempts: int) -> float:
    # 1st: 1.0, 2nd: 0.9, 3rd: 0.8, 4th+: min 0.7
    return max(1.0 - 0.1 * (attempts - 1), 0.7)

# -------------------------------------------------
# MAIN VERIFY (Excel -> Vector, shrink-aware)
# -------------------------------------------------
def verify_student_vector():
    if not os.path.exists(VECTOR_FILE):
        print(f"ERROR: Vector CSV not found -> {VECTOR_FILE}")
        return

    df_vector = pd.read_csv(VECTOR_FILE, index_col=0)

    # normalize student ids in CSV
    df_vector.index = [normalize_unicode(i) for i in df_vector.index]
    target_id = normalize_unicode(TARGET_STUDENT_ID)

    if target_id not in df_vector.index:
        print(f"ERROR: Student not found in vector CSV -> {repr(target_id)}")
        print("Sample CSV indexes:", [repr(x) for x in df_vector.index[:10]])
        return

    student_vector = df_vector.loc[target_id]
    print(f"--- Checking: {target_id} ---\n")

    excel_path = os.path.join(EXCEL_FOLDER, f"{target_id}.xlsx")
    if not os.path.exists(excel_path):
        print(f"ERROR: Excel file not found -> {excel_path}")
        return

    try:
        df_excel = pd.read_excel(excel_path)
    except Exception as e:
        print(f"ERROR: Excel could not be read -> {e}")
        return

    df_excel.columns = [normalize_unicode(c) for c in df_excel.columns]

    if "Ders Kodu" not in df_excel.columns or "Harf Notu" not in df_excel.columns:
        print("ERROR: Excel must contain 'Ders Kodu' and 'Harf Notu' columns")
        print("Found columns:", list(df_excel.columns))
        return

    # Build expected values from Excel using the SAME rules as vector creation
    tmp = df_excel[["Ders Kodu", "Harf Notu"]].copy()
    tmp["Ders Kodu"] = tmp["Ders Kodu"].apply(normalize_course)
    tmp["Numeric_Grade"] = tmp["Harf Notu"].apply(clean_grade)

    # attempts should count only real grades (exclude NaN and BL=-1)
    valid = tmp["Numeric_Grade"].notna() & (tmp["Numeric_Grade"] != -1)

    grouped = (
        tmp[valid]
        .groupby("Ders Kodu")["Numeric_Grade"]
        .agg(best_grade="max", attempts="count")
        .reset_index()
    )

    mismatches = []
    matches = 0
    missing_columns = 0

    for _, r in grouped.iterrows():
        course = r["Ders Kodu"]
        best = float(r["best_grade"])
        attempts = int(r["attempts"])
        expected = best * shrink_factor(attempts)

        if course not in student_vector.index:
            missing_columns += 1
            continue

        vector_val = float(student_vector[course])

        # if vector uses -1 as missing course, treat -1 as mismatch (because Excel says it exists)
        if vector_val == -1:
            mismatches.append(f"{course}: Expected={expected} (best={best}, attempts={attempts}), Vector=-1 (missing)")
            continue

        if abs(vector_val - expected) > 0.01:
            mismatches.append(f"{course}: Expected={expected} (best={best}, attempts={attempts}), Vector={vector_val}")
        else:
            matches += 1

    # Report
    print("-" * 40)
    print(f"Matched courses: {matches}")
    if missing_columns:
        print(f"WARNING: {missing_columns} courses exist in Excel but missing as columns in vector CSV.")
    if mismatches:
        print(f"ERROR: {len(mismatches)} mismatches found:")
        for m in mismatches:
            print("  -", m)
    else:
        print("OK: Excel grades and vector values match (including shrink factor).")
    print("-" * 40)

if __name__ == "__main__":
    verify_student_vector()
