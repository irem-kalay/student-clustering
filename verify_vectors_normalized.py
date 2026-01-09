import pandas as pd
import os
import unicodedata

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VECTOR_FILE = os.path.join(BASE_DIR, "student_feature_vectors2.csv")
EXCEL_FOLDER = os.path.join(BASE_DIR, "data")

TARGET_STUDENT_ID = "Öğrenci Sınıf Listesi (97)"

GRADE_MAP = {
    "AA": 4.0, "BA": 3.5, "BB": 3.0, "CB": 2.5,
    "CC": 2.0, "DC": 1.5, "DD": 1.0,
    "FF": 0.0, "VF": 0.0, "F": 0.0
}

def normalize_unicode(s):
    return unicodedata.normalize("NFC", str(s)).strip()

def normalize_course(code):
    code = normalize_unicode(code)
    if code.endswith("E") and len(code) > 1 and code[-2].isdigit():
        return code[:-1]
    return code

def shrink_factor(attempts: int) -> float:
    # 1st: 1.0, 2nd: 0.9, 3rd: 0.8, 4th+: min 0.7
    return max(1.0 - 0.1 * (attempts - 1), 0.7)

def verify_bidirectional():
    if not os.path.exists(VECTOR_FILE):
        print(f"ERROR: CSV not found -> {VECTOR_FILE}")
        return

    df_vector = pd.read_csv(VECTOR_FILE, index_col=0)
    df_vector.index = [normalize_unicode(i) for i in df_vector.index]

    target_id = normalize_unicode(TARGET_STUDENT_ID)

    if target_id not in df_vector.index:
        print("ERROR: Student not found in CSV")
        print("Target:", repr(target_id))
        print("Sample indexes:", [repr(x) for x in df_vector.index[:10]])
        return

    student_vector = df_vector.loc[target_id]
    print(f"\n--- BIDIRECTIONAL VERIFICATION: {target_id} ---\n")

    excel_path = os.path.join(EXCEL_FOLDER, f"{target_id}.xlsx")
    if not os.path.exists(excel_path):
        print(f"ERROR: Excel not found -> {excel_path}")
        return

    df_excel = pd.read_excel(excel_path)
    df_excel.columns = [normalize_unicode(c) for c in df_excel.columns]

    if "Ders Kodu" not in df_excel.columns or "Harf Notu" not in df_excel.columns:
        print("ERROR: Excel must contain 'Ders Kodu' and 'Harf Notu' columns")
        print("Found columns:", list(df_excel.columns))
        return

    # -----------------------------
    # TEST 1: Excel -> Vector (shrink-aware)
    # -----------------------------
    print("TEST 1: Excel -> Vector (shrink-aware)")

    tmp = df_excel[["Ders Kodu", "Harf Notu"]].copy()
    tmp["Ders Kodu"] = tmp["Ders Kodu"].apply(normalize_course)

    def to_numeric(letter):
        letter = str(letter).split("/")[0].strip()
        return GRADE_MAP.get(letter, None)

    tmp["Numeric"] = tmp["Harf Notu"].apply(to_numeric)

    valid = tmp["Numeric"].notna()

    grouped = (
        tmp[valid]
        .groupby("Ders Kodu")["Numeric"]
        .agg(best_grade="max", attempts="count")
        .reset_index()
    )

    excel_courses = set(grouped["Ders Kodu"].tolist())
    mismatches = []

    for _, r in grouped.iterrows():
        course = r["Ders Kodu"]
        best = float(r["best_grade"])
        attempts = int(r["attempts"])
        expected = best * shrink_factor(attempts)

        if course in student_vector.index:
            vector_val = float(student_vector[course])
            if vector_val != -1 and abs(vector_val - expected) > 0.01:
                mismatches.append(
                    f"{course}: Expected={expected} (best={best}, attempts={attempts}), Vector={vector_val}"
                )
        else:
            print(f"WARNING: {course} missing in vector columns")

    if mismatches:
        print(f"ERROR: {len(mismatches)} mismatches found")
        for m in mismatches:
            print("  -", m)
    else:
        print("OK: Excel -> Vector consistent (including shrink factor)")

    # -----------------------------
    # TEST 2: Vector -> Excel (ghost courses)
    # -----------------------------
    print("\nTEST 2: Vector -> Excel")

    ghost_courses = []
    for col in student_vector.index:
        if float(student_vector[col]) != -1 and col not in excel_courses:
            ghost_courses.append(f"{col} (Vector={student_vector[col]})")

    if ghost_courses:
        print(f"ERROR: {len(ghost_courses)} ghost courses found")
        for g in ghost_courses:
            print("  -", g)
    else:
        print("OK: No ghost courses")

    print("\n" + "-" * 40)
    if not mismatches and not ghost_courses:
        print("RESULT: DATA IS FULLY CONSISTENT")
    else:
        print("RESULT: DATA INCONSISTENCIES FOUND")
    print("-" * 40)

if __name__ == "__main__":
    verify_bidirectional()
