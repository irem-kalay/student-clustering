import pandas as pd
import os
import unicodedata
import numpy as np 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# File paths
VECTOR_FILE = os.path.join(BASE_DIR, "student_feature_vectors.csv") 
EXCEL_FOLDER = os.path.join(BASE_DIR, "data")

TARGET_STUDENT_ID = "Öğrenci Sınıf Listesi (97)"

GRADE_MAP = {
    "AA": 4.0, "BA": 3.5, "BB": 3.0, "CB": 2.5,
    "CC": 2.0, "DC": 1.5, "DD": 1.0,
    "FF": 0.0, "VF": 0.0, "F": 0.0,
    "BL": -1 
}

def normalize_unicode(s):
    return unicodedata.normalize("NFC", str(s)).strip()

def normalize_course(code):
    code = normalize_unicode(code)
    # Merges codes like "MAT103E" into "MAT103"
    if code.endswith("E") and len(code) > 1 and code[-2].isdigit():
        return code[:-1]
    return code

def shrink_factor(attempts: int) -> float:
    # 1st attempt: 1.0, 2nd: 0.9, 3rd: 0.8... min: 0.7
    return max(1.0 - 0.1 * (attempts - 1), 0.7)

def credit_weight(credit: float) -> float:
    return np.sqrt(max(float(credit), 1))

def verify_full_check():
    if not os.path.exists(VECTOR_FILE):
        print(f"ERROR: CSV not found -> {VECTOR_FILE}")
        return

    df_vector = pd.read_csv(VECTOR_FILE, index_col=0)
    # Normalize indices
    df_vector.index = [normalize_unicode(i) for i in df_vector.index]

    target_id = normalize_unicode(TARGET_STUDENT_ID)

    if target_id not in df_vector.index:
        print("ERROR: Student not found in CSV")
        return

    student_vector = df_vector.loc[target_id]
    print(f"\n--- FULL COMPREHENSIVE VERIFICATION: {target_id} ---\n")

    excel_path = os.path.join(EXCEL_FOLDER, f"{target_id}.xlsx")
    if not os.path.exists(excel_path):
        print(f"ERROR: Excel file not found -> {excel_path}")
        return

    df_excel = pd.read_excel(excel_path)
    df_excel.columns = [normalize_unicode(c) for c in df_excel.columns]

    # Check for Credit column
    if "Kredi" not in df_excel.columns:
        print("ERROR: Column 'Kredi' not found in Excel file!")
        return

    # --- DATA PREPARATION ---
    tmp = df_excel[["Ders Kodu", "Harf Notu", "Kredi"]].copy()
    tmp["Ders Kodu"] = tmp["Ders Kodu"].apply(normalize_course)

    def to_numeric(letter):
        if pd.isna(letter): return None
        letter = str(letter).split("/")[0].strip()
        return GRADE_MAP.get(letter, None)

    tmp["Numeric"] = tmp["Harf Notu"].apply(to_numeric)
    
    # Filter valid grades (ignore -1/BL)
    valid = tmp["Numeric"].notna() & (tmp["Numeric"] != -1)

    grouped = (
        tmp[valid]
        .groupby("Ders Kodu")
        .agg(
            best_grade=("Numeric", "max"),
            attempts=("Numeric", "count"),
            credit=("Kredi", "max")
        )
        .reset_index()
    )

    # List of taken courses in Excel (Set for fast lookup)
    excel_taken_courses = set(grouped["Ders Kodu"].tolist())

    # ==========================================
    # TEST 1: EXCEL -> VECTOR CHECK
    # (Are grades in Excel correctly transferred to the Vector?)
    # ==========================================
    print("TEST 1: Excel -> Vector Check (Are Scores Correct?)")
    mismatches = []
    
    for _, r in grouped.iterrows():
        course = r["Ders Kodu"]
        best = float(r["best_grade"])
        attempts = int(r["attempts"])
        credit = float(r["credit"])
        
        # Formula: BestGrade * ShrinkFactor * sqrt(Credit)
        expected = best * shrink_factor(attempts) * credit_weight(credit)

        if course in student_vector.index:
            vector_val = float(student_vector[course])
            if vector_val != -1 and abs(vector_val - expected) > 0.01:
                mismatches.append(
                    f"{course}: Expected={expected:.4f}, InVector={vector_val:.4f}"
                )
        else:
            print(f"WARNING: {course} not found in vector columns!")

    if mismatches:
        print(f"❌ ERROR: {len(mismatches)} score mismatches found:")
        for m in mismatches: print("  -", m)
    else:
        print("✅ OK: All courses in Excel are correctly transferred to the vector.")

    print("-" * 30)

    # ==========================================
    # TEST 2: VECTOR -> EXCEL CHECK
    # (Are there 'Ghost Courses' in Vector that are NOT in Excel?)
    # ==========================================
    print("TEST 2: Vector -> Excel Check (Are There Ghost Courses?)")
    ghost_courses = []

    # Iterate through every column in the vector
    for course_code, vector_val in student_vector.items():
        vector_val = float(vector_val)
        
        # If vector says course is taken (not -1.0)
        if vector_val != -1.0:
            # But it is NOT in the Excel list
            if course_code not in excel_taken_courses:
                ghost_courses.append(f"{course_code}: InVector={vector_val:.4f} but NOT in Excel!")

    if ghost_courses:
        print(f"❌ ERROR: {len(ghost_courses)} 'Ghost Courses' found (Present in Vector, Missing in Excel):")
        for g in ghost_courses: print("  -", g)
    else:
        print("✅ OK: No extra/erroneous courses found in the vector.")

    print("\n" + "="*40)
    if not mismatches and not ghost_courses:
        print("RESULT: DATA IS 100% CONSISTENT 🎯")
    else:
        print("RESULT: DATA INCONSISTENCIES FOUND ⚠️")
    print("="*40)

if __name__ == "__main__":
    verify_full_check()