import pandas as pd
import glob
import os
import numpy as np

# -------------------------------------------------
# SETTINGS
# -------------------------------------------------
DATA_FOLDER = "data"
OUTPUT_FILE = "student_feature_vectors2.csv"

# -------------------------------------------------
# GRADE MAP
# -------------------------------------------------
grade_map = {
    'AA': 4.0, 'BA+': 3.75, 'BA': 3.5, 'BB+': 3.25, 'BB': 3.0,
    'CB+': 2.75, 'CB': 2.5, 'CC+': 2.25, 'CC': 2.0,
    'DC+': 1.75, 'DC': 1.5, 'DD+': 1.25, 'DD': 1.0,
    'FF': 0.0, 'VF': 0.0, 'F': 0.0,
    'BL': -1
}

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def clean_grade(grade_str):
    if pd.isna(grade_str):
        return np.nan
    if not isinstance(grade_str, str):
        return grade_str
    grade_text = grade_str.split('/')[0].strip()
    return grade_map.get(grade_text, np.nan)

def normalize_course(code):
    code = str(code).strip()
    if code.endswith('E') and len(code) > 1 and code[-2].isdigit():
        return code[:-1]
    return code

def shrink_factor(attempts: int) -> float:
    return max(1.0 - 0.1 * (attempts - 1), 0.7)

def credit_weight(credit: float) -> float:
    return np.sqrt(max(credit, 1))

# -------------------------------------------------
# MAIN PROCESS
# -------------------------------------------------
def process_student_files():

    file_list = glob.glob(os.path.join(DATA_FOLDER, "*.xlsx"))
    if not file_list:
        print(f"No Excel files found in '{DATA_FOLDER}'")
        return

    print(f"Processing {len(file_list)} student files...")
    all_students_data = []

    for filepath in file_list:
        student_id = os.path.splitext(os.path.basename(filepath))[0]

        try:
            df = pd.read_excel(filepath)
            df.columns = [c.strip() for c in df.columns]

            required_cols = {'Ders Kodu', 'Harf Notu', 'Kredi'}
            if not required_cols.issubset(df.columns):
                print(f"SKIPPED: {student_id} (Missing columns)")
                continue

            temp_df = df[['Ders Kodu', 'Harf Notu', 'Kredi']].copy()
            temp_df['Ders Kodu'] = temp_df['Ders Kodu'].apply(normalize_course)
            temp_df['Numeric_Grade'] = temp_df['Harf Notu'].apply(clean_grade)

            valid = temp_df['Numeric_Grade'].notna() & (temp_df['Numeric_Grade'] != -1)

            agg = (
                temp_df[valid]
                .groupby('Ders Kodu')
                .agg(
                    best_grade=('Numeric_Grade', 'max'),
                    attempts=('Numeric_Grade', 'count'),
                    credit=('Kredi', 'max')
                )
                .reset_index()
            )

            # Final weighted grade (NO 0–4 normalization)
            agg['Numeric_Grade'] = agg.apply(
                lambda r: (
                    r['best_grade']
                    * shrink_factor(int(r['attempts']))
                    * credit_weight(float(r['credit']))
                ),
                axis=1
            )

            agg['Student_ID'] = student_id
            all_students_data.append(
                agg[['Student_ID', 'Ders Kodu', 'Numeric_Grade']]
            )

        except Exception as e:
            print(f"ERROR in {student_id}: {e}")

    if not all_students_data:
        print("No data processed.")
        return

    # -------------------------------------------------
    # VECTOR CREATION
    # -------------------------------------------------
    full_data = pd.concat(all_students_data, ignore_index=True)

    feature_matrix = full_data.pivot(
        index='Student_ID',
        columns='Ders Kodu',
        values='Numeric_Grade'
    )

    feature_matrix = feature_matrix.fillna(-1)
    feature_matrix = feature_matrix.reindex(
        sorted(feature_matrix.columns), axis=1
    )

    feature_matrix.to_csv(OUTPUT_FILE)

    print("-" * 40)
    print("PROCESS COMPLETED")
    print(f"Matrix shape: {feature_matrix.shape}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("-" * 40)

if __name__ == "__main__":
    process_student_files()
