import pandas as pd
import numpy as np
import os

# =============================================================================
# SABİT TANIMLAMALAR
# =============================================================================
TARGET_CATEGORIES = [
    'Basic_Sciences', 'Software_Practice', 'Algorithm_Theory',
    'Systems', 'Hardware', 'Social_Cultural', 'Language_Comm', 'Math_Calc'
]

CORE_COURSES = {
    'FIZ 101': {'Basic_Sciences': 0.8, 'Math_Calc': 0.2},
    'FIZ 101EL': {'Basic_Sciences': 1.0},
    'BLG 101': {'Basic_Sciences': 0.5, 'Software_Practice': 0.5},
    'BLG 113': {'Social_Cultural': 0.6, 'Systems': 0.4},
    'MAT 103': {'Math_Calc': 1.0},
    'MAT 281': {'Math_Calc': 0.8, 'Algorithm_Theory': 0.2},
    'ING 100': {'Language_Comm': 1.0},
    'BLG 112': {'Math_Calc': 0.6, 'Algorithm_Theory': 0.4},
    'BLG 102': {'Software_Practice': 0.8, 'Algorithm_Theory': 0.2},
    'MAT 104': {'Math_Calc': 1.0},
    'FIZ 102': {'Basic_Sciences': 0.8, 'Math_Calc': 0.2},
    'FIZ 102EL': {'Basic_Sciences': 1.0},
    'ING 112': {'Language_Comm': 1.0},
    'DAN 102': {'Social_Cultural': 1.0},
    'BLG 210': {'Math_Calc': 1.0},
    'BLG 231': {'Hardware': 1.0},
    'BLG 223': {'Algorithm_Theory': 0.5, 'Software_Practice': 0.5},
    'EHB 222': {'Hardware': 0.7, 'Basic_Sciences': 0.3},
    'EHB 211': {'Hardware': 0.6, 'Math_Calc': 0.4},
    'ING 201': {'Language_Comm': 1.0},
    'BLG 252': {'Software_Practice': 0.9, 'Algorithm_Theory': 0.1},
    'BLG 222': {'Hardware': 0.6, 'Systems': 0.4},
    'BLG 242': {'Hardware': 1.0},
    'BLG 202': {'Math_Calc': 0.7, 'Software_Practice': 0.3},
    'BLG 311': {'Algorithm_Theory': 0.7, 'Math_Calc': 0.3},
    'TUR 121': {'Social_Cultural': 1.0},
    'BLG 335': {'Algorithm_Theory': 0.6, 'Software_Practice': 0.2, 'Math_Calc': 0.2},
    'MAT 271': {'Math_Calc': 0.8, 'Algorithm_Theory': 0.2},
    'BLG 351': {'Hardware': 0.7, 'Software_Practice': 0.3},
    'TUR 122': {'Social_Cultural': 1.0},
    'BLG 317': {'Systems': 0.7, 'Software_Practice': 0.3},
    'BLG 212': {'Hardware': 0.5, 'Systems': 0.5},
    'EHB 311': {'Hardware': 1.0},
    'BLG 322': {'Hardware': 0.4, 'Systems': 0.6},
    'BLG 312': {'Systems': 0.8, 'Software_Practice': 0.2},
    'BLG 336': {'Algorithm_Theory': 0.7, 'Software_Practice': 0.2, 'Math_Calc': 0.1},
    'ATA 121': {'Social_Cultural': 1.0},
    'BLG 354': {'Math_Calc': 0.6, 'Systems': 0.4},
    'BLG 374': {'Language_Comm': 0.8, 'Social_Cultural': 0.2},
    'ATA 122': {'Social_Cultural': 1.0},
    'BLG 411': {'Software_Practice': 0.6, 'Systems': 0.2, 'Social_Cultural': 0.2},
    'BLG 4901': {'Software_Practice': 0.5, 'Systems': 0.3, 'Language_Comm': 0.2},
    'BLG 4902': {'Software_Practice': 0.4, 'Systems': 0.4, 'Language_Comm': 0.2},
    'EKO 201': {'Social_Cultural': 1.0},
}

ELECTIVE_POOLS = {
    'MT': {
        'BLG 413': {'Systems': 0.6, 'Software_Practice': 0.4},
        'BLG 430': {'Systems': 0.9, 'Software_Practice': 0.1},
        'BLG 433': {'Systems': 0.9, 'Algorithm_Theory': 0.1},
        'BLG 434': {'Algorithm_Theory': 0.7, 'Software_Practice': 0.3},
        'BLG 435': {'Algorithm_Theory': 0.6, 'Math_Calc': 0.2, 'Software_Practice': 0.2},
        'BLG 438': {'Hardware': 1.0},
        'BLG 439': {'Software_Practice': 0.6, 'Systems': 0.2},
        'BLG 440': {'Software_Practice': 0.6, 'Systems': 0.2},
        'BLG 443': {'Algorithm_Theory': 0.5, 'Math_Calc': 0.3, 'Systems': 0.2},
        'BLG 444': {'Software_Practice': 0.4, 'Math_Calc': 0.4, 'Algorithm_Theory': 0.2},
        'BLG 447': {'Systems': 0.4, 'Algorithm_Theory': 0.4},
        'BLG 449': {'Systems': 0.5, 'Software_Practice': 0.5},
        'BLG 450': {'Systems': 0.7, 'Software_Practice': 0.3},
        'BLG 451': {'Systems': 0.6, 'Hardware': 0.4},
        'BLG 452': {'Hardware': 1.0},
        'BLG 453': {'Algorithm_Theory': 0.4, 'Math_Calc': 0.4, 'Software_Practice': 0.2},
        'BLG 456': {'Hardware': 0.4, 'Algorithm_Theory': 0.3, 'Systems': 0.3},
        'BLG 458': {'Software_Practice': 0.5, 'Algorithm_Theory': 0.5},
        'BLG 459': {'Systems': 0.8, 'Algorithm_Theory': 0.2},
        'BLG 460': {'Software_Practice': 0.7, 'Systems': 0.3},
        'BLG 468': {'Software_Practice': 0.8, 'Systems': 0.2},
        'BLG 475': {'Software_Practice': 0.8, 'Social_Cultural': 0.2},
        'BLG 477': {'Systems': 0.5, 'Software_Practice': 0.3, 'Algorithm_Theory': 0.2},
        'BLG 478': {'Systems': 0.8, 'Algorithm_Theory': 0.2},
        'BLG 483': {'Algorithm_Theory': 0.5, 'Software_Practice': 0.3, 'Hardware': 0.2},
        'YZV 406': {'Hardware': 0.4, 'Algorithm_Theory': 0.4, 'Math_Calc': 0.2},
    },
    'TM': {
        'BLG 337': {'Systems': 0.8, 'Algorithm_Theory': 0.2},
        'BLG 368': {'Math_Calc': 0.7, 'Algorithm_Theory': 0.3},
        'BLG 442': {'Social_Cultural': 0.6, 'Language_Comm': 0.4},
        'BLG 448': {'Social_Cultural': 0.5, 'Language_Comm': 0.3},
        'BLG 454': {'Algorithm_Theory': 0.5, 'Math_Calc': 0.3, 'Software_Practice': 0.2},
        'KON 224': {'Hardware': 0.7, 'Math_Calc': 0.3},
        'KON 317': {'Math_Calc': 0.5, 'Hardware': 0.5},
        'MAL 201': {'Basic_Sciences': 1.0},
    },
    'SOCIAL': {
        'BLG 346': {'Social_Cultural': 0.7, 'Software_Practice': 0.3},
        'SNT': {'Social_Cultural': 1.0},
    }
}

# =============================================================================
# FONKSİYONLAR
# =============================================================================
def custom_normalization(df):
    df_norm = df.copy()
    mask_taken = df != -1

    for col in df.columns:
        taken_values = df.loc[mask_taken[col], col]
        if len(taken_values) > 0:
            min_val = taken_values.min()
            max_val = taken_values.max()

            if max_val == min_val:
                df_norm.loc[mask_taken[col], col] = 1.0
            else:
                scaled_values = 0.2 + 0.8 * ((taken_values - min_val) / (max_val - min_val))
                df_norm.loc[mask_taken[col], col] = scaled_values

    df_norm[~mask_taken] = 0.0
    return df_norm

def calculate_dynamic_features(df_norm, alpha=0.15, bonus_type="log"):
    targets = TARGET_CATEGORIES
    result = pd.DataFrame(0.0, index=df_norm.index, columns=targets)
    weight_counters = pd.DataFrame(0.0, index=df_norm.index, columns=targets)
    course_counts = pd.DataFrame(0, index=df_norm.index, columns=targets, dtype=int)

    def find_course_col(course_code):
        for col in df_norm.columns:
            if course_code == 'SNT' and col.startswith('SNT'):
                return col
            if course_code in col:
                return col
        return None

    for idx, row in df_norm.iterrows():
        # Core
        for course_code, weights in CORE_COURSES.items():
            found_col = find_course_col(course_code)
            if found_col is None: continue
            score = row[found_col]
            if score > 0:
                for cat, w in weights.items():
                    if cat in targets:
                        result.loc[idx, cat] += score * w
                        weight_counters.loc[idx, cat] += w
                        course_counts.loc[idx, cat] += 1

        # Electives
        for pool_name, pool_courses in ELECTIVE_POOLS.items():
            for course_code, weights in pool_courses.items():
                found_col = find_course_col(course_code)
                if found_col is None: continue
                score = row[found_col]
                if score > 0:
                    for cat, w in weights.items():
                        if cat in targets:
                            result.loc[idx, cat] += score * w
                            weight_counters.loc[idx, cat] += w
                            course_counts.loc[idx, cat] += 1

    avg_scores = result / weight_counters.replace(0, 1.0)

    if bonus_type == "none":
        return avg_scores

    max_counts = course_counts.max(axis=0).replace(0, 1)
    if bonus_type == "log":
        bonus = np.log1p(course_counts) / np.log1p(max_counts)
    elif bonus_type == "sqrt":
        bonus = np.sqrt(course_counts / max_counts)
    else:
        raise ValueError("bonus_type must be one of: 'log', 'sqrt', 'none'")

    final_scores = avg_scores * (1.0 + alpha * bonus)
    return final_scores

def load_and_process_data(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Hata: {csv_path} bulunamadı.")
        return None, None

    if 'Student_ID' in df.columns:
        student_ids = df['Student_ID']
        data = df.drop(columns=['Student_ID'])
    else:
        student_ids = df.index
        data = df

    data_normalized = custom_normalization(data)
    weighted_features = calculate_dynamic_features(data_normalized, alpha=0.15, bonus_type="log")
    
    # ID'leri geri ekleyelim
    weighted_features['Student_ID'] = student_ids
    
    return weighted_features, data_normalized

def extract_gender_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        if content:
            first_line = content.split('\n')[0].strip().lower()
            if 'erkek' in first_line or 'e' == first_line: return 'Erkek'
            elif any(x in first_line for x in ['kadın', 'kiz', 'k', 'f']): return 'Kadın'
        return 'Bilinmiyor'
    except:
        return 'Bilinmiyor'

def extract_year_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.read().strip().split('\n')
        if len(lines) >= 2:
            year_str = lines[1].strip()
            year_str_clean = ''.join(c for c in year_str if c.isdigit())
            if len(year_str_clean) >= 4:
                if year_str_clean.startswith('150'):
                    year = '20' + year_str_clean[3:5]
                    try:
                        year_int = int(year)
                        if 2000 <= year_int <= 2030: return year_int
                    except: pass
        return None
    except:
        return None