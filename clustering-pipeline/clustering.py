import pandas as pd
import numpy as np
import re
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from math import pi
from sklearn.decomposition import PCA
import seaborn as sns

# =============================================================================
# 1. DATA LOADING AND PREPARATION
# =============================================================================
try:
    df = pd.read_csv('feature-vectors/vector-1400-equivalent.csv')
except FileNotFoundError:
    df = pd.read_csv('vector-1400-equivalent.csv')

if 'Student_ID' in df.columns:
    student_ids = df['Student_ID']
    data = df.drop(columns=['Student_ID'])
else:
    student_ids = df.index
    data = df

# =============================================================================
# 2. NORMALIZATION
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

data_normalized = custom_normalization(data)

# =============================================================================
# 3. COURSE CATEGORY DEFINITIONS (POOLS)
# =============================================================================
core_courses = {
    # --- 1. YARIYIL ---
    'FIZ 101': {'Basic_Sciences': 0.8, 'Math_Calc': 0.2},       # Physics I
    'FIZ 101EL': {'Basic_Sciences': 1.0},                       # Physics I Lab
    'BLG 101': {'Basic_Sciences': 0.5, 'Software_Practice': 0.5}, # Intr. to Info Systems
    'BLG 113': {'Social_Cultural': 0.6, 'Systems': 0.4},        # Intr.to Comp.Eng. & Ethics (Etik ağırlıklı)
    'MAT 103': {'Math_Calc': 1.0},                              # Mathematics I
    'MAT 281': {'Math_Calc': 0.8, 'Algorithm_Theory': 0.2},     # Linear Algebra
    'ING 100': {'Language_Comm': 1.0},                          # EAP Through Global Goals

    # --- 2. YARIYIL ---
    'BLG 112': {'Math_Calc': 0.6, 'Algorithm_Theory': 0.4},     # Discrete Mathematics
    'BLG 102': {'Software_Practice': 0.8, 'Algorithm_Theory': 0.2}, # Intr to Sci&Eng Comp (C)
    'MAT 104': {'Math_Calc': 1.0},                              # Mathematics II
    'FIZ 102': {'Basic_Sciences': 0.8, 'Math_Calc': 0.2},       # Physics II
    'FIZ 102EL': {'Basic_Sciences': 1.0},                       # Physics II Lab
    'ING 112': {'Language_Comm': 1.0},                          # Basics of Academic Writing
    'DAN 102': {'Social_Cultural': 1.0},                        # Girişimcilik & Kariyer

    # --- 3. YARIYIL ---
    'BLG 210': {'Math_Calc': 1.0},                              # Engineering Mathematics
    'BLG 231': {'Hardware': 1.0},                               # Digital Circuits
    'BLG 223': {'Algorithm_Theory': 0.5, 'Software_Practice': 0.5}, # Data Structures
    'EHB 222': {'Hardware': 0.7, 'Basic_Sciences': 0.3},        # Introduction to Electronics
    'EHB 211': {'Hardware': 0.6, 'Math_Calc': 0.4},             # Basics of Electrical Circuits
    'ING 201': {'Language_Comm': 1.0},                          # Essentials of Res.Paper Writ.

    # --- 4. YARIYIL ---
    'BLG 252': {'Software_Practice': 0.9, 'Algorithm_Theory': 0.1}, # Object Oriented Programming
    'BLG 222': {'Hardware': 0.6, 'Systems': 0.4},               # Computer Organization
    'BLG 242': {'Hardware': 1.0},                               # Logic Circuits Laboratory
    'BLG 202': {'Math_Calc': 0.7, 'Software_Practice': 0.3},    # Numerical Methods
    'BLG 311': {'Algorithm_Theory': 0.7, 'Math_Calc': 0.3},     # Formal Languages and Automata (Teori)
    'TUR 121': {'Social_Cultural': 1.0},                        # Türk Dili I

    # --- 5. YARIYIL ---
    'BLG 335': {'Algorithm_Theory': 0.6, 'Software_Practice': 0.2, 'Math_Calc': 0.2}, # Analysis of Algo I
    'MAT 271': {'Math_Calc': 0.8, 'Algorithm_Theory': 0.2},     # Probability and Statistics
    'BLG 351': {'Hardware': 0.7, 'Software_Practice': 0.3},     # Microcomputer Lab
    'TUR 122': {'Social_Cultural': 1.0},                        # Türk Dili II
    'BLG 317': {'Systems': 0.7, 'Software_Practice': 0.3},     # Database Systems
    'BLG 212': {'Hardware': 0.5, 'Systems': 0.5},               # Microprocessor Systems
    'EHB 311': {'Hardware': 1.0},                               # Intr.to Electronics Laboratory

    # --- 6. YARIYIL ---
    'BLG 322': {'Hardware': 0.4, 'Systems': 0.6},               # Computer Architecture
    'BLG 312': {'Systems': 0.8, 'Software_Practice': 0.2},      # Operating Systems
    'BLG 336': {'Algorithm_Theory': 0.7, 'Software_Practice': 0.2, 'Math_Calc': 0.1}, # Analysis of Algo II
    'ATA 121': {'Social_Cultural': 1.0},                        # Atatürk İlk & İnkılap Trh I
    'BLG 354': {'Math_Calc': 0.6, 'Systems': 0.4},              # Signal & Systems (Matematik/Sistem ağırlıklı)
    'BLG 374': {'Language_Comm': 0.8, 'Social_Cultural': 0.2}, # Tech. Communic.

    # --- 7. YARIYIL ---
    'ATA 122': {'Social_Cultural': 1.0},                        # Atatürk İlk & İnkılap Trh II
    'BLG 411': {'Software_Practice': 0.6, 'Systems': 0.2, 'Social_Cultural': 0.2}, # Software Engineering
    'BLG 4901': {'Software_Practice': 0.5, 'Systems': 0.3, 'Language_Comm': 0.2}, # Design I (Proje)

    # --- 8. YARIYIL ---
    'BLG 4902': {'Software_Practice': 0.4, 'Systems': 0.4, 'Language_Comm': 0.2}, # Design II (Bitirme)
    'EKO 201': {'Social_Cultural': 1.0},                        # Economics
}
# 3.2 ELECTIVE POOLS (Seçmeli Ders Havuzları - Sadece Alınanlar Hesaplanır)
elective_pools = {
    # --- MT Courses ---
    'MT': {
        'BLG 413': {'Systems': 0.6, 'Software_Practice': 0.4}, # System Programming
        'BLG 430': {'Systems': 0.9, 'Software_Practice': 0.1}, # Computer Networks
        'BLG 433': {'Systems': 0.9, 'Algorithm_Theory': 0.1}, # Computer Communications
        'BLG 434': {'Algorithm_Theory': 0.7, 'Software_Practice': 0.3}, # Expert Systems
        'BLG 435': {'Algorithm_Theory': 0.6, 'Math_Calc': 0.2, 'Software_Practice': 0.2}, # AI
        'BLG 438': {'Hardware': 1.0}, # Digital System Design Lab
        'BLG 439': {'Software_Practice': 0.6, 'Systems': 0.2}, # Computer Project I
        'BLG 440': {'Software_Practice': 0.6, 'Systems': 0.2}, # Computer Project II
        'BLG 443': {'Algorithm_Theory': 0.5, 'Math_Calc': 0.3, 'Systems': 0.2}, # Discrete Event Sim
        'BLG 444': {'Software_Practice': 0.4, 'Math_Calc': 0.4, 'Algorithm_Theory': 0.2}, # Graphics
        'BLG 447': {'Systems': 0.4, 'Algorithm_Theory': 0.4}, # Compiler Design
        'BLG 449': {'Systems': 0.5, 'Software_Practice': 0.5}, # Parallel Dist.
        'BLG 450': {'Systems': 0.7, 'Software_Practice': 0.3}, # Real-Time Soft
        'BLG 451': {'Systems': 0.6, 'Hardware': 0.4}, # Real-Time Sys
        'BLG 452': {'Hardware': 1.0}, # Microprocessor Lab
        'BLG 453': {'Algorithm_Theory': 0.4, 'Math_Calc': 0.4, 'Software_Practice': 0.2}, # Vision
        'BLG 456': {'Hardware': 0.4, 'Algorithm_Theory': 0.3, 'Systems': 0.3}, # Robotics
        'BLG 458': {'Software_Practice': 0.5, 'Algorithm_Theory': 0.5}, # Functional Prog
        'BLG 459': {'Systems': 0.8, 'Algorithm_Theory': 0.2}, # Computer Security
        'BLG 460': {'Software_Practice': 0.7, 'Systems': 0.3}, # Secure Programming
        'BLG 468': {'Software_Practice': 0.8, 'Systems': 0.2}, # OO Modeling
        'BLG 475': {'Software_Practice': 0.8, 'Social_Cultural': 0.2}, # Quality Testing
        'BLG 477': {'Systems': 0.5, 'Software_Practice': 0.3, 'Algorithm_Theory': 0.2}, # Multimedia
        'BLG 478': {'Systems': 0.8, 'Algorithm_Theory': 0.2}, # Network Security
        'BLG 483': {'Algorithm_Theory': 0.5, 'Software_Practice': 0.3, 'Hardware': 0.2}, # AI Aided CE
        'YZV 406': {'Hardware': 0.4, 'Algorithm_Theory': 0.4, 'Math_Calc': 0.2}, # Robotics
    },
    
    # --- TM Courses ---
    'TM': {
        'BLG 337': {'Systems': 0.8, 'Algorithm_Theory': 0.2}, # Principles of Comp Comm
        'BLG 368': {'Math_Calc': 0.7, 'Algorithm_Theory': 0.3}, # Operations Research
        'BLG 442': {'Social_Cultural': 0.6, 'Language_Comm': 0.4}, # Tech Innov Mng
        'BLG 448': {'Social_Cultural': 0.5, 'Language_Comm': 0.3}, # Project Mng
        'BLG 454': {'Algorithm_Theory': 0.5, 'Math_Calc': 0.3, 'Software_Practice': 0.2}, # Learning From Data
        'KON 224': {'Hardware': 0.7, 'Math_Calc': 0.3}, # Measurement
        'KON 317': {'Math_Calc': 0.5, 'Hardware': 0.5}, # Control Systems
        'MAL 201': {'Basic_Sciences': 1.0}, # Materials Science
    },
    
    # --- ITB / SNT  Elective Courses ---
    'SOCIAL': {
        'BLG 346': {'Social_Cultural': 0.7, 'Software_Practice': 0.3}, # Visual Composition
        'SNT': {'Social_Cultural': 1.0}, # Covers ALL SNT courses (Regex matched)
    }
}

target_categories = ['Basic_Sciences', 'Software_Practice', 'Algorithm_Theory', 
                     'Systems', 'Hardware', 'Social_Cultural', 'Language_Comm', 'Math_Calc']

# =============================================================================
# 4. DYNAMIC SCORE CALCULATION ENGINE
# =============================================================================
def calculate_dynamic_features(df_norm, core_map, elective_pools, targets):
    result = pd.DataFrame(0.0, index=df_norm.index, columns=targets)
    weight_counters = pd.DataFrame(0.0, index=df_norm.index, columns=targets)
    
    for idx, row in df_norm.iterrows():
        
        # --- A. CORE COURSES ---
        for course_code, weights in core_map.items():
            # Find column
            found_col = None
            for col in df_norm.columns:
                if course_code in col:
                    found_col = col
                    break
            
            if not found_col:
                for col in df_norm.columns:
                    prefix = re.match(r"([A-Z]+)", col)
                    if prefix and prefix.group(1) == course_code: 
                        found_col = col
                        break

            if found_col:
                score = row[found_col]
                # Core courses always count
                for cat, w in weights.items():
                    if cat in targets:
                        result.loc[idx, cat] += score * w
                        weight_counters.loc[idx, cat] += w

        # --- B. ELECTIVE POOLS  ---
        for pool_name, pool_courses in elective_pools.items():
            for course_code, weights in pool_courses.items():
                
                # Find column
                found_col = None
                for col in df_norm.columns:
                    # Special check for SNT pool (starts with SNT)
                    if course_code == 'SNT' and col.startswith('SNT'):
                        found_col = col
                    elif course_code in col:
                        found_col = col
                        break
                
                if found_col:
                    score = row[found_col]
                    
                    # ONLY count if student took the course (score > 0)
                    if score > 0:
                        for cat, w in weights.items():
                            if cat in targets:
                                result.loc[idx, cat] += score * w
                                weight_counters.loc[idx, cat] += w
    
    weight_counters = weight_counters.replace(0, 1.0)
    return result / weight_counters

weighted_features = calculate_dynamic_features(data_normalized, core_courses, elective_pools, target_categories)

# =============================================================================
# 5. CLUSTERING
# =============================================================================
n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(weighted_features)
weighted_features['Cluster'] = clusters
centroids = weighted_features.groupby('Cluster').mean()

# =============================================================================
# 6. NAMING ENGINE (Descriptive & Analytical Labels)
# =============================================================================
def name_profile_distinct(row):
    avg = row.mean()
    software = row['Software_Practice']
    hardware = row['Hardware']
    algo = row['Algorithm_Theory']
    systems = row['Systems']
    math = row['Math_Calc']
    basic = row['Basic_Sciences']
    
    # 1. Low Academic Profile 
    if avg < 0.15: 
        return "📉 Low Academic Profile (Data Missing or At Risk)"
    
    # 2. Good at both Software and Hardware
    if software > 0.60 and hardware > 0.60: 
        return "🤖 Versatile Engineer (Strong: Software & Hardware)"
    
    # 3. Only good at Hardware
    if hardware > 0.70: 
        return "⚡ Hardware Specialist (Strong: Embedded/Electronics - Weak: Software)"
    
    # 4. Good at Systems 
    if systems > 0.70: 
        return "⚙️ Systems Architect (Strong: OS/Networks/Hardware)"
    
    # 5. Good at Software/Programming Onnly, bad at hardware
    if software > 0.60 and hardware < 0.45: 
        return "☁️ Pure Software Dev (Strong: Coding - Weak: Hardware)"
    
    # 6. Good at algorithms and software
    if software > 0.50 and algo > 0.50: 
        return "🚀 Algorithmic Programmer (Strong: Algorithms & Coding)"
    
    # 7. good at basic sciences or math
    if (basic > 0.55 or math > 0.55):
        # Eğer Matematiği çok yüksekse
        if math > 0.80: 
            return "🏆 Elite Academic Theorist (Excellent: Math/Physics)"
        # Standart Teorisyen
        return "📚 Theoretical Profile (Strong: Basic Sciences - Weak: Practical Coding)"
        
    # 8. successful in general
    if avg > 0.65: 
        return "🌟 High Achiever (Balanced Success across all fields)"
    
    # 9. Standard
    return "⚖️ Standard Profile (Average Performance)"


profile_names = {}
for i in range(n_clusters):
    profile_names[i] = name_profile_distinct(centroids.iloc[i])

weighted_features['Profile_Name'] = weighted_features['Cluster'].map(profile_names)

# =============================================================================
# 7. OUTPUTS
# =============================================================================
print("--- STUDENT PROFILE DISTRIBUTION ---")
print(weighted_features['Profile_Name'].value_counts())
print("\nCluster Centroids:")
print(centroids.round(2))

def create_pca_scatter(df, features, filename='clustering-pipeline/student_clusters_scatter_eng.png'):
    pca = PCA(n_components=2)
    X = df[features]
    X_pca = pca.fit_transform(X)
    df_plot = df.copy()
    df_plot['PC1'] = X_pca[:, 0]
    df_plot['PC2'] = X_pca[:, 1]
    
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        data=df_plot, x='PC1', y='PC2', 
        hue='Profile_Name', style='Profile_Name',
        s=150, alpha=0.85, palette='tab10'
    )
    plt.title('Student Cluster Distribution (Dynamic Elective Calculation)', fontsize=18)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', title='Profiles')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Plot Saved: {filename}")

create_pca_scatter(weighted_features, target_categories)

weighted_features['Student_ID'] = student_ids
weighted_features.to_csv('clustering-pipeline/final_student_profiles_eng.csv', index=False)