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
# Adjust the file path according to your system
try:
    df = pd.read_csv('feature-vectors/student_feature_vectors2_collapsed.csv')
except FileNotFoundError:
    # Fallback for different directory structure
    df = pd.read_csv('student_feature_vectors2_collapsed.csv')

if 'Student_ID' in df.columns:
    student_ids = df['Student_ID']
    data = df.drop(columns=['Student_ID'])
else:
    student_ids = df.index
    data = df

# =============================================================================
# 2. NORMALIZATION (0=None, 0.2=FF, 1.0=AA)
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
                # Scale between 0.2 and 1.0
                scaled_values = 0.2 + 0.8 * ((taken_values - min_val) / (max_val - min_val))
                df_norm.loc[mask_taken[col], col] = scaled_values
    df_norm[~mask_taken] = 0.0
    return df_norm

data_normalized = custom_normalization(data)

# =============================================================================
# 3. WEIGHTED COURSE MAP (Including All Electives)
# =============================================================================

course_weights = {
    # --- CORE & COMPULSORY COURSES ---
    'BLG 102': {'Software_Practice': 0.8, 'Algorithm_Theory': 0.2},
    'BLG 252': {'Software_Practice': 0.9, 'Algorithm_Theory': 0.1},
    'BLG 335': {'Algorithm_Theory': 0.6, 'Software_Practice': 0.2, 'Math_Calc': 0.2},
    'BLG 336': {'Algorithm_Theory': 0.7, 'Software_Practice': 0.2, 'Math_Calc': 0.1},
    'BLG 223': {'Algorithm_Theory': 0.5, 'Software_Practice': 0.5},
    'BLG 411': {'Software_Practice': 0.6, 'Systems': 0.2, 'Social_Cultural': 0.2},
    'BLG 4901': {'Software_Practice': 0.5, 'Systems': 0.3, 'Language_Comm': 0.2},
    'MAT 103': {'Math_Calc': 1.0},
    'MAT 104': {'Math_Calc': 1.0},
    'MAT 271': {'Math_Calc': 0.8, 'Algorithm_Theory': 0.2},
    'MAT 281': {'Math_Calc': 0.9, 'Algorithm_Theory': 0.1},
    'BLG 210': {'Math_Calc': 1.0},
    'BLG 112': {'Math_Calc': 0.6, 'Algorithm_Theory': 0.4},
    'BLG 202': {'Math_Calc': 0.7, 'Software_Practice': 0.3},
    'FIZ 101': {'Basic_Sciences': 0.8, 'Math_Calc': 0.2},
    'FIZ 102': {'Basic_Sciences': 0.8, 'Math_Calc': 0.2},
    'KIM 101': {'Basic_Sciences': 1.0},
    'BLG 101': {'Basic_Sciences': 0.5, 'Software_Practice': 0.5},
    'BLG 222': {'Hardware': 0.6, 'Systems': 0.4},
    'BLG 322': {'Hardware': 0.4, 'Systems': 0.6},
    'BLG 231': {'Hardware': 1.0},
    'BLG 242': {'Hardware': 1.0},
    'EHB':     {'Hardware': 1.0},
    'BLG 312': {'Systems': 0.8, 'Software_Practice': 0.2},
    'BLG 317': {'Systems': 0.7, 'Software_Practice': 0.3},
    'BLG 351': {'Hardware': 0.7, 'Software_Practice': 0.3},
    'BLG 212': {'Hardware': 0.5, 'Systems': 0.5},
    'ING': {'Language_Comm': 1.0},
    'TUR': {'Social_Cultural': 1.0},
    'ATA': {'Social_Cultural': 1.0},
    'EKO': {'Social_Cultural': 1.0},
    'SNT': {'Social_Cultural': 1.0},
    'DAN': {'Social_Cultural': 0.5, 'Language_Comm': 0.5},
    'BLG 374': {'Language_Comm': 0.8, 'Social_Cultural': 0.2},

    #seçmeli dersler cluster düzeni bozuyo nasıl yapıcam, 0 ekleyince ort düşüyor
}

target_categories = ['Basic_Sciences', 'Software_Practice', 'Algorithm_Theory', 
                     'Systems', 'Hardware', 'Social_Cultural', 'Language_Comm', 'Math_Calc']

# =============================================================================
# 4. SCORE CALCULATION ENGINE
# =============================================================================
def calculate_weighted_features(df_norm, weights_map, targets):
    result = pd.DataFrame(0.0, index=df_norm.index, columns=targets)
    weight_counters = pd.DataFrame(0.0, index=df_norm.index, columns=targets)
    
    for col in df_norm.columns:
        current_weights = {}
        found = False
        
        # 1. Exact Match
        for k, v in weights_map.items():
            if k in col:
                current_weights = v
                found = True
                break
        
        # 2. Prefix Match (Fallback)
        if not found:
            prefix = re.match(r"([A-Z]+)", col)
            if prefix:
                p = prefix.group(1)
                if p == 'MAT': current_weights = {'Math_Calc': 1.0}
                elif p == 'FIZ' or p == 'KIM': current_weights = {'Basic_Sciences': 1.0}
                elif p == 'EHB': current_weights = {'Hardware': 1.0}
                elif p == 'ING': current_weights = {'Language_Comm': 1.0}
                elif p in ['TUR', 'ATA', 'SNT', 'EKO', 'ISL']: current_weights = {'Social_Cultural': 1.0}
                elif p == 'BLG': current_weights = {'Software_Practice': 0.5, 'Algorithm_Theory': 0.5}
        
        if current_weights:
            student_scores = df_norm[col]
            for cat, w in current_weights.items():
                if cat in targets:
                    result[cat] += student_scores * w
                    # We add the weight REGARDLESS of whether the student took the course.
                    # This penalizes students who haven't taken electives (zeros remain zeros),
                    # helping to separate "Experts" from "Standard" students.
                    weight_counters[cat] += w 

    # Avoid division by zero
    weight_counters = weight_counters.replace(0, 1.0)
    return result / weight_counters

weighted_features = calculate_weighted_features(data_normalized, course_weights, target_categories)

# =============================================================================
# 5. CLUSTERING (10 CLUSTERS)
# =============================================================================
n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(weighted_features)
weighted_features['Cluster'] = clusters
centroids = weighted_features.groupby('Cluster').mean()

# =============================================================================
# 6. NAMING ENGINE (Distinct English Labels)
# =============================================================================
def name_profile_distinct(row):
    basic_sci = row['Basic_Sciences']
    math = row['Math_Calc']
    software = row['Software_Practice']
    algo = row['Algorithm_Theory']
    hardware = row['Hardware']
    systems = row['Systems']
    lang = row['Language_Comm']
    
    avg = row.mean()
    
    # 1. Low Achievement / Missing Data
    if avg < 0.15: 
        return "📉 Low Academic Profile / Data Missing"
    
    # TAM DONANIMLI (Full Stack + Hardware)
    if software > 0.65 and hardware > 0.65:
        return "Successful in Software & Hardware"
    
    # GÖMÜLÜ SİSTEMCİ (Embedded Systems)
    if hardware > 0.8:
        return "Successful in Hardware (Embedded Systems)"
    
    # SİSTEM MİMARI (Systems Architect)
    if systems > 0.75:
        return "Successful in Systems & Hardware"
    
    # SAF YAZILIMCI (Pure Software)
    if software > 0.40 and hardware < 0.5:
        return "Successful in Software, Weak in Hardware"
    
    # ALGORİTMİK PROGRAMCI (Algorithmic Programmer)
    if software > 0.40 and algo > 0.40:
        return "Successful in Software & Algorithms"

    # --- THEORIST GROUP & SUB-SEGMENTS ---
    if (basic_sci > 0.6 or math > 0.6):
        
        # Distinction 1: Communication
        if lang > 0.75:
            return "🗣️ Strong Communicator Theorist"
        
        # Distinction 2: Elite Math Skills
        if math > 0.85:
            return "🏆 Elite Academic Theorist (Math/Physics)"
            
        # Distinction 3: Hardware Oriented
        if hardware > 0.65:
            return "🔧 Hardware-Oriented Theorist"
            
        # Distinction 4: Weak Math
        if math < 0.3:
            return "⚠️ Theorist with Weak Math Foundation"
            
        # Standard
        return "📚 Standard Theorist (Lacking Coding)"
        
    # ALL ROUNDER
    if avg > 0.70: 
        return "🌟 High Achiever (All-Rounder)"
    
    return "⚖️ Mid-Level / Standard Profile"

# Apply Names
profile_names = {}
for i in range(n_clusters):
    profile_names[i] = name_profile_distinct(centroids.iloc[i])

weighted_features['Profile_Name'] = weighted_features['Cluster'].map(profile_names)

# =============================================================================
# 7. OUTPUTS AND VISUALIZATION
# =============================================================================
print("--- STUDENT PROFILE DISTRIBUTION ---")
print(weighted_features['Profile_Name'].value_counts())
print("\nCluster Centroids (Mean Scores):")
print(centroids.round(2))

# PLOT: PCA SCATTER
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
    plt.title('Student Cluster Distribution (PCA Analysis)', fontsize=18, pad=20)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., title='Student Profiles')
    plt.xlabel('Principal Component 1 (General Achievement)')
    plt.ylabel('Principal Component 2 (Domain Interest)')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"PCA Plot Saved: {filename}")

# Run Visualization
create_pca_scatter(weighted_features, target_categories)

# SAVE CSV
weighted_features['Student_ID'] = student_ids
weighted_features.to_csv('clustering-pipeline/final_student_profiles_eng.csv', index=False)