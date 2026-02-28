import pandas as pd
import numpy as np
import os
import sys
import unicodedata
import re
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import random

# ==========================================
# REPRODUCIBILITY (Add this!)
# ==========================================
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
# ==========================================

# ==========================================
# 1. CONFIGURATION & HELPERS
# ==========================================

GRADE_MAP = {
    'AA': 4.0, 'BA+': 3.75, 'BA': 3.5, 'BB+': 3.25, 'BB': 3.0,
    'CB+': 2.75, 'CB': 2.5, 'CC+': 2.25, 'CC': 2.0, 'DC+': 1.75,
    'DC': 1.5, 'DD+': 1.25, 'DD': 1.0, 'FF': 0.0, 'VF': 0.0, 'BL': None
}

def parse_grade(grade_str):
    if not isinstance(grade_str, str): return 0.0
    grade_code = grade_str.split(' /')[0].strip()
    return GRADE_MAP.get(grade_code, 0.0)

# ==========================================
# 2. NEW CURRICULUM FILTERING LOGIC
# ==========================================

def normalize_course(code: str) -> str:
    """Removes trailing 'E' if it's a suffix to a course code."""
    code = str(code).strip()
    if code.endswith('E') and len(code) > 1 and code[-2].isdigit():
        return code[:-1]
    return code

def variants(code: str) -> set:
    """Generates {CODE, CODE+E} set."""
    c = normalize_course(code)
    return {c, c + "E"}

def load_equivalency_map(denk_csv_path: str) -> dict:
    """Builds a map: variant_code -> canonical_code."""
    df = pd.read_csv(denk_csv_path)
    df.columns = [c.strip() for c in df.columns]
    
    eq_map = {}
    
    for _, row in df.iterrows():
        raw_main = str(row['Ders Kodu']).strip()
        canonical = normalize_course(raw_main)
        
        for v in variants(raw_main):
            eq_map[v] = canonical
            
        if 'Ders Denklikleri' in df.columns and pd.notna(row['Ders Denklikleri']):
            parts = re.split(r"[;,]", str(row['Ders Denklikleri']))
            for p in parts:
                code = p.strip().strip('"').strip("'")
                if code and code != '-':
                    for v in variants(code):
                        eq_map[v] = canonical
    return eq_map

def filter_and_aggregate_features(raw_df, eq_map):
    """Filters columns based on curriculum and merges equivalents."""
    id_col = 'Student_ID'
    course_cols = [c for c in raw_df.columns if c != id_col]
    
    col_to_canon = {}
    for col in course_cols:
        norm = normalize_course(col)
        if col in eq_map:
            col_to_canon[col] = eq_map[col]
        elif norm in eq_map:
            col_to_canon[col] = eq_map[norm]
        elif (norm + 'E') in eq_map:
            col_to_canon[col] = eq_map[norm + 'E']
        else:
            col_to_canon[col] = None 

    valid_cols = [c for c in course_cols if col_to_canon[c] is not None]
    print(f"Filtering: Keeping {len(valid_cols)} columns out of {len(course_cols)} original columns.")
    
    canon_to_cols = {}
    for c in valid_cols:
        canon = col_to_canon[c]
        canon_to_cols.setdefault(canon, []).append(c)
    
    out_df = pd.DataFrame()
    out_df[id_col] = raw_df[id_col]
    
    for canon, cols in sorted(canon_to_cols.items()):
        sub = raw_df[cols]
        out_df[canon] = sub.max(axis=1)
        
    return out_df

def load_raw_student_data(student_files):
    """Loads raw data without mapping."""
    all_data = []
    total = len(student_files)
    for i, file in enumerate(student_files):
        if (i+1)%50==0: sys.stdout.write(f"\rLoading: {i+1}/{total}")
        try:
            df = pd.read_excel(file) if file.endswith('.xlsx') else pd.read_csv(file)
            
            # DOSYA ADINI ID OLARAK ALMA KISMI EKLENDİ
            base_name = os.path.basename(file)
            sid, _ = os.path.splitext(base_name)
            
            for _, row in df.iterrows():
                grade = parse_grade(row['Harf Notu'])
                if grade is not None:
                    all_data.append({'Student_ID': sid, 'Course': str(row['Ders Kodu']).strip(), 'Grade': grade})
        except: pass
    print(f"\nLoaded {len(student_files)} files.")
    
    if not all_data: return pd.DataFrame()
    
    master = pd.DataFrame(all_data)
    pivot = master.pivot_table(index='Student_ID', columns='Course', values='Grade', aggfunc='max').fillna(0.0).reset_index()
    return pivot

# ==========================================
# 3. MODEL DEFINITIONS (AE & DEC)
# ==========================================

class ClusteringLayer(layers.Layer):
    """
    Clustering layer converts input sample (feature) to soft label.
    """
    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = layers.InputSpec(ndim=2)

    def build(self, input_shape):
        self.clusters = self.add_weight(shape=(self.n_clusters, input_shape[1]), 
                                        initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        # Student t-distribution
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        return K.transpose(K.transpose(q) / K.sum(q, axis=1))

def target_distribution(q):
    """
    Compute the target distribution p, which sharpens q (soft assignments).
    """
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

def build_autoencoder(input_dim, encoding_dim=10):
    input_layer = layers.Input(shape=(input_dim,))
    x = layers.Dense(500, activation='relu')(input_layer)
    x = layers.Dense(100, activation='relu')(x)
    encoder_output = layers.Dense(encoding_dim, activation='relu', name='encoder')(x)
    x = layers.Dense(100, activation='relu')(encoder_output)
    x = layers.Dense(500, activation='relu')(x)
    decoder_output = layers.Dense(input_dim, activation='sigmoid')(x)
    return models.Model(inputs=input_layer, outputs=decoder_output), models.Model(inputs=input_layer, outputs=encoder_output)

# ==========================================
# 4. MAIN PIPELINE
# ==========================================

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. SETUP PATHS
    eq_file = "dersdenklikleri.csv" 
    student_dir = os.path.join(script_dir, 'test-sınıf-50')
    
    if os.path.exists(student_dir):
        student_files = [os.path.join(student_dir, f) for f in os.listdir(student_dir) 
                         if f.endswith(".xlsx") or f.endswith(".csv")]
    else:
        student_files = []

    if not student_files or not os.path.exists(os.path.join(script_dir, eq_file)):
        print("Error: Files not found.")
    else:
        # 2. LOAD & CLEAN DATA
        print("Loading equivalency map...")
        eq_map = load_equivalency_map(os.path.join(script_dir, eq_file))
        
        print("Loading raw student data...")
        raw_df = load_raw_student_data(student_files)
        
        print("Applying Curriculum Filter...")
        clean_df = filter_and_aggregate_features(raw_df, eq_map)
        
        clean_df.to_csv('vector-equivalent.csv', index=False)
        print(f"Cleaned feature vector saved to 'vector-equivalent.csv'. Shape: {clean_df.shape}")
        
        # Prepare Data
        clean_df = clean_df.set_index('Student_ID')
        X = clean_df.values / 4.0 
        
        # 3. TRAIN AUTOENCODER
        print("Training Autoencoder...")
        autoencoder, encoder = build_autoencoder(X.shape[1])
        autoencoder.compile(optimizer='adam', loss='mse')
        # verbose=1 so you can see AE loss too
        autoencoder.fit(X, X, batch_size=32, epochs=200, verbose=1) 
        
# 4. DEC CLUSTERING
        print("\nInitializing DEC...")
        # Sınıf mevcuduna göre dinamik küme sayısı belirleme
        student_count = len(clean_df)

        if student_count < 40:
            n_clusters = 2
        elif student_count <= 100:
            n_clusters = 3
        else:
            n_clusters = 5
            
        kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=SEED)
        y_pred = kmeans.fit_predict(encoder.predict(X))
        
        # Build DEC Model
        dec_model = models.Model(inputs=encoder.input, 
                                 outputs=ClusteringLayer(n_clusters, weights=[kmeans.cluster_centers_])(encoder.output))
        dec_model.compile(optimizer='adam', loss='kld')
        
        # --- DEC TRAINING LOOP ---
        print("Running DEC Training Loop...")
        loss = 0
        index = 0
        maxiter = 2000 # Max iterations
        update_interval = 140 # Update target distribution p every 140 iters
        tol = 0.001 # Tolerance for convergence
        
        # YORUMUNUZU KODA DÖKTÜK: Öğrenci sayısına göre dinamik batch_size
        if student_count <= 100:
            batch_size = 16
        else:
            batch_size = 256 
            
        # Initialize q and p
        q = dec_model.predict(X, verbose=0)
        p = target_distribution(q)
        y_pred_last = y_pred
        
        index_array = np.arange(X.shape[0])
        
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q = dec_model.predict(X, verbose=0)
                p = target_distribution(q)
                
                # Check convergence
                y_pred = q.argmax(1)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred
                
                print(f"  Iter {ite}: Delta Label (Cluster Changes) = {delta_label:.5f}")
                if ite > 0 and delta_label < tol:
                    print('  -> Delta label below tolerance, stopping training.')
                    break
            
            # Train on batch
            idx = index_array[index * batch_size: min((index+1) * batch_size, X.shape[0])]
            loss = dec_model.train_on_batch(x=X[idx], y=p[idx])
            
            # Reset index if end of epoch
            index = index + 1 if (index + 1) * batch_size <= X.shape[0] else 0
            
            # PRINT LOSS
            if ite % 100 == 0:
                print(f"  DEC Iteration {ite} - Loss: {loss:.5f}")
        # -------------------------
        
        # 5. FINAL RESULTS
        final_q = dec_model.predict(X, verbose=0)
        clusters = final_q.argmax(1)
        
        results = pd.DataFrame({'Student_ID': clean_df.index, 'Assigned_Cluster': clusters})
        results.to_csv('student_final_clusters.csv', index=False)
        print("\nFinal clusters saved to 'student_final_clusters.csv'")
        
        # 6. VISUALIZATION
        print("Generating PCA Plot...")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(encoder.predict(X, verbose=0))
        
        plt.figure(figsize=(10, 8))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='tab10', alpha=0.6)
        plt.title('Student Clusters (Curriculum Courses Only)')
        plt.colorbar(label='Cluster')
        plt.savefig('cluster_plot.png')
        print("Done. Saved plot as 'cluster_plot.png'")
        
        # ==========================================
        # 7. CLUSTER ANALYSIS & REPORT GENERATION
        # ==========================================
        print("\nGenerating Report for Instructor AI...")
        
        # Course name mapping dictionary
        COURSE_NAMES = {
            "FIZ 101": "Physics I",
            "FIZ 101EL": "Physics I Laboratory",
            "BLG 101": "Intr. to Information Systems",
            "BLG 113": "Intr.to Comp.Eng. and Ethics",
            "MAT 103": "Mathematics I",
            "MAT 281": "Linear Algebra and Applicat.",
            "ING 100": "EAP Through Global Goals",
            "BLG 112": "Discrete Mathematics",
            "BLG 102": "Intr to Sci&Eng Comp (C)",
            "MAT 104": "Mathematics II",
            "FIZ 102": "Physics II",
            "FIZ 102EL": "Physics II Laboratory",
            "ING 112A": "Basics of Academic Writing",
            "DAN 102": "Girişimcilik & Kariyer Danış.",
            "BLG 210": "Engineering Mathematics",
            "BLG 231": "Digital Circuits",
            "BLG 223": "Data Structures",
            "EHB 222": "Introduction to Electronics",
            "EHB 211": "Basics of Electrical Circuits",
            "ING 201A": "Essentials of Res.Paper Writ.",
            "BLG 252": "Object Oriented Programming",
            "BLG 222": "Computer Organization",
            "BLG 242": "Logic Circuits Laboratory",
            "BLG 202": "Numerical Methods in CE",
            "BLG 311": "Formal Languages and Automata",
            "TUR 121": "Türk Dili I",
            "BLG 335": "Analysis of Algorithms I",
            "MAT 271": "Probability and Statistics",
            "BLG 351": "Microcomputer Lab.",
            "TUR 122": "Türk Dili II",
            "BLG 317": "Database Systems",
            "BLG 212": "Microprocessor Systems",
            "EHB 311": "Intr.to Electronics Laboratory",
            "BLG 322": "Computer Architecture",
            "BLG 312": "Computer Operating Systems",
            "BLG 336": "Analysis of Algorithms II",
            "ATA 121": "Atatürk İlk & İnkılap Trh I",
            "BLG 354": "Signal&Systems for Comp.Eng.",
            "BLG 374": "Tech. Communic.for Comp.Eng.",
            "ATA 122": "Atatürk İlk & İnkılap Trh II",
            "BLG 411": "Software Engineering",
            "BLG 4901": "Computer Engineering Design I",
            "BLG 4902": "Computer Engineering Design II",
            "EKO 201": "Economics",
            "BLG 337": "Principles of Computer Comm.",
            "BLG 345": "Logic & Computability",
            "BLG 348": "Introduction to Bioinformatics",
            "BLG 368": "Operations Research",
            "BLG 442": "Tech.&Innov. Mng.for Inf.Tech.",
            "BLG 448": "Project Management in Eng.",
            "BLG 454": "Learning From Data",
            "KON 224": "Measurement&Instrumentation",
            "KON 317": "Control Systems",
            "MAL 201": "Materials Science",
            "BLG 413": "System Programming",
            "BLG 430": "Computer Networks",
            "BLG 433": "Computer Communications",
            "BLG 434": "Introduction to Expert Systems",
            "BLG 435": "Artificial Intelligence",
            "BLG 438": "Digital System Design Laboratory",
            "BLG 439": "Computer Project I",
            "BLG 440": "Computer Project II",
            "BLG 443": "Discrete Event Simulation",
            "BLG 444": "Computer Graphics",
            "BLG 447": "Compiler Design",
            "BLG 449": "Prog.in Parallel&DistrubedSys.",
            "BLG 450": "Real-Time Systems Software",
            "BLG 451": "Real-Time Systems",
            "BLG 452": "Microprocessor Design Laboratory",
            "BLG 453": "Computer Vision",
            "BLG 456": "Robotics",
            "BLG 458": "Functional Programming",
            "BLG 459": "Computer Security",
            "BLG 460": "Secure Programming",
            "BLG 475": "Software Quality and Testing",
            "BLG 477": "Multimedia Computing",
            "BLG 478": "Network Security",
            "BLG 481": "Al Accelerators Lab.",
            "BLG 483": "Artificial Intelligence Aided Computer Engineering",
            "YZV 406": "Robotics"
        }

        # Add cluster labels temporarily to our dataframe to calculate means
        analysis_df = clean_df.copy()
        analysis_df['Cluster'] = clusters
        
        # Calculate overall means for baseline comparison
        overall_means = clean_df.mean()
        
        with open('report.txt', 'w', encoding='utf-8') as f:
            f.write("====================================================\n")
            f.write("      AI INSTRUCTOR: STUDENT CLUSTER ANALYSIS REPORT\n")
            f.write("====================================================\n\n")
            
            # DÜZELTME BURADA: range(5) yerine range(n_clusters)
            for c_id in range(n_clusters):  
                cluster_data = analysis_df[analysis_df['Cluster'] == c_id].drop('Cluster', axis=1)
                student_count_in_cluster = len(cluster_data)
                
                f.write(f"--- CLUSTER {c_id} PROFILE ---\n")
                f.write(f"Population: {student_count_in_cluster} students ({(student_count_in_cluster/len(clean_df))*100:.1f}% of total)\n\n")
                
                if student_count_in_cluster == 0:
                    f.write("  [Empty Cluster]\n\n")
                    continue
                
                # Calculate the average course grade inside this cluster
                cluster_means = cluster_data.mean()
                
                # Compare to overall population to find relative strengths/weaknesses
                diff = cluster_means - overall_means
                
                # Helper function to get the formatted name
                def get_course_label(course_code):
                    name = COURSE_NAMES.get(course_code, "Unknown Course")
                    return f"{course_code} ({name})"
                
                # Top 5 Relative Strengths
                strengths = diff.nlargest(5)
                f.write("  [+] SUCCESSFUL AT (Relative Strengths):\n")
                for course, diff_val in strengths.items():
                    avg_grade = cluster_means[course]
                    if avg_grade > 0:
                        course_label = get_course_label(course)
                        f.write(f"      - {course_label}: {avg_grade:.2f} (Avg is {overall_means[course]:.2f} -> +{diff_val:.2f})\n")
                
                # Top 5 Relative Weaknesses
                weaknesses = diff.nsmallest(5)
                f.write("\n  [-] WEAK WITH (Relative Weaknesses):\n")
                for course, diff_val in weaknesses.items():
                    avg_grade = cluster_means[course]
                    course_label = get_course_label(course)
                    f.write(f"      - {course_label}: {avg_grade:.2f} (Avg is {overall_means[course]:.2f} -> {diff_val:.2f})\n")
                
                # Top 5 Highest Absolute Grades
                top_abs = cluster_means.nlargest(5)
                f.write("\n  [★] ABSOLUTE TOP PERFORMING COURSES:\n")
                for course, val in top_abs.items():
                    if val > 0:
                        course_label = get_course_label(course)
                        f.write(f"      - {course_label}: {val:.2f}/4.00\n")
                
                # Space between cluster profiles
                f.write("\n" + "="*52 + "\n\n")
        
        print("Analysis complete! Insights exported to 'report.txt'.")