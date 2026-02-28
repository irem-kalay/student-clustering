import pandas as pd
import os
import glob

# ==========================================
# 1. SETTINGS AND FILE PATHS
# ==========================================
CLUSTERS_FILE = 'student_final_clusters.csv'
TXT_FOLDER = 'downloads_properties'
OUTPUT_REPORT = 'demographic_report.txt'
OUTPUT_CSV = 'student_clusters_with_demographics.csv'

def parse_txt_file(filepath):
    """
    Reads the txt file and extracts Gender and Entry Year data.
    Format:
    Line 1: Gender (e.g., Erkek / Kadın)
    Line 2: ID (e.g., 15022 -> takes '22' and turns it into 2022)
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.read().strip().splitlines()
            
        if len(lines) >= 2:
            gender = lines[0].strip()
            student_no = lines[1].strip()
            
            # Calculate the year (take the 2 digits after '150' and add 2000)
            # e.g., 15022 -> 22 -> 2022
            if len(student_no) >= 5 and student_no.startswith("150"):
                year_str = student_no[3:5]
                if year_str.isdigit():
                    entry_year = 2000 + int(year_str)
                else:
                    entry_year = None
            else:
                entry_year = None
                
            return gender, entry_year
    except Exception as e:
        print(f"Error ({filepath}): {e}")
        
    return None, None

# ==========================================
# 2. MERGING DATA
# ==========================================
def main():
    if not os.path.exists(CLUSTERS_FILE):
        print(f"ERROR: '{CLUSTERS_FILE}' not found!")
        return
        
    if not os.path.exists(TXT_FOLDER):
        print(f"ERROR: '{TXT_FOLDER}' folder not found!")
        return

    print("Loading clusters...")
    df_clusters = pd.read_csv(CLUSTERS_FILE)
    
    demographics = []
    
    print("Reading TXT files and extracting data...")
    for idx, row in df_clusters.iterrows():
        student_id = row['Student_ID']
        cluster = row['Assigned_Cluster']
        
        # Create txt file path (Student_ID already holds the filename without extension)
        txt_path = os.path.join(TXT_FOLDER, f"{student_id}.txt")
        
        gender, entry_year = None, None
        if os.path.exists(txt_path):
            gender, entry_year = parse_txt_file(txt_path)
        else:
            print(f"Warning: '{txt_path}' not found.")
            
        demographics.append({
            'Student_ID': student_id,
            'Assigned_Cluster': cluster,
            'Gender': gender,
            'Entry_Year': entry_year
        })
        
    df_combined = pd.DataFrame(demographics)
    
    # Perform analysis using only the rows with valid gender and year data
    df_valid = df_combined.dropna(subset=['Gender', 'Entry_Year']).copy()
    df_valid['Entry_Year'] = df_valid['Entry_Year'].astype(int)
    
    # Save the merged data as CSV
    df_combined.to_csv(OUTPUT_CSV, index=False)
    print(f"\nMerged data successfully saved as '{OUTPUT_CSV}'.")

    # ==========================================
    # 3. CLUSTER-BASED DEMOGRAPHIC ANALYSIS & REPORTING
    # ==========================================
    print("\nGenerating Demographic Analysis Report...")
    
    clusters = sorted(df_valid['Assigned_Cluster'].unique())
    
    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        f.write("====================================================\n")
        f.write("        CLUSTER-BASED DEMOGRAPHIC ANALYSIS REPORT\n")
        f.write("====================================================\n\n")
        
        for c in clusters:
            df_c = df_valid[df_valid['Assigned_Cluster'] == c]
            total_students = len(df_c)
            
            f.write(f"--- CLUSTER {c} PROFILE ---\n")
            f.write(f"Total Valid Student Count: {total_students}\n\n")
            
            if total_students == 0:
                f.write("  [No Data]\n\n")
                continue
                
            # Gender Distribution
            gender_counts = df_c['Gender'].value_counts()
            f.write("  [GENDER DISTRIBUTION]:\n")
            for g, count in gender_counts.items():
                percentage = (count / total_students) * 100
                f.write(f"    - {g}: {count} students ({percentage:.1f}%)\n")
                
            # Entry Year Distribution
            year_counts = df_c['Entry_Year'].value_counts().sort_index()
            most_common_year = year_counts.idxmax()
            
            f.write("\n  [ENTRY YEAR DISTRIBUTION]:\n")
            for y, count in year_counts.items():
                percentage = (count / total_students) * 100
                f.write(f"    - {y}: {count} students ({percentage:.1f}%)\n")
                
            f.write(f"\n  > Dominant entry year in this cluster: {most_common_year}\n")
            f.write("\n" + "="*52 + "\n\n")

    print(f"Analysis complete! Results written to '{OUTPUT_REPORT}'.")

if __name__ == "__main__":
    main()