import pandas as pd
import sys

# ==========================================
# 1. COURSE NAME MAPPING
# ==========================================
# Raw list from the curriculum you provided, updated with new entries
raw_course_list = {
    "FIZ 101E": "Physics I",
    "FIZ 101EL": "Physics I Laboratory",
    "BLG 101E": "Intr. to Information Systems",
    "BLG 113E": "Intr.to Comp.Eng. and Ethics",
    "MAT 103E": "Mathematics I",
    "MAT 281E": "Linear Algebra and Applicat.",
    "ING 100": "EAP Through Global Goals",
    "BLG 112E": "Discrete Mathematics",
    "BLG 102E": "Intr to Sci&Eng Comp (C)",
    "MAT 104E": "Mathematics II",
    "FIZ 102E": "Physics II",
    "FIZ 102EL": "Physics II Laboratory",
    "ING 112A": "Basics of Academic Writing",
    "DAN 102": "Girişimcilik & Kariyer Danış.",
    "BLG 210E": "Engineering Mathematics",
    "BLG 231E": "Digital Circuits",
    "BLG 223E": "Data Structures",
    "EHB 222E": "Introduction to Electronics",
    "EHB 211E": "Basics of Electrical Circuits",
    "ING 201A": "Essentials of Res.Paper Writ.",
    "BLG 252E": "Object Oriented Programming",
    "BLG 222E": "Computer Organization",
    "BLG 242E": "Logic Circuits Laboratory",
    "BLG 202E": "Numerical Methods in CE",
    "BLG 311E": "Formal Languages and Automata",
    "TUR 121": "Türk Dili I",
    "BLG 335E": "Analysis of Algorithms I",
    "MAT 271E": "Probability and Statistics",
    "BLG 351E": "Microcomputer Lab.",
    "TUR 122": "Türk Dili II",
    "BLG 317E": "Database Systems",
    "BLG 212E": "Microprocessor Systems",
    "EHB 311E": "Intr.to Electronics Laboratory",
    "BLG 322E": "Computer Architecture",
    "BLG 312E": "Computer Operating Systems",
    "BLG 336E": "Analysis of Algorithms II",
    "ATA 121": "Atatürk İlk & İnkılap Trh I",
    "BLG 354E": "Signal&Systems for Comp.Eng.",
    "BLG 374E": "Tech. Communic.for Comp.Eng.",
    "ATA 122": "Atatürk İlk & İnkılap Trh II",
    "BLG 411E": "Software Engineering",
    "BLG 4901E": "Computer Engineering Design I",
    "BLG 4902E": "Computer Engineering Design II",
    "EKO 201E": "Economics",
    # New additions
    "BLG 337E": "Principles of Computer Comm.",
    "BLG 345E": "Logic & Computability",
    "BLG 348E": "Introduction to Bioinformatics",
    "BLG 368E": "Operations Research",
    "BLG 442E": "Tech.&Innov. Mng.for Inf.Tech.",
    "BLG 448E": "Project Management in Eng.",
    "BLG 454E": "Learning From Data",
    "KON 224E": "Measurement&Instrumentation",
    "KON 317E": "Control Systems",
    "MAL 201E": "Materials Science",
    "BLG 413E": "System Programming",
    "BLG 430E": "Computer Networks",
    "BLG 433E": "Computer Communications",
    "BLG 434E": "Introduction to Expert Systems",
    "BLG 435E": "Artificial Intelligence",
    "BLG 438E": "Digital System Design Laboratory",
    "BLG 439E": "Computer Project I",
    "BLG 440E": "Computer Project II",
    "BLG 443E": "Discrete Event Simulation",
    "BLG 444E": "Computer Graphics",
    "BLG 447E": "Compiler Design",
    "BLG 449E": "Prog.in Parallel&DistrubedSys.",
    "BLG 450E": "Real-Time Systems Software",
    "BLG 451E": "Real-Time Systems",
    "BLG 452E": "Microprocessor Design Laboratory",
    "BLG 453E": "Computer Vision",
    "BLG 456E": "Robotics",
    "BLG 458E": "Functional Programming",
    "BLG 459E": "Computer Security",
    "BLG 460E": "Secure Programming",
    "BLG 475E": "Software Quality and Testing",
    "BLG 477E": "Multimedia Computing",
    "BLG 478E": "Network Security",
    "BLG 481E": "Al Accelerators Lab.",
    "BLG 483E": "Artificial Intelligence Aided Computer Engineering",
    "YZV 406E": "Robotics"
}

def normalize_course(code: str) -> str:
    """Removes trailing 'E' to match your vector file format."""
    code = str(code).strip()
    if code.endswith('E') and len(code) > 1 and code[-2].isdigit():
        return code[:-1]
    return code

# Create the lookup dictionary with normalized keys (e.g., 'BLG 101E' -> 'BLG 101')
COURSE_NAMES = {normalize_course(k): v for k, v in raw_course_list.items()}

# ==========================================
# 2. ANALYSIS FUNCTION
# ==========================================
def analyze_clusters_to_file(output_filename="cluster_report.txt"):
    # 1. Load generated files
    try:
        clusters_df = pd.read_csv('student_final_clusters.csv')
        vectors_df = pd.read_csv('vector-equivalent.csv')
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure you have run the main script first.")
        return

    # 2. Merge on Student_ID
    merged_df = pd.merge(clusters_df, vectors_df, on='Student_ID')

    # 3. Calculate Averages
    # Global average (for comparison)
    global_mean = vectors_df.mean(numeric_only=True)
    # Average per cluster
    cluster_means = merged_df.groupby('Assigned_Cluster').mean(numeric_only=True)

    # 4. Open file and write report
    with open(output_filename, "w", encoding="utf-8") as f:
        # Helper function to write to both file and (optionally) terminal, 
        # but here we write only to file as requested.
        def log(text):
            f.write(text + "\n")

        log("="*60)
        log("      CLUSTER INTERPRETATION REPORT")
        log("="*60)
        
        for cluster_id in sorted(cluster_means.index):
            log(f"\n{'='*20} CLUSTER {cluster_id} {'='*20}")
            n_students = len(merged_df[merged_df['Assigned_Cluster'] == cluster_id])
            log(f"Size: {n_students} Students")
            
            # Compare this cluster to the global average
            diff = cluster_means.loc[cluster_id] - global_mean
            
            # --- Strengths ---
            strengths = diff.sort_values(ascending=False).head(5)
            log("\n  [+] DISTINCTIVE STRENGTHS (Higher than Avg):")
            for course, val in strengths.items():
                actual_grade = cluster_means.loc[cluster_id, course]
                # Get the readable name
                course_name = COURSE_NAMES.get(course, "Elective / Other")
                log(f"    {course:<9} : {course_name:<35} | Grade: {actual_grade:.2f} (Diff: +{val:.2f})")
                
            # --- Weaknesses ---
            weaknesses = diff.sort_values(ascending=True).head(5)
            log("\n  [-] DISTINCTIVE WEAKNESSES (Lower than Avg):")
            for course, val in weaknesses.items():
                actual_grade = cluster_means.loc[cluster_id, course]
                # Get the readable name
                course_name = COURSE_NAMES.get(course, "Elective / Other")
                log(f"    {course:<9} : {course_name:<35} | Grade: {actual_grade:.2f} (Diff: {val:.2f})")
    
    print(f"Report saved to '{output_filename}'")

if __name__ == "__main__":
    analyze_clusters_to_file()