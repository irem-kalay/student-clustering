import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. SETTINGS AND FILE PATHS
# ==========================================
GRADES_FILE = 'vector-equivalent.csv'
DEMOGRAPHICS_FILE = 'student_clusters_with_demographics.csv'

# Output files
GENDER_REPORT = 'gender_based_performance_report.txt'
YEAR_REPORT = 'year_based_performance_report.txt'

# Course names dictionary (For readability)
COURSE_NAMES = {
    "FIZ 101": "Physics I", "FIZ 101EL": "Physics I Laboratory",
    "BLG 101": "Intr. to Information Systems", "BLG 113": "Intr.to Comp.Eng. and Ethics",
    "MAT 103": "Mathematics I", "MAT 281": "Linear Algebra and Applicat.",
    "ING 100": "EAP Through Global Goals", "BLG 112": "Discrete Mathematics",
    "BLG 102": "Intr to Sci&Eng Comp (C)", "MAT 104": "Mathematics II",
    "FIZ 102": "Physics II", "FIZ 102EL": "Physics II Laboratory",
    "ING 112A": "Basics of Academic Writing", "BLG 210": "Engineering Mathematics",
    "BLG 231": "Digital Circuits", "BLG 223": "Data Structures",
    "EHB 222": "Introduction to Electronics", "EHB 211": "Basics of Electrical Circuits",
    "ING 201A": "Essentials of Res.Paper Writ.", "BLG 252": "Object Oriented Programming",
    "BLG 222": "Computer Organization", "BLG 242": "Logic Circuits Laboratory",
    "BLG 202": "Numerical Methods in CE", "BLG 311": "Formal Languages and Automata",
    "BLG 335": "Analysis of Algorithms I", "MAT 271": "Probability and Statistics",
    "BLG 351": "Microcomputer Lab.", "BLG 317": "Database Systems",
    "BLG 212": "Microprocessor Systems", "EHB 311": "Intr.to Electronics Laboratory",
    "BLG 322": "Computer Architecture", "BLG 312": "Computer Operating Systems",
    "BLG 336": "Analysis of Algorithms II", "BLG 354": "Signal&Systems for Comp.Eng.",
    "BLG 374": "Tech. Communic.for Comp.Eng.", "BLG 411": "Software Engineering",
    "BLG 4901": "Computer Engineering Design I", "BLG 4902": "Computer Engineering Design II",
    "BLG 454": "Learning From Data", "BLG 413": "System Programming",
    "BLG 459": "Computer Security", "BLG 475": "Software Quality and Testing"
}

def get_course_label(course_code):
    name = COURSE_NAMES.get(course_code, "Unknown Course")
    return f"{course_code} ({name})"

def main():
    if not os.path.exists(GRADES_FILE) or not os.path.exists(DEMOGRAPHICS_FILE):
        print(f"ERROR: Required CSV files not found! Please ensure you have completed the previous steps.")
        return

    print("Loading data...")
    df_grades = pd.read_csv(GRADES_FILE)
    df_demo = pd.read_csv(DEMOGRAPHICS_FILE)

    # Merge grades and demographics on Student_ID
    df_merged = pd.merge(df_grades, df_demo, on='Student_ID')
    
    exclude_cols = ['Student_ID', 'Assigned_Cluster', 'Gender', 'Entry_Year']
    course_cols = [c for c in df_merged.columns if c not in exclude_cols]

    clusters = sorted(df_merged['Assigned_Cluster'].unique())
    genders = df_merged['Gender'].dropna().unique()
    years = sorted(df_merged['Entry_Year'].dropna().astype(int).unique())

    # ==========================================
    # PART A: GENDER BASED ANALYSIS
    # ==========================================
    print("Calculating GENDER-based cluster performance...")
    viz_data_pct_gender = {g: [] for g in genders}
    viz_data_abs_gender = {g: [] for g in genders}

    with open(GENDER_REPORT, 'w', encoding='utf-8') as f:
        f.write("====================================================\n")
        f.write("    INTRA-GENDER CLUSTER PERFORMANCE REPORT\n")
        f.write("====================================================\n\n")

        for gender in genders:
            df_g_all = df_merged[df_merged['Gender'] == gender]
            total_g = len(df_g_all)
            
            if total_g == 0:
                for c in clusters: 
                    viz_data_pct_gender[gender].append(0)
                    viz_data_abs_gender[gender].append(0)
                continue
                
            g_overall_means = df_g_all[course_cols].mean()

            f.write(f"### >>> {gender.upper()} STUDENTS <<< (Total: {total_g})\n\n")

            for c in clusters:
                df_g_cluster = df_g_all[df_g_all['Assigned_Cluster'] == c]
                c_count = len(df_g_cluster)
                
                pct = (c_count / total_g) * 100
                viz_data_pct_gender[gender].append(pct)
                viz_data_abs_gender[gender].append(c_count)

                f.write(f"--- CLUSTER {c} ({gender.upper()}) ---\n")
                f.write(f"Count: {c_count} ({pct:.1f}% of all {gender}s)\n\n")

                if c_count > 0:
                    c_means = df_g_cluster[course_cols].mean()
                    diff = c_means - g_overall_means

                    strengths = diff.nlargest(5)
                    f.write(f"  [+] RELATIVE STRENGTHS:\n")
                    for course, val in strengths.items():
                        if c_means[course] > 0:
                            f.write(f"      - {get_course_label(course)}: {c_means[course]:.2f} (Avg: {g_overall_means[course]:.2f} -> +{val:.2f})\n")

                    weaknesses = diff.nsmallest(5)
                    f.write(f"\n  [-] RELATIVE WEAKNESSES:\n")
                    for course, val in weaknesses.items():
                        f.write(f"      - {get_course_label(course)}: {c_means[course]:.2f} (Avg: {g_overall_means[course]:.2f} -> {val:.2f})\n")
                f.write("\n" + "="*52 + "\n\n")

    # Draw Gender Dual Chart
    draw_dual_chart(clusters, genders, viz_data_abs_gender, viz_data_pct_gender, 
                    'Gender', 'gender_cluster_dual_analysis.png')


    # ==========================================
    # PART B: ENTRY YEAR BASED ANALYSIS
    # ==========================================
    print("Calculating ENTRY YEAR-based cluster performance...")
    viz_data_pct_year = {y: [] for y in years}
    viz_data_abs_year = {y: [] for y in years}

    with open(YEAR_REPORT, 'w', encoding='utf-8') as f:
        f.write("====================================================\n")
        f.write("    INTRA-COHORT (ENTRY YEAR) CLUSTER PERFORMANCE REPORT\n")
        f.write("====================================================\n\n")

        for year in years:
            df_y_all = df_merged[df_merged['Entry_Year'] == year]
            total_y = len(df_y_all)
            
            if total_y == 0:
                for c in clusters: 
                    viz_data_pct_year[year].append(0)
                    viz_data_abs_year[year].append(0)
                continue
                
            y_overall_means = df_y_all[course_cols].mean()

            f.write(f"### >>> ENTRY YEAR: {year} <<< (Total Students: {total_y})\n\n")

            for c in clusters:
                df_y_cluster = df_y_all[df_y_all['Assigned_Cluster'] == c]
                c_count = len(df_y_cluster)
                
                pct = (c_count / total_y) * 100
                viz_data_pct_year[year].append(pct)
                viz_data_abs_year[year].append(c_count)

                f.write(f"--- CLUSTER {c} ({year} Cohort) ---\n")
                f.write(f"Count: {c_count} ({pct:.1f}% of all {year} entries)\n\n")

                if c_count > 0:
                    c_means = df_y_cluster[course_cols].mean()
                    diff = c_means - y_overall_means

                    strengths = diff.nlargest(5)
                    f.write(f"  [+] RELATIVE STRENGTHS (vs. all {year} students):\n")
                    for course, val in strengths.items():
                        if c_means[course] > 0:
                            f.write(f"      - {get_course_label(course)}: {c_means[course]:.2f} (Avg: {y_overall_means[course]:.2f} -> +{val:.2f})\n")

                    weaknesses = diff.nsmallest(5)
                    f.write(f"\n  [-] RELATIVE WEAKNESSES (vs. all {year} students):\n")
                    for course, val in weaknesses.items():
                        f.write(f"      - {get_course_label(course)}: {c_means[course]:.2f} (Avg: {y_overall_means[course]:.2f} -> {val:.2f})\n")
                f.write("\n" + "="*52 + "\n\n")

    # Draw Year Dual Chart
    # Convert years to string labels for the chart legend
    str_years = [str(y) for y in years]
    viz_data_abs_year_str = {str(k): v for k, v in viz_data_abs_year.items()}
    viz_data_pct_year_str = {str(k): v for k, v in viz_data_pct_year.items()}
    
    draw_dual_chart(clusters, str_years, viz_data_abs_year_str, viz_data_pct_year_str, 
                    'Entry Year', 'year_cluster_dual_analysis.png')
                    
    print("All analyses complete! Check the .txt reports and .png charts.")

# ==========================================
# HELPER PLOTTING FUNCTION
# ==========================================
def draw_dual_chart(clusters, categories, abs_data, pct_data, category_title, filename):
    x = np.arange(len(clusters))
    # Adjust bar width dynamically based on how many categories we have
    width = 0.8 / len(categories) if len(categories) > 0 else 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Absolute Chart
    for i, cat in enumerate(categories):
        offset = (i - len(categories)/2 + 0.5) * width
        bars = ax1.bar(x + offset, abs_data[cat], width, label=cat)
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    ax1.set_ylabel('Number of Students')
    ax1.set_title(f'Absolute Cluster Population by {category_title}')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Cluster {c}' for c in clusters])
    ax1.legend(title=category_title)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # 2. Percentage Chart
    for i, cat in enumerate(categories):
        offset = (i - len(categories)/2 + 0.5) * width
        bars = ax2.bar(x + offset, pct_data[cat], width, label=cat)
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    ax2.set_ylabel(f'Percentage within {category_title} (%)')
    ax2.set_title(f'Proportional Cluster Distribution by {category_title}')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Cluster {c}' for c in clusters])
    ax2.set_ylim(0, 100)
    ax2.legend(title=category_title)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)

if __name__ == "__main__":
    main()