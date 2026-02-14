import pandas as pd
import numpy as np
import openpyxl
import os
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# GENDER ANALYSIS BASED ON CLUSTERS
# =============================================================================

def extract_year_from_value(value):
    """
    Extract year from student ID code
    Example: 15020 -> 2020
    Logic: Remove first 3 digits, remaining is year (20 -> 2020)
    """
    try:
        value_str = str(value).strip()
        # Remove non-numeric characters
        numeric_str = ''.join(filter(str.isdigit, value_str))
        
        if len(numeric_str) >= 4:
            # Take last 2 digits and convert to year
            year_digits = numeric_str[-2:]
            year = int(year_digits)
            if 0 <= year <= 30:  # Assume 00-30 is 2000-2030
                return 2000 + year
            else:  # 31-99 could be 1931-1999
                return 1900 + year
        return None
    except:
        return None


def read_xlsx_file(file_path):
    """
    Read file and extract:
    - Gender (first line)
    - Year (second line, converted from code)
    Files are actually CSV text files despite .xlsx extension
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        first_row = lines[0].strip() if len(lines) > 0 else None  # Gender
        second_row = lines[1].strip() if len(lines) > 1 else None  # Year code
        
        return first_row, second_row
    except Exception as e:
        # Try with latin-1 encoding if UTF-8 fails
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                lines = f.readlines()
            first_row = lines[0].strip() if len(lines) > 0 else None
            second_row = lines[1].strip() if len(lines) > 1 else None
            return first_row, second_row
        except:
            return None, None


def normalize_for_matching(text):
    """
    Normalize text for matching - extract the unique identifier part
    """
    if not isinstance(text, str):
        return str(text)
    
    # Remove .xlsx extension
    text = text.replace('.xlsx', '')
    
    # Extract the dynamic part (timestamp or number)
    # Pattern 1: "... - 2026-02-01TXXXXXX.XXX" (timestamp-based)
    # Pattern 2: "... (X)" (number-based)
    # Pattern 3: "... without suffix"
    
    if ' - 2026-02-01T' in text:
        # Extract timestamp part
        parts = text.split(' - 2026-02-01T')
        if len(parts) > 1:
            return '2026-02-01T' + parts[1]
    
    if text.endswith(')'):
        # Extract number from parentheses
        import re
        match = re.search(r'\((\d+)\)', text)
        if match:
            return match.group(1)
    
    # Return the entire text if no pattern matched
    return text


def process_gender_data():
    """
    Main function to process gender and year data from xlsx files
    and match with cluster data
    """
    
    # Read clustering results
    print("Loading cluster data...")
    try:
        results_df = pd.read_csv('clustering-pipeline/results/final_results.csv', encoding='utf-8')
    except:
        results_df = pd.read_csv('clustering-pipeline/results/final_results.csv')
    
    # Dictionary to store gender and year info  
    # Key: normalized filename, Value: {gender, year}
    student_info = {}
    
    # Directory with xlsx files
    xlsx_dir = 'obs_track/downloads_properties'
    xlsx_files = glob.glob(os.path.join(xlsx_dir, '*.xlsx'))
    
    print(f"Found {len(xlsx_files)} xlsx files")
    print("Extracting gender and year data from xlsx files...")
    
    # Build a mapping
    for xlsx_file in xlsx_files:
        filename = os.path.basename(xlsx_file)
        gender, year_code = read_xlsx_file(xlsx_file)
        
        if gender and year_code:
            year = extract_year_from_value(year_code)
            # Normalize the filename for matching
            normalized_key = normalize_for_matching(filename)
            student_info[normalized_key] = {
                'gender': gender,
                'year': year,
                'year_code': year_code,
                'filename': filename
            }
    
    print(f"Successfully extracted data for {len(student_info)} unique students")
    
    # Add gender and year to results dataframe
    def find_gender_year(student_id):
        # Normalize the student ID for matching
        normalized_id = normalize_for_matching(student_id)
        
        if normalized_id in student_info:
            return student_info[normalized_id]['gender'], student_info[normalized_id]['year']
        
        return 'Unknown', None
    
    # Create a series with both gender and year
    gender_year_results = results_df['Student_ID'].apply(find_gender_year)
    results_df['Gender'] = gender_year_results.apply(lambda x: x[0])
    results_df['Year'] = gender_year_results.apply(lambda x: x[1])
    
    return results_df, student_info


def analyze_gender_by_cluster(results_df):
    """
    Analyze gender distribution for each cluster
    """
    
    print("\n" + "="*80)
    print("GENDER DISTRIBUTION BY CLUSTER")
    print("="*80)
    
    analysis_results = []
    
    for cluster in sorted(results_df['Cluster_DEC'].unique()):
        cluster_data = results_df[results_df['Cluster_DEC'] == cluster]
        
        # Overall gender count
        gender_counts = cluster_data['Gender'].value_counts()
        total_students = len(cluster_data)
        
        print(f"\n--- CLUSTER {cluster} ---")
        print(f"Total Students: {total_students}")
        print("\nGender Distribution (Count & Percentage):")
        
        for gender, count in gender_counts.items():
            percentage = (count / total_students) * 100
            print(f"  {gender}: {count} ({percentage:.1f}%)")
        
        # Save to analysis results
        analysis_results.append({
            'Cluster': cluster,
            'Total_Students': total_students,
            **{f'{gender}_Count': int(count) for gender, count in gender_counts.items()},
            **{f'{gender}_Percentage': (count / total_students) * 100 
               for gender, count in gender_counts.items()}
        })
    
    return pd.DataFrame(analysis_results)


def analyze_gender_year_by_cluster(results_df):
    """
    Analyze gender distribution by year within each cluster
    """
    
    print("\n" + "="*80)
    print("GENDER & YEAR DISTRIBUTION BY CLUSTER")
    print("="*80)
    
    year_gender_data = []
    
    for cluster in sorted(results_df['Cluster_DEC'].unique()):
        cluster_data = results_df[results_df['Cluster_DEC'] == cluster]
        print(f"\n--- CLUSTER {cluster} ---")
        
        # Group by year and gender
        year_groups = cluster_data.groupby('Year')['Gender'].value_counts().unstack(fill_value=0)
        
        print(year_groups)
        
        # Store detailed data
        for year in sorted(cluster_data['Year'].dropna().unique()):
            year_data = cluster_data[cluster_data['Year'] == year]
            gender_counts = year_data['Gender'].value_counts()
            total = len(year_data)
            
            for gender, count in gender_counts.items():
                year_gender_data.append({
                    'Cluster': cluster,
                    'Year': int(year),
                    'Gender': gender,
                    'Count': count,
                    'Percentage': (count / total) * 100
                })
    
    return pd.DataFrame(year_gender_data)


def save_analysis_reports(results_df, gender_analysis_df, year_gender_analysis_df):
    """
    Save analysis reports to CSV files
    """
    
    output_dir = 'clustering-pipeline/results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save gender analysis
    gender_csv_path = os.path.join(output_dir, 'cluster_gender_analysis.csv')
    gender_analysis_df.to_csv(gender_csv_path, index=False)
    print(f"\n✓ Saved gender analysis to: {gender_csv_path}")
    
    # Save year-gender analysis
    year_gender_csv_path = os.path.join(output_dir, 'cluster_gender_year_analysis.csv')
    year_gender_analysis_df.to_csv(year_gender_csv_path, index=False)
    print(f"✓ Saved year-gender analysis to: {year_gender_csv_path}")
    
    # Save enriched results
    results_csv_path = os.path.join(output_dir, 'final_results_with_gender.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"✓ Saved enriched results to: {results_csv_path}")


def create_visualizations(results_df, gender_analysis_df, year_gender_analysis_df):
    """
    Create visualization plots
    """
    
    output_dir = 'clustering-pipeline/results'
    
    # 1. Gender distribution by cluster (bar plot)
    fig, ax = plt.subplots(figsize=(12, 6))
    gender_cols = [col for col in gender_analysis_df.columns if col.endswith('_Count') and col != 'Cluster']
    
    if gender_cols:
        gender_names = [col.replace('_Count', '') for col in gender_cols]
        
        for i, (col, gender_name) in enumerate(zip(gender_cols, gender_names)):
            ax.bar(gender_analysis_df['Cluster'] + i*0.2 - 0.2, 
                   gender_analysis_df[col], 
                   width=0.2, 
                   label=gender_name)
        
        ax.set_xlabel('Cluster', fontsize=12)
        ax.set_ylabel('Number of Students', fontsize=12)
        ax.set_title('Gender Distribution by Cluster', fontsize=14, fontweight='bold')
        ax.set_xticks(sorted(gender_analysis_df['Cluster'].unique()))
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'gender_distribution_by_cluster.png')
        plt.savefig(plot_path, dpi=300)
        print(f"✓ Saved visualization to: {plot_path}")
        plt.close()
    
    # 2. Gender percentage by cluster (stacked bar)
    fig, ax = plt.subplots(figsize=(12, 6))
    gender_pct_cols = [col for col in gender_analysis_df.columns if col.endswith('_Percentage')]
    
    if gender_pct_cols:
        gender_names = [col.replace('_Percentage', '') for col in gender_pct_cols]
        bottom = np.zeros(len(gender_analysis_df))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(gender_pct_cols)))
        
        for col, gender_name, color in zip(gender_pct_cols, gender_names, colors):
            ax.bar(gender_analysis_df['Cluster'], 
                   gender_analysis_df[col], 
                   bottom=bottom, 
                   label=gender_name,
                   color=color)
            bottom += gender_analysis_df[col]
        
        ax.set_xlabel('Cluster', fontsize=12)
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_title('Gender Distribution by Cluster (Percentage)', fontsize=14, fontweight='bold')
        ax.set_xticks(sorted(gender_analysis_df['Cluster'].unique()))
        ax.legend(loc='upper right')
        ax.set_ylim(0, 100)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'gender_distribution_percentage_by_cluster.png')
        plt.savefig(plot_path, dpi=300)
        print(f"✓ Saved visualization to: {plot_path}")
        plt.close()
    
    # 3. Gender distribution by year and cluster
    if not year_gender_analysis_df.empty:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        years = sorted(year_gender_analysis_df['Year'].dropna().unique())
        genders = year_gender_analysis_df['Gender'].unique()
        
        for gender in genders:
            gender_data = year_gender_analysis_df[year_gender_analysis_df['Gender'] == gender]
            year_gender_data = gender_data.groupby(['Year', 'Cluster'])['Count'].sum().reset_index()
            
            # Create pivot for plotting
            pivot_data = year_gender_data.pivot(index='Year', columns='Cluster', values='Count').fillna(0)
            
            ax.plot(pivot_data.index, pivot_data.sum(axis=1), marker='o', label=gender, linewidth=2)
        
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Number of Students', fontsize=12)
        ax.set_title('Gender Distribution Across Years', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'gender_year_distribution.png')
        plt.savefig(plot_path, dpi=300)
        print(f"✓ Saved visualization to: {plot_path}")
        plt.close()


if __name__ == "__main__":
    print("Starting Gender Analysis...")
    print("="*80)
    
    # Process gender and year data
    results_df, student_info = process_gender_data()
    
    # Analyze gender by cluster
    gender_analysis_df = analyze_gender_by_cluster(results_df)
    
    # Analyze gender and year by cluster
    year_gender_analysis_df = analyze_gender_year_by_cluster(results_df)
    
    # Save reports
    save_analysis_reports(results_df, gender_analysis_df, year_gender_analysis_df)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_visualizations(results_df, gender_analysis_df, year_gender_analysis_df)
    
    print("\n" + "="*80)
    print("Gender Analysis Complete!")
    print("="*80)
