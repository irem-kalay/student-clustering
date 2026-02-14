"""
Gender Performance Success Rate Analysis
Analyzes overall student performance by gender within each cluster
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

# Feature columns (course scores)
FEATURE_COLUMNS = [
    'Basic_Sciences',
    'Software_Practice', 
    'Algorithm_Theory',
    'Systems',
    'Hardware',
    'Social_Cultural',
    'Language_Comm',
    'Math_Calc'
]

OUTPUT_DIR = 'clustering-pipeline/results'

# =============================================================================
# LOAD DATA
# =============================================================================

print("=" * 80)
print("GENDER PERFORMANCE SUCCESS RATE ANALYSIS")
print("=" * 80)

print("\nLoading enriched results with gender and cluster information...")
results_df = pd.read_csv(
    os.path.join(OUTPUT_DIR, 'final_results_with_gender.csv'),
    encoding='utf-8'
)

print(f"Loaded {len(results_df)} student records")
print(f"Clusters: {sorted(results_df['Cluster_DEC'].unique())}")
print(f"Genders: {results_df['Gender'].unique()}")

# =============================================================================
# CALCULATE SUCCESS RATES
# =============================================================================

def calculate_success_rate_by_gender():
    """
    Calculate success rate as average performance score for each gender in each cluster
    Success Rate = Average of all feature scores for that gender+cluster combination
    """
    
    success_data = []
    
    for cluster in sorted(results_df['Cluster_DEC'].unique()):
        cluster_data = results_df[results_df['Cluster_DEC'] == cluster]
        
        print(f"\n--- CLUSTER {cluster} ---")
        
        for gender in ['Erkek', 'Kadın']:  # Male, Female in Turkish
            gender_data = cluster_data[cluster_data['Gender'] == gender]
            
            if len(gender_data) > 0:
                # Calculate average performance across all features
                avg_performance = gender_data[FEATURE_COLUMNS].mean().mean()
                
                # Calculate success rate as percentage above cluster mean
                cluster_mean = cluster_data[FEATURE_COLUMNS].mean().mean()
                success_percentage = (avg_performance / cluster_mean * 100) if cluster_mean > 0 else 0
                
                # Get stats
                count = len(gender_data)
                std_dev = gender_data[FEATURE_COLUMNS].mean(axis=1).std()
                
                print(f"  {gender}:")
                print(f"    Count: {count}")
                print(f"    Avg Performance Score: {avg_performance:.4f}")
                print(f"    Success Rate: {success_percentage:.2f}%")
                print(f"    Std Dev: {std_dev:.4f}")
                
                success_data.append({
                    'Cluster': cluster,
                    'Gender': gender,
                    'Count': count,
                    'Avg_Performance': avg_performance,
                    'Success_Rate_Percent': success_percentage,
                    'Std_Dev': std_dev
                })
    
    return pd.DataFrame(success_data)


def calculate_high_performers_percentage():
    """
    Calculate what percentage of students are above their gender+cluster average
    """
    
    high_performer_data = []
    
    for cluster in sorted(results_df['Cluster_DEC'].unique()):
        cluster_data = results_df[results_df['Cluster_DEC'] == cluster]
        
        for gender in ['Erkek', 'Kadın']:
            gender_data = cluster_data[cluster_data['Gender'] == gender]
            
            if len(gender_data) > 0:
                # Calculate individual average scores
                gender_data_copy = gender_data.copy()
                gender_data_copy['Individual_Avg_Score'] = gender_data_copy[FEATURE_COLUMNS].mean(axis=1)
                
                # Gender+cluster mean
                gender_cluster_mean = gender_data_copy['Individual_Avg_Score'].mean()
                
                # Count high performers (above gender+cluster mean)
                high_performers = (gender_data_copy['Individual_Avg_Score'] > gender_cluster_mean).sum()
                high_performer_percentage = (high_performers / len(gender_data_copy) * 100)
                
                high_performer_data.append({
                    'Cluster': cluster,
                    'Gender': gender,
                    'Total_Students': len(gender_data_copy),
                    'High_Performers': high_performers,
                    'High_Performer_Percent': high_performer_percentage,
                    'Cluster_Gender_Mean': gender_cluster_mean
                })
    
    return pd.DataFrame(high_performer_data)


def calculate_overall_success_rate():
    """
    Calculate overall success rates for all students combined (not cluster-specific)
    Compares male vs female across entire dataset
    """
    
    print("\n" + "=" * 80)
    print("OVERALL GENDER SUCCESS RATE ANALYSIS (ALL DATA COMBINED)")
    print("=" * 80)
    
    overall_data = []
    
    # Calculate for all students (entire dataset)
    all_cluster_mean = results_df[FEATURE_COLUMNS].mean().mean()
    
    for gender in ['Erkek', 'Kadın']:
        gender_data = results_df[results_df['Gender'] == gender]
        
        if len(gender_data) > 0:
            # Average performance for this gender across all courses
            avg_performance = gender_data[FEATURE_COLUMNS].mean().mean()
            
            # Success rate relative to overall dataset mean
            success_percentage = (avg_performance / all_cluster_mean * 100) if all_cluster_mean > 0 else 0
            
            # Additional metrics
            count = len(gender_data)
            std_dev = gender_data[FEATURE_COLUMNS].mean(axis=1).std()
            
            # Calculate individual course averages
            gender_data_copy = gender_data.copy()
            gender_data_copy['Individual_Avg_Score'] = gender_data_copy[FEATURE_COLUMNS].mean(axis=1)
            gender_avg = gender_data_copy['Individual_Avg_Score'].mean()
            
            # High performers (above overall dataset mean for that gender)
            high_performers = (gender_data_copy['Individual_Avg_Score'] > gender_avg).sum()
            high_performer_pct = (high_performers / len(gender_data_copy) * 100)
            
            print(f"\n{gender} Students (Overall):")
            print(f"  Total Count: {count}")
            print(f"  Average Performance Score: {avg_performance:.4f}")
            print(f"  Success Rate: {success_percentage:.2f}%")
            print(f"  Standard Deviation: {std_dev:.4f}")
            print(f"  High Performers (Above Gender Average): {high_performers} ({high_performer_pct:.1f}%)")
            
            overall_data.append({
                'Gender': gender,
                'Total_Students': count,
                'Avg_Performance': avg_performance,
                'Success_Rate_Percent': success_percentage,
                'Std_Dev': std_dev,
                'High_Performers': high_performers,
                'High_Performer_Percent': high_performer_pct,
                'Gender_Avg_Score': gender_avg
            })
    
    return pd.DataFrame(overall_data)


def compare_gender_success():
    """
    Compare success rates between males and females within each cluster
    """
    
    success_df = calculate_success_rate_by_gender()
    high_perf_df = calculate_high_performers_percentage()
    
    print("\n" + "=" * 80)
    print("GENDER COMPARISON WITHIN CLUSTERS")
    print("=" * 80)
    
    for cluster in sorted(success_df['Cluster'].unique()):
        cluster_success = success_df[success_df['Cluster'] == cluster]
        
        male_data = cluster_success[cluster_success['Gender'] == 'Erkek']
        female_data = cluster_success[cluster_success['Gender'] == 'Kadın']
        
        if len(male_data) > 0 and len(female_data) > 0:
            male_rate = male_data['Success_Rate_Percent'].values[0]
            female_rate = female_data['Success_Rate_Percent'].values[0]
            
            print(f"\nCluster {cluster}:")
            print(f"  Male Success Rate: {male_rate:.2f}%")
            print(f"  Female Success Rate: {female_rate:.2f}%")
            
            if male_rate > female_rate:
                diff = male_rate - female_rate
                print(f"  ✓ Males perform {diff:.2f}% better")
            else:
                diff = female_rate - male_rate
                print(f"  ✓ Females perform {diff:.2f}% better")
    
    return success_df, high_perf_df


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

# Calculate cluster-based success rates
success_rates_df, high_performers_df = compare_gender_success()

# Calculate overall success rates (all data combined)
overall_success_df = calculate_overall_success_rate()

# Save results
success_rates_df.to_csv(
    os.path.join(OUTPUT_DIR, 'gender_success_rate_analysis.csv'),
    index=False
)
print(f"\n✓ Saved success rate analysis to: {OUTPUT_DIR}/gender_success_rate_analysis.csv")

high_performers_df.to_csv(
    os.path.join(OUTPUT_DIR, 'gender_high_performers_analysis.csv'),
    index=False
)
print(f"✓ Saved high performers analysis to: {OUTPUT_DIR}/gender_high_performers_analysis.csv")

overall_success_df.to_csv(
    os.path.join(OUTPUT_DIR, 'gender_overall_success_rate_analysis.csv'),
    index=False
)
print(f"✓ Saved overall success rate analysis to: {OUTPUT_DIR}/gender_overall_success_rate_analysis.csv")

# =============================================================================
# VISUALIZATIONS
# =============================================================================

print("\nGenerating visualizations...")

# 1. Success Rate Comparison (Bar Chart)
fig, ax = plt.subplots(figsize=(12, 6))

clusters = sorted(success_rates_df['Cluster'].unique())
male_rates = []
female_rates = []

for cluster in clusters:
    cluster_data = success_rates_df[success_rates_df['Cluster'] == cluster]
    male = cluster_data[cluster_data['Gender'] == 'Erkek']['Success_Rate_Percent'].values
    female = cluster_data[cluster_data['Gender'] == 'Kadın']['Success_Rate_Percent'].values
    
    male_rates.append(male[0] if len(male) > 0 else 0)
    female_rates.append(female[0] if len(female) > 0 else 0)

x = np.arange(len(clusters))
width = 0.35

bars1 = ax.bar(x - width/2, male_rates, width, label='Male (Erkek)', color='#3498db', alpha=0.8)
bars2 = ax.bar(x + width/2, female_rates, width, label='Female (Kadın)', color='#e74c3c', alpha=0.8)

ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Gender Performance Success Rate Comparison by Cluster', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(clusters)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, max(max(male_rates), max(female_rates)) * 1.1)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, 'gender_success_rate_comparison.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved visualization to: {plot_path}")
plt.close()

# 2. High Performers Percentage Comparison
fig, ax = plt.subplots(figsize=(12, 6))

male_high_perf = []
female_high_perf = []

for cluster in clusters:
    cluster_data = high_performers_df[high_performers_df['Cluster'] == cluster]
    male = cluster_data[cluster_data['Gender'] == 'Erkek']['High_Performer_Percent'].values
    female = cluster_data[cluster_data['Gender'] == 'Kadın']['High_Performer_Percent'].values
    
    male_high_perf.append(male[0] if len(male) > 0 else 0)
    female_high_perf.append(female[0] if len(female) > 0 else 0)

x = np.arange(len(clusters))
width = 0.35

bars1 = ax.bar(x - width/2, male_high_perf, width, label='Male (Erkek)', color='#2ecc71', alpha=0.8)
bars2 = ax.bar(x + width/2, female_high_perf, width, label='Female (Kadın)', color='#f39c12', alpha=0.8)

ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
ax.set_ylabel('Percentage of High Performers (%)', fontsize=12, fontweight='bold')
ax.set_title('High Performers by Cluster and Gender\n(Students Above Their Gender+Cluster Average)', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(clusters)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, 100)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, 'gender_high_performers_comparison.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved visualization to: {plot_path}")
plt.close()

# 3. Gender Performance Difference Heatmap
fig, ax = plt.subplots(figsize=(10, 6))

diff_matrix = []
for cluster in clusters:
    cluster_data = success_rates_df[success_rates_df['Cluster'] == cluster]
    male = cluster_data[cluster_data['Gender'] == 'Erkek']['Success_Rate_Percent'].values
    female = cluster_data[cluster_data['Gender'] == 'Kadın']['Success_Rate_Percent'].values
    
    if len(male) > 0 and len(female) > 0:
        # Positive = Male better, Negative = Female better
        diff = male[0] - female[0]
        diff_matrix.append(diff)
    else:
        diff_matrix.append(0)

colors = ['#e74c3c' if x < 0 else '#3498db' for x in diff_matrix]
bars = ax.barh(clusters, diff_matrix, color=colors, alpha=0.8)

ax.set_xlabel('Success Rate Difference (%)\n(Positive = Males Better, Negative = Females Better)', 
              fontsize=11, fontweight='bold')
ax.set_ylabel('Cluster', fontsize=12, fontweight='bold')
ax.set_title('Gender Performance Difference by Cluster', fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, diff_matrix)):
    label_x = val + (0.2 if val > 0 else -0.2)
    ax.text(label_x, i, f'{val:.2f}%', va='center', 
            ha='left' if val > 0 else 'right', fontsize=10, fontweight='bold')

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, 'gender_performance_difference.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved visualization to: {plot_path}")
plt.close()

# 4. Average Feature Scores by Gender and Cluster
fig, axes = plt.subplots(2, 4, figsize=(16, 10))
axes = axes.flatten()

for idx, feature in enumerate(FEATURE_COLUMNS):
    ax = axes[idx]
    
    male_avgs = []
    female_avgs = []
    
    for cluster in clusters:
        cluster_data = results_df[results_df['Cluster_DEC'] == cluster]
        
        male_data = cluster_data[cluster_data['Gender'] == 'Erkek']
        female_data = cluster_data[cluster_data['Gender'] == 'Kadın']
        
        male_avg = male_data[feature].mean() if len(male_data) > 0 else 0
        female_avg = female_data[feature].mean() if len(female_data) > 0 else 0
        
        male_avgs.append(male_avg)
        female_avgs.append(female_avg)
    
    x = np.arange(len(clusters))
    width = 0.35
    
    ax.bar(x - width/2, male_avgs, width, label='Male', color='#3498db', alpha=0.8)
    ax.bar(x + width/2, female_avgs, width, label='Female', color='#e74c3c', alpha=0.8)
    
    ax.set_title(feature, fontweight='bold', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(clusters, fontsize=9)
    ax.set_ylabel('Average Score', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=9)

plt.suptitle('Feature Scores by Gender and Cluster', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, 'feature_scores_by_gender_cluster.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved visualization to: {plot_path}")
plt.close()

# =============================================================================
# OVERALL ANALYSIS VISUALIZATIONS
# =============================================================================

# 5. Overall Success Rate Comparison (All Data)
fig, ax = plt.subplots(figsize=(10, 6))

genders = overall_success_df['Gender'].values
rates = overall_success_df['Success_Rate_Percent'].values
colors = ['#3498db', '#e74c3c']

bars = ax.bar(genders, rates, color=colors, alpha=0.8, width=0.5)

ax.set_ylabel('Overall Success Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Overall Gender Performance Success Rate\n(All Students Combined)', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, max(rates) * 1.15)

# Add value labels
for bar, rate in zip(bars, rates):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{rate:.2f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add count labels
for i, (gender, count) in enumerate(zip(genders, overall_success_df['Total_Students'].values)):
    ax.text(i, 1, f'n={count}', ha='center', va='bottom', fontsize=10, style='italic')

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, 'gender_overall_success_rate_comparison.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved visualization to: {plot_path}")
plt.close()

# 6. Overall High Performers Comparison (All Data)
fig, ax = plt.subplots(figsize=(10, 6))

genders = overall_success_df['Gender'].values
high_perf_pcts = overall_success_df['High_Performer_Percent'].values
high_perf_counts = overall_success_df['High_Performers'].values
colors = ['#2ecc71', '#f39c12']

bars = ax.bar(genders, high_perf_pcts, color=colors, alpha=0.8, width=0.5)

ax.set_ylabel('Percentage of High Performers (%)', fontsize=12, fontweight='bold')
ax.set_title('Overall High Performers by Gender\n(Students Above Gender Average)\nAll Data Combined', 
             fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, 100)

# Add value labels
for bar, pct, count in zip(bars, high_perf_pcts, high_perf_counts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{pct:.1f}%\n({int(count)} students)',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, 'gender_overall_high_performers_comparison.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved visualization to: {plot_path}")
plt.close()

# 7. Overall Performance Statistics Comparison
fig, ax = plt.subplots(figsize=(12, 6))

stats_data = []
for _, row in overall_success_df.iterrows():
    stats_data.append({
        'metric': f"{row['Gender']}\nAvg Performance",
        'value': row['Avg_Performance']
    })
    stats_data.append({
        'metric': f"{row['Gender']}\nStd Dev",
        'value': row['Std_Dev']
    })

x_pos = np.arange(len(stats_data))
metrics = [d['metric'] for d in stats_data]
values = [d['value'] for d in stats_data]
colors_list = ['#3498db', '#3498db', '#e74c3c', '#e74c3c']

bars = ax.bar(x_pos, values, color=colors_list, alpha=0.8)

ax.set_xticks(x_pos)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylabel('Score Value', fontsize=12, fontweight='bold')
ax.set_title('Overall Gender Performance Statistics\n(Average Score and Variability)', 
             fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.4f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, 'gender_overall_statistics_comparison.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved visualization to: {plot_path}")
plt.close()

# 8. Individual Course Performance - Overall
fig, ax = plt.subplots(figsize=(14, 6))

male_course_avgs = []
female_course_avgs = []

for feature in FEATURE_COLUMNS:
    male_data = results_df[results_df['Gender'] == 'Erkek']
    female_data = results_df[results_df['Gender'] == 'Kadın']
    
    male_avg = male_data[feature].mean()
    female_avg = female_data[feature].mean()
    
    male_course_avgs.append(male_avg)
    female_course_avgs.append(female_avg)

x = np.arange(len(FEATURE_COLUMNS))
width = 0.35

bars1 = ax.bar(x - width/2, male_course_avgs, width, label='Male (Erkek)', color='#3498db', alpha=0.8)
bars2 = ax.bar(x + width/2, female_course_avgs, width, label='Female (Kadın)', color='#e74c3c', alpha=0.8)

ax.set_xlabel('Courses', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Score', fontsize=12, fontweight='bold')
ax.set_title('Overall Course Performance by Gender\n(All Data Combined)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(FEATURE_COLUMNS, rotation=45, ha='right')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, 'gender_overall_course_performance.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved visualization to: {plot_path}")
plt.close()

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print("\nGenerated Files:")
print("  CSV Files:")
print("    - gender_success_rate_analysis.csv (Cluster-based)")
print("    - gender_high_performers_analysis.csv (Cluster-based)")
print("    - gender_overall_success_rate_analysis.csv (Overall)")
print("\n  Cluster-Based Analysis Visualizations:")
print("    - gender_success_rate_comparison.png")
print("    - gender_high_performers_comparison.png")
print("    - gender_performance_difference.png")
print("    - feature_scores_by_gender_cluster.png")
print("\n  Overall Analysis Visualizations:")
print("    - gender_overall_success_rate_comparison.png")
print("    - gender_overall_high_performers_comparison.png")
print("    - gender_overall_statistics_comparison.png")
print("    - gender_overall_course_performance.png")
