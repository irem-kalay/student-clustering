import pandas as pd

def analyze_clusters():
    # 1. Load your generated files
    clusters_df = pd.read_csv('student_final_clusters.csv')
    vectors_df = pd.read_csv('vector-equivalent.csv')

    # 2. Merge them on Student_ID
    # This links the "Cluster ID" to the actual "Grades"
    merged_df = pd.merge(clusters_df, vectors_df, on='Student_ID')

    # 3. Calculate Averages
    # Global average (for comparison)
    global_mean = vectors_df.mean(numeric_only=True)
    # Average per cluster
    cluster_means = merged_df.groupby('Assigned_Cluster').mean(numeric_only=True)

    # 4. Print Report
    print("=== CLUSTER INTERPRETATION ===")
    for cluster_id in sorted(cluster_means.index):
        print(f"\n--- Cluster {cluster_id} ---")
        n_students = len(merged_df[merged_df['Assigned_Cluster'] == cluster_id])
        print(f"Number of Students: {n_students}")
        
        # Compare this cluster to the global average
        diff = cluster_means.loc[cluster_id] - global_mean
        
        # Find features where this cluster is MUCH higher than average
        strengths = diff.sort_values(ascending=False).head(5)
        print("\n  Distinctive Strengths (Higher than Average):")
        for course, val in strengths.items():
            actual_grade = cluster_means.loc[cluster_id, course]
            print(f"    {course}: {actual_grade:.2f} (Diff: +{val:.2f})")
            
        # Find features where this cluster is MUCH lower than average
        weaknesses = diff.sort_values(ascending=True).head(5)
        print("\n  Distinctive Weaknesses (Lower than Average):")
        for course, val in weaknesses.items():
            actual_grade = cluster_means.loc[cluster_id, course]
            print(f"    {course}: {actual_grade:.2f} (Diff: {val:.2f})")

if __name__ == "__main__":
    analyze_clusters()