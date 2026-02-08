# -*- coding: utf-8 -*-
"""
Generate Cluster Personas for Agentic AI System

This module generates detailed English persona descriptions for each student cluster,
useful for agentic AI systems to understand and describe different student types.
"""

import pandas as pd
import json
import os


def generate_cluster_persona(cluster_id, centroid_row, rank_row, gap_row, cluster_size, gender_dist, cats, weighted_features, overall_mean, strong_cats, weak_cats):
    """
    Generate a detailed English persona description for a cluster based on its characteristics.
    
    Args:
        cluster_id: Cluster identifier
        centroid_row: Centroid values for the cluster
        rank_row: Rank values (1=best, 20=worst)
        gap_row: Gap to best cluster values
        cluster_size: Number of students in cluster
        gender_dist: Gender distribution (dict with percentages)
        cats: List of category names
        weighted_features: DataFrame with weighted features
        overall_mean: Overall mean scores for categories
        strong_cats: List of categories above average
        weak_cats: List of categories below average
    """
    
    # Identify strengths (top 3 categories - lowest rank means best)
    strengths = []
    rank_items = [(cat, rank_row[cat]) for cat in cats]
    rank_items.sort(key=lambda x: x[1])
    strengths = [cat for cat, _ in rank_items[:3]]
    
    # Identify weaknesses (bottom 3 categories - highest rank means worst)
    weaknesses = [cat for cat, _ in rank_items[-3:]]
    
    # Get average centroid
    avg_centroid = centroid_row.mean()
    performance_level = "Excellent" if avg_centroid > 0.65 else "Good" if avg_centroid > 0.55 else "Moderate" if avg_centroid > 0.45 else "Below Average"
    
    # Generate a characteristic title based on actual strengths (top ranked by competency)
    # Use strengths (ranked best/worst) instead of strong_cats (above/below average)
    # This ensures Excellent clusters aren't homogenous in naming
    title_cats = strengths[:2]
    
    if title_cats:
        # Create meaningful cluster titles based on category combinations
        cat_combinations = ' & '.join([cat.replace('_', ' ') for cat in title_cats])
        
        if 'Systems' in title_cats and 'Hardware' in title_cats:
            title = "Systems & Hardware Specialists"
        elif 'Basic_Sciences' in title_cats and 'Software_Practice' in title_cats:
            title = "Technical Excellence & Development"
        elif 'Algorithm_Theory' in title_cats and 'Math_Calc' in title_cats:
            title = "Algorithmic & Mathematical Focus"
        elif 'Social_Cultural' in title_cats and 'Language_Comm' in title_cats:
            title = "Social & Communication Leaders"
        elif 'Systems' in title_cats:
            title = "Systems & Infrastructure Focused"
        elif 'Hardware' in title_cats:
            title = "Hardware & Design Specialists"
        elif 'Software_Practice' in title_cats:
            title = "Software Development Focused"
        elif 'Algorithm_Theory' in title_cats:
            title = "Algorithmic Problem Solvers"
        elif 'Language_Comm' in title_cats:
            title = "Communication & Language Strong"
        elif 'Social_Cultural' in title_cats:
            title = "Social & Cultural Oriented"
        elif 'Math_Calc' in title_cats:
            title = "Mathematical & Computational Focus"
        else:
            title = f"{cat_combinations} Specialists"
    else:
        # Fallback to performance level if no ranked strengths
        title = f"{performance_level} Performers"
    
    # Gender description
    gender_desc = "Balanced gender distribution" if abs(gender_dist.get('Erkek', 0) - gender_dist.get('Kadın', 0)) < 15 else "Female-dominated" if gender_dist.get('Kadın', 0) > 50 else "Male-dominated"
    
    # Build "Ideal For" section with proper formatting
    ideal_for_list = []
    if 'Systems' in strong_cats:
        ideal_for_list.append("Backend/Systems development roles")
    if 'Software_Practice' in strong_cats:
        ideal_for_list.append("Software engineering positions")
    if 'Algorithm_Theory' in strong_cats:
        ideal_for_list.append("Algorithmic challenges")
    if 'Hardware' in strong_cats:
        ideal_for_list.append("Hardware-focused work")
    if 'Basic_Sciences' in strong_cats:
        ideal_for_list.append("Core scientific development")
    
    ideal_for_str = " • ".join(ideal_for_list) if ideal_for_list else "Diverse technical roles"
    
    # Build description with weakness included in title for better visibility
    description = f"""Cluster {cluster_id}: {title} [Weak: {weaknesses[-1].replace('_', ' ')}] ({cluster_size} students)

Performance Profile:
- Overall Performance Level: {performance_level} (average score: {avg_centroid:.3f})
- Primary Strengths: {', '.join(strengths[:2])} - excelling in these areas with strong competency levels
- Primary Weakness: {weaknesses[-1]} - significantly below-average performance
- Strong Categories: {', '.join(strong_cats) if strong_cats else 'None above average'}
- Weak Categories: {', '.join(weak_cats) if weak_cats else 'None below average'}

Demographic Profile:
- Cluster Size: {cluster_size} students ({cluster_size/len(weighted_features)*100:.1f}% of total)
- Gender Distribution: {gender_desc} ({gender_dist.get('Erkek', 0):.1f}% Male, {gender_dist.get('Kadın', 0):.1f}% Female, {gender_dist.get('Bilinmiyor', 0):.1f}% Unknown)

Distinguishing Characteristics:
- This cluster represents students who are particularly strong in {strengths[0]} and {strengths[1]}
- These students show solid fundamentals across {'technical areas' if 'Algorithm_Theory' in strengths or 'Software_Practice' in strengths else 'academic areas'}
- Notable gap in {weaknesses[-1]} suggests this group either deprioritized or found {weaknesses[-1].replace('_', ' ').lower()} challenging
- {'Diverse gender composition indicates inclusive skill distribution' if 'Balanced' in gender_desc else 'Specific gender predominance in this cluster'}

Ideal For:
- {ideal_for_str}
- Career paths emphasizing {strengths[0].replace('_', ' ').lower()} and {strengths[1].replace('_', ' ').lower()}
"""
    
    return description.strip()


def generate_personas(weighted_features, centroids, rank_df, gap_to_best, cluster_sizes, gender_cluster_pct, 
                     cats, n_clusters, overall_mean):
    """
    Generate personas for all clusters and save to JSON and TXT files.
    
    Args:
        weighted_features: DataFrame with weighted features and cluster assignments
        centroids: Cluster centroids
        rank_df: Rank dataframe
        gap_to_best: Gap to best cluster
        cluster_sizes: Cluster sizes
        gender_cluster_pct: Gender percentages by cluster
        cats: Category names
        n_clusters: Number of clusters
        overall_mean: Overall mean scores
    """
    
    print("\n=== Generating Cluster Persona Descriptions ===")
    
    cluster_personas = {}
    
    for cluster_id in range(n_clusters):
        centroid_row = centroids.loc[cluster_id]
        rank_row = rank_df.loc[cluster_id]
        gap_row = gap_to_best.loc[cluster_id]
        cluster_size = cluster_sizes.loc[cluster_id]
        
        # Get gender distribution for this cluster
        gender_dist = {}
        if cluster_id in gender_cluster_pct.index:
            for gender in ['Erkek', 'Kadın', 'Bilinmiyor']:
                gender_dist[gender] = gender_cluster_pct.loc[cluster_id, gender] if gender in gender_cluster_pct.columns else 0.0
        
        # Get strong and weak categories
        strong_cats = [cat for cat in cats if centroid_row[cat] > overall_mean[cat]]
        weak_cats = [cat for cat in cats if centroid_row[cat] < overall_mean[cat]]
        
        persona = generate_cluster_persona(cluster_id, centroid_row, rank_row, gap_row, cluster_size, gender_dist,
                                          cats, weighted_features, overall_mean, strong_cats, weak_cats)
        
        avg_centroid = centroid_row.mean()
        cluster_personas[f"cluster_{cluster_id}"] = {
            "cluster_id": int(cluster_id),
            "size": int(cluster_size),
            "performance_level": "Excellent" if avg_centroid > 0.65 else "Good" if avg_centroid > 0.55 else "Moderate" if avg_centroid > 0.45 else "Below Average",
            "average_score": float(round(avg_centroid, 3)),
            "strengths": [str(cat) for cat in cats if centroid_row[cat] > overall_mean[cat]][:5],
            "weaknesses": [str(cat) for cat in cats if centroid_row[cat] < overall_mean[cat]][-3:],
            "gender_distribution": {
                "male_percentage": float(round(gender_dist.get('Erkek', 0.0), 1)),
                "female_percentage": float(round(gender_dist.get('Kadın', 0.0), 1)),
                "unknown_percentage": float(round(gender_dist.get('Bilinmiyor', 0.0), 1))
            },
            "description": persona
        }
    
    # Save as JSON
    json_output = {
        "metadata": {
            "total_clusters": n_clusters,
            "total_students": len(weighted_features),
            "categories": cats,
            "description": "Student cluster personas for agentic AI - describes each student cluster type with strengths, weaknesses, and demographic information"
        },
        "clusters": cluster_personas
    }
    
    # Create AgenticAI directory if it doesn't exist
    os.makedirs("clustering-pipeline/results/AgenticAI", exist_ok=True)
    
    with open("clustering-pipeline/results/AgenticAI/cluster_personas.json", "w", encoding='utf-8') as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    
    print("Saved: clustering-pipeline/results/AgenticAI/cluster_personas.json")
    
    # Also save as readable text file
    with open("clustering-pipeline/results/AgenticAI/cluster_personas.txt", "w", encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("STUDENT CLUSTER PERSONAS - For Agentic AI System\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total Clusters: {n_clusters}\n")
        f.write(f"Total Students: {len(weighted_features)}\n")
        f.write(f"Analysis Categories: {', '.join(cats)}\n\n")
        f.write("=" * 80 + "\n\n")
        
        for cluster_id in range(n_clusters):
            f.write(cluster_personas[f"cluster_{cluster_id}"]["description"])
            f.write("\n\n" + "-" * 80 + "\n\n")
    
    print("Saved: clustering-pipeline/results/AgenticAI/cluster_personas.txt")
    
    # Print summary to console
    print("\n=== Cluster Personas Summary ===")
    for cluster_id in range(n_clusters):
        info = cluster_personas[f"cluster_{cluster_id}"]
        print(f"\nCluster {cluster_id}: {info['performance_level']} ({info['size']} students)")
        print(f"  - Avg Score: {info['average_score']:.3f}")
        print(f"  - Strengths: {', '.join(info['strengths'][:2])}")
        print(f"  - Gender: {info['gender_distribution']['male_percentage']:.1f}% M, {info['gender_distribution']['female_percentage']:.1f}% F")
    
    print("\n✅ Cluster personas generated successfully!")
    
    return cluster_personas


def load_existing_data():
    """Load pre-computed data from CSV files for standalone persona generation."""
    import os
    import numpy as np
    
    results_dir = "clustering-pipeline/results"
    
    # Check if required files exist
    required_files = [
        "final_student_clusters_no_naming.csv",
        "cluster_centroid_ranks.csv",
        "cluster_gap_to_best.csv",
        "cluster_gender_percentage.csv"
    ]
    
    for file in required_files:
        if not os.path.exists(os.path.join(results_dir, file)):
            raise FileNotFoundError(f"Missing required file: {os.path.join(results_dir, file)}")
    
    print("Loading existing clustering data...")
    
    # Load weighted features and cluster assignments
    weighted_features = pd.read_csv(os.path.join(results_dir, "final_student_clusters_no_naming.csv"))
    
    # Load rank dataframe with proper type conversion
    rank_df = pd.read_csv(os.path.join(results_dir, "cluster_centroid_ranks.csv"), index_col=0)
    rank_df = rank_df.astype('int64')  # Ensure int type
    
    # Load gap to best
    gap_to_best = pd.read_csv(os.path.join(results_dir, "cluster_gap_to_best.csv"), index_col=0)
    gap_to_best = gap_to_best.astype('float64')  # Ensure float type
    
    # Load gender percentage by cluster
    gender_cluster_pct = pd.read_csv(os.path.join(results_dir, "cluster_gender_percentage.csv"), index_col=0)
    gender_cluster_pct = gender_cluster_pct.astype('float64')  # Ensure float type
    
    # Define category names
    cats = ['Basic_Sciences', 'Software_Practice', 'Algorithm_Theory',
            'Systems', 'Hardware', 'Social_Cultural', 'Language_Comm', 'Math_Calc']
    
    # Extract cluster assignments and compute centroids
    n_clusters = int(weighted_features['Cluster'].max() + 1)
    cluster_sizes = weighted_features['Cluster'].value_counts().sort_index()
    cluster_sizes = cluster_sizes.astype('int64')  # Ensure int type
    
    # Compute centroids from weighted features
    centroids = weighted_features.groupby("Cluster")[cats].mean().sort_index()
    
    # Compute overall mean
    overall_mean = weighted_features[cats].mean()
    
    print(f"✅ Data loaded successfully!")
    print(f"   - {len(weighted_features)} students")
    print(f"   - {n_clusters} clusters")
    print(f"   - {len(cats)} categories")
    
    return weighted_features, centroids, rank_df, gap_to_best, cluster_sizes, gender_cluster_pct, cats, n_clusters, overall_mean


if __name__ == "__main__":
    print("=" * 80)
    print("STANDALONE AGENTIC PERSONAS GENERATOR")
    print("=" * 80)
    
    try:
        # Load all required data
        weighted_features, centroids, rank_df, gap_to_best, cluster_sizes, gender_cluster_pct, cats, n_clusters, overall_mean = load_existing_data()
        
        # Generate personas
        generate_personas(weighted_features, centroids, rank_df, gap_to_best, cluster_sizes, 
                         gender_cluster_pct, cats, n_clusters, overall_mean)
        
        print("\n" + "=" * 80)
        print("✅ Persona generation completed successfully!")
        print("=" * 80)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have run clustering.py first to generate the required CSV files.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
