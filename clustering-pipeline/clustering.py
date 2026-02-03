import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# =============================================================================
# 1. DATA LOADING AND PREPARATION
# =============================================================================
try:
    df = pd.read_csv('feature-vectors/student_feature_vectors2_collapsed.csv')
except FileNotFoundError:
    df = pd.read_csv('student_feature_vectors2_collapsed.csv')

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
    """
    Input: raw grade-like values with -1 meaning 'not taken'
    Output: each course scaled into [0.2, 1.0] for taken entries, and 0.0 for not taken.
    """
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
    'FIZ 101': {'Basic_Sciences': 0.8, 'Math_Calc': 0.2},
    'FIZ 101EL': {'Basic_Sciences': 1.0},
    'BLG 101': {'Basic_Sciences': 0.5, 'Software_Practice': 0.5},
    'BLG 113': {'Social_Cultural': 0.6, 'Systems': 0.4},
    'MAT 103': {'Math_Calc': 1.0},
    'MAT 281': {'Math_Calc': 0.8, 'Algorithm_Theory': 0.2},
    'ING 100': {'Language_Comm': 1.0},

    'BLG 112': {'Math_Calc': 0.6, 'Algorithm_Theory': 0.4},
    'BLG 102': {'Software_Practice': 0.8, 'Algorithm_Theory': 0.2},
    'MAT 104': {'Math_Calc': 1.0},
    'FIZ 102': {'Basic_Sciences': 0.8, 'Math_Calc': 0.2},
    'FIZ 102EL': {'Basic_Sciences': 1.0},
    'ING 112': {'Language_Comm': 1.0},
    'DAN 102': {'Social_Cultural': 1.0},

    'BLG 210': {'Math_Calc': 1.0},
    'BLG 231': {'Hardware': 1.0},
    'BLG 223': {'Algorithm_Theory': 0.5, 'Software_Practice': 0.5},
    'EHB 222': {'Hardware': 0.7, 'Basic_Sciences': 0.3},
    'EHB 211': {'Hardware': 0.6, 'Math_Calc': 0.4},
    'ING 201': {'Language_Comm': 1.0},

    'BLG 252': {'Software_Practice': 0.9, 'Algorithm_Theory': 0.1},
    'BLG 222': {'Hardware': 0.6, 'Systems': 0.4},
    'BLG 242': {'Hardware': 1.0},
    'BLG 202': {'Math_Calc': 0.7, 'Software_Practice': 0.3},
    'BLG 311': {'Algorithm_Theory': 0.7, 'Math_Calc': 0.3},
    'TUR 121': {'Social_Cultural': 1.0},

    'BLG 335': {'Algorithm_Theory': 0.6, 'Software_Practice': 0.2, 'Math_Calc': 0.2},
    'MAT 271': {'Math_Calc': 0.8, 'Algorithm_Theory': 0.2},
    'BLG 351': {'Hardware': 0.7, 'Software_Practice': 0.3},
    'TUR 122': {'Social_Cultural': 1.0},
    'BLG 317': {'Systems': 0.7, 'Software_Practice': 0.3},
    'BLG 212': {'Hardware': 0.5, 'Systems': 0.5},
    'EHB 311': {'Hardware': 1.0},

    'BLG 322': {'Hardware': 0.4, 'Systems': 0.6},
    'BLG 312': {'Systems': 0.8, 'Software_Practice': 0.2},
    'BLG 336': {'Algorithm_Theory': 0.7, 'Software_Practice': 0.2, 'Math_Calc': 0.1},
    'ATA 121': {'Social_Cultural': 1.0},
    'BLG 354': {'Math_Calc': 0.6, 'Systems': 0.4},
    'BLG 374': {'Language_Comm': 0.8, 'Social_Cultural': 0.2},

    'ATA 122': {'Social_Cultural': 1.0},
    'BLG 411': {'Software_Practice': 0.6, 'Systems': 0.2, 'Social_Cultural': 0.2},
    'BLG 4901': {'Software_Practice': 0.5, 'Systems': 0.3, 'Language_Comm': 0.2},

    'BLG 4902': {'Software_Practice': 0.4, 'Systems': 0.4, 'Language_Comm': 0.2},
    'EKO 201': {'Social_Cultural': 1.0},
}

elective_pools = {
    'MT': {
        'BLG 413': {'Systems': 0.6, 'Software_Practice': 0.4},
        'BLG 430': {'Systems': 0.9, 'Software_Practice': 0.1},
        'BLG 433': {'Systems': 0.9, 'Algorithm_Theory': 0.1},
        'BLG 434': {'Algorithm_Theory': 0.7, 'Software_Practice': 0.3},
        'BLG 435': {'Algorithm_Theory': 0.6, 'Math_Calc': 0.2, 'Software_Practice': 0.2},
        'BLG 438': {'Hardware': 1.0},
        'BLG 439': {'Software_Practice': 0.6, 'Systems': 0.2},
        'BLG 440': {'Software_Practice': 0.6, 'Systems': 0.2},
        'BLG 443': {'Algorithm_Theory': 0.5, 'Math_Calc': 0.3, 'Systems': 0.2},
        'BLG 444': {'Software_Practice': 0.4, 'Math_Calc': 0.4, 'Algorithm_Theory': 0.2},
        'BLG 447': {'Systems': 0.4, 'Algorithm_Theory': 0.4},
        'BLG 449': {'Systems': 0.5, 'Software_Practice': 0.5},
        'BLG 450': {'Systems': 0.7, 'Software_Practice': 0.3},
        'BLG 451': {'Systems': 0.6, 'Hardware': 0.4},
        'BLG 452': {'Hardware': 1.0},
        'BLG 453': {'Algorithm_Theory': 0.4, 'Math_Calc': 0.4, 'Software_Practice': 0.2},
        'BLG 456': {'Hardware': 0.4, 'Algorithm_Theory': 0.3, 'Systems': 0.3},
        'BLG 458': {'Software_Practice': 0.5, 'Algorithm_Theory': 0.5},
        'BLG 459': {'Systems': 0.8, 'Algorithm_Theory': 0.2},
        'BLG 460': {'Software_Practice': 0.7, 'Systems': 0.3},
        'BLG 468': {'Software_Practice': 0.8, 'Systems': 0.2},
        'BLG 475': {'Software_Practice': 0.8, 'Social_Cultural': 0.2},
        'BLG 477': {'Systems': 0.5, 'Software_Practice': 0.3, 'Algorithm_Theory': 0.2},
        'BLG 478': {'Systems': 0.8, 'Algorithm_Theory': 0.2},
        'BLG 483': {'Algorithm_Theory': 0.5, 'Software_Practice': 0.3, 'Hardware': 0.2},
        'YZV 406': {'Hardware': 0.4, 'Algorithm_Theory': 0.4, 'Math_Calc': 0.2},
    },
    'TM': {
        'BLG 337': {'Systems': 0.8, 'Algorithm_Theory': 0.2},
        'BLG 368': {'Math_Calc': 0.7, 'Algorithm_Theory': 0.3},
        'BLG 442': {'Social_Cultural': 0.6, 'Language_Comm': 0.4},
        'BLG 448': {'Social_Cultural': 0.5, 'Language_Comm': 0.3},
        'BLG 454': {'Algorithm_Theory': 0.5, 'Math_Calc': 0.3, 'Software_Practice': 0.2},
        'KON 224': {'Hardware': 0.7, 'Math_Calc': 0.3},
        'KON 317': {'Math_Calc': 0.5, 'Hardware': 0.5},
        'MAL 201': {'Basic_Sciences': 1.0},
    },
    'SOCIAL': {
        'BLG 346': {'Social_Cultural': 0.7, 'Software_Practice': 0.3},
        'SNT': {'Social_Cultural': 1.0},
    }
}

target_categories = [
    'Basic_Sciences', 'Software_Practice', 'Algorithm_Theory',
    'Systems', 'Hardware', 'Social_Cultural', 'Language_Comm', 'Math_Calc'
]

# =============================================================================
# 4. SCORE ENGINE (WITH COURSE-COUNT BONUS)
# =============================================================================
def calculate_dynamic_features(df_norm, core_map, elective_pools, targets,
                               alpha=0.15, bonus_type="log"):
    result = pd.DataFrame(0.0, index=df_norm.index, columns=targets)
    weight_counters = pd.DataFrame(0.0, index=df_norm.index, columns=targets)
    course_counts = pd.DataFrame(0, index=df_norm.index, columns=targets, dtype=int)

    def find_course_col(course_code):
        for col in df_norm.columns:
            if course_code == 'SNT' and col.startswith('SNT'):
                return col
            if course_code in col:
                return col
        return None

    for idx, row in df_norm.iterrows():
        # Core
        for course_code, weights in core_map.items():
            found_col = find_course_col(course_code)
            if found_col is None:
                continue
            score = row[found_col]
            if score > 0:
                for cat, w in weights.items():
                    if cat in targets:
                        result.loc[idx, cat] += score * w
                        weight_counters.loc[idx, cat] += w
                        course_counts.loc[idx, cat] += 1

        # Electives
        for pool_name, pool_courses in elective_pools.items():
            for course_code, weights in pool_courses.items():
                found_col = find_course_col(course_code)
                if found_col is None:
                    continue
                score = row[found_col]
                if score > 0:
                    for cat, w in weights.items():
                        if cat in targets:
                            result.loc[idx, cat] += score * w
                            weight_counters.loc[idx, cat] += w
                            course_counts.loc[idx, cat] += 1

    avg_scores = result / weight_counters.replace(0, 1.0)

    if bonus_type == "none":
        return avg_scores

    max_counts = course_counts.max(axis=0).replace(0, 1)
    if bonus_type == "log":
        bonus = np.log1p(course_counts) / np.log1p(max_counts)   # 0..1
    elif bonus_type == "sqrt":
        bonus = np.sqrt(course_counts / max_counts)             # 0..1
    else:
        raise ValueError("bonus_type must be one of: 'log', 'sqrt', 'none'")

    final_scores = avg_scores * (1.0 + alpha * bonus)
    return final_scores

weighted_features = calculate_dynamic_features(
    data_normalized, core_courses, elective_pools, target_categories,
    alpha=0.15,
    bonus_type="log"
)

# Bu satır kalmalı (istatistik/heatmap’lerde kategori listesi)
cats = target_categories

# =============================================================================
# 4B. AUTOENCODER (Representation Learning)
# =============================================================================
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --- 1) Input matrix (N x 8) ---
# weighted_features şu an sadece 8 kategori sütunu içeriyor -> (n_students, 8)
X_raw = weighted_features[cats].values.astype(np.float32)

# (opsiyonel ama iyi) AE eğitimini stabilize etmek için standartlaştırma
# Normalizasyonunu bozmaz: sadece AE’ye giren uzayı dengeler
ae_scaler = StandardScaler()
X_ae = ae_scaler.fit_transform(X_raw).astype(np.float32)

X_tensor = torch.tensor(X_ae)

# --- 2) Model ---
class AutoEncoder(nn.Module):
    def __init__(self, in_dim=8, latent_dim=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, in_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def encode(self, x):
        return self.encoder(x)

# --- 3) Train settings ---
latent_dim = 3
epochs = 200
batch_size = 64
lr = 1e-3
seed = 42

torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoEncoder(in_dim=X_ae.shape[1], latent_dim=latent_dim).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

dataset = TensorDataset(X_tensor)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- 4) Train loop ---
model.train()
for ep in range(1, epochs + 1):
    total_loss = 0.0
    for (xb,) in loader:
        xb = xb.to(device)
        optimizer.zero_grad()
        x_hat = model(xb)
        loss = criterion(x_hat, xb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)

    if ep % 20 == 0:
        print(f"[AE] epoch {ep:3d}/{epochs}  loss={total_loss/len(dataset):.6f}")

# --- 5) Extract latent features ---
model.eval()
with torch.no_grad():
    Z = model.encode(X_tensor.to(device)).cpu().numpy()   # (N, latent_dim)

# Latent’i ölçeklemek genelde clustering’i stabilize eder
Z_scaler = StandardScaler()
X_latent = Z_scaler.fit_transform(Z)

print("Latent shape:", X_latent.shape)

# =============================================================================
# 5A. SILHOUETTE SCAN (K from 8 to 30)  -- latent uzayda
# =============================================================================
X = X_latent

sil_scores = {}
best_k = None
best_sc = -1

for k in range(8, 31):
    if k >= len(X):
        break

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    sc = silhouette_score(X, labels)

    sil_scores[k] = sc
    if sc > best_sc:
        best_sc = sc
        best_k = k

print("\n=== Silhouette scores (K=8..30) ===")
for k in sorted(sil_scores):
    print(f"K={k:2d} -> silhouette={sil_scores[k]:.4f}")
print(f"\n✅ Best K by silhouette: K={best_k} (score={best_sc:.4f})")

plt.figure(figsize=(10, 4))
plt.plot(list(sil_scores.keys()), list(sil_scores.values()), marker='o')
plt.xticks(range(8, max(sil_scores.keys()) + 1))
plt.title("Silhouette score vs K (latent space)")
plt.xlabel("K (number of clusters)")
plt.ylabel("Silhouette score")
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig("clustering-pipeline/silhouette_k_8_30.png")
print("Saved: clustering-pipeline/silhouette_k_8_30.png")

# =============================================================================
# 5B. CLUSTERING (latent + KMeans)
# =============================================================================
# İstersen best_k kullan:
# n_clusters = best_k

# İstersen sabit kullan:
n_clusters = 20

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_latent)

# Cluster label’larını weighted_features’a ekliyoruz
weighted_features = weighted_features.copy()
weighted_features["Cluster"] = clusters
weighted_features["Student_ID"] = student_ids

# =============================================================================
# 6. CLUSTER STATISTICS
# =============================================================================
cluster_sizes = weighted_features["Cluster"].value_counts().sort_index()
print("\n=== Cluster Sizes ===")
print(cluster_sizes.to_string())

centroids = weighted_features.groupby("Cluster")[cats].mean().sort_index()
print("\n=== Cluster Centroids (mean scores) ===")
print(centroids.round(3).to_string())

desc = weighted_features.groupby("Cluster")[cats].describe()

q10 = weighted_features.groupby("Cluster")[cats].quantile(0.10)
q90 = weighted_features.groupby("Cluster")[cats].quantile(0.90)

overall_mean = weighted_features[cats].mean()

spread_q90_q10 = (q90 - q10).sort_index()
spread_q90_q10.to_csv("clustering-pipeline/cluster_spread_q90_q10.csv")

std_df = weighted_features.groupby("Cluster")[cats].std().sort_index()
std_df.to_csv("clustering-pipeline/cluster_std.csv")

print("\n=== Cluster Spread (q90-q10) ===")
print(spread_q90_q10.round(3).to_string())

print("\n=== Cluster Std (within-cluster) ===")
print(std_df.round(3).to_string())

rank_df = centroids.rank(axis=0, ascending=False, method="min").astype(int)
rank_df.to_csv("clustering-pipeline/cluster_centroid_ranks.csv")

print("\n=== Cluster Ranks by Category (1=highest centroid) ===")
print(rank_df.to_string())

gap_to_overall = centroids.sub(overall_mean, axis=1)
best_centroid = centroids.max(axis=0)
gap_to_best = best_centroid - centroids

gap_to_overall.to_csv("clustering-pipeline/cluster_gap_to_overall_mean.csv")
gap_to_best.to_csv("clustering-pipeline/cluster_gap_to_best.csv")

print("\n=== Gap to Overall Mean (centroid - overall_mean) ===")
print(gap_to_overall.round(3).to_string())

print("\n=== Gap to Best Cluster (best_centroid - centroid) ===")
print(gap_to_best.round(3).to_string())

avg_centroid = centroids.mean(axis=1)
avg_spread = spread_q90_q10.mean(axis=1)
avg_rank = rank_df.mean(axis=1)

cluster_summary = pd.DataFrame({
    "Size": cluster_sizes,
    "Avg_Centroid": avg_centroid,
    "Avg_Spread(q90-q10)": avg_spread,
    "Avg_Rank(1=best)": avg_rank,
}).sort_index()

cluster_summary["Compactness_Score"] = 1.0 / (1.0 + cluster_summary["Avg_Spread(q90-q10)"])
cluster_summary["Performance_Score"] = cluster_summary["Avg_Centroid"]
cluster_summary.to_csv("clustering-pipeline/cluster_summary.csv", index=True)

print("\n=== Cluster Summary (quick view) ===")
print(cluster_summary.round(3).to_string())

print("\nSaved:")
print(" - clustering-pipeline/cluster_spread_q90_q10.csv")
print(" - clustering-pipeline/cluster_std.csv")
print(" - clustering-pipeline/cluster_centroid_ranks.csv")
print(" - clustering-pipeline/cluster_gap_to_overall_mean.csv")
print(" - clustering-pipeline/cluster_gap_to_best.csv")
print(" - clustering-pipeline/cluster_summary.csv")

# =============================================================================
# 7. VISUALS
# =============================================================================
plt.figure(figsize=(12, 6))
sns.heatmap(spread_q90_q10, annot=True, fmt=".2f")
plt.title("Cluster Spread Heatmap (q90 - q10)  [lower = more compact]")
plt.xlabel("Category")
plt.ylabel("Cluster")
plt.tight_layout()
plt.savefig("clustering-pipeline/cluster_spread_heatmap.png")
print("Saved: clustering-pipeline/cluster_spread_heatmap.png")

plt.figure(figsize=(12, 6))
sns.heatmap(rank_df, annot=True, fmt="d")
plt.title("Cluster Rank Heatmap by Category (1 = best centroid)")
plt.xlabel("Category")
plt.ylabel("Cluster")
plt.tight_layout()
plt.savefig("clustering-pipeline/cluster_rank_heatmap.png")
print("Saved: clustering-pipeline/cluster_rank_heatmap.png")

plt.figure(figsize=(12, 6))
sns.heatmap(gap_to_overall, annot=True, fmt=".2f", center=0)
plt.title("Gap to Overall Mean (centroid - overall_mean) [positive=above avg]")
plt.xlabel("Category")
plt.ylabel("Cluster")
plt.tight_layout()
plt.savefig("clustering-pipeline/cluster_gap_to_overall_heatmap.png")
print("Saved: clustering-pipeline/cluster_gap_to_overall_heatmap.png")

# =============================================================================
# 8. EXPORT final table
# =============================================================================
weighted_features.to_csv("clustering-pipeline/final_student_clusters_no_naming.csv", index=False)
print("Saved: clustering-pipeline/final_student_clusters_no_naming.csv")

# =============================================================================
# 9. (OPSİYONEL) Autoencoder sonrası Spectral Clustering istersen:
# =============================================================================
# from sklearn.cluster import SpectralClustering
#
# n_clusters = 20
# spectral = SpectralClustering(
#     n_clusters=n_clusters,
#     affinity="nearest_neighbors",
#     n_neighbors=20,
#     assign_labels="kmeans",
#     random_state=42
# )
#
# clusters = spectral.fit_predict(X_latent)
# weighted_features["Cluster"] = clusters
# print("Spectral silhouette:", silhouette_score(X_latent, clusters))
