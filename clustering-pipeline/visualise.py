import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# Ayarlar
RESULTS_DIR = "clustering-pipeline/results"
CSV_PATH = os.path.join(RESULTS_DIR, "final_results.csv")
SAVE_2D = os.path.join(RESULTS_DIR, "pca_2d_advanced.png")
SAVE_3D = os.path.join(RESULTS_DIR, "pca_3d_advanced.png")
SAVE_BIPLOT = os.path.join(RESULTS_DIR, "pca_biplot.png")

# Hedef Kategoriler (Data Prep'teki ile aynı olmalı)
CATS = [
    'Basic_Sciences', 'Software_Practice', 'Algorithm_Theory',
    'Systems', 'Hardware', 'Social_Cultural', 'Language_Comm', 'Math_Calc'
]

def generate_pca_graphs():
    # 1. Veriyi Yükle
    if not os.path.exists(CSV_PATH):
        print(f"HATA: {CSV_PATH} bulunamadı! Önce main.py'yi çalıştırın.")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"Veri yüklendi: {len(df)} öğrenci")

    # Features ve Cluster
    X = df[CATS].values
    clusters = df['Cluster'].values

    # 2. Standartlaştırma (PCA için şart)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. PCA Uygula (3 Bileşen)
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    
    var_exp = pca.explained_variance_ratio_
    print(f"Açıklanan Varyans: PC1={var_exp[0]:.2f}, PC2={var_exp[1]:.2f}, PC3={var_exp[2]:.2f}")
    print(f"Toplam Bilgi Kaybı: {(1 - sum(var_exp)):.2f}")

    # Renk Paleti
    palette = sns.color_palette("tab10", n_colors=len(np.unique(clusters)))
    colors = [palette[c] for c in clusters]

    # --- GRAFİK 1: 2D PCA ---
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette="tab10", s=60, alpha=0.8, edgecolor='k')
    plt.title(f"2D PCA - Öğrenci Kümeleri\n(Varyans: {sum(var_exp[:2])*100:.1f}%)")
    plt.xlabel(f"PC1 (Yön: {get_loading_text(pca, 0, CATS)})")
    plt.ylabel(f"PC2 (Yön: {get_loading_text(pca, 1, CATS)})")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(SAVE_2D, dpi=300)
    print(f"Kaydedildi: {SAVE_2D}")

    # --- GRAFİK 2: 3D PCA ---
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Her cluster'ı ayrı çiz (Legend için)
    unique_clusters = np.unique(clusters)
    for c in unique_clusters:
        mask = clusters == c
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2], 
                   label=f"Cluster {c}", s=50, alpha=0.7, edgecolors='k')

    ax.set_title(f"3D PCA - Öğrenci Uzayı\n(Toplam Varyans: {sum(var_exp)*100:.1f}%)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend()
    # Açıyı ayarla (en iyi görüş için)
    ax.view_init(elev=30, azim=135)
    plt.savefig(SAVE_3D, dpi=300)
    print(f"Kaydedildi: {SAVE_3D}")

    # --- GRAFİK 3: BIPLOT (Derslerin Yönleri) ---
    plt.figure(figsize=(12, 10))
    
    # Önce noktaları çiz (daha soluk)
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette="tab10", s=30, alpha=0.3, legend=False)
    
    # Vektörleri (Dersleri) Çiz
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_) * 3.5 # Ölçekleme katsayısı
    
    for i, feature in enumerate(CATS):
        plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], color='r', alpha=0.9, head_width=0.1)
        # Metni biraz uzağa yaz
        plt.text(loadings[i, 0]*1.15, loadings[i, 1]*1.15, feature, color='darkred', ha='center', va='center', fontsize=11, weight='bold')

    plt.title("Biplot: Hangi Ders Hangi Kümeyi Oluşturuyor?")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(SAVE_BIPLOT, dpi=300)
    print(f"Kaydedildi: {SAVE_BIPLOT}")

def get_loading_text(pca, pc_idx, features):
    # PC eksenini en çok etkileyen özelliği bulur
    loadings = pca.components_[pc_idx]
    max_idx = np.argmax(np.abs(loadings))
    feature = features[max_idx]
    direction = "(+)" if loadings[max_idx] > 0 else "(-)"
    return f"{feature} {direction}"

if __name__ == "__main__":
    generate_pca_graphs()