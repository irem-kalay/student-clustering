import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import os
import random  # <--- YENİ EKLENDİ

# Kendi modüllerimiz
import data_prep
import models

# =============================================================================
# 0. RASTGELELİĞİ ÖLDÜRME (SEED SABİTLEME)
# =============================================================================
def set_seed(seed=42):
    """
    Bu fonksiyon tüm kütüphanelerin rastgele sayı üreteçlerini sabitler.
    Böylece kod her çalıştığında BİREBİR aynı sonucu verir.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Eğer çoklu GPU varsa
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🔒 Rastgelelik kilitlendi. Seed: {seed}")

# Seed'i en başta çalıştırıyoruz (42 evrenin cevabıdır :) )
set_seed(42)

# Klasörleri oluştur
os.makedirs("clustering-pipeline/results", exist_ok=True)

# =============================================================================
# 1. VERİ HAZIRLIĞI
# =============================================================================
csv_path = 'feature-vectors/vector-1400-equivalent.csv'
# Eğer dosya yolu farklıysa burayı düzelt:
if not os.path.exists(csv_path):
    csv_path = 'vector-1400-equivalent.csv'

print("Veri yükleniyor...")
weighted_features, _ = data_prep.load_and_process_data(csv_path)

cats = data_prep.TARGET_CATEGORIES
X_raw = weighted_features[cats].values.astype(np.float32)

# --- DİKKAT: Loss'u düşürmek için StandardScaler İPTAL EDİLDİ ---
# Veri zaten 0-1 arasında, Sigmoid ile tam uyumlu çalışacak.
print("Veri normalizasyonu: Ham veri kullanılıyor (0-1 arası).")
X_tensor = torch.tensor(X_raw)

# =============================================================================
# 2. AUTOENCODER EĞİTİMİ (ÖZEL LOSS İLE)
# =============================================================================
latent_dim = 3
epochs = 3000
batch_size = 32
lr = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.AutoEncoder(in_dim=8, latent_dim=latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100)

# YENİ LOSS FONKSİYONU
def weighted_mse_loss(input, target):
    loss = (input - target) ** 2
    # Eğer gerçek veri 0 ise (ders alınmamışsa), hatayı 20 kat fazla önemse!
    zero_mask = (target == 0.0)
    loss[zero_mask] *= 20.0  # CEZA KATSAYISI
    return loss.mean()

dataset = TensorDataset(X_tensor)
# Shuffle=True olsa bile seed sabit olduğu için hep aynı sırayla karıştıracak
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(f"\nModel eğitimi başlıyor ({epochs} epoch)...")
print("Hedef: Sıfırları ezip geçmek!")
loss_history = []

model.train()
for ep in range(1, epochs + 1):
    total_loss = 0.0
    for (xb,) in loader:
        xb = xb.to(device)
        optimizer.zero_grad()
        x_hat = model(xb)
        
        # Weighted Loss kullanıyoruz
        loss = weighted_mse_loss(x_hat, xb)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    
    avg_loss = total_loss / len(dataset)
    loss_history.append(avg_loss)
    scheduler.step(avg_loss)
    
    if ep % 100 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {ep:4d}/{epochs} | Loss: {avg_loss:.6f} | LR: {current_lr:.8f}")

# Latent vektörleri çıkar
model.eval()
with torch.no_grad():
    Z = model.encode(X_tensor.to(device)).cpu().numpy()

# Latent uzayı scale et (Sadece K-Means için)
Z_scaler = StandardScaler()
X_latent = Z_scaler.fit_transform(Z)

# --- Loss Grafiği ---
plt.figure(figsize=(10, 5))
plt.plot(loss_history, label='Weighted Loss')
plt.title("Training Loss (Zero-Forcing)")
plt.savefig("clustering-pipeline/results/ae_loss_graph.png")

# =============================================================================
# 3. CLUSTERING
# =============================================================================
print("\nClustering başlıyor...")
# KMeans'de zaten random_state=42 vardı, bu da sabitlik sağlar
kmeans = KMeans(n_clusters=7, random_state=42, n_init=20)
clusters = kmeans.fit_predict(X_latent)
weighted_features['Cluster'] = clusters

# =============================================================================
# 4. DEBUG (SIFIR KONTROLÜ)
# =============================================================================
print("\n=== DEBUG: Model SIFIRLARI NE YAPTI? ===")
model.eval()
# Rastgeleliği sabitlediğimiz için hep aynı 3 öğrenciyi seçecek, kıyaslama kolaylaşır
np.random.seed(42) 
indices = np.random.choice(len(X_tensor), 3, replace=False)
sample_inputs = X_tensor[indices].to(device)

with torch.no_grad():
    sample_outputs = model(sample_inputs)

in_np = sample_inputs.cpu().numpy()
out_np = sample_outputs.cpu().numpy()

cat_names = ['Basic', 'Softw', 'Algo', 'Syst', 'Hardw', 'Soc', 'Lang', 'Math']

for i in range(3):
    print(f"\n>> Öğrenci {indices[i]}:")
    print(f"{'Ders':<10} | {'Gerçek':<8} | {'Tahmin':<8} | {'Fark'}")
    print("-" * 40)
    for j, cat in enumerate(cat_names):
        real_val = in_np[i][j]
        pred_val = out_np[i][j]
        diff = abs(real_val - pred_val)
        
        if real_val > 0.01 or diff > 0.01: 
            print(f"{cat:<10} | {real_val:.4f}   | {pred_val:.4f}   | {diff:.4f}")

# Sonuçları Kaydet
centroids = weighted_features.groupby("Cluster")[cats].mean()
centroids.to_csv("clustering-pipeline/results/cluster_centroids.csv")
weighted_features.to_csv("clustering-pipeline/results/final_results.csv", index=False)

# =============================================================================
# 5. CİNSİYET ANALİZİ
# =============================================================================
downloads_dir = "/Users/iremkalay/Desktop/student_cluster_project/obs_track/downloads_properties"
if os.path.exists(downloads_dir):
    print("\nCinsiyet verisi işleniyor...")
    gender_map = {}
    for file in os.listdir(downloads_dir):
        if file.endswith('.xlsx') or file.endswith('.label'):
            sid = file.replace('.xlsx', '').replace('.label', '')
            path = os.path.join(downloads_dir, file)
            gender_map[sid] = data_prep.extract_gender_from_file(path)
    
    weighted_features['Cinsiyet'] = weighted_features['Student_ID'].map(gender_map).fillna('Bilinmiyor')
    
    gender_stats = weighted_features.groupby(['Cluster', 'Cinsiyet']).size().unstack(fill_value=0)
    print("\nCluster Cinsiyet Dağılımı:")
    print(gender_stats)
    gender_stats.to_csv("clustering-pipeline/results/gender_stats.csv")
else:
    print(f"\nUYARI: '{downloads_dir}' klasörü bulunamadı. Cinsiyet analizi atlanıyor.")

print("\nİşlem Tamamlandı! Tüm sonuçlar 'clustering-pipeline/results' klasöründe.")