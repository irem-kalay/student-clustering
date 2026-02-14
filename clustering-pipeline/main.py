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
import random  # <--- NEWLY ADDED

# Custom modules
import data_prep
import models

# =============================================================================
# 0. REPRODUCIBILITY (SEED FIXING)
# =============================================================================
def set_seed(seed=42):
    """
    This function fixes the random number generators of all libraries.
    This ensures the code produces IDENTICAL results every time it runs.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If multiple GPUs exist
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("[LOCK] Seed locked: " + str(seed))

# Run seed at the very beginning (42 is the answer to the universe :) )
set_seed(42)

# Create folders
os.makedirs("clustering-pipeline/results", exist_ok=True)

# =============================================================================
# 1. DATA PREPARATION
# =============================================================================
csv_path = 'feature-vectors/vector-1400-equivalent.csv'
# Correct path if different:
if not os.path.exists(csv_path):
    csv_path = 'vector-1400-equivalent.csv'

print("Loading data...")
weighted_features, _ = data_prep.load_and_process_data(csv_path)

cats = data_prep.TARGET_CATEGORIES
X_raw = weighted_features[cats].values.astype(np.float32)

# --- NOTE: StandardScaler CANCELLED to reduce Loss ---
# Data is already between 0-1, will work perfectly with Sigmoid.
print("Data normalization: Using raw data (between 0-1).")
X_tensor = torch.tensor(X_raw)

# =============================================================================
# 2. AUTOENCODER TRAINING (WITH CUSTOM LOSS)
# =============================================================================
latent_dim = 3
epochs = 1800
batch_size = 32
lr = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.AutoEncoder(in_dim=8, latent_dim=latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100)

# NEW LOSS FUNCTION
def weighted_mse_loss(input, target):
    loss = (input - target) ** 2
    # If real data is 0 (course not taken), give the error 20x more importance!
    zero_mask = (target == 0.0)
    loss[zero_mask] *= 20.0  # PENALTY COEFFICIENT
    return loss.mean()

dataset = TensorDataset(X_tensor)
# Even with Shuffle=True, it will mix in the same order due to fixed seed
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(f"\nModel training starting ({epochs} epochs)...")
print("Goal: Crush the zeros!")
loss_history = []

model.train()
for ep in range(1, epochs + 1):
    total_loss = 0.0
    for (xb,) in loader:
        xb = xb.to(device)
        optimizer.zero_grad()
        x_hat = model(xb)
        
        # Using Weighted Loss
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

# Extract latent vectors
model.eval()
with torch.no_grad():
    Z = model.encode(X_tensor.to(device)).cpu().numpy()


# --- Loss Plot ---
plt.figure(figsize=(10, 5))
plt.plot(loss_history, label='Weighted Loss')
plt.title("Training Loss Graph")
plt.savefig("clustering-pipeline/results/ae_loss_graph.png")

# =============================================================================
# 3. CLUSTERING (DEC - REWRITTEN & OPTIMIZED)
# =============================================================================
print("\n[DEC] DEC fine-tuning starting...")

# ---------------------------
# DEC Model (Standard Student's t-distribution)
# ---------------------------
class DEC(nn.Module):
    def __init__(self, encoder, n_clusters=7, latent_dim=3, alpha=1.0):
        super().__init__()
        self.encoder = encoder  # Pre-trained AE
        self.alpha = alpha      # Degrees of freedom (default=1 for Student's t)
        
        # Learnable cluster centers
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, latent_dim))

    def forward(self, x):
        # 1. Get latent representation from Autoencoder
        z = self.encoder.encode(x)
        
        # 2. Calculate soft assignment (q) using Student's t-distribution
        # Formula: q_ij = (1 + ||z_i - u_j||^2 / alpha) ^ -((alpha+1)/2)
        norm_squared = torch.sum((z.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        q = (1.0 + (norm_squared / self.alpha)) ** -((self.alpha + 1.0) / 2.0)
        
        # Normalize q so rows sum to 1
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q, z

# ---------------------------
# Target Distribution (P)
# ---------------------------
def target_distribution(q):
    weight = (q ** 2) / torch.sum(q, dim=0)
    return (weight.t() / torch.sum(weight, dim=1)).t()

# ---------------------------
# Initialize & Train
# ---------------------------
n_clusters = 7
# Re-instantiate DEC model
dec_model = DEC(model, n_clusters=n_clusters, latent_dim=latent_dim).to(device)

# A. Initialize Centers with KMeans
print("Initializing cluster centers with KMeans...")
kmeans_init = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
# Get initial latent vectors (no grad needed here)
with torch.no_grad():
    z_init = model.encode(X_tensor.to(device)).cpu().numpy()

kmeans_init.fit(z_init)
dec_model.cluster_centers.data = torch.tensor(kmeans_init.cluster_centers_, dtype=torch.float32).to(device)

# B. Optimizer
# We use separate learning rates: 
# - Higher for centers (move them to data)
# - Lower for encoder (gently nudge data to centers)
optimizer_dec = torch.optim.Adam([
    {'params': dec_model.cluster_centers, 'lr': 0.001},  # Centers move fast
    {'params': dec_model.encoder.parameters(), 'lr': 0.0001} # Encoder fine-tunes slowly
])

print("[DEC] Training Started (Encoder Unfrozen)...")

max_patience = 100
patience_counter = 0
best_kl_loss = float('inf')

for epoch in range(2000): # Max 2000 epochs
    dec_model.train()
    
    # Forward pass
    q, z = dec_model(X_tensor.to(device))
    
    # Calculate target distribution P (The "Sharpened" version of Q)
    p = target_distribution(q).detach()
    
    # KL Divergence Loss
    kl_loss = nn.KLDivLoss(reduction='batchmean')(torch.log(q + 1e-10), p)
    
    # Optional: Add Reconstruction Loss to keep features valid (prevents drifting)
    # x_recon = dec_model.encoder.decode(z)
    # rec_loss = weighted_mse_loss(x_recon, X_tensor.to(device))
    # total_loss = kl_loss + (0.1 * rec_loss) 
    
    total_loss = kl_loss # Using pure DEC loss for now

    optimizer_dec.zero_grad()
    total_loss.backward()
    optimizer_dec.step()
    
    # Logging
    if epoch % 100 == 0:
        print(f"DEC Epoch {epoch:4d} | KL Loss: {kl_loss.item():.6f}")

    # Early Stopping Logic
    if kl_loss.item() < best_kl_loss:
        best_kl_loss = kl_loss.item()
        patience_counter = 0
    else:
        patience_counter += 1
        
    if patience_counter >= max_patience:
        print(f"Early stopping at epoch {epoch} (Best KL Loss: {best_kl_loss:.6f})")
        break

print("[DEC] Fine-tuning complete.")

# ---------------------------
# Final Assignments
# ---------------------------
dec_model.eval()
with torch.no_grad():
    q_final, z_final = dec_model(X_tensor.to(device))
    final_clusters = torch.argmax(q_final, dim=1).cpu().numpy()

weighted_features['Cluster_DEC'] = final_clusters

# ---------------------------
# SAVE DATA TO CSV
# ---------------------------
csv_save_path = "clustering-pipeline/results/final_results.csv"
weighted_features.to_csv(csv_save_path, index=False)
print(f"\n[DATA SAVED] Student clustering data saved to: {csv_save_path}")

# =============================================================================
# 4. DEBUG (ZERO CHECK) - (Your existing code here...)
# =============================================================================
# ... (Keep your existing zero check code here) ...

# =============================================================================
# 5. VISUALIZATION (PCA)
# =============================================================================
print("\n[VISUALIZATION] Generating PCA plot for clusters...")

# 1. Prepare Data for Plotting
z_np = z_final.cpu().numpy()
centers_np = dec_model.cluster_centers.detach().cpu().numpy()

# 2. Run PCA (3D -> 2D)
pca = PCA(n_components=2, random_state=42)
z_pca = pca.fit_transform(z_np)
centers_pca = pca.transform(centers_np) # Project centers into same 2D space

# 3. Create DataFrame
df_vis = pd.DataFrame({
    'PCA1': z_pca[:, 0],
    'PCA2': z_pca[:, 1],
    'Cluster': final_clusters
})

# 4. Plot
plt.figure(figsize=(12, 8))

# Scatter plot of students
sns.scatterplot(
    data=df_vis, x='PCA1', y='PCA2', 
    hue='Cluster', palette='tab10', 
    style='Cluster', s=80, alpha=0.7
)

# Scatter plot of Cluster Centers (Black X)
plt.scatter(
    centers_pca[:, 0], centers_pca[:, 1], 
    marker='X', s=300, c='black', label='Centroids', zorder=10
)

plt.title("DEC Clusters Visualization (PCA Projection)", fontsize=16)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()

# Save
plot_path = "clustering-pipeline/results/dec_clusters_pca.png"
plt.savefig(plot_path, dpi=300)
print(f"Plot saved to: {plot_path}")
plt.show()