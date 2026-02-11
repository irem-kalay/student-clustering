import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class AutoEncoder(nn.Module):
    def __init__(self, in_dim=8, latent_dim=3):
        super().__init__()
        
        # Encoder: "Compression" (Sıkıştırma) Odaklı Mimari
        # Hocanın isteği üzerine nöron sayıları azaltıldı ve Dropout kaldırıldı.
        # Yapı: 8 (Girdi) -> 16 -> 8 -> 3 (Latent)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            # Dropout YOK (Tam öğrenme için)
            
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            
            nn.Linear(8, latent_dim)  # Sıkıştırılmış temsil
        )
        
        # Decoder: Encoder'ın tam tersi (Simetrik)
        # Yapı: 3 (Latent) -> 8 -> 16 -> 8 (Çıktı)
        # Sonuna Sigmoid ekledik ki çıktı 0-1 arasına sıkışsın
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            
            nn.Linear(8, 16),
            nn.ReLU(),
            
            nn.Linear(16, in_dim),
            nn.Sigmoid()  # <--- Çıktıyı 0-1 arasına hapseder
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def encode(self, x):
        return self.encoder(x)


# --- YENİ EKLENEN KISIM: DEC MODÜLÜ ---
class DEC(nn.Module):
    def __init__(self, autoencoder, n_clusters=5, latent_dim=3, alpha=1.0):
        super(DEC, self).__init__()
        
        # Eğitilmiş AutoEncoder'ın sadece encoder kısmını alıyoruz
        self.encoder = autoencoder.encoder
        
        # Küme sayısı ve Alpha (Student's t-distribution parametresi)
        self.n_clusters = n_clusters
        self.alpha = alpha
        
        # Küme Merkezleri (Cluster Centroids)
        # Bunlar da modelin bir parametresi olacak ve backpropagation ile güncellenecek!
        self.cluster_centers = Parameter(torch.Tensor(n_clusters, latent_dim))
        
        # Başlangıç için rastgele dolduruyoruz (Sonra K-Means ile ezilecek)
        nn.init.xavier_normal_(self.cluster_centers.data)

    def forward(self, x):
        # 1. Veriyi latent uzaya sıkıştır
        z = self.encoder(x)
        
        # 2. Student's t-distribution Kernel'i (Benzerlik Hesaplama)
        # Formül: q_ij = (1 + ||z_i - mu_j||^2 / alpha)^(- (alpha+1)/2)
        # Bu formül, bir noktanın bir merkeze olan uzaklığını olasılığa çevirir.
        
        # |z - mu|^2 hesapla
        norm_squared = torch.sum((z.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        
        # Pay kısmı
        q = 1.0 / (1.0 + (norm_squared / self.alpha))
        q = q ** ((self.alpha + 1.0) / 2.0)
        
        # Payda kısmı (Normalizasyon - Toplamları 1 olsun diye)
        q = (q.t() / torch.sum(q, 1)).t()
        
        return q, z  # q: Olasılık dağılımı, z: Latent veri