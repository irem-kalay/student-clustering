import pandas as pd
import os

# --- AYARLAR ---
VECTOR_FILE = "student_feature_vectors2.csv"  # Oluşturduğun vektör dosyası
EXCEL_FOLDER = "/Users/iremkalay/Desktop/student_cluster_project/data"  # Temizlenmiş Excel'lerin olduğu klasör
TARGET_STUDENT_ID = "Öğrenci Sınıf Listesi (97)" # Test edilecek öğrenci ID'si (dosya adıyla aynı olmalı)

# Not dönüşüm tablosu (Senin mantığınla aynı olmalı)
GRADE_MAP = {
    "AA": 4.0, "BA": 3.5, "BB": 3.0, "CB": 2.5,
    "CC": 2.0, "DC": 1.5, "DD": 1.0, "FF": 0.0,
    "VF": 0.0, "F": 0.0
}

def verify_student_vector():
    # 1. Vektör dosyasını yükle
    try:
        df_vector = pd.read_csv(VECTOR_FILE, index_col=0)
    except FileNotFoundError:
        print("HATA: Vektör dosyası bulunamadı.")
        return

    # 2. İlgili öğrencinin vektör satırını çek
    if TARGET_STUDENT_ID not in df_vector.index:
        print(f"HATA: {TARGET_STUDENT_ID} vektör dosyasında bulunamadı.")
        return
    
    student_vector = df_vector.loc[TARGET_STUDENT_ID]
    print(f"--- {TARGET_STUDENT_ID} İçin Kontrol Başlıyor ---\n")

    # 3. Orijinal Excel verisini yükle
    excel_path = os.path.join(EXCEL_FOLDER, f"{TARGET_STUDENT_ID}.xlsx")
    try:
        # Excel okuma (repair modülünle düzelttiğin formatta olduğunu varsayıyorum)
        df_excel = pd.read_excel(excel_path)
    except Exception as e:
        print(f"Excel okunamadı: {e}")
        return

    # Sütun isimlerini standartlaştır (Excel başlıklarına göre ayarla)
    # Genelde: 'Ders Kodu', 'Harf Notu' gibi sütunlar olur.
    # Senin CSV örneğine baktığımda sütunlar: "Ders Kodu", "Harf Notu"
    
    mismatches = []
    matches = 0
    
    # 4. Her dersi tek tek kontrol et
    for index, row in df_excel.iterrows():
        course_code = str(row['Ders Kodu']).strip()
        raw_grade = str(row['Harf Notu']).split('/')[0].strip() # "AA / Original Entry" -> "AA" al
        
        # Notu sayıya çevir
        actual_grade = GRADE_MAP.get(raw_grade)
        
        if actual_grade is None:
            # Not harf değilse (örn: 'E' veya boşsa) atla
            continue

        # Vektörde bu ders var mı?
        if course_code in student_vector.index:
            vector_value = student_vector[course_code]
            
            # Karşılaştırma (Float hassasiyeti için küçük bir toleransla)
            if abs(vector_value - actual_grade) > 0.01:
                # Bazen öğrenci dersten kalıp tekrar almıştır, vektörde en yüksek not mu var?
                # Eğer vektördeki not, Excel'deki bu satırdan yüksekse sorun yok (tekrar alıp geçmiştir).
                if vector_value > actual_grade:
                    print(f"[BİLGİ] {course_code}: Excel={actual_grade}, Vektör={vector_value} (Daha yüksek notu var, OK)")
                else:
                    mismatches.append(f"{course_code} -> Excel: {actual_grade}, Vektör: {vector_value}")
            else:
                matches += 1
        else:
            print(f"[UYARI] {course_code} dersi vektör sütunlarında YOK (Sütun eksik olabilir).")

    # 5. Sonuç Raporu
    print("-" * 30)
    print(f"Toplam Eşleşen Ders: {matches}")
    if mismatches:
        print(f"HATA: {len(mismatches)} adet uyuşmazlık bulundu:")
        for m in mismatches:
            print(f"  - {m}")
    else:
        print("Tebrikler! Excel'deki ders notları ile Vektör değerleri TAM UYUŞUYOR.")

if __name__ == "__main__":
    verify_student_vector()