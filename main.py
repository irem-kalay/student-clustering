import pandas as pd
import glob
import os
import numpy as np

# --- AYARLAR ---
DATA_FOLDER = "data"  # Dosyaların olduğu klasör
OUTPUT_FILE = "student_feature_vectors2.csv" # Çıktı dosyasının adı

# 1. NOT DÖNÜŞÜM SÖZLÜĞÜ
grade_map = {
    'AA': 4.0, 'BA+': 3.75, 'BA': 3.5, 'BB+': 3.25, 'BB': 3.0,
    'CB+': 2.75, 'CB': 2.5, 'CC+': 2.25, 'CC': 2.0,
    'DC+': 1.75, 'DC': 1.5, 'DD+': 1.25, 'DD': 1.0,
    'FF': 0.0, 'VF': 0.0, 'F': 0.0,
    'BL': -1, 
}

# --- YARDIMCI FONKSİYONLAR ---

def clean_grade(grade_str):
    """
    Excel'den gelen 'AA / Original Entry' verisini temizler, sayıya çevirir.
    """
    if pd.isna(grade_str):
        return np.nan
    if not isinstance(grade_str, str):
        return grade_str
    
    grade_text = grade_str.split('/')[0].strip()
    return grade_map.get(grade_text, np.nan)

#ingilizce ve türkçe aynı olan dersleri 1 sütun gibi gösteriyor.
def normalize_course(code):
    """
    Ders kodunun sonundaki 'E' harfini temizler.
    Örnek: MAT103E -> MAT103
    """
    code = str(code).strip()
    # Sonu E ile bitiyorsa ve öncesi rakamsa E'yi at
    if code.endswith('E') and len(code) > 1 and code[-2].isdigit():
        return code[:-1]
    return code

# --- ANA İŞLEM FONKSİYONU ---

def process_student_files():
    file_list = glob.glob(os.path.join(DATA_FOLDER, "*.xlsx"))
    
    if not file_list:
        print(f"UYARI: '{DATA_FOLDER}' klasöründe hiç .xlsx dosyası bulunamadı!")
        return

    print(f"Toplam {len(file_list)} öğrenci dosyası bulundu. İşleniyor...")

    all_students_data = []

    for filepath in file_list:
        try:
            filename = os.path.basename(filepath)
            student_id = os.path.splitext(filename)[0]
            
            # Dosyayı oku
            df = pd.read_excel(filepath)
            
            # Sütun isimlerini temizle
            df.columns = [c.strip() for c in df.columns]

            if 'Ders Kodu' not in df.columns or 'Harf Notu' not in df.columns:
                print(f"ATLANDI: {filename} (Gerekli sütunlar eksik)")
                continue

            # Gerekli veriyi al
            temp_df = df[['Ders Kodu', 'Harf Notu']].copy()

            # --- DÜZELTME BURADA ---
            # 1. Önce Ders Kodlarını Normalize Et (E harflerini at)
            temp_df['Ders Kodu'] = temp_df['Ders Kodu'].apply(normalize_course)

            # 2. Sonra Notları Temizle
            temp_df['Numeric_Grade'] = temp_df['Harf Notu'].apply(clean_grade)

            # 3. Aynı ders (artık normalize edilmiş) tekrar ediyorsa MAX notu al
            # (Örn: MAT103'ten kalıp MAT103E ile geçtiyse, ikisi de MAT103 oldu ve yüksek not alındı)
            temp_df = temp_df.groupby('Ders Kodu')['Numeric_Grade'].max().reset_index()

            temp_df['Student_ID'] = student_id
            all_students_data.append(temp_df)

        except Exception as e:
            print(f"HATA: {filename} dosyasında sorun oluştu -> {e}")

    # --- VEKTÖR OLUŞTURMA ---
    if all_students_data:
        full_data = pd.concat(all_students_data, ignore_index=True)
        
        feature_matrix = full_data.pivot(index='Student_ID', columns='Ders Kodu', values='Numeric_Grade')
        
        # Eksik dersleri -1 ile doldur
        feature_matrix = feature_matrix.fillna(-1)

        # EKLEME: Sütunları (Ders Kodlarını) Alfabetik Sırala
        # Bu, vektörün her zaman aynı sırada olmasını sağlar (AKM..., BIL..., MAT...)
        feature_matrix = feature_matrix.reindex(sorted(feature_matrix.columns), axis=1)
        
        feature_matrix.to_csv(OUTPUT_FILE)
        
        print("-" * 30)
        print("İŞLEM TAMAMLANDI!")
        print(f"Oluşturulan Matris Boyutu: {feature_matrix.shape}")
        print(f"Dosya kaydedildi: {OUTPUT_FILE}")
        print("-" * 30)
        print(feature_matrix.head())
    else:
        print("Hiçbir veri işlenemedi.")

if __name__ == "__main__":
    process_student_files()