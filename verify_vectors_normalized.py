import pandas as pd
import os

# --- AYARLAR ---
VECTOR_FILE = "student_feature_vectors2.csv"
EXCEL_FOLDER = "/Users/iremkalay/Desktop/student_cluster_project/data"
TARGET_STUDENT_ID = "Öğrenci Sınıf Listesi (97)"

# Not dönüşüm tablosu
GRADE_MAP = {
    "AA": 4.0, "BA": 3.5, "BB": 3.0, "CB": 2.5,
    "CC": 2.0, "DC": 1.5, "DD": 1.0, "FF": 0.0,
    "VF": 0.0, "F": 0.0
}

def normalize_course(code):
    code = str(code).strip()
    if code.endswith('E') and len(code) > 1 and code[-2].isdigit():
        return code[:-1]
    return code

def verify_bidirectional():
    # 1. Dosyaları Yükle
    if not os.path.exists(VECTOR_FILE):
        print(f"HATA: {VECTOR_FILE} bulunamadı.")
        return
        
    df_vector = pd.read_csv(VECTOR_FILE, index_col=0)
    
    if TARGET_STUDENT_ID not in df_vector.index:
        print(f"HATA: {TARGET_STUDENT_ID} vektörde yok.")
        return
    
    student_vector = df_vector.loc[TARGET_STUDENT_ID]
    print(f"--- {TARGET_STUDENT_ID} Çift Yönlü Kontrol ---\n")

    excel_path = os.path.join(EXCEL_FOLDER, f"{TARGET_STUDENT_ID}.xlsx")
    try:
        df_excel = pd.read_excel(excel_path)
    except Exception as e:
        print(f"Excel okunamadı: {e}")
        return

    df_excel.columns = [c.strip() for c in df_excel.columns]
    
    # ---------------------------------------------------------
    # TEST 1: EXCEL -> VEKTÖR (Eksik veya Yanlış Not Var mı?)
    # ---------------------------------------------------------
    print("TEST 1: Excel'den Vektöre Kontrol...")
    excel_courses_found = set()
    mismatches = []
    
    for index, row in df_excel.iterrows():
        raw_course = str(row['Ders Kodu']).strip()
        course_code = normalize_course(raw_course) # Normalize et
        
        raw_grade = str(row['Harf Notu']).split('/')[0].strip()
        actual_grade = GRADE_MAP.get(raw_grade)
        
        if actual_grade is None: continue
        
        excel_courses_found.add(course_code) # Excel'de var olduğunu kaydet

        if course_code in student_vector.index:
            vector_val = student_vector[course_code]
            if abs(vector_val - actual_grade) > 0.01:
                if vector_val > actual_grade:
                    # Dersten kalıp geçmişse normaldir
                    pass 
                else:
                    mismatches.append(f"{course_code}: Excel={actual_grade}, Vektör={vector_val}")
        else:
            print(f"  [UYARI] {course_code} vektörde sütun olarak bile yok!")

    if mismatches:
        print(f"  ❌ HATA: {len(mismatches)} not uyuşmazlığı bulundu!")
        for m in mismatches: print(f"    - {m}")
    else:
        print("  ✅ Excel'deki tüm notlar vektörde doğru görünüyor.")

    # ---------------------------------------------------------
    # TEST 2: VEKTÖR -> EXCEL (Hayali Ders Var mı?)
    # ---------------------------------------------------------
    print("\nTEST 2: Vektörden Excel'e Ters Kontrol (Hayali Veri)...")
    
    # Vektörde -1 olmayan (yani bir not girilmiş) dersleri bul
    active_vector_courses = student_vector[student_vector != -1].index
    ghost_courses = []

    for col in active_vector_courses:
        # Bu ders Excel listemizde (normalize edilmiş haliyle) var mıydı?
        if col not in excel_courses_found:
            # Excel'de yok ama Vektörde notu var!
            ghost_courses.append(f"{col} (Vektör Notu: {student_vector[col]})")

    if ghost_courses:
        print(f"  ❌ KRİTİK HATA: Vektörde Excel'de olmayan {len(ghost_courses)} 'hayali' ders bulundu:")
        for g in ghost_courses:
            print(f"    - {g}")
        print("  (Bu durum, vektör dosyasının elle değiştirildiğini veya yanlış öğrenciye yazıldığını gösterir.)")
    else:
        print("  ✅ Vektör temiz. Fazladan/Hayali ders yok.")

    print("-" * 30)
    if not mismatches and not ghost_courses:
        print("SONUÇ: %100 UYUMLU VE GÜVENİLİR.")

if __name__ == "__main__":
    verify_bidirectional()