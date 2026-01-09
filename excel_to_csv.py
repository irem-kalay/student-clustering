import pandas as pd

# Excel dosya yolu
excel_path = "/Users/iremkalay/Desktop/student_cluster_project/data/Öğrenci Sınıf Listesi (1).xlsx"

# CSV çıkış dosyası
csv_path = "/Users/iremkalay/Desktop/student_cluster_project/clean_data/Öğrenci Sınıf Listesi (1).csv"

# Excel dosyasını oku
df = pd.read_excel(excel_path)

# CSV olarak kaydet
df.to_csv(csv_path, index=False, encoding="utf-8-sig")

print("Dosya başarıyla CSV formatına dönüştürüldü.")
