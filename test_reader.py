full_path = "obs_track/downloads_properties/Öğrenci Sınıf Listesi - 2026-02-01T161543.346.xlsx"

with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
    for _ in range(10):
        print(f.readline())
