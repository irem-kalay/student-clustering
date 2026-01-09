import os
import glob
import pandas as pd

# Base directory of the script (portable, works anywhere)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input and output directories
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "clean_data")

# Create output directory if it does not exist
os.makedirs(OUT_DIR, exist_ok=True)

# Collect all Excel files
all_xlsx = glob.glob(os.path.join(DATA_DIR, "*.xlsx"))

# Filter only files starting with "Öğrenci Sınıf Listesi ("
target_files = [
    p for p in all_xlsx
    if os.path.basename(p).startswith("Öğrenci Sınıf Listesi (")
]

if not target_files:
    print(f"WARNING: No suitable .xlsx files found in {DATA_DIR}")
    print("Hint: Files should look like 'Öğrenci Sınıf Listesi (1).xlsx'")
    raise SystemExit(0)

print(f"Number of files found: {len(target_files)}")

success, failed = 0, 0

for excel_path in sorted(target_files):
    try:
        filename = os.path.basename(excel_path)
        base_name = os.path.splitext(filename)[0]   # remove .xlsx
        csv_path = os.path.join(OUT_DIR, base_name + ".csv")

        df = pd.read_excel(excel_path)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        print(f"{filename} -> {os.path.basename(csv_path)}")
        success += 1

    except Exception as e:
        print(f"ERROR: {os.path.basename(excel_path)} could not be processed -> {e}")
        failed += 1

print("-" * 40)
print(f"Finished. Success: {success}, Failed: {failed}")
print(f"Output directory: {OUT_DIR}")
