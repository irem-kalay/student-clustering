# Student Clustering Pipeline

This repository contains a pipeline that processes student grade files, filters courses using a curriculum equivalency map, trains an autoencoder + DEC clustering model, and generates cluster reports and visualizations.

## Prerequisites
- Python 3.8 or newer
- Recommended system with enough RAM; GPU can accelerate training if TensorFlow is configured
- Place raw student files (CSV or XLSX) into the `fixed_xlsx` directory at the repository root
- Ensure the curriculum equivalency CSV is present as `dersdenklikleri.csv` in the repository root

## Recommended Python packages
If there is a `requirements.txt`, install from it. Otherwise, install the most common dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install pandas numpy tensorflow scikit-learn matplotlib openpyxl xlrd
```

## Running the pipeline
From the repository root run:

```bash
python3 main.py
```

Behavior:
- The script expects student files inside `fixed_xlsx` and `dersdenklikleri.csv` in the same folder as `main.py`.
- If either the student files or `dersdenklikleri.csv` are missing, the script will print `Error: Files not found.` and exit.

## Outputs
After a successful run, the pipeline will generate several files in the repository root:

- `vector-equivalent.csv` — cleaned feature vectors (courses merged using equivalency map)
- `student_final_clusters.csv` — final cluster assignment per student
- `loss_plots.png` — autoencoder and DEC loss visualization
- `cluster_plot.png` — 2D PCA visualization of clusters
- `report.txt` and `report.json` — human-readable and machine-readable cluster analysis reports

## Notes & Troubleshooting
- If you see `Error: Files not found.`, confirm that `fixed_xlsx` contains `.xlsx`/`.csv` files and `dersdenklikleri.csv` exists.
- If you have many students, training may take significant time and memory. Consider reducing the dataset size for testing or use a machine with GPU.
- The code uses TensorFlow; ensure the installed TensorFlow version is compatible with your Python version.

## Optional improvements
- Add a `requirements.txt` to pin package versions for reproducible environments.
- Add a small test dataset and a `--dry-run` or `--quick` mode to validate setup quickly.

If you want, I can add a `requirements.txt` file and a quick-run example next. Reply with "yes" to proceed.
