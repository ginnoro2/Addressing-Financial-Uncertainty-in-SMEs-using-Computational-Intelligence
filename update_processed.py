"""update_processed.py — regenerate all processed/ CSVs with 6-month data."""
import os, warnings
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import joblib
joblib.parallel_backend("threading")

import pandas as pd
from src.data_pipeline import AuditDataPipeline
from src.model_lab import ModelLab
from pathlib import Path

PROC = Path("processed")

print("Loading 6-month pipeline...")
p = AuditDataPipeline().load_all()

# feature_matrix.csv
p.feature_matrix.to_csv(PROC / "feature_matrix.csv", index=False)
print(f"  feature_matrix.csv  → {p.feature_matrix.shape[0]} rows × {p.feature_matrix.shape[1]} cols")

# monthly_pl.csv
p.monthly_pl.to_csv(PROC / "monthly_pl.csv", index=False)
print(f"  monthly_pl.csv      → {len(p.monthly_pl)} months")

# daily_sales.csv
if not p.daily_sales.empty:
    p.daily_sales.to_csv(PROC / "daily_sales.csv", index=False)
    print(f"  daily_sales.csv     → {len(p.daily_sales)} days")

print("Running ModelLab (LOO on 6 months)...")
m = ModelLab(p).run()
m.comparison.to_csv(PROC / "model_comparison.csv", index=False)
print(f"  model_comparison.csv → {len(m.comparison)} rows")

print()
print("=== Stress labels ===")
print(p.feature_matrix[["period","stress","profit_margin","cogs_ratio"]].to_string(index=False))

print()
print("=== Model comparison ===")
print(m.comparison.to_string(index=False))

print()
print("Done — all processed/ CSVs updated with Jan–Jun 2026 data.")
