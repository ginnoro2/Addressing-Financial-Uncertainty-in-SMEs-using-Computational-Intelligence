"""Extract key values from June 2026 + full 6-month PL and sales overview."""
import pandas as pd
import numpy as np
from pathlib import Path

data_dir = Path("Data")
files = {
    "Jan": "Emilio's Monthly Report - Jan 2026_1.xlsx",
    "Feb": "Emilio's Monthly Report - Feb 2026.xlsx",
    "Mar": "Emilio's Monthly Report - March 2026.xlsx",
    "Apr": "Emilio's Monthly Report - April 2026.xlsx",
    "May": "Emilio's Monthly Report - May 2026.xlsx",
    "Jun": "Emilio's Monthly Report - Jun 2026_1.xlsx",
}

# ── June P&L ─────────────────────────────────────────────────────────────────
print("=== JUNE P&L RAW (first 40 rows) ===")
pl = pd.read_excel(data_dir / files["Jun"], sheet_name="Emilio's PL", header=None)
print(pl.iloc[:40, :18].to_string())

# ── June Monthly Sales ────────────────────────────────────────────────────────
print("\n=== JUNE Monthly Sales ===")
ms = pd.read_excel(data_dir / files["Jun"], sheet_name="Emilio's Monthly Sales", header=None)
print(ms.iloc[:18].to_string())

# ── Cash Flow all months ──────────────────────────────────────────────────────
print("\n=== CASH FLOW SUMMARY all months ===")
for month, fname in files.items():
    try:
        cf = pd.read_excel(data_dir / fname, sheet_name="Cash Flow", header=None)
        opening = cf.iloc[2, 1]  # Bank Balance row
        cash_bal = cf.iloc[1, 1]
        total_inflows = cf.iloc[10, 1] if len(cf) > 10 else "N/A"
        print(f"  {month}: Cash={cash_bal}, Bank={opening}, TotalInflows={total_inflows}")
    except Exception as e:
        print(f"  {month}: ERROR - {e}")

# ── Wastage totals all months ─────────────────────────────────────────────────
print("\n=== WASTAGE totals all months ===")
for month, fname in files.items():
    try:
        w = pd.read_excel(data_dir / fname, sheet_name="Sample Gift waste", header=None)
        # Find Total Pizza row
        for _, row in w.iterrows():
            cell = str(row.iloc[0]).strip().lower()
            if "total pizza" in cell:
                print(f"  {month}: Total Pizza wastage KTM={row.iloc[1]}, PKR={row.iloc[2]}")
                break
    except Exception as e:
        print(f"  {month}: ERROR - {e}")

# ── Purchase totals all months ────────────────────────────────────────────────
print("\n=== PURCHASE TOTALS all months ===")
for month, fname in files.items():
    try:
        pa = pd.read_excel(data_dir / fname, sheet_name="Purchase Analysis", header=None)
        # Row 3 = Material Purchase (first data row)
        mat = pa.iloc[3, 1:14]  # monthly columns
        print(f"  {month} file - first purchase row values: {mat.tolist()}")
    except Exception as e:
        print(f"  {month}: ERROR - {e}")

# ── Salary Details (May + Jun) ────────────────────────────────────────────────
print("\n=== SALARY DETAILS ===")
for month in ["May", "Jun"]:
    try:
        sal = pd.read_excel(data_dir / files[month], sheet_name="Salary Details", header=None)
        print(f"\n  {month} Salary Details:")
        print(sal.to_string())
    except Exception as e:
        print(f"  {month}: ERROR - {e}")
