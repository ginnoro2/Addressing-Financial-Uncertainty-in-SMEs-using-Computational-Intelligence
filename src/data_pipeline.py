"""Data collection, quality assessment, cleaning, and feature engineering.

v2 update: loads all 6 monthly reports (Jan–Jun 2026) from their individual files
using the PL_COL_MAP defined in config.py.  The original 5-month pipeline is
preserved unchanged when DATA_DIR contains the old file set only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import DATA_DIR


MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# ── v2: per-file, per-month P&L column mapping ───────────────────────────────
# Each monthly file is the "most recent" cumulative P&L; the reporting month
# sits in the first populated column after the header pair.
# Confirmed by inspecting row 4 of each file's "Emilio's PL" sheet.
_V2_FILES = {
    "Jan": DATA_DIR / "Emilio's Monthly Report - Jan 2026_1.xlsx",
    "Feb": DATA_DIR / "Emilio's Monthly Report - Feb 2026.xlsx",
    "Mar": DATA_DIR / "Emilio's Monthly Report - March 2026.xlsx",
    "Apr": DATA_DIR / "Emilio's Monthly Report - April 2026.xlsx",
    "May": DATA_DIR / "Emilio's Monthly Report - May 2026.xlsx",
    "Jun": DATA_DIR / "Emilio's Monthly Report - Jun 2026_1.xlsx",
}

# Column index inside each file's P&L sheet that holds the reporting month
# Confirmed by inspecting row 4 of each file's "Emilio's PL" sheet:
#   Row 4 header: Particulars | Dec | % | Nov | % | … | reporting_month | % | …
# Jun file has one extra pair (Jul) vs May file, shifting Jun column to 13.
_V2_PL_COL = {
    "Jan": 23,
    "Feb": 21,
    "Mar": 19,
    "Apr": 17,
    "May": 15,
    "Jun": 13,   # Jun file: July added → June shifts from 12 to 13
}

# Period labels
_V2_PERIOD = {
    "Jan": "2026-01",
    "Feb": "2026-02",
    "Mar": "2026-03",
    "Apr": "2026-04",
    "May": "2026-05",
    "Jun": "2026-06",
}

# ── legacy (original 5-month) constants kept for backward compat ──────────────
PL_MONTH_COLS = {
    "May": 15,
    "April": 17,
    "March": 19,
    "February": 21,
    "January": 23,
}
PL_MONTH_LABELS = {
    "January": "2026-01",
    "February": "2026-02",
    "March": "2026-03",
    "April": "2026-04",
    "May": "2026-05",
}


def _num(value) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class DataQualityReport:
    source_files: list[str] = field(default_factory=list)
    sheets_indexed: dict[str, list[str]] = field(default_factory=dict)
    issues: list[dict[str, Any]] = field(default_factory=list)
    completeness: dict[str, float] = field(default_factory=dict)
    summary: str = ""

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.issues) if self.issues else pd.DataFrame(
            columns=["dataset", "issue", "severity", "detail"]
        )


class AuditDataPipeline:
    """Load, assess, clean, and engineer features from Emilio's Pizza audit Excel files.

    v2: if all 6 monthly files exist in DATA_DIR the pipeline uses them;
    otherwise it falls back to the original 3-file layout so the original
    dashboard.py continues working unchanged.
    """

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.april_path = data_dir / "Emilio's Monthly Report - April 2026.xlsx"
        self.may_path   = data_dir / "Emilio's Monthly Report - May 2026.xlsx"
        self.daily_path = data_dir / "feb10628-e3b2-47de-8b87-ce11a1a84e39.xlsx"

        # v2: detect available monthly files
        self._v2_files: dict[str, Path] = {
            m: p for m, p in _V2_FILES.items() if p.exists()
        }
        self._use_v2 = len(self._v2_files) >= 5  # at least 5 of 6 present

        self.monthly_sales: pd.DataFrame = pd.DataFrame()
        self.monthly_pl:    pd.DataFrame = pd.DataFrame()
        self.daily_sales:   pd.DataFrame = pd.DataFrame()
        self.creditors:     pd.DataFrame = pd.DataFrame()
        self.debtors:       pd.DataFrame = pd.DataFrame()
        self.wastage:       pd.DataFrame = pd.DataFrame()
        self.category_sales:pd.DataFrame = pd.DataFrame()
        self.feature_matrix:pd.DataFrame = pd.DataFrame()
        self.quality = DataQualityReport()

    # ------------------------------------------------------------------ collection
    def load_all(self) -> "AuditDataPipeline":
        self._load_monthly_sales()
        self._load_monthly_pl()
        self._load_daily_sales()
        self._load_creditors_debtors()
        self._load_wastage()
        self._load_category_snapshot()
        self.assess_quality()
        self.feature_matrix = self.build_feature_matrix()
        return self

    def _load_monthly_sales(self) -> None:
        # v2: use Jun file which has the full Jan-Jun cumulative sales sheet
        src = self._v2_files.get("Jun", self.may_path)
        if not src.exists():
            src = self.may_path
        raw = pd.read_excel(src, sheet_name="Emilio's Monthly Sales", header=None)
        rows = []
        for _, row in raw.iloc[2:14].iterrows():
            month = str(row.iloc[0]).strip()
            if month not in MONTH_ORDER:
                continue
            rows.append(
                {
                    "month": month,
                    "sales_2024": _num(row.iloc[1]),
                    "sales_2025": _num(row.iloc[2]),
                    "total_sales_2026": _num(row.iloc[3]),
                    "discount": _num(row.iloc[4]),
                    "discount_pct": _num(row.iloc[5]),
                    "vat": _num(row.iloc[6]),
                    "net_sales": _num(row.iloc[7]),
                    "cash_sales": _num(row.iloc[8]),
                    "cash_pct": _num(row.iloc[9]),
                    "foodmandu_sales": _num(row.iloc[10]),
                    "foodmandu_pct": _num(row.iloc[11]),
                    "pokhara_sales": _num(row.iloc[14]),
                    "pokhara_pct": _num(row.iloc[15]),
                    "credit_sales": _num(row.iloc[16]),
                    "credit_pct": _num(row.iloc[17]),
                }
            )
        self.monthly_sales = pd.DataFrame(rows)
        self.monthly_sales["month_num"] = self.monthly_sales["month"].map(
            {m: i + 1 for i, m in enumerate(MONTH_ORDER)}
        )

    def _pl_value(self, pl: pd.DataFrame, label: str, col: int) -> float | None:
        target = label.lower()
        for _, row in pl.iterrows():
            cell = str(row.iloc[0]).strip().lower() if pd.notna(row.iloc[0]) else ""
            if cell == target:
                return _num(row.iloc[col])
        return None

    def _load_monthly_pl(self) -> None:
        """v2: extract P&L from each month's own file using per-month column offsets.
        Falls back to the original May-file-only loader when only old files exist.
        """
        if self._use_v2:
            self._load_monthly_pl_v2()
        else:
            self._load_monthly_pl_legacy()

    def _load_monthly_pl_v2(self) -> None:
        """Read P&L for each available month from its individual report file."""
        rows = []
        for month, fpath in self._v2_files.items():
            col = _V2_PL_COL[month]
            period = _V2_PERIOD[month]
            try:
                pl = pd.read_excel(fpath, sheet_name="Emilio's PL", header=None)
            except Exception:
                continue

            def _get(label: str) -> float | None:
                return self._pl_value(pl, label, col)

            sales  = _get("sales value")
            incomes= _get("total incomes")
            cogs   = _get("cost of goods sold")
            gp     = _get("gross profit")
            opex   = _get("operational expenses")
            salary = _get("staff salary & wages")
            fm_fee = _get("foodmandu charges")
            tot_exp= _get("total expenses")
            profit = _get("profit")
            bread  = _get("bread income")

            rows.append({
                "period":               period,
                "month":                month,
                "sales_value":          sales,
                "total_incomes":        incomes,
                "cogs":                 cogs,
                "gross_profit":         gp,
                "gross_margin":         (gp / incomes) if incomes and gp is not None else None,
                "cogs_ratio":           (cogs / incomes) if incomes and cogs is not None else None,
                "operational_expenses": opex,
                "staff_salary":         salary,
                "foodmandu_charges":    fm_fee,
                "total_expenses":       tot_exp,
                "profit":               profit,
                "profit_margin":        (profit / incomes) if incomes and profit is not None else None,
                "bread_income":         bread,
                "salary_to_sales":      (salary / sales) if sales and salary is not None else None,
            })

        self.monthly_pl = (
            pd.DataFrame(rows)
            .sort_values("period")
            .reset_index(drop=True)
        )

    def _load_monthly_pl_legacy(self) -> None:
        """Original 5-month loader using only the May file (backward compat)."""
        pl = pd.read_excel(self.may_path, sheet_name="Emilio's PL", header=None)
        rows = []
        for month_name, col in PL_MONTH_COLS.items():
            period = PL_MONTH_LABELS[month_name]
            sales  = self._pl_value(pl, "sales value", col)
            incomes= self._pl_value(pl, "total incomes", col)
            cogs   = self._pl_value(pl, "cost of goods sold", col)
            gp     = self._pl_value(pl, "gross profit", col)
            opex   = self._pl_value(pl, "operational expenses", col)
            salary = self._pl_value(pl, "staff salary & wages", col)
            fm_fee = self._pl_value(pl, "foodmandu charges", col)
            tot_exp= self._pl_value(pl, "total expenses", col)
            profit = self._pl_value(pl, "profit", col)
            bread  = self._pl_value(pl, "bread income", col)
            rows.append({
                "period":               period,
                "month":                month_name[:3],
                "sales_value":          sales,
                "total_incomes":        incomes,
                "cogs":                 cogs,
                "gross_profit":         gp,
                "gross_margin":         (gp / incomes) if incomes and gp is not None else None,
                "cogs_ratio":           (cogs / incomes) if incomes and cogs is not None else None,
                "operational_expenses": opex,
                "staff_salary":         salary,
                "foodmandu_charges":    fm_fee,
                "total_expenses":       tot_exp,
                "profit":               profit,
                "profit_margin":        (profit / incomes) if incomes and profit is not None else None,
                "bread_income":         bread,
                "salary_to_sales":      (salary / sales) if sales and salary is not None else None,
            })
        self.monthly_pl = pd.DataFrame(rows).sort_values("period").reset_index(drop=True)

    def _load_daily_sales(self) -> None:
        """Load daily sales.  The original feb10628 file may no longer exist;
        fall back gracefully to processed/daily_sales.csv if available."""
        from config import PROCESSED_DIR

        # Try the original daily Excel first
        if self.daily_path.exists():
            try:
                raw = pd.read_excel(self.daily_path, sheet_name="Sales Summary ", header=None)
                rows = []
                for _, row in raw.iloc[4:].iterrows():
                    date_val = row.iloc[0]
                    total = _num(row.iloc[5])
                    if date_val is None or total is None:
                        continue
                    try:
                        date = pd.to_datetime(date_val)
                    except Exception:
                        continue
                    if date.year < 2020:
                        continue
                    if total > 500_000:
                        continue
                    cash = _num(row.iloc[1]) or 0
                    online = _num(row.iloc[2]) or 0
                    credit = _num(row.iloc[4]) or 0
                    customers = _num(row.iloc[8])
                    dine_in = _num(row.iloc[9])
                    rows.append({
                        "date": date, "cash": cash, "online": online,
                        "credit": credit, "total_sales": total,
                        "customers": customers, "dine_in_sales": dine_in,
                        "avg_ticket": (dine_in / customers) if customers and dine_in else None,
                    })
                self.daily_sales = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
            except Exception:
                self.daily_sales = pd.DataFrame()
        else:
            # Fall back to pre-exported CSV (generated by generate_evidence.py)
            csv_path = PROCESSED_DIR / "daily_sales.csv"
            if csv_path.exists():
                self.daily_sales = pd.read_csv(csv_path, parse_dates=["date"])
            else:
                self.daily_sales = pd.DataFrame()

        if not self.daily_sales.empty:
            if "day_of_week" not in self.daily_sales.columns:
                self.daily_sales["day_of_week"] = self.daily_sales["date"].dt.day_name()
            if "dow" not in self.daily_sales.columns:
                self.daily_sales["dow"] = self.daily_sales["date"].dt.dayofweek

    def _load_creditors_debtors(self) -> None:
        """v2: load creditors/debtors from all available monthly files."""
        frames = []
        if self._use_v2:
            period_paths = [
                (_V2_PERIOD[m], p)
                for m, p in self._v2_files.items()
            ]
        else:
            period_paths = [
                ("2026-04", self.april_path),
                ("2026-05", self.may_path),
            ]

        for period, path in period_paths:
            for sheet, kind in [("Creditors Analysis", "creditor"), ("Debtors Analysis", "debtor")]:
                try:
                    raw = pd.read_excel(path, sheet_name=sheet, header=None)
                except Exception:
                    continue
                for _, row in raw.iloc[3:].iterrows():
                    name = row.iloc[0]
                    closing = _num(row.iloc[4]) if len(row) > 4 else None
                    opening = _num(row.iloc[1]) if len(row) > 1 else None
                    if not isinstance(name, str) or closing is None:
                        continue
                    name_l = name.strip().lower()
                    if "grand total" in name_l or name_l == "total":
                        continue
                    frames.append({
                        "period":   period,
                        "type":     kind,
                        "party":    name.strip(),
                        "opening":  opening,
                        "closing":  closing,
                        "movement": (closing - opening) if opening is not None else None,
                    })

        parties = pd.DataFrame(frames)
        if parties.empty:
            self.creditors = pd.DataFrame()
            self.debtors   = pd.DataFrame()
        else:
            self.creditors = parties[parties["type"] == "creditor"].copy()
            self.debtors   = parties[parties["type"] == "debtor"].copy()

    def _load_wastage(self) -> None:
        """v2: load wastage from all available monthly files."""
        frames = []
        if self._use_v2:
            period_paths = [(_V2_PERIOD[m], p) for m, p in self._v2_files.items()]
        else:
            period_paths = [
                ("2026-04", self.april_path),
                ("2026-05", self.may_path),
            ]

        for period, path in period_paths:
            try:
                raw = pd.read_excel(path, sheet_name="Sample Gift waste", header=None)
            except Exception:
                continue
            for _, row in raw.iloc[3:].iterrows():
                item = row.iloc[0]
                if not isinstance(item, str) or not item.strip():
                    continue
                if item.strip().lower() in {"pizza", "bansbari", "total"}:
                    continue
                ktm   = _num(row.iloc[1]) or 0
                pkr   = (_num(row.iloc[2]) or 0) if len(row) > 2 else 0
                total = (_num(row.iloc[3]) or (ktm + pkr)) if len(row) > 3 else (ktm + pkr)
                frames.append({
                    "period": period,
                    "item":   item.strip(),
                    "ktm":    ktm,
                    "pkr":    pkr,
                    "total":  total or 0,
                })
        self.wastage = pd.DataFrame(frames)

    def _load_category_snapshot(self) -> None:
        """Latest available category value columns for May 2026."""
        raw = pd.read_excel(self.may_path, sheet_name="Product Sales - Category wise", header=None)
        # May 2026 qty/value are near the end — columns 45/46 in May file
        rows = []
        for _, row in raw.iloc[2:].iterrows():
            cat = row.iloc[0]
            if not isinstance(cat, str) or not cat.strip():
                continue
            # Prefer last non-null qty/value pair in known May region
            qty = _num(row.iloc[45]) if len(row) > 45 else None
            val = _num(row.iloc[46]) if len(row) > 46 else None
            if val is None:
                continue
            rows.append({"category": cat.strip(), "qty": qty, "value": val})
        self.category_sales = pd.DataFrame(rows)

    # ------------------------------------------------------------------ quality
    def assess_quality(self) -> DataQualityReport:
        # v2: report on all available files
        source_files = (
            list(self._v2_files.values())
            if self._use_v2
            else [self.april_path, self.may_path]
        )
        if self.daily_path.exists():
            source_files = [self.daily_path] + [p for p in source_files if p != self.daily_path]

        report = DataQualityReport(
            source_files=[p.name for p in source_files if p.exists()]
        )
        for path in source_files:
            if path.exists():
                try:
                    xl = pd.ExcelFile(path)
                    report.sheets_indexed[path.name] = xl.sheet_names
                except Exception:
                    pass

        def check(name: str, df: pd.DataFrame, required_cols: list[str]) -> None:
            if df.empty:
                report.issues.append({
                    "dataset": name, "issue": "Empty dataset",
                    "severity": "High", "detail": "No rows extracted",
                })
                report.completeness[name] = 0.0
                return
            avail = [c for c in required_cols if c in df.columns]
            if not avail:
                report.completeness[name] = 1.0
                return
            missing = df[avail].isna().mean()
            report.completeness[name] = float(1 - missing.mean())
            for col, rate in missing.items():
                if rate > 0:
                    report.issues.append({
                        "dataset": name, "issue": "Missing values",
                        "severity": "Medium" if rate < 0.3 else "High",
                        "detail": f"{col}: {rate:.0%} missing",
                    })

        sales_with_data = self.monthly_sales[self.monthly_sales["net_sales"].notna()] \
            if not self.monthly_sales.empty else pd.DataFrame()
        check("monthly_sales_2026", sales_with_data, ["net_sales", "cash_pct"])
        check("monthly_pl",         self.monthly_pl,  ["sales_value", "cogs", "gross_profit", "profit"])
        check("daily_sales",        self.daily_sales,  ["total_sales", "cash", "online"])
        check("creditors",          self.creditors,    ["closing"])
        check("debtors",            self.debtors,      ["closing"])

        n_months = len(self.monthly_pl)
        report.issues.append({
            "dataset": "coverage",
            "issue":   f"{'Full' if n_months >= 6 else 'Partial'} {n_months}-month window",
            "severity":"Low" if n_months >= 6 else "Medium",
            "detail":  f"P&L for {n_months} months loaded; daily grain for May only",
        })
        report.issues.append({
            "dataset": "schema", "issue": "Semi-structured Excel",
            "severity": "Low",
            "detail": "Headers irregular; values extracted via label/column mapping",
        })
        report.summary = (
            f"Indexed {len(report.source_files)} workbooks. "
            f"Monthly P&L months: {len(self.monthly_pl)}. "
            f"Daily sales days: {len(self.daily_sales)}. "
            f"Issues logged: {len(report.issues)}."
        )
        self.quality = report
        return report

    # ------------------------------------------------------------------ features
    def build_feature_matrix(self) -> pd.DataFrame:
        """Engineer monthly financial intelligence features for all loaded months."""
        # Full period map (supports Jun)
        full_period_map = {
            "Jan": "2026-01", "Feb": "2026-02", "Mar": "2026-03",
            "Apr": "2026-04", "May": "2026-05", "Jun": "2026-06",
        }
        sales_2026 = self.monthly_sales[self.monthly_sales["net_sales"].notna()].copy()
        sales_2026["period"] = sales_2026["month"].map(full_period_map)
        sales_2026 = sales_2026[sales_2026["period"].notna()]

        merged = self.monthly_pl.merge(
            sales_2026[[
                "period", "net_sales", "cash_pct", "foodmandu_pct",
                "pokhara_pct", "credit_pct", "discount_pct",
                "cash_sales", "foodmandu_sales", "credit_sales",
            ]],
            on="period", how="left",
        )

        for kind, frame in [("creditor", self.creditors), ("debtor", self.debtors)]:
            if frame is not None and not frame.empty:
                totals = frame.groupby("period")["closing"].sum().rename(f"{kind}_total")
                merged = merged.merge(totals, left_on="period", right_index=True, how="left")

        if self.wastage is not None and not self.wastage.empty:
            waste = self.wastage.groupby("period")["total"].sum().rename("wastage_units")
            merged = merged.merge(waste, left_on="period", right_index=True, how="left")

        merged = merged.sort_values("period").reset_index(drop=True)

        merged["sales_mom_pct"]    = merged["sales_value"].pct_change()
        merged["cogs_mom_pct"]     = merged["cogs"].pct_change()
        merged["profit_mom_pct"]   = merged["profit"].pct_change()
        merged["margin_pressure"]  = merged["cogs_mom_pct"] - merged["sales_mom_pct"]
        merged["liquidity_proxy"]  = merged["cash_pct"] * merged["net_sales"]
        merged["platform_dependency"] = merged["foodmandu_pct"].fillna(0)
        merged["credit_exposure"]  = merged["credit_pct"].fillna(0)

        for col in ["creditor_total", "debtor_total", "wastage_units"]:
            if col in merged.columns:
                merged[col] = merged[col].ffill().bfill()

        # Stress label — same rule as before, now applied to all months
        merged["stress"] = (
            (merged["profit_margin"].fillna(1) < 0.08)
            | (merged["cogs_ratio"].fillna(0) > 0.52)
            | (
                (merged["margin_pressure"].fillna(0) > 0.05)
                & (merged["profit_margin"].fillna(1) < 0.12)
            )
        ).astype(int)

        return merged

    def descriptive_stats(self) -> dict[str, pd.DataFrame]:
        stats: dict[str, pd.DataFrame] = {}
        if not self.monthly_pl.empty:
            pl_cols = [c for c in
                       ["sales_value","cogs","gross_profit","profit","gross_margin","staff_salary"]
                       if c in self.monthly_pl.columns]
            stats["monthly_pl"] = self.monthly_pl[pl_cols].describe()
        if not self.daily_sales.empty:
            d_cols = [c for c in ["total_sales","cash","online","credit","customers"]
                      if c in self.daily_sales.columns]
            stats["daily_sales"] = self.daily_sales[d_cols].describe()
        if not self.feature_matrix.empty:
            feat_cols = [c for c in [
                "sales_value","gross_margin","cogs_ratio","profit_margin",
                "salary_to_sales","cash_pct","platform_dependency",
                "wastage_units","creditor_total","debtor_total",
            ] if c in self.feature_matrix.columns]
            stats["engineered_features"] = self.feature_matrix[feat_cols].describe()
        return stats

    def inventory_summary(self) -> pd.DataFrame:
        """v2: summarise all available monthly files."""
        paths = (
            list(self._v2_files.values())
            if self._use_v2
            else [self.april_path, self.may_path]
        )
        if self.daily_path.exists():
            paths = [self.daily_path] + [p for p in paths if p != self.daily_path]

        rows = []
        for path in paths:
            if not path.exists():
                continue
            try:
                xl = pd.ExcelFile(path)
                rows.append({
                    "file":         path.name,
                    "sheets":       len(xl.sheet_names),
                    "sheet_names":  ", ".join(xl.sheet_names[:8])
                                    + ("..." if len(xl.sheet_names) > 8 else ""),
                })
            except Exception:
                rows.append({"file": path.name, "sheets": "?", "sheet_names": "error reading file"})
        return pd.DataFrame(rows)
