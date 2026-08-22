"""Load SME restaurant audit Excel files into RAG-ready chunks."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.pdf_loader import DocumentChunk

PRIORITY_SHEETS = {
    "Emilio's Monthly Sales": "Sales Analysis",
    "Emilio's PL": "Profit and Loss",
    "Pokhara PL": "Pokhara Profit and Loss",
    "Food Margin analysis": "Food Margin Analysis",
    "Product Sales - Category wise": "Product Sales by Category",
    "Purchase Analysis": "Purchase Analysis",
    "Cash Flow": "Cash Flow",
    "Creditors Analysis": "Creditors Analysis",
    "Debtors Analysis": "Debtors Analysis",
    "Sample Gift waste": "Wastage and Sample/Gift",
    "Closing Stock ": "Closing Stock",
    "Cash Collection Report": "Cash Collection",
    "Sales Summary ": "Daily Sales Summary",
    "Bread Sales": "Bread Sales",
}

AUDIT_KEYWORDS = (
    "sales",
    "profit",
    "loss",
    "margin",
    "cash",
    "flow",
    "creditor",
    "debtor",
    "stock",
    "inventory",
    "wastage",
    "waste",
    "purchase",
    "revenue",
    "audit",
    "emilio",
    "restaurant",
    "april",
    "may",
    "pokhara",
    "foodmandu",
)


def _clean(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _format_number(value) -> str:
    if isinstance(value, (int, float)) and not pd.isna(value):
        if abs(value) >= 1000:
            return f"{value:,.2f}"
        if float(value).is_integer():
            return str(int(value))
        return f"{value:.2f}"
    return _clean(value) or ""


def detect_report_period(file_name: str, xl: pd.ExcelFile) -> str:
    name = file_name.lower()
    # v2: detect all 6 months by filename first (fast path)
    if "jan" in name and "2026" in name:
        return "January 2026"
    if "feb" in name and "2026" in name:
        return "February 2026"
    if "march" in name or ("mar" in name and "2026" in name):
        return "March 2026"
    if "april" in name:
        return "April 2026"
    if "may" in name:
        return "May 2026"
    if "jun" in name and "2026" in name:
        return "June 2026"

    # Fallback: scan first few sheets for month/year text
    for sheet in xl.sheet_names[:5]:
        try:
            df = pd.read_excel(xl, sheet_name=sheet, header=None, nrows=8)
        except Exception:
            continue
        flat = " ".join(_clean(v) or "" for v in df.values.flatten()).lower()
        for month, label in [
            ("january",  "January 2026"),
            ("february", "February 2026"),
            ("march",    "March 2026"),
            ("april",    "April 2026"),
            ("may",      "May 2026"),
            ("june",     "June 2026"),
        ]:
            if month in flat and "2026" in flat:
                return label
    return "Unknown Period"


def _row_to_line(row: pd.Series) -> str | None:
    cells = [_clean(v) for v in row.tolist()]
    cells = [c for c in cells if c]
    if not cells:
        return None
    return " | ".join(cells)


def _format_number_short(value) -> str:
    """Format numbers compactly for table cells."""
    if isinstance(value, (int, float)) and not pd.isna(value):
        if abs(value) >= 1_000_000:
            return f"{value/1_000_000:.2f}M"
        if abs(value) >= 1_000:
            return f"{value:,.0f}"
        if float(value) == int(value):
            return str(int(value))
        return f"{value:.3f}"
    cleaned = _clean(value)
    return cleaned if cleaned else ""


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 35, max_cols: int = 10) -> str:
    """Convert a DataFrame to a proper markdown table.

    Scans the first non-empty row to use as the header.
    Caps at max_cols columns to keep wide Excel sheets readable.
    Falls back to a compact pipe-row format for completely headerless sheets.
    """
    if df.empty:
        return ""

    # Drop columns that are entirely null
    df = df.dropna(axis=1, how="all").reset_index(drop=True)

    # Find first row that has at least 2 non-null values to use as header
    header_idx = None
    for i, row in df.head(5).iterrows():
        cells = [_clean(v) for v in row.tolist() if _clean(v)]
        if len(cells) >= 2:
            header_idx = i
            break

    if header_idx is None:
        lines = []
        for _, row in df.head(max_rows).iterrows():
            line = _row_to_line(row)
            if line:
                lines.append(line)
        return "\n".join(lines)

    header_row = df.iloc[header_idx]
    headers = [_clean(v) or f"col{i}" for i, v in enumerate(header_row.tolist())]

    # Remove trailing unnamed columns
    while headers and headers[-1].startswith("col"):
        headers.pop()

    # Cap column count
    n_cols = min(len(headers), max_cols)
    headers = headers[:n_cols]

    # Build markdown table
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * n_cols) + " |")

    data_start = header_idx + 1
    rows_added = 0
    for _, row in df.iloc[data_start:data_start + max_rows].iterrows():
        cells = [_format_number_short(v) for v in row.tolist()[:n_cols]]
        while len(cells) < n_cols:
            cells.append("")
        row_str = "| " + " | ".join(cells) + " |"
        if set(cells) - {""} == set():
            continue
        lines.append(row_str)
        rows_added += 1

    total_data = len(df) - data_start
    if total_data > max_rows:
        lines.append(f"\n*...{total_data - max_rows} additional rows not shown*")

    return "\n".join(lines)


def dataframe_to_text(df: pd.DataFrame, max_rows: int = 40) -> str:
    """Keep for backward compatibility — now returns markdown table format."""
    return dataframe_to_markdown(df, max_rows=max_rows)


def chunk_sheet_text(
    text: str,
    prefix: str,
    chunk_size: int = 1200,
    overlap: int = 0,
) -> list[str]:
    """Split sheet text into chunks, always breaking on line boundaries.

    For markdown tables we never want to cut mid-row, so we split on newlines
    and reassemble greedily. The separator row (|---|---| …) is carried into
    every chunk that follows the header so the table renders correctly.
    """
    if len(text) <= chunk_size:
        return [text]

    lines = text.split("\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    # Detect header + separator rows for table re-injection
    header_lines: list[str] = []
    for line in lines[:3]:
        if line.startswith("|") and "---" not in line:
            header_lines.append(line)
        elif line.startswith("|") and "---" in line:
            header_lines.append(line)
            break

    for line in lines:
        line_len = len(line) + 1
        if current_len + line_len > chunk_size and current:
            chunks.append("\n".join(current))
            # Start next chunk with table header so it still renders
            current = list(header_lines) if header_lines else []
            current_len = sum(len(l) + 1 for l in current)
        current.append(line)
        current_len += line_len

    if current:
        chunks.append("\n".join(current))

    return [c for c in chunks if c.strip()]


def _find_month_columns(df: pd.DataFrame, month: str) -> list[int]:
    month = month.lower()
    for row_idx in range(min(8, len(df))):
        for col_idx, value in enumerate(df.iloc[row_idx].tolist()):
            cell = (_clean(value) or "").lower()
            if cell == month or cell.startswith(f"{month},"):
                return [col_idx, col_idx + 1]
    return []


def _get_pl_metric(df: pd.DataFrame, label: str, month_cols: list[int]) -> str | None:
    if not month_cols:
        return None
    label_lower = label.lower()
    for _, row in df.iterrows():
        first = (_clean(row.iloc[0]) or "").lower()
        if label_lower in first:
            for col in month_cols:
                if col < len(row):
                    value = row.iloc[col]
                    if isinstance(value, (int, float)) and not pd.isna(value):
                        return _format_number(value)
                    text = _clean(value)
                    if text and not text.endswith("%"):
                        return text
    return None


def build_kpi_summary(file_name: str, period: str, xl: pd.ExcelFile) -> str | None:
    # period is e.g. "January 2026", "June 2026" etc.
    month_token = period.split()[0].lower()
    short_token = month_token[:3]  # "jan", "feb", "mar", "apr", "may", "jun"

    # Use hardcoded column offsets confirmed by inspection
    # Jun file has an extra July column pair, shifting June to col 13
    _PL_COL_FOR_MONTH = {
        "jan": 23, "feb": 21, "mar": 19, "apr": 17, "may": 15, "jun": 13,
    }

    lines = [
        f"Audit KPI Summary for Emilio's Pizza Pvt. Ltd.",
        f"Source file: {file_name}",
        f"Report period: {period}",
        f"Business: SME restaurant, Kathmandu (Bansbari + Pokhara branches)",
    ]

    if "Emilio's PL" in xl.sheet_names:
        pl = pd.read_excel(xl, sheet_name="Emilio's PL", header=None)
        # Use the known P&L column offset for this month
        pl_col = _PL_COL_FOR_MONTH.get(short_token)
        if pl_col is not None:
            month_cols = [pl_col]
        else:
            month_cols = _find_month_columns(pl, short_token)
            if not month_cols:
                month_cols = _find_month_columns(pl, month_token)
        metrics = {
            "Sales Value": _get_pl_metric(pl, "Sales Value", month_cols),
            "Cost of Goods Sold": _get_pl_metric(pl, "Cost of Goods Sold", month_cols),
            "Gross Profit": _get_pl_metric(pl, "Gross Profit", month_cols),
            "Net Profit": _get_pl_metric(pl, "Net Profit", month_cols),
            "Staff Salary": _get_pl_metric(pl, "Staff Salary", month_cols),
        }
        found = {k: v for k, v in metrics.items() if v}
        if found:
            lines.append("Profit & Loss highlights:")
            for key, value in found.items():
                lines.append(f"- {key}: {value}")

    if "Food Margin analysis" in xl.sheet_names:
        margin = pd.read_excel(xl, sheet_name="Food Margin analysis", header=None)
        # Food Margin sheet only has the current month's column — use col 1
        fm_cols = [1]
        sales = _get_pl_metric(margin, "Sales Value", fm_cols)
        cogs  = _get_pl_metric(margin, "Cost of Goods Sold", fm_cols)
        if sales:
            lines.append(f"- Total Sales Value: {sales}")
        if cogs:
            lines.append(f"- Cost of Goods Sold: {cogs}")

    if "Emilio's Monthly Sales" in xl.sheet_names:
        sales_df = pd.read_excel(xl, sheet_name="Emilio's Monthly Sales", header=None)
        for _, row in sales_df.iterrows():
            month_cell = (_clean(row.iloc[0]) or "").lower()
            if month_cell in (short_token, month_token):
                net_sales = _format_number(row.iloc[7]) if len(row) > 7 else ""
                cash_pct = _format_number(row.iloc[9]) if len(row) > 9 else ""
                if net_sales:
                    lines.append(f"- Net Sales ({period}): {net_sales}")
                if cash_pct:
                    lines.append(f"- Cash Sales share: {cash_pct}")
                break

    if "Cash Flow" in xl.sheet_names:
        cash = pd.read_excel(xl, sheet_name="Cash Flow", header=None)
        for _, row in cash.iterrows():
            label = (_clean(row.iloc[0]) or "").lower()
            value = _format_number(row.iloc[1]) if len(row) > 1 else ""
            if label in {"cash balance", "bank balance", "total opening"} and value:
                lines.append(f"- {row.iloc[0]}: {value}")

    if "Creditors Analysis" in xl.sheet_names:
        creditors = pd.read_excel(xl, sheet_name="Creditors Analysis", header=None)
        rows: list[str] = []
        total_closing = 0.0
        for _, row in creditors.iloc[3:13].iterrows():
            supplier = _clean(row.iloc[0])
            closing = row.iloc[4] if len(row) > 4 else None
            if supplier and isinstance(closing, (int, float)) and not pd.isna(closing):
                total_closing += float(closing)
                rows.append(f"- {supplier}: closing balance {closing:,.2f}")
        if rows:
            lines.append("Top creditors:")
            lines.extend(rows)
            lines.append(f"- Total creditor closing balance (top entries): {total_closing:,.2f}")

    if "Debtors Analysis" in xl.sheet_names:
        debtors = pd.read_excel(xl, sheet_name="Debtors Analysis", header=None)
        total_closing = 0.0
        count = 0
        for _, row in debtors.iloc[3:].iterrows():
            customer = _clean(row.iloc[0])
            closing = row.iloc[4] if len(row) > 4 else None
            if customer and isinstance(closing, (int, float)) and not pd.isna(closing):
                total_closing += float(closing)
                count += 1
        if count:
            lines.append(f"- Total debtor closing balance: {total_closing:,.2f} across {count} customers")

    if len(lines) <= 4:
        return None
    return "\n".join(lines)


def load_excel_chunks(data_dir: Path, chunk_size: int = 900) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    excel_files = sorted(data_dir.glob("*.xlsx"))
    counter = 0

    for file_path in excel_files:
        xl = pd.ExcelFile(file_path)
        period = detect_report_period(file_path.name, xl)
        section_label = PRIORITY_SHEETS

        overview = [
            f"Emilio's Pizza SME Audit Dataset",
            f"File: {file_path.name}",
            f"Period: {period}",
            f"Sheets: {', '.join(xl.sheet_names)}",
            "Data categories: sales, P&L, food margin, purchases, cash flow, creditors, debtors, wastage, inventory.",
        ]
        counter += 1
        chunks.append(
            DocumentChunk(
                text="\n".join(overview),
                page=0,
                chunk_id=f"audit_{counter:05d}",
                section="Audit Data Overview",
                source_type="audit_data",
                file_name=file_path.name,
                sheet_name="Overview",
                report_period=period,
            )
        )

        kpi_text = build_kpi_summary(file_path.name, period, xl)
        if kpi_text:
            counter += 1
            chunks.append(
                DocumentChunk(
                    text=kpi_text,
                    page=0,
                    chunk_id=f"audit_{counter:05d}",
                    section="Audit KPI Summary",
                    source_type="audit_data",
                    file_name=file_path.name,
                    sheet_name="KPI Summary",
                    report_period=period,
                )
            )

        for sheet_name in xl.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
            if df.empty:
                continue

            section = PRIORITY_SHEETS.get(sheet_name, sheet_name.strip())
            header = (
                f"Audit Record — Emilio's Pizza | Period: {period} | "
                f"File: {file_path.name} | Sheet: {sheet_name}"
            )
            body = dataframe_to_text(df, max_rows=45 if len(df) > 60 else len(df))
            if not body:
                continue

            full_text = f"{header}\n{body}"
            for piece in chunk_sheet_text(full_text, prefix=sheet_name, chunk_size=chunk_size):
                counter += 1
                chunks.append(
                    DocumentChunk(
                        text=piece,
                        page=0,
                        chunk_id=f"audit_{counter:05d}",
                        section=section,
                        source_type="audit_data",
                        file_name=file_path.name,
                        sheet_name=sheet_name,
                        report_period=period,
                    )
                )

    return chunks


def list_audit_inventory(data_dir: Path) -> list[dict[str, str]]:
    inventory: list[dict[str, str]] = []
    for file_path in sorted(data_dir.glob("*.xlsx")):
        try:
            xl = pd.ExcelFile(file_path)
            inventory.append({
                "file":        file_path.name,
                "period":      detect_report_period(file_path.name, xl),
                "sheets":      ", ".join(xl.sheet_names),
                "sheet_count": str(len(xl.sheet_names)),
            })
        except Exception:
            inventory.append({
                "file":        file_path.name,
                "period":      "Error reading",
                "sheets":      "",
                "sheet_count": "?",
            })
    return inventory
