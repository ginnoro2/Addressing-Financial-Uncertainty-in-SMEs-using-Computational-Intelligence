# Addressing Financial Uncertainty in SMEs: A Computational Intelligence Framework for Risk and Profitability Assessment Using Audit Data from Restaurants in Kathmandu

**MSc Research Prototype · Rupak Rajbanshi (Student ID: 210333)**

---

## Research Overview

### Aim
To design and evaluate a computational intelligence based financial intelligence framework that assesses financial risk and profitability of SME restaurants using summarized audit data.

### Research Questions
- **RQ1:** How can a computational financial intelligence framework be designed and developed using statistical analysis and machine learning techniques to improve the accuracy of risk and profitability assessments based on summarized audit data from SME restaurants in Kathmandu?
- **RQ2:** What are the ethical considerations, design principles, and responsibilities associated with developing and deploying predictive financial models that influence real-world credit and operational decision-making in the context of data-constrained SMEs in Kathmandu?

### Hypotheses
- **H1:** Ensemble tree-based ML (Random Forest, XGBoost) significantly outperforms linear models for financial stress prediction. **Result: Supported — RF LOO accuracy 0.80 vs Logistic 0.60.**
- **H2:** SHAP-based explainability enhances credit decisions and reduces cognitive bias, but deployment without ethical safeguards risks algorithmic exclusion and stakeholder mistrust. **Result: Evidenced — SHAP feature attributions implemented; bias checklist validated.**

---

## Does the Framework Address the Research Aim?

| Objective | Implementation | Evidence |
|-----------|----------------|---------|
| Collect & preprocess 6-month audit data | `src/data_pipeline.py` — Jan–Jun 2026, 6 workbooks, 9 data domains | `processed/feature_matrix.csv`, Section 14 figures |
| Derive financial features & ratios | 12 engineered features: gross_margin, cogs_ratio, profit_margin, salary_to_sales, platform_dependency, credit_exposure, margin_pressure, liquidity_proxy | `thesis_docs/14.../05_Feature_Engineering/` |
| Risk scoring & predictive uncertainty models | RF (LOO acc=0.80), XGBoost, GB, Logistic Regression; P10–P90 probabilistic daily forecast; cash-flow stress probability | `processed/model_comparison.csv`, Section 15 figures |
| Explainable AI & bias mitigation | SHAP TreeExplainer, feature importance; 5-bias checklist (anchoring, availability, overconfidence, narrative fallacy, algorithmic exclusion) | `src/model_lab.py`, `src/analytics_layers.py` |
| Prototype for FinTech/SME decision support | Streamlit dashboard (Sections 14–15) + RAG assistant over proposal & audit data | `dashboard_v2.py`, `app_v2.py` |

---

## Launch the Prototype

### v2 Dashboard 
```bash
cd /Users/priyankarai/MSC
./run_dashboard.sh
# Opens at http://localhost:8501
```

### v2 RAG Assistant 
```bash
./run_rag.sh
# Opens at http://localhost:8501
```

### Manual launch 
```bash
cd /Users/priyankarai/MSC
source venv/bin/activate

# Required env vars to prevent segfault on macOS / Python 3.13
export LOKY_MAX_CPU_COUNT=1 JOBLIB_MULTIPROCESSING=0 OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

streamlit run dashboard_v2.py          # Main prototype  (port 8501)
streamlit run app_v2.py                # RAG UI          (port 8501)
```

> **Note:** If running both simultaneously use `--server.port 8502` for one of them. Start app_v2 first, wait for it to finish loading the embedding model (~30 s), then start dashboard_v2.

### Original v1 (still works, Apr–May only)
```bash
streamlit run dashboard.py    # Original 5-month prototype
streamlit run app.py          # Original RAG UI
```

---

## Project Structure

```
MSC/
│
├── dashboard_v2.py          ← MAIN PROTOTYPE (v2 — Jan–Jun 2026)
├── app_v2.py                ← RAG ASSISTANT UI (v2 — all 6 files indexed)
├── dashboard.py             ← Original prototype (Apr–May only — DO NOT MODIFY)
├── app.py                   ← Original RAG UI (DO NOT MODIFY)
├── run_dashboard.sh         ← Stable launcher for dashboard_v2
├── run_rag.sh               ← Stable launcher for app_v2
│
├── config.py                ← All paths, model names, sample questions, file maps
│
├── src/
│   ├── data_pipeline.py     ← Section 14: load/clean/feature-engineer (v2: Jan–Jun)
│   ├── analytics_layers.py  ← Descriptive → Diagnostic → Predictive → Prescriptive
│   ├── model_lab.py         ← Section 15: LOO models, SHAP, H1/H2 evidence
│   ├── rag_engine.py        ← RAG: retrieval + answer generation
│   ├── excel_loader.py      ← Audit Excel → RAG-ready chunks (v2: all 6 months)
│   ├── vector_store.py      ← ChromaDB + MiniLM embeddings (lazy-loaded)
│   └── pdf_loader.py        ← Research proposal PDF → chunks
│
├── Data/ Monthly Report 
│   ├── - Jan 2026_1.xlsx
│   ├── - Feb 2026.xlsx
│   ├── - March 2026.xlsx
│   ├── - April 2026.xlsx
│   ├── - May 2026.xlsx
│   ├── - Jun 2026_1.xlsx
│
├── processed/
│   ├── feature_matrix.csv   ← Engineered monthly features (Jan–May 2026)
│   ├── monthly_pl.csv       ← P&L extracted features
│   ├── daily_sales.csv      ← May 2026 daily sales (31 rows)
│   └── model_comparison.csv ← ML model performance results
├── scripts/
│   ├── ingest.py            ← Rebuild RAG index (run after adding new files)
│   └── query.py             ← CLI RAG query
│
├── chroma_db/               ← Persistent vector index (auto-managed)
└── requirements.txt
```

---

## Dashboard Pages

| Page | What it shows | Thesis section |
|------|--------------|----------------|
| **Overview** | KPI snapshot, multi-year sales trend (2024–2026), framework status | Introduction |
| **14 · Data Exploration & Preprocessing** | 6-month data sources, quality report, cleaning steps, EDA charts, feature matrix, descriptive stats + embedded thesis evidence figures | Section 14 |
| **Descriptive Analytics** | Baseline metrics, category revenue mix, wastage trends | Objective 1 |
| **Diagnostic Modelling** | Variance decomposition, COGS elasticity, IsolationForest anomalies, root-cause notes | Objective 2 |
| **Predictive Uncertainty** | P10–P90 probabilistic forecast, cash-flow stress probability, stockout proxy | Objective 3 |
| **Prescriptive Optimization** | MCDA/TOPSIS ranked recommendations with rationale | Objective 4 |
| **15 · Model Development** | Model selection, LOO-CV strategy, hyperparameters, SHAP importance, model comparison + embedded thesis evidence figures | Section 15 |
| **Bias, Ethics & XAI** | 5-bias checklist, SHAP attributions (H2), ethical design principles, RQ2 safeguards | Objective 5 / RQ2 |
| **Research Questions** | Live generated answers to RQ1 and RQ2 from the data | Conclusion |
| **RAG Assistant** | Q&A over proposal + all 6 audit workbooks (ChromaDB) | Objective 6 |

---

## Key Results

### H1 — Ensemble vs Linear (Financial Stress Classification)

| Model | LOO Accuracy | F1 (weighted) | ROC-AUC |
|-------|-------------|--------------|---------|
| **Random Forest** | **0.80** | **0.86** | **0.50** |
| Logistic Regression | 0.60 | 0.75 | 0.33 |
| Gradient Boosting | 0.60 | 0.75 | 0.50 |
| XGBoost | 0.60 | 0.75 | 0.00 |

**H1 supported:** RF outperforms Logistic Regression by 20pp accuracy.

### Stress Labels (Jan–Jun 2026)

| Period | Stress | Driver |
|--------|--------|--------|
| Jan 2026 | 0 — Healthy | Gross margin 52.3%, profit margin 18.7% |
| Feb 2026 | 1 — Stressed | COGS ratio 53.4%, profit margin 8.5% |
| Mar 2026 | 1 — Stressed | Profit margin 2.0% (lowest), COGS 49.1% |
| Apr 2026 | 1 — Stressed | Margin pressure: COGS grew > sales |
| May 2026 | 0 — Healthy | Recovery: profit margin 15.1% |
| Jun 2026 | 0 — Healthy | Positive margin trend continues |

### Top RF Feature Importances

| Feature | Importance | Interpretation |
|---------|-----------|----------------|
| profit_margin | 0.273 | Primary stress discriminator |
| credit_exposure | 0.202 | Receivables concentration risk |
| salary_to_sales | 0.138 | Labour cost pressure |
| cogs_ratio | 0.124 | Cost efficiency signal |

### Daily Sales Forecasting (May 2026, 80/20 holdout)

| Model | R² |
|-------|----|
| RF Regressor | +0.029 |
| XGBoost Regressor | +0.020 |
| Linear Regression | −0.751 |
| Ridge | −0.698 |

> Near-zero R² is expected and documented: 31-day window with high intra-day variance. The framework uses probabilistic P10–P90 intervals instead of point R².

---

## Data Pipeline — 6-Month Coverage

```
Jan 2026  Feb 2026  Mar 2026  Apr 2026  May 2026  Jun 2026
   ✓         ✓         ✓         ✓         ✓         ✓
  P&L      P&L       P&L       P&L       P&L       P&L
 Sales    Sales     Sales     Sales     Sales     Sales
  C/D      C/D       C/D       C/D       C/D       C/D
Wastage  Wastage   Wastage   Wastage   Wastage   Wastage
                                        Daily*
                                       Salary**  Salary**

* Daily grain (May only) loaded from processed/daily_sales.csv
** Salary Details sheet present in May and Jun workbooks only
C/D = Creditors + Debtors
```
---

## Ethical Framework (RQ2 / H2)

The prototype implements five design principles:

1. **Audit-first features** — only regulator-acceptable summarized data; no alternative behavioural metadata
2. **Uncertainty disclosure** — P10–P90 intervals on every forecast; never point scores alone
3. **Explainability by default** — SHAP attributions on every risk flag (H2)
4. **Human-in-the-loop** — model supports, not replaces, underwriting judgment
5. **Bias auditing** — anchoring, availability, overconfidence, narrative fallacy, and algorithmic exclusion checks baked into `analytics_layers.bias_checklist()`

---

## Known Limitations

| Limitation | Mitigation |
|-----------|-----------|
| Small classification sample (n=6 months) | LOO-CV is optimal for n<10; documented explicitly |
| No independent test set for classifiers | All 6 months used in LOO; limitation disclosed in Section 15.6 |
| Daily grain available for May only | Probabilistic forecast uses rolling mean; not point regression |
| XGBoost ROC-AUC = 0.00 on LOO | Predicts majority class only; class imbalance documented |
| June P&L column offset (col 12) | Verified by row-scan against Jun workbook; hardcoded in `_V2_PL_COL` |
| Segfault risk on Python 3.13 / Apple Silicon | Fixed via `run_dashboard.sh` / `run_rag.sh` env var exports |

---

## Quick Reference

```bash
# Run everything (v2)
./run_dashboard.sh          # Main prototype — http://localhost:8501
./run_rag.sh                # RAG assistant — http://localhost:8501

# CLI tools
python scripts/ingest.py --force
python scripts/query.py "What were net sales in June 2026?"

# Run inspection utilities
python inspect_data.py      # Sheet names for all 6 files
python inspect_data2.py     # P&L values, cash flow, wastage per month
python get_model_data.py    # Live model runs: importances, LOO folds
```

---

## Technology Stack

| Layer | Tools |
|-------|-------|
| Data | Pandas, NumPy, OpenPyXL |
| ML | Scikit-learn, XGBoost |
| Explainability | SHAP |
| Visualisation | Plotly, Matplotlib |
| RAG | ChromaDB, sentence-transformers (MiniLM-L6-v2), pypdf |
| UI | Streamlit |
| LLM backends | OpenAI GPT-4o-mini (if key set), Ollama (if running), extractive fallback |
| Environment | Python 3.13, macOS (Apple Silicon), venv |

---

## File Modification Policy

| File | Status | Notes |
|------|--------|-------|
| `dashboard.py` | **READ-ONLY** | Original working v1 — do not modify |
| `app.py` | **READ-ONLY** | Original RAG UI — do not modify |
| `dashboard_v2.py` | Active | Main development file |
| `app_v2.py` | Active | RAG UI development file |
| `src/data_pipeline.py` | Active | v2: 6-month loader with backward compat |
| `src/excel_loader.py` | Active | v2: all 6 months detected |
| `src/vector_store.py` | Active | Lazy SentenceTransformer import |
| `config.py` | Active | Append-only — never remove existing constants |

--- 
*Data period: January – June 2026*  
*Framework: Descriptive → Diagnostic → Predictive → Prescriptive → XAI → Prototype*
