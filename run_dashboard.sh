#!/bin/zsh
# run_dashboard.sh — stable launcher for dashboard_v2.py on macOS / Python 3.13
# Sets all thread-limiting env vars before Streamlit spawns any subprocess.

cd "$(dirname "$0")"

export LOKY_MAX_CPU_COUNT=1
export JOBLIB_MULTIPROCESSING=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

source venv/bin/activate
exec streamlit run dashboard_v2.py "$@"
