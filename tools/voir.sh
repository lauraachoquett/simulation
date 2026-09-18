#!/bin/bash
# Dashboard de comparaison sur les experiences rapatriees (tools/rapatrie.py)
cd "$(dirname "$0")/.." || exit 1
(sleep 5; open http://localhost:8501 2>/dev/null) &
exec uv run --no-project --with streamlit --with pandas --with plotly --with numpy \
    streamlit run dashboard/compare.py --server.headless true \
    --browser.gatherUsageStats false -- \
    --root "${1:-$HOME/Documents/BioTiC/exp_figs}"
