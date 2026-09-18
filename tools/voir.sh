#!/bin/bash
# Dashboard de comparaison sur les experiences rapatriees (tools/rapatrie.py)
cd "$(dirname "$0")/.." && exec uv run --no-project --with streamlit --with pandas \
    --with plotly --with numpy streamlit run dashboard/compare.py -- \
    --root "${1:-$HOME/Documents/BioTiC/exp_figs}"
