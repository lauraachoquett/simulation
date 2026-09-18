# Experiment comparison dashboard

MLflow/Aim-style local UI to compare BioTiC simulation runs. It reads the
`config.json` + `*.png` files your runs already write — nothing to instrument,
no re-running.

## Install (once, on the machine with the experiments)

```bash
pip install -r dashboard/requirements.txt
```

## Local, from the clusters

```bash
python3 tools/rapatrie.py 'exp/2026-09-17/*'            # config + figures, no data
python3 tools/rapatrie.py exp/2026-09-17/<run> --videos # + videos, re-encoded smaller
tools/voir.sh                                           # dashboard on ~/Documents/BioTiC/exp_figs
```

Each pattern is looked up on every G5K site (`--sites lyon lille` to restrict).

## Run

```bash
streamlit run dashboard/compare.py -- --root exp/
```

Then open the printed URL (default http://localhost:8501). If you run it on a
remote VM, forward the port:

```bash
ssh -L 8501:localhost:8501 user@vm
```

## What it does

- **Discovers runs**: every folder under `--root` containing a `config.json`.
- **Filters** (sidebar): pick several parameters, then the values to keep for
  each. Runs must match every parameter (AND); within a parameter, any ticked
  value matches (OR). By default only parameters that differ across runs are
  offered.
- **Config comparison tab**: all selected runs as columns; toggle "show only
  differing parameters" to cut through the noise; differing cells highlighted.
- **Plots side by side**: same plot type across runs in one row; per-chunk plots
  get a slider to scrub variants.
- **Single run**: full config JSON, resource-shuffle log, and videos.
