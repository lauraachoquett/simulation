# Experiment comparison dashboard

MLflow/Aim-style local UI to compare BioTiC simulation runs. It reads the
`config.json` + `*.png` files your runs already write — nothing to instrument,
no re-running.

## Install (once, on the machine with the experiments)

```bash
pip install -r dashboard/requirements.txt
```

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
- **Config comparison tab**: all selected runs as columns; toggle "show only
  differing parameters" to cut through the noise; differing cells highlighted.
- **Plots side by side**: same plot type across runs in one row; per-chunk plots
  get a slider to scrub variants.
- **Single run**: full config JSON, resource-shuffle log, and videos.
