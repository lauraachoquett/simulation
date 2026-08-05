"""
Experiment comparison dashboard (MLflow/Aim-style) for BioTiC simulations.

Scans a directory tree for experiment folders (any folder containing a
`config.json`), then lets you:

  * browse all runs in a diff-highlighted parameter table (only params that
    actually differ across the selected runs are highlighted),
  * filter runs by any param value,
  * put the same plots side by side across the runs you pick,
  * inspect the raw config and resource-shuffle log of a single run.

Nothing here touches the simulation or requires re-running: it reads the
`config.json` + `*.png` files your runs already write to disk.

Usage:
    streamlit run dashboard/compare.py -- --root exp/
    # or set the root interactively in the sidebar.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------- #
# Config discovery / flattening
# --------------------------------------------------------------------------- #

# Params that are per-run bookkeeping rather than something you compare on.
_HIDDEN_PARAMS = {"seeds", "seeds_full", "env"}


def _flatten(cfg: dict) -> dict:
    """Flatten a config.json into a single {param: scalar} dict.

    `resources` is a list of dicts -> expand to resources[i].<field>.
    `env`/seeds are dropped from the comparison view.
    """
    flat: dict[str, object] = {}
    for k, v in cfg.items():
        if k in _HIDDEN_PARAMS:
            continue
        if k == "resources" and isinstance(v, list):
            for i, res in enumerate(v):
                if isinstance(res, dict):
                    for rk, rv in res.items():
                        flat[f"resources[{i}].{rk}"] = rv
                else:
                    flat[f"resources[{i}]"] = res
        elif isinstance(v, (list, dict)):
            flat[k] = json.dumps(v, sort_keys=True)
        else:
            flat[k] = v
    return flat


@st.cache_data(show_spinner=False)
def discover_runs(root: str) -> list[dict]:
    """Find every folder under `root` containing a config.json."""
    root_path = Path(root).expanduser()
    runs: list[dict] = []
    if not root_path.exists():
        return runs
    for cfg_path in sorted(root_path.rglob("config.json")):
        exp_dir = cfg_path.parent
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        rel = exp_dir.relative_to(root_path)
        runs.append(
            {
                "id": str(rel),
                "name": exp_dir.name,
                "group": str(rel.parent) if str(rel.parent) != "." else "",
                "dir": str(exp_dir),
                "mtime": cfg_path.stat().st_mtime,
                "params": _flatten(cfg),
                "raw": cfg,
            }
        )
    return runs


# --------------------------------------------------------------------------- #
# Plot discovery / grouping
# --------------------------------------------------------------------------- #

_CHUNK_RE = re.compile(r"(chunk|lab|agent)[_-]?\d+", re.IGNORECASE)
_NUM_RE = re.compile(r"\d+")


def _plot_key(png_path: Path, exp_dir: Path) -> str:
    """Normalize a plot filename into a stable 'type' key that aligns across
    runs, e.g. `plot_sim_chunk_500.png` -> `plot_sim_chunk_N.png`, keeping the
    subdirectory so `energy/...` stays distinct from top-level plots.
    """
    rel = png_path.relative_to(exp_dir)
    stem = rel.name
    stem = _CHUNK_RE.sub(lambda m: re.sub(_NUM_RE, "N", m.group(0)), stem)
    stem = _NUM_RE.sub("N", stem)
    return str(rel.parent / stem) if str(rel.parent) != "." else stem


@st.cache_data(show_spinner=False)
def discover_plots(exp_dir: str) -> dict[str, list[str]]:
    """Return {plot_key: [png paths...]} for one run, newest-numbered last."""
    exp = Path(exp_dir)
    groups: dict[str, list[str]] = {}
    for png in exp.rglob("*.png"):
        key = _plot_key(png, exp)
        groups.setdefault(key, []).append(str(png))
    for key in groups:
        groups[key].sort()
    return groups


def _list_videos(exp_dir: str) -> list[str]:
    return sorted(str(p) for p in Path(exp_dir).rglob("*.mp4"))


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #

def _parse_cli_root() -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--root", default="exp")
    # streamlit passes extra args; ignore unknowns
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args.root


def _highlight_diffs(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """Highlight cells in rows (params) that differ across the shown runs."""
    def _row_style(row: pd.Series):
        differs = row.nunique(dropna=False) > 1
        # semi-transparent amber: readable on both light and dark themes
        color = "background-color: rgba(255, 193, 7, 0.35)" if differs else ""
        return [color] * len(row)

    return df.style.apply(_row_style, axis=1)


def main() -> None:
    st.set_page_config(page_title="BioTiC — Experiment Compare", layout="wide")
    st.title("🧬 BioTiC — Experiment Comparison")

    default_root = _parse_cli_root()
    root = st.sidebar.text_input("Experiment root directory", value=default_root)
    if st.sidebar.button("🔄 Rescan"):
        discover_runs.clear()
        discover_plots.clear()

    runs = discover_runs(root)
    if not runs:
        st.warning(
            f"No experiments found under `{root}`. "
            "An experiment is any folder containing a `config.json`."
        )
        st.stop()

    st.sidebar.caption(f"{len(runs)} run(s) found")

    # ---- Filters ---------------------------------------------------------- #
    groups = sorted({r["group"] for r in runs if r["group"]})
    if groups:
        picked_groups = st.sidebar.multiselect(
            "Filter by subdirectory", groups, default=groups
        )
        runs = [r for r in runs if not r["group"] or r["group"] in picked_groups]

    # optional param filter
    all_params = sorted({p for r in runs for p in r["params"]})
    with st.sidebar.expander("Filter by parameter value"):
        fparam = st.selectbox("Parameter", ["(none)"] + all_params)
        if fparam != "(none)":
            values = sorted({str(r["params"].get(fparam)) for r in runs})
            fvals = st.multiselect("Keep values", values, default=values)
            runs = [r for r in runs if str(r["params"].get(fparam)) in fvals]

    id_to_run = {r["id"]: r for r in runs}
    labels = [r["id"] for r in runs]

    selected = st.multiselect(
        "Select runs to compare",
        labels,
        default=labels[: min(3, len(labels))],
        help="Pick the runs whose configs and plots you want side by side.",
    )
    sel_runs = [id_to_run[s] for s in selected]
    if not sel_runs:
        st.info("Select at least one run above.")
        st.stop()

    tab_cfg, tab_plots, tab_single = st.tabs(
        ["📋 Config comparison", "🖼 Plots side by side", "🔍 Single run"]
    )

    # ---- Config comparison ----------------------------------------------- #
    with tab_cfg:
        only_diffs = st.checkbox("Show only differing parameters", value=True)
        table = {r["id"]: r["params"] for r in sel_runs}
        df = pd.DataFrame(table)
        df.index.name = "param"

        if only_diffs and df.shape[1] > 1:
            mask = df.apply(lambda row: row.nunique(dropna=False) > 1, axis=1)
            df = df[mask]
            if df.empty:
                st.success("All selected runs share identical parameters.")
        st.dataframe(_highlight_diffs(df), use_container_width=True, height=600)

    # ---- Plots side by side ---------------------------------------------- #
    with tab_plots:
        per_run_plots = {r["id"]: discover_plots(r["dir"]) for r in sel_runs}
        plot_keys = sorted({k for g in per_run_plots.values() for k in g})
        if not plot_keys:
            st.info("No PNG plots found in the selected runs.")
        else:
            chosen = st.multiselect(
                "Plot types to show", plot_keys,
                default=[k for k in plot_keys if "chunk" not in k.lower()][:4]
                or plot_keys[:4],
            )
            for key in chosen:
                st.subheader(key)
                cols = st.columns(len(sel_runs))
                for col, r in zip(cols, sel_runs):
                    with col:
                        st.caption(r["id"])
                        paths = per_run_plots[r["id"]].get(key, [])
                        if not paths:
                            st.write("—")
                            continue
                        # if several (per-chunk) variants, let user scrub
                        if len(paths) > 1:
                            idx = st.select_slider(
                                "variant", options=list(range(len(paths))),
                                value=len(paths) - 1,
                                format_func=lambda i, p=paths: Path(p[i]).name,
                                key=f"{key}-{r['id']}",
                            )
                            path = paths[idx]
                        else:
                            path = paths[0]
                        st.image(path, use_container_width=True)

    # ---- Single run inspector -------------------------------------------- #
    with tab_single:
        one = st.selectbox("Run", [r["id"] for r in sel_runs])
        run = id_to_run[one]
        st.markdown(f"**Directory:** `{run['dir']}`")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Full config**")
            st.json(run["raw"], expanded=False)
        with c2:
            shuffle_log = Path(run["dir"]) / "resource_shuffles.jsonl"
            if shuffle_log.exists():
                st.markdown(f"**{shuffle_log.name}** (resource shuffles)")
                records = [
                    json.loads(l) for l in shuffle_log.read_text().splitlines() if l.strip()
                ]
                if records:
                    st.dataframe(pd.json_normalize(records), use_container_width=True)
            vids = _list_videos(run["dir"])
            if vids:
                st.markdown("**Videos**")
                vsel = st.selectbox(
                    "video", vids, format_func=lambda p: os.path.relpath(p, run["dir"])
                )
                st.video(vsel)


if __name__ == "__main__":
    main()
