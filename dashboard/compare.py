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
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------- #
# Config discovery / flattening
# --------------------------------------------------------------------------- #

# Params that are per-run bookkeeping rather than something you compare on.
# seeds/seeds_full hold one key per chunk -- thousands of values, useless raw.
# They come back as the compact derived fields built by _seed_params().
_HIDDEN_PARAMS = {"seeds", "seeds_full", "env"}


def _seed_params(cfg: dict) -> dict:
    """Compact, comparable summary of a run's random stream.

    Two runs can share every parameter and still differ entirely because they
    were drawn from different seeds. Hiding the seed lists made those runs look
    identical in the table, so we surface a fingerprint of them instead.

    `seed_fingerprint` digests the full keys: same value = same random stream.
    `seed_first` is the first key, readable enough to tell runs apart at a
    glance. `prng_impl` matters too -- identical seeds under a different PRNG
    implementation do not give the same simulation.
    """
    out: dict[str, object] = {}
    raw = cfg.get("seeds_full") or cfg.get("seeds")
    if raw:
        blob = json.dumps(raw, sort_keys=True).encode()
        out["seed_fingerprint"] = hashlib.sha1(blob).hexdigest()[:8]
        first = raw[0]
        out["seed_first"] = first[0] if isinstance(first, list) else first
    env = cfg.get("env") or {}
    for k in ("jax_version", "prng_impl", "threefry_partitionable"):
        if k in env:
            out[k] = env[k]
    return out


def _flatten(cfg: dict) -> dict:
    """Flatten a config.json into a single {param: scalar} dict.

    `resources` is a list of dicts -> expand to resources[i].<field>.
    Seeds and env are replaced by the compact fields of _seed_params().
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
    flat.update(_seed_params(cfg))
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
    # newest first, so freshly launched experiments are on top / auto-selected
    runs.sort(key=lambda r: r["mtime"], reverse=True)
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


def _natural_key(path: str) -> list:
    """Sort key treating digit runs as numbers.

    Plain string sort puts `chunk_100` before `chunk_9`, so the last element was
    not the newest chunk -- which is what "latest variant" relies on.
    """
    return [int(tok) if tok.isdigit() else tok.lower()
            for tok in re.split(r"(\d+)", Path(path).name)]


@st.cache_data(show_spinner=False)
def discover_plots(exp_dir: str) -> dict[str, list[str]]:
    """Return {plot_key: [png paths...]} for one run, newest-numbered last."""
    exp = Path(exp_dir)
    groups: dict[str, list[str]] = {}
    for png in exp.rglob("*.png"):
        key = _plot_key(png, exp)
        groups.setdefault(key, []).append(str(png))
    for key in groups:
        groups[key].sort(key=_natural_key)
    return groups


# The plot to show when the page opens; falls back to any non-per-chunk plots.
_DEFAULT_PLOT = "plot_evo.png"


def _default_plot_keys(plot_keys: list[str]) -> list[str]:
    evo = [k for k in plot_keys if Path(k).name == _DEFAULT_PLOT]
    if evo:
        return evo
    return [k for k in plot_keys if "chunk" not in k.lower()][:4] or plot_keys[:4]


def _list_videos(exp_dir: str) -> list[str]:
    return sorted(str(p) for p in Path(exp_dir).rglob("*.mp4"))


# --------------------------------------------------------------------------- #
# Step-range series, rebuilt from data/chunk_*.npz
# --------------------------------------------------------------------------- #
# The saved PNG/HTML figures cover the whole run at a fixed resolution. The raw
# per-step history is on disk though, so we replot it here over any step window.
# Kept dependency-free (numpy + plotly) so the dashboard still runs outside the
# simulation environment -- hence the local copies of LABELS / build_id_timeline.

_LABELS = ("good", "medium", "poison")
_COLOR_BY_ID = {0: "#2A9131", 1: "#3933F3", 2: "#9C27B0"}
_DASHES = ("solid", "dash", "dot", "dashdot", "longdash")

_CHUNK_NPZ = re.compile(r"chunk_(\d+)\.npz$")

# metric -> (npz keys, how to aggregate a block, per-identity?)
_METRICS = {
    "population":   (("population",), "mean", False),
    "resources":    (("resources",), "mean", True),
    "consumed":     (("consumed",), "mean", True),
    "P(eat|seen)":  (("n_seen", "n_eaten_seen"), "sum", True),
}


@st.cache_data(show_spinner="Loading step history…", max_entries=16)
def load_series(exp_dir: str, keys: tuple[str, ...]) -> dict | None:
    """Concatenate `keys` across data/chunk_*.npz, in numeric chunk order.

    Returns None when a run has no data/ dir (older runs, or lab-only exports).
    """
    files = sorted(Path(exp_dir).glob("data/chunk_*.npz"),
                   key=lambda p: int(_CHUNK_NPZ.search(p.name).group(1)))
    if not files:
        return None
    parts: dict[str, list] = {k: [] for k in keys}
    for f in files:
        with np.load(f) as z:
            for k in keys:
                if k not in z.files:
                    return None
                parts[k].append(z[k])
    out = {k: np.concatenate(v, axis=0) for k, v in parts.items()}
    n = len(next(iter(out.values())))
    chunk_size = n // len(files)
    first = int(_CHUNK_NPZ.search(files[0].name).group(1))
    out["_start"] = first * chunk_size
    out["_n"] = n
    return out


def _id_timeline(steps: np.ndarray, shuffle_log: list, initial_ids: list) -> np.ndarray:
    """(T, n_types): which resource identity sits on each channel, per step."""
    if not shuffle_log:
        return np.tile(np.asarray(initial_ids), (len(steps), 1))
    cuts = np.array([e["step"] for e in shuffle_log])
    orders = [initial_ids] + [e["order_ids"] for e in shuffle_log]
    active = np.searchsorted(cuts, steps, side="right")
    return np.array([orders[a] for a in active])


def _by_identity(series: np.ndarray, timeline: np.ndarray, ident: int) -> np.ndarray:
    """Follow one resource across channel permutations.

    Each row of `timeline` is a permutation, so `timeline == ident` selects
    exactly one column per row and the result stays (T,) in step order. Plotting
    a channel instead would splice two different resources at every shuffle.
    """
    return series[timeline == ident]


def _block_reduce(y: np.ndarray, n_target: int, how: str) -> np.ndarray:
    """Aggregate into ~n_target blocks. Blocks, not striding: a stride would
    step over the spikes, which is most of what these series are about."""
    n = len(y)
    if n <= n_target:
        return y
    edges = np.linspace(0, n, n_target + 1).astype(int)
    fn = np.add.reduceat(y, edges[:-1])
    if how == "mean":
        fn = fn / np.diff(edges)
    return fn


def _block_x(steps: np.ndarray, n_target: int) -> np.ndarray:
    n = len(steps)
    if n <= n_target:
        return steps
    edges = np.linspace(0, n, n_target + 1).astype(int)
    return steps[edges[:-1]]


def _series_figure(metric: str, sel_runs: list[dict], lo: int, hi: int,
                   idents: list[int], n_target: int) -> go.Figure | None:
    keys, how, per_id = _METRICS[metric]
    fig = go.Figure()
    drew = False

    for r_i, r in enumerate(sel_runs):
        data = load_series(r["dir"], keys)
        if data is None:
            continue
        start, n = data["_start"], data["_n"]
        steps = np.arange(start, start + n)
        m = (steps >= lo) & (steps <= hi)
        if not m.any():
            continue
        drew = True
        x = _block_x(steps[m], n_target)
        dash = _DASHES[r_i % len(_DASHES)]

        if not per_id:
            y = _block_reduce(data[keys[0]][m], n_target, how)
            fig.add_scatter(x=x, y=y, name=r["id"], mode="lines",
                            line=dict(width=2, dash=dash))
            continue

        initial_ids = [res["id"] for res in r["raw"].get("resources", [])]
        tl = _id_timeline(steps[m], _load_shuffles(r["dir"]), initial_ids)
        for ident in idents:
            if ident not in initial_ids:
                continue
            cols = [_by_identity(data[k][m], tl, ident) for k in keys]
            if len(cols) == 1:
                y = _block_reduce(cols[0], n_target, how)
            else:
                # pooled ratio: sum both counts over the block, THEN divide.
                # Averaging per-step ratios would weight a step where 3 agents
                # see a resource like one where 300 do.
                seen = _block_reduce(cols[0], n_target, "sum")
                eaten = _block_reduce(cols[1], n_target, "sum")
                y = np.divide(eaten, seen, out=np.full_like(seen, np.nan,
                                                            dtype=float),
                              where=seen > 0)
            fig.add_scatter(
                x=x, y=y, mode="lines",
                name=f"{_LABELS[ident]} — {r['id']}",
                line=dict(width=2, dash=dash,
                          color=_COLOR_BY_ID.get(ident)),
            )

    if not drew:
        return None

    # shuffle markers only for a single run: several runs shuffle at different
    # steps and the lines stop meaning anything
    if len(sel_runs) == 1:
        for e in _load_shuffles(sel_runs[0]["dir"]):
            if lo <= e["step"] <= hi:
                fig.add_vline(x=e["step"], line=dict(color="grey", width=1,
                                                     dash="dot"))

    fig.update_layout(
        title=metric, xaxis_title="step", height=420,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", y=-0.2),
        hovermode="x unified",
    )
    return fig


def _load_shuffles(exp_dir: str) -> list[dict]:
    path = Path(exp_dir) / "resource_shuffles.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


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


def _plot_grid(key: str, sel_runs: list[dict], per_run_plots: dict,
               n_cols: int, variant_mode: str) -> None:
    """Lay one plot type out across runs, wrapping every `n_cols` figures."""
    for start in range(0, len(sel_runs), n_cols):
        # always allocate n_cols so a short last row keeps the same figure width
        cols = st.columns(n_cols)
        for col, r in zip(cols, sel_runs[start:start + n_cols]):
            with col:
                st.caption(r["id"])
                paths = per_run_plots[r["id"]].get(key, [])
                if not paths:
                    st.write("—")
                    continue
                if len(paths) > 1 and variant_mode == "Pick per run":
                    idx = st.select_slider(
                        "variant", options=list(range(len(paths))),
                        value=len(paths) - 1,
                        format_func=lambda i, p=paths: Path(p[i]).name,
                        key=f"{key}-{r['id']}",
                    )
                    path = paths[idx]
                else:
                    path = paths[-1]          # natural sort -> newest chunk
                st.image(path, use_container_width=True)
                if len(paths) > 1 and variant_mode == "Latest":
                    # say which chunk is shown, since there is no slider to read
                    st.caption(f"`{Path(path).name}`  ({len(paths)} variants)")


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
            "Subdirectories to compare", groups, default=[groups[-1]],
            help="Defaults to the most recent subdirectory; check others to add them.",
        )
        if picked_groups:
            runs = [r for r in runs if r["group"] in picked_groups]

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

    tab_cfg, tab_plots, tab_range, tab_single = st.tabs(
        ["📋 Config comparison", "🖼 Plots side by side",
         "📈 Step range", "🔍 Single run"]
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
                "Plot types to show", plot_keys, default=_default_plot_keys(plot_keys),
            )

            c1, c2 = st.columns([1, 2])
            with c1:
                # One column per run makes each figure unreadable past ~4 runs,
                # so wrap into a grid instead of stretching a single row.
                n_cols = st.slider(
                    "Figures per row", 1, 6, value=min(3, len(sel_runs)),
                    help="Fewer columns = larger, more readable figures.",
                )
            with c2:
                variant_mode = st.radio(
                    "Per-chunk variant", ["Latest", "Pick per run"],
                    horizontal=True,
                    help="Plots saved once per chunk. 'Latest' avoids one slider "
                         "per run when comparing many runs.",
                )

            for key in chosen:
                st.subheader(key)
                _plot_grid(key, sel_runs, per_run_plots, n_cols, variant_mode)

    # ---- Step range ------------------------------------------------------- #
    with tab_range:
        spans = {}
        for r in sel_runs:
            d = load_series(r["dir"], ("population",))
            if d is not None:
                spans[r["id"]] = (d["_start"], d["_start"] + d["_n"] - 1)

        if not spans:
            st.info(
                "No `data/chunk_*.npz` found in the selected runs. This tab "
                "replots the raw per-step history, which only runs writing "
                "that directory have."
            )
        else:
            gmin = min(s for s, _ in spans.values())
            gmax = max(e for _, e in spans.values())
            missing = [r["id"] for r in sel_runs if r["id"] not in spans]
            if missing:
                st.caption(f"No history for: {', '.join(missing)}")

            lo, hi = st.slider("Step range", gmin, gmax, (gmin, gmax), step=1000)

            c1, c2, c3 = st.columns([2, 2, 1])
            with c1:
                metrics = st.multiselect("Metrics", list(_METRICS),
                                         default=["population", "resources"])
            with c2:
                idents = st.multiselect(
                    "Resources", list(range(len(_LABELS))),
                    default=list(range(len(_LABELS))),
                    format_func=lambda i: _LABELS[i],
                    help="Followed by identity across shuffles, not by channel.",
                )
            with c3:
                n_target = st.select_slider("Points", [500, 1000, 2000, 5000],
                                            value=2000)

            st.caption(
                f"{hi - lo + 1:,} steps selected — "
                f"{'full resolution' if hi - lo + 1 <= n_target else f'blocks of ~{(hi - lo + 1) // n_target:,} steps'}"
            )
            for metric in metrics:
                fig = _series_figure(metric, sel_runs, lo, hi, idents, n_target)
                if fig is None:
                    st.write(f"— no data for {metric}")
                else:
                    st.plotly_chart(fig, use_container_width=True)

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
