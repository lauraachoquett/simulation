"""Les trois geometries de test comparees, agent SEUL.

    python -m simulation.tools.plot_geometries <fusion|exp_dir>
    python -m simulation.tools.plot_geometries <dir> --pas-chunks 100 --chunks 2000 4000
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simulation.utils.plots import _band, _chunks_dans

MESURES = [("duree_vie", "Lifespan", "steps"),
           ("greediness", "Greediness  G = Cr/Tr", "ratio"),
           ("mouvement", "Movement", "/step")]
NOMS = {"scatter40_s0": "scattered", "patch8x5_s0": "patchy",
        "blob1x40_s1": "single blob"}


def data_dir_de(chemin):
    for c in (os.path.join(chemin, "replay", "lab_data"),
              os.path.join(chemin, "lab_data"), chemin):
        if glob.glob(os.path.join(c, "chunk_*_summary.json")):
            return c
    return None


def geometries(data_dir):
    """{stem: [fichiers]} des resumes par geometrie, agent seul."""
    out = {}
    for f in glob.glob(os.path.join(data_dir, "chunk_*_env_*_summary.json")):
        m = re.fullmatch(r"chunk_\d+_env_(.+)_summary\.json", os.path.basename(f))
        if m:
            out.setdefault(m.group(1), []).append(f)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("source")
    p.add_argument("--geos", nargs="+", default=list(NOMS),
                   help="ordre des geometries, du disperse au groupe")
    p.add_argument("--chunks", type=int, nargs=2, default=None,
                   metavar=("DEBUT", "FIN"))
    p.add_argument("--pas-chunks", dest="pas", type=int, default=0, metavar="N")
    p.add_argument("--cmap", default="viridis")
    p.add_argument("--chunk-size", dest="chunk_size", type=int, default=1000)
    p.add_argument("-o", "--out", default=None,
                   help="defaut <source>/fig/lab_geometries_alone.png")
    a = p.parse_args()

    data_dir = data_dir_de(a.source)
    if data_dir is None:
        raise SystemExit(f"pas de lab_data sous {a.source}")
    par_geo = geometries(data_dir)
    geos = [g for g in a.geos if g in par_geo] + \
           [g for g in sorted(par_geo) if g not in a.geos]
    if not geos:
        raise SystemExit(f"aucun resume par geometrie dans {data_dir}")

    cmap = plt.get_cmap(a.cmap)
    couleurs = [cmap(t) for t in np.linspace(.12, .82, len(geos))]

    fig, axes = plt.subplots(1, len(MESURES), figsize=(5.4 * len(MESURES), 4.6))
    for geo, couleur in zip(geos, couleurs):
        files = _chunks_dans(sorted(par_geo[geo]), a.chunks, a.pas)
        S = [json.load(open(f)) for f in sorted(files, key=lambda f: int(
            re.search(r"chunk_(\d+)", os.path.basename(f)).group(1)))]
        if not S:
            continue
        x = np.array([s["chunk"] for s in S]) * a.chunk_size
        for ax, (prefix, titre, unite) in zip(axes, MESURES):
            m, lo, hi = _band(S, prefix)
            ok = ~np.isnan(m)
            ax.plot(x[ok], m[ok], marker="o", ms=3.5, color=couleur,
                    label=NOMS.get(geo, geo))
            okb = ~(np.isnan(lo) | np.isnan(hi))
            ax.fill_between(x[okb], lo[okb], hi[okb], color=couleur, alpha=.15)
            ax.set_title(titre)
            ax.set_ylabel(unite)
            ax.set_xlabel("simulation step")
            ax.grid(alpha=.3)
    axes[1].set_ylim(0, 1)
    axes[0].legend(frameon=False, title="test environment")

    fig.suptitle("Focal agent alone, across test geometries "
                 "(median and p25–p75 over genomes)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, .93])
    out = a.out or os.path.join(a.source, "fig", "lab_geometries_alone.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Figure saved: {out}")


if __name__ == "__main__":
    main()
