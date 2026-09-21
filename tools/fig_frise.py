"""Bande des canaux et de leurs permutations, sur toute la duree du run.

    python -m simulation.tools.fig_frise <exp_dir>
    python -m simulation.tools.fig_frise <exp_dir> --pas-fin 4739000 -o fig/frise.png
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulation.data_class import color_of, label_of
from simulation.utils.plots import frise_canaux
from simulation.utils.utils_sim import load_shuffle_log

_NUM = re.compile(r"chunk_(\d+)\.npz$")


def config(exp_dir):
    f = os.path.join(exp_dir, "config.json")
    return json.load(open(f)) if os.path.exists(f) else {}


def dernier_pas(dossiers, cfg):
    """Fin de l'axe : le dernier chunk ECRIT, comme dans plot_evo."""
    taille = int(cfg.get("chunk_size", 1000))
    chunks = [int(m.group(1)) for d in dossiers
              for f in glob.glob(os.path.join(d, "data", "chunk_*.npz"))
              for m in [_NUM.search(os.path.basename(f))] if m]
    if chunks:
        return (max(chunks) + 1) * taille
    return int(cfg.get("num_chunks", 0)) * taille


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("exp_dirs", nargs="+", metavar="EXP_DIR")
    p.add_argument("--no-resume", dest="no_resume", action="store_true",
                   help="ne pas remonter la chaine des reprises")
    p.add_argument("--pas-fin", dest="pas_fin", type=int, default=None,
                   help="forcer la fin de l'axe, en pas")
    p.add_argument("--taille", type=float, nargs=2, default=(11., 2.2),
                   metavar=("L", "H"))
    p.add_argument("-o", "--out", default=None,
                   help="defaut <exp_dir>/fig/frise_canaux.png")
    a = p.parse_args()

    from simulation.tools.replot import chaine_de_reprise
    dossiers = (chaine_de_reprise(a.exp_dirs[0])
                if len(a.exp_dirs) == 1 and not a.no_resume
                else [os.path.abspath(d) for d in a.exp_dirs])
    if len(dossiers) > 1:
        print("chaine suivie : " + " -> ".join(map(os.path.basename, dossiers)))

    cfg = config(dossiers[0])
    ids = [int(r["id"]) for r in cfg.get("resources", [])] or [0, 1, 2]
    journal = []
    for d in dossiers:
        for e in load_shuffle_log(d):
            if not any(abs(e["step"] - v["step"]) < 1 for v in journal):
                journal.append(e)
    journal.sort(key=lambda e: e["step"])
    fin = a.pas_fin or dernier_pas(dossiers, cfg)
    print(f"{len(journal)} permutation(s), axe jusqu'au pas {fin}")

    fig = plt.figure(figsize=tuple(a.taille))
    fr = frise_canaux(fig, journal, ids, step=fin, curseur=False,
                      rect=(0.07, 0.30, 0.91, 0.52))
    fr.set_title("Channel identity over time", fontsize=12, pad=8)
    # les identites du journal aussi : une permutation peut en faire apparaitre
    vus = sorted(set(ids) | {int(i) for e in journal for i in e["order_ids"]})
    fig.legend(handles=[plt.Line2D([0], [0], color=color_of(i), lw=8,
                                   label=label_of(i)) for i in vus],
               loc="lower center", ncol=len(vus), frameon=False, fontsize=9,
               bbox_to_anchor=(.5, -.02))

    out = a.out or os.path.join(dossiers[-1], "fig", "frise_canaux.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Figure saved: {out}")


if __name__ == "__main__":
    main()
