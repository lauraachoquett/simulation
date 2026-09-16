"""Video de la ligne de descendance dans le simplex, generation apres generation.

    python -m simulation.tools.video_lod <exp_dir>
    python -m simulation.tools.video_lod <exp_dir> --fenetre 25 --fps 10

Lit lod/lab/evaluation.npz, ecrit par tools/lineage_lab : rien n'est reevalue.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from simulation.utils.plots import _bary, _cadre_simplex, frise_canaux
from simulation.utils.utils_sim import load_shuffle_log
from simulation.utils.utils_video import VideoWriter


def charge(exp_dir):
    """(p, born, generation, post_shuffle, age) des ancetres ayant mange."""
    f = os.path.join(exp_dir, "lod", "lab", "evaluation.npz")
    if not os.path.exists(f):
        raise SystemExit(f"{f} absent : lancer d'abord "
                         "python -m simulation.tools.lineage_lab <exp_dir>")
    with np.load(f) as d:
        regime, born = np.asarray(d["regime"], float), np.asarray(d["born"])
        gen = np.asarray(d["generation"])
        post = np.asarray(d["post_shuffle"], bool)
        age = np.asarray(d["age"], float)
    total = regime.sum(axis=1)
    ok = np.isfinite(regime).all(axis=1) & (total > 0)
    return regime[ok] / total[ok, None], born[ok], gen[ok], post[ok], age[ok]


def config(exp_dir):
    f = os.path.join(exp_dir, "config.json")
    return json.load(open(f)) if os.path.exists(f) else {}


def decor(exp_dir, born, age_max):
    """Figure persistante : triangle, barre de couleur, bande des canaux."""
    cfg = config(exp_dir)
    ids = [int(r["id"]) for r in cfg.get("resources", [])] or [0, 1, 2]

    fig, ax = plt.subplots(figsize=(8.6, 8.6))
    fig.subplots_adjust(bottom=0.20, top=0.92)
    cmap, norm = plt.get_cmap("viridis"), plt.Normalize(vmin=0, vmax=age_max)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                      shrink=.62, pad=.02)
    cb.set_label("lifespan (steps)", fontsize=10)

    fr = frise_canaux(fig, load_shuffle_log(exp_dir), ids,
                      step=int(born.max()), curseur=False,
                      rect=(0.13, 0.055, 0.72, 0.055))
    curseur = fr.axvline(born[0], color="#C1121F", lw=2.2, zorder=6)
    return fig, ax, curseur, cmap, norm


def rend(fig, ax, curseur, p, born, gen, post, age, i, fenetre, taille,
         cmap, norm):
    """Une frame : les `fenetre` dernieres generations, les plus anciennes pales.

    Couleur = duree de vie au lab, meme echelle que plot_food_simplex. Le point
    courant est une etoile, les losanges marquent un changement de canaux.
    """
    deb = max(0, i - fenetre + 1)
    vus = np.arange(deb, i + 1)
    x, y = _bary(p[vus, 0], p[vus, 1], p[vus, 2])
    recul = (i - vus) / max(fenetre - 1, 1)          # 0 = courant, 1 = le plus vieux
    alphas = np.clip(1.0 - 0.85 * recul, 0.08, 1.0)

    a = age[vus]
    fini = np.isfinite(a)
    couleurs = np.tile(np.array(plt.matplotlib.colors.to_rgba("0.7")),
                       (len(vus), 1))
    couleurs[fini] = cmap(norm(a[fini]))
    couleurs[:, 3] = alphas

    ax.clear()
    _cadre_simplex(ax)
    if len(vus) > 1:
        seg = np.stack([np.column_stack([x[:-1], y[:-1]]),
                        np.column_stack([x[1:], y[1:]])], axis=1)
        ax.add_collection(LineCollection(seg, colors="0.55", linewidths=1.2,
                                         alpha=float(alphas[:-1].mean()), zorder=3))
    ax.scatter(x, y, s=taille * (1 - .6 * recul), c=couleurs, zorder=4,
               edgecolors="none")
    losanges = post[vus]
    if losanges.any():
        ax.scatter(x[losanges], y[losanges], s=taille * 1.5, marker="D",
                   facecolors="none", edgecolors="0.25", linewidths=1.2, zorder=5)
    ax.scatter(x[-1], y[-1], s=taille * 2.6, marker="*", color=couleurs[-1][:3],
               edgecolors="black", linewidths=1.0, zorder=6)

    curseur.set_xdata([born[i], born[i]])
    vie = "" if not np.isfinite(age[i]) else f"   |   lab lifespan {int(age[i])}"
    ax.set_title(f"Line of descent — generation {int(gen[i])} / {int(gen[-1])}"
                 f"\nborn at step {int(born[i]):,}".replace(",", " ") + vie,
                 fontsize=12)
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3]


def main():
    p_arg = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p_arg.add_argument("exp_dir")
    p_arg.add_argument("-o", "--out", default=None,
                       help="fichier de sortie (defaut <exp_dir>/videos/lod_simplex.mp4)")
    p_arg.add_argument("--fenetre", type=int, default=20,
                       help="generations visibles avant disparition (defaut %(default)s)")
    p_arg.add_argument("--fps", type=int, default=10)
    p_arg.add_argument("--pause", type=int, default=6,
                       help="frames tenues sur un changement de canaux (defaut %(default)s)")
    p_arg.add_argument("--taille", type=float, default=90,
                       help="taille des points (defaut %(default)s)")
    p_arg.add_argument("--age-max", dest="age_max", type=float, default=None,
                       help="haut de l'echelle de couleur (defaut : lab_time_steps)")
    a = p_arg.parse_args()

    p, born, gen, post, age = charge(a.exp_dir)
    if len(p) < 2:
        raise SystemExit("moins de deux ancetres exploitables")
    age_max = a.age_max or config(a.exp_dir).get("lab_time_steps") \
        or float(np.nanmax(age))
    print(f"{len(p)} ancetre(s), generations {int(gen[0])} a {int(gen[-1])}, "
          f"{int(post.sum())} changement(s) de canaux")

    fig, ax, curseur, cmap, norm = decor(a.exp_dir, born, age_max)
    sortie = a.out or os.path.join(a.exp_dir, "videos", "lod_simplex.mp4")
    os.makedirs(os.path.dirname(sortie) or ".", exist_ok=True)

    n = 0
    with VideoWriter(sortie, fps=a.fps) as vid:
        for i in range(len(p)):
            img = rend(fig, ax, curseur, p, born, gen, post, age, i, a.fenetre,
                       a.taille, cmap, norm)
            vid.add(img); n += 1
            if post[i]:
                for _ in range(a.pause):
                    vid.add(img); n += 1
    plt.close(fig)
    print(f"Video : {sortie}  ({n} frames, {n / a.fps:.1f} s)")


if __name__ == "__main__":
    main()
