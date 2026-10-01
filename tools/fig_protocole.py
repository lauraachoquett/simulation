"""Schema du protocole : prelever un agent du monde, l'evaluer en test.

    python -m simulation.tools.fig_protocole --ressources 1
    python -m simulation.tools.fig_protocole --ressources 3 --no-titre \\
        -o fig/protocole_3res.pdf

Le monde de gauche est SYNTHETIQUE : il illustre le tirage, il ne rejoue aucun
etat de simulation. L'env de test de droite est la vraie grille scattered.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

from simulation.data_class import color_of, label_of
from simulation.tools.fig_lab_envs import FOND, GRILLE, MUR, charge, panneau_multi

AGENT, CHOISI = "#C1121F", "#6E0B12"


def monde(ax, res, ids, agents, choisi, titre, sous_titre):
    """Grille du monde : meme style que les env de lab, sans bande de depart."""
    L = res.shape[1]
    ax.set_facecolor(FOND)
    ax.set_xlim(-.5, L - .5), ax.set_ylim(L - .5, -.5)
    ax.set_aspect("equal")
    for k in range(L + 1):
        ax.axhline(k - .5, color=GRILLE, lw=.25, zorder=1)
        ax.axvline(k - .5, color=GRILLE, lw=.25, zorder=1)
    for x, y, w, h in ((-.5, -.5, L, 1), (-.5, L - 1.5, L, 1),
                       (-.5, .5, 1, L - 2), (L - 1.5, .5, 1, L - 2)):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=MUR, edgecolor="none",
                               zorder=3))
    ax.add_patch(Rectangle((-.5, -.5), L, L, facecolor="none", edgecolor=MUR,
                           lw=1.6, zorder=5))
    for c, i in enumerate(ids):
        y, x = np.nonzero(res[c])
        ax.scatter(x, y, s=24, marker="s", color=color_of(i), edgecolor="white",
                   linewidth=.35, zorder=4)
    for k, (y, x) in enumerate(agents):
        vise = k == choisi
        ax.scatter(x, y, s=120 if vise else 78,
                   color=CHOISI if vise else AGENT, edgecolor="white",
                   linewidth=1.3, zorder=7 if vise else 6)
        if vise:        # halo : c'est celui qu'on preleve
            ax.scatter(x, y, s=700, facecolor="none", edgecolor=CHOISI,
                       linewidth=1.5, linestyle=(0, (3, 2)), zorder=7)
    ax.set_xticks([]), ax.set_yticks([])
    for c in ax.spines.values():
        c.set_visible(False)
    ax.set_title(titre, fontsize=13, pad=30, fontweight="semibold")
    ax.text(.5, 1.012, sous_titre, transform=ax.transAxes, ha="center",
            va="bottom", fontsize=10, color="#5A5A5A")


def monde_synthetique(cote, n_res, n_agents, n_types, rng):
    """Ressources en amas irreguliers et agents disperses, loin des murs."""
    res = np.zeros((n_types, cote, cote))
    amas = rng.integers(4, cote - 4, size=(10, 2))
    for _ in range(n_res):
        c = amas[rng.integers(len(amas))] + rng.normal(0, 3.4, 2)
        y, x = np.clip(np.round(c), 1, cote - 2).astype(int)
        res[rng.integers(n_types), y, x] = 1
    agents = rng.integers(3, cote - 3, size=(n_agents, 2))
    return res, agents


def env_de_test(nom, dossier, n_types, rng):
    """La vraie grille de lab ; avec trois identites, une couche par ressource."""
    g = charge(nom, dossier)
    if n_types == 1:
        return g[None, :, :]
    L = g.shape[0]
    res = np.zeros((n_types, L, L))
    libre = [(y, x) for y in range(1, L - 1) for x in range(1, L - 1)]
    tire = rng.permutation(len(libre))[:n_types * int(g.sum())]
    for k, idx in enumerate(tire):
        y, x = libre[idx]
        res[k % n_types, y, x] = 1
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ressources", type=int, default=1, choices=[1, 2, 3])
    p.add_argument("--env", default="scatter40_s0", help="geometrie de test")
    p.add_argument("--dir", default=os.path.join(os.path.dirname(__file__),
                                                 "..", "lab_envs"))
    p.add_argument("--monde", type=int, default=60, help="cote du monde")
    p.add_argument("--n-res", dest="n_res", type=int, default=160)
    p.add_argument("--n-agents", dest="n_agents", type=int, default=14)
    p.add_argument("--graine", type=int, default=3)
    p.add_argument("--vue", type=int, default=5, help="demi-fenetre de l'agent")
    p.add_argument("--no-titre", dest="no_titre", action="store_true")
    p.add_argument("-o", "--out", default=None)
    a = p.parse_args()

    rng = np.random.default_rng(a.graine)
    ids = list(range(a.ressources))
    res_m, agents = monde_synthetique(a.monde, a.n_res, a.n_agents,
                                      a.ressources, rng)
    # l'agent preleve : au milieu du monde, pour que la fleche parte de la
    centre = np.array([a.monde / 2, a.monde * .72])
    choisi = int(np.argmin(np.abs(agents - centre).sum(axis=1)))
    res_t = env_de_test(a.env, a.dir, a.ressources, rng)

    fig, (g, d) = plt.subplots(1, 2, figsize=(14.5, 7.2),
                               gridspec_kw={"width_ratios": [1.3, 1]})
    fig.patch.set_facecolor("white")
    # pas de sous-titre : ce que montre chaque grille se dit dans la caption
    monde(g, res_m, ids, agents, choisi, "Natural environment", "")
    panneau_multi(d, res_t, ids, "Test environment", "", vue=a.vue)
    for ax in (g, d):
        ax.set_title(ax.get_title(), fontsize=13, fontweight="semibold", pad=12)
    # l'agent teste, au centre de la fenetre d'observation : meme rond rouge
    # que dans le schema des conditions sociales
    L = res_t.shape[1]
    yy, xx = np.nonzero(res_t.sum(axis=0))
    cy, cx = (int(np.clip(round(v), a.vue + 1, L - a.vue - 2))
              for v in (yy.mean(), xx.mean()))
    d.scatter(cx, cy, s=150, color=CHOISI, edgecolor="white", linewidth=1.4,
              zorder=8)
    if d.child_axes:          # meme agent dans l'encart de vision
        d.child_axes[-1].scatter(cx, cy, s=150, color=CHOISI,
                                 edgecolor="white", linewidth=1.4, zorder=9)
    for ax in (g, d):          # memes hauteurs : les titres s'alignent
        ax.set_anchor("N")
    if a.ressources > 1:
        d.legend(loc="upper center", bbox_to_anchor=(.5, -.03), ncol=3,
                 frameon=False, fontsize=9)

    fig.subplots_adjust(wspace=.34)
    # positions reelles : avec set_anchor les axes sont plus petits que leur case
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    bg = inv.transform(g.get_window_extent()).ravel()
    bd = inv.transform(d.get_window_extent()).ravel()
    y = (bd[1] + bd[3]) / 2
    x0, x1 = bg[2] + .015, bd[0] - .02
    fig.patches.append(FancyArrowPatch(
        (x0, y), (x1, y), transform=fig.transFigure, mutation_scale=24,
        arrowstyle="-|>", color=CHOISI, lw=2.2, zorder=10,
        shrinkA=0, shrinkB=0))
    fig.text((x0 + x1) / 2, y + .03, "one agent\nis drawn", ha="center",
             va="bottom", fontsize=10, color=CHOISI)

    if not a.no_titre:
        fig.suptitle("Agents are drawn from the simulation and evaluated "
                     "under identical conditions", fontsize=14, y=.99)
    out = a.out or f"fig/protocole_{a.ressources}res.png"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    print(f"Figure saved: {out}")


if __name__ == "__main__":
    main()
