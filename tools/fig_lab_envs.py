"""Figure des environnements de lab figes (lab_envs/*.npy).

    python -m simulation.tools.fig_lab_envs
    python -m simulation.tools.fig_lab_envs scatter40_s0 patch8x5_s0 -o fig/envs.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from simulation.data_class import LAB_ENVS, color_of, label_of

TITRES = {"scatter40_s0": "Scattered", "patch8x5_s0": "Patchy",
          "blob1x40_s1": "Single blob"}
SOUS_TITRES = {"scatter40_s0": "40 isolated items",
               "patch8x5_s0": "8 patches of 5",
               "blob1x40_s1": "one blob of 40"}
MUR, FOND, GRILLE, VUE = "#2B2B2B", "#FAF7F2", "#E2DCD3", "#1D5C8F"


def grille_tiree(quel, exp_dir, graine):
    """Grille de depart d'un env de lab, par le vrai chemin de code."""
    from simulation.data_class import resolve_model
    from simulation.lab_env import (vmap_over_agents_env_lab_high_res,
                                    vmap_over_agents_env_lab_low_res)
    from simulation.tools.make_lab_envs import ModeleFactice
    from simulation.tools.preview_lab_env import config_par_defaut, grille_de_depart
    from simulation.utils.utils_sim import load_config

    cfg = load_config(exp_dir)[0] if exp_dir else config_par_defaut()
    cfg = resolve_model(cfg)
    if graine is None:
        graine = cfg.lab_seed
    fn = (vmap_over_agents_env_lab_high_res if quel == "high_res"
          else vmap_over_agents_env_lab_low_res)
    res, _ = grille_de_depart(fn, cfg, ModeleFactice(), graine)
    return res, [r.id for r in cfg.resources], graine


def charge(nom, dossier):
    g = np.load(os.path.join(dossier, nom if nom.endswith(".npy") else nom + ".npy"))
    return g.sum(axis=0) if g.ndim == 3 else g


def encart_vision(ax, grille, couleur, vue, centre, res=None, ids=None):
    """Fenetre d'observation (2*vue+1) autour d'un agent, montree en zoom."""
    c = 2 * vue + 1
    y0, x0 = centre[0] - vue, centre[1] - vue
    iax = ax.inset_axes([1.07, .28, .46, .46])
    iax.set_facecolor(FOND)
    for k in range(c + 1):
        iax.axhline(y0 - .5 + k, color=GRILLE, lw=.5)
        iax.axvline(x0 - .5 + k, color=GRILLE, lw=.5)
    # une couleur par identite quand la grille en porte plusieurs
    couches = ([(res[k], color_of(i)) for k, i in enumerate(ids)]
               if res is not None else [(grille, couleur)])
    for couche, col in couches:
        sous = couche[max(y0, 0):y0 + c, max(x0, 0):x0 + c]
        yy, xx = np.nonzero(sous)
        iax.scatter(xx + max(x0, 0), yy + max(y0, 0), s=70, marker="s",
                    color=col, edgecolor="white", linewidth=.5)
    iax.scatter([centre[1]], [centre[0]], s=110, marker="o", color=VUE,
                edgecolor="white", linewidth=1.2, zorder=5)
    iax.set_xlim(x0 - .5, x0 + c - .5), iax.set_ylim(y0 + c - .5, y0 - .5)
    iax.set_xticks([]), iax.set_yticks([])
    for co in iax.spines.values():
        co.set_color(VUE), co.set_linewidth(1.4)
    _, traits = ax.indicate_inset_zoom(iax, edgecolor=VUE, linewidth=1.4, alpha=1)
    for t in traits:                      # les diagonales se croisent, illisible
        t.set_visible(False)
    iax.set_title(f"agent view, {c} × {c}", fontsize=9.5, color=VUE, pad=5)


def panneau_multi(ax, res, ids, titre, sous_titre, vue=0):
    """Plusieurs identites de ressource sur la meme grille."""
    L = res.shape[1]
    panneau(ax, np.zeros((L, L)), color_of(ids[0]), titre, sous_titre)
    for c, i in enumerate(ids):
        y, x = np.nonzero(res[c])
        ax.scatter(x, y, s=46, marker="s", color=color_of(i), edgecolor="white",
                   linewidth=.5, zorder=4, label=f"{label_of(i)} ({int(res[c].sum())})")
    total = res.sum(axis=0)
    if vue and total.any():
        yy, xx = np.nonzero(total)
        centre = tuple(int(np.clip(round(v), vue + 1, L - vue - 2))
                       for v in (yy.mean(), xx.mean()))
        encart_vision(ax, total, color_of(ids[0]), vue, centre, res=res, ids=ids)



def panneau(ax, grille, couleur, titre, sous_titre, vue=None, centre=None):
    L = grille.shape[0]
    ax.set_facecolor(FOND)
    ax.set_xlim(-.5, L - .5), ax.set_ylim(L - .5, -.5)
    ax.set_aspect("equal")
    for k in range(L + 1):
        ax.axhline(k - .5, color=GRILLE, lw=.35, zorder=1)
        ax.axvline(k - .5, color=GRILLE, lw=.35, zorder=1)

    # bande d'apparition des agents : randint(10, L - 10) dans init_state_lab
    ax.add_patch(Rectangle((9.5, 9.5), L - 20, L - 20, facecolor="#00000008",
                           edgecolor="#9A9A9A", lw=.9, ls=(0, (4, 3)), zorder=2))
    # mur letal : l'anneau exterieur d'une case
    for x, y, w, h in ((-.5, -.5, L, 1), (-.5, L - 1.5, L, 1),
                       (-.5, .5, 1, L - 2), (L - 1.5, .5, 1, L - 2)):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=MUR, edgecolor="none", zorder=3))
    ax.add_patch(Rectangle((-.5, -.5), L, L, facecolor="none", edgecolor=MUR,
                           lw=1.6, zorder=5))

    y, x = np.nonzero(grille)
    if len(y):
        ax.scatter(x, y, s=46, marker="s", color=couleur, edgecolor="white",
                   linewidth=.5, zorder=4)
    ax.set_xticks([]), ax.set_yticks([])
    for c in ax.spines.values():
        c.set_visible(False)
    if vue and grille.any():
        # centre sur les ressources : un encart vide ne montrerait pas l'echelle
        yy, xx = np.nonzero(grille)
        c = centre or (int(round(yy.mean())), int(round(xx.mean())))
        c = tuple(int(np.clip(v, vue + 1, L - vue - 2)) for v in c)
        encart_vision(ax, grille, couleur, vue, c)
    ax.set_title(titre, fontsize=13, pad=30, fontweight="semibold")
    ax.text(.5, 1.012, sous_titre, transform=ax.transAxes, ha="center",
            va="bottom", fontsize=10, color="#5A5A5A")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("noms", nargs="*", default=None)
    p.add_argument("--dir", default=os.path.join(os.path.dirname(__file__), "..", "lab_envs"))
    p.add_argument("--low-res", dest="low_res", action="store_true",
                   help="ajouter l'env low_res (exploration), tire a la graine")
    p.add_argument("--high-res", dest="high_res", action="store_true",
                   help="ajouter l'env high_res tire a la graine : le seul cas a "
                        "plusieurs ressources, les .npy etant figes a une seule")
    p.add_argument("--from", dest="config_exp", default=None,
                   help="config d'un run, pour ses ressources et sa lab_seed")
    p.add_argument("--graine", type=int, default=None,
                   help="graine du low_res (defaut : cfg.lab_seed)")
    p.add_argument("--colonnes", type=int, default=0, metavar="N",
                   help="panneaux par ligne (defaut : tous sur une ligne)")
    p.add_argument("--vue", type=int, default=5,
                   help="rayon du champ de vision, encart sur le dernier panneau "
                        "(defaut %(default)s ; 0 = pas d'encart)")
    p.add_argument("--texte", action=argparse.BooleanOptionalAction, default=True,
                   help="phrase explicative sous la figure (--no-texte pour l'enlever)")
    p.add_argument("-o", "--out", default="fig/lab_envs.png")
    a = p.parse_args()

    # les .npy sont a une ressource : on ne les melange pas a un tirage a trois
    noms = list(a.noms) if a.noms else ([] if a.high_res else list(LAB_ENVS))
    grilles = [(n, charge(n, a.dir)) for n in noms]
    tires = []
    if a.high_res:
        tires.append(("High resources", "test env", ) + grille_tiree(
            "high_res", a.config_exp, a.graine))
    if a.low_res:
        tires.append(("Low resources", "exploration env") + grille_tiree(
            "low_res", a.config_exp, a.graine))
    bas = tires[-1][2:] if tires else None
    couleur = color_of(0)
    n_pan = len(grilles) + len(tires)
    nc = a.colonnes or n_pan
    nl = -(-n_pan // nc)
    fig, axes = plt.subplots(nl, nc, squeeze=False,
                             figsize=(4.5 * nc + (1.9 if a.vue else 0), 5.1 * nl))
    fig.patch.set_facecolor("white")
    axes = axes.ravel()
    for ax in axes[n_pan:]:
        ax.axis("off")
    for ax, (nom, g) in zip(axes, grilles):
        cle = os.path.splitext(os.path.basename(nom))[0]
        dernier = nom == grilles[-1][0] and not tires
        panneau(ax, g, couleur, TITRES.get(cle, cle), SOUS_TITRES.get(cle, ""),
                vue=a.vue if dernier else 0)
    for k, (titre, sous, res, ids, graine) in enumerate(tires):
        ax = axes[len(grilles) + k]
        total = int(res.sum())
        panneau_multi(ax, res, ids, titre, f"{sous} — {total} cells, seed {graine}",
                      vue=a.vue if k == len(tires) - 1 else 0)

    titre = ("Fixed test environments" if not tires else
             "Lab environments" if grilles else "Lab environments, drawn from the seed")
    fig.suptitle(titre, fontsize=16, fontweight="semibold", y=.99)
    ids_vus = sorted({i for *_, ids, _ in tires for i in ids})
    if len(ids_vus) > 1:
        fig.legend(handles=[plt.Line2D([0], [0], marker="s", color="none",
                                       markerfacecolor=color_of(i),
                                       markeredgecolor="white", markersize=11,
                                       label=label_of(i)) for i in ids_vus],
                   loc="lower center", ncol=len(ids_vus), frameon=False,
                   fontsize=10, bbox_to_anchor=(.5, .075 if a.texte else .01))
    legende = ("30 × 30 arena, border wall in dark, agents start inside the "
               "dashed area.")
    if grilles:
        legende += " The fixed environments hold 40 resource cells each."
    if a.vue:
        legende += " Blue: what one agent sees from its position."
    if a.texte:
        fig.text(.5, .028, legende, ha="center", fontsize=9.5, color="#4A4A4A",
                 wrap=True).set_in_layout(False)
    fig.tight_layout(rect=[0, .13 if len(ids_vus) > 1 else .08,
                           .88 if a.vue else 1, .95])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=200, facecolor="white")
    print(f"Figure saved: {a.out}")


if __name__ == "__main__":
    main()
