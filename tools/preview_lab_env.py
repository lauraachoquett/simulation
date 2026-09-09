"""Apercu de l'environnement de lab pour plusieurs graines, pour en choisir une.

    python -m simulation.tools.preview_lab_env --seeds 0 1 2 3 4 5
    python -m simulation.tools.preview_lab_env --from <exp_dir> --seeds 0 1 2

L'env de lab est l'etalon sur lequel tous les genomes sont notes : il vaut la
peine de le regarder avant de figer `lab_seed`. La graine change la disposition
ET le nombre de ressources -- la pre-croissance tire case par case, donc le
total varie d'un tirage a l'autre (39 a 46 sur six graines, a une ressource).
Une graine pauvre rend le lab plus dur pour tout le monde, ce qui n'est pas
grave en soi puisque l'etalon est partage, mais deplace le plancher.

La grille est lue sur le VRAI chemin de code (`launch_env_high_res` avec
log_grid), pas reconstruite : ce qui est montre est donc exactement ce que les
agents rencontreront.
"""
import argparse
import os

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from jax import random

from simulation.data_class import Config, BASE_RESOURCES, label_of, color_of, resolve_model
from simulation.lab_env import vmap_over_agents_env_lab_high_res
from simulation.run import build_model
from simulation.utils.utils_sim import load_config


def grille_de_depart(cfg, model, graine):
    """(n_types, L, L), (L, L) : ressources et murs au premier pas du lab."""
    params = jnp.zeros((1, model.num_params))
    cfg_v = cfg._replace(log_grid=True, lab_time_steps=2)
    _, out = vmap_over_agents_env_lab_high_res(
        params, random.PRNGKey(graine), random.split(random.PRNGKey(0), 1),
        model, cfg_v)
    g = np.asarray(out.grid[0, 0])            # (n_types + 2, L, L)
    n = len(cfg.resources)
    return g[:n], g[n + 1]


def trace(grilles, cfg, sortie):
    """Une vignette par graine, plus le compte par identite."""
    n_types = len(cfg.resources)
    k = len(grilles)
    cols = min(k, 4)
    lignes = (k + cols - 1) // cols
    fig, axes = plt.subplots(lignes, cols, figsize=(3.6 * cols, 3.9 * lignes),
                             squeeze=False)

    for ax in axes.ravel()[k:]:
        ax.axis("off")

    for i, (graine, res, murs) in enumerate(grilles):
        ax = axes[i // cols, i % cols]
        L = murs.shape[0]
        img = np.ones((L, L, 3))
        img[murs > 0] = (0.25, 0.25, 0.25)
        # canal -> identite : la couleur suit l'identite, pas l'indice de canal
        for c, r in enumerate(cfg.resources):
            m = res[c] > 0
            if m.any():
                img[m] = mcolors.to_rgb(color_of(r.id))
        ax.imshow(img, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        comptes = "  ".join(f"{label_of(r.id)} {int(res[c].sum())}"
                            for c, r in enumerate(cfg.resources))
        ax.set_title(f"seed {graine}\n{comptes}", fontsize=9)

    fig.suptitle("Lab starting grid, by seed  "
                 f"(grid {grilles[0][2].shape[0]}, agents spawn in the central band)",
                 fontsize=12)
    # rect et h_pad : sans eux les titres de la 2e ligne recouvrent la 1ere
    fig.tight_layout(rect=[0, 0, 1, 0.94], h_pad=2.6)
    os.makedirs(os.path.dirname(sortie) or ".", exist_ok=True)
    fig.savefig(sortie, dpi=150)
    plt.close(fig)
    print(f"Figure saved: {sortie}")


def config_par_defaut():
    """Une Config minimale mais COMPLETE.

    Les champs sont enumeres depuis Config elle-meme (`_field_defaults`) plutot
    qu'ecrits a la main : une liste figee se perime des qu'un champ obligatoire
    est ajoute, et l'erreur ne sort qu'a l'execution. Seuls comptent ici ceux qui
    touchent la grille du lab ; le reste n'influe pas sur ce qu'on affiche.
    """
    valeurs = {
        "grid_length": 200, "chunk_size": 1000, "num_chunks": 1,
        "checkpoint_freq": 50, "video_freq": 50, "lab_evaluation_freq": 500,
        "n_agents_max": 10, "n_agents_init": 5, "agent_view": 5,
        "temperature": 0.3, "energy_decay": 0.01,
        "factor_energy_decay_not_moving": 0.3, "energy_max": 8.0,
        "time_to_die": 400, "time_above_repr": 320, "min_energy_repr": 6.0,
        "starting_energy": 1.5, "random_pos_offspring": False,
        "mutation_var": 0.02, "param_mutate": 0.3, "pre_growth_step": 500,
    }
    sans_defaut = [f for f in Config._fields if f not in Config._field_defaults]
    manquants = [f for f in sans_defaut if f not in valeurs]
    if manquants:
        raise SystemExit(
            "config_par_defaut : champs obligatoires non couverts "
            f"({', '.join(manquants)}). Les ajouter ici, ou passer --from.")
    return Config(**valeurs, resources=BASE_RESOURCES, model_version="v2")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5],
                   help="graines a comparer (defaut %(default)s)")
    p.add_argument("--from", dest="config_exp", default=None,
                   help="reprendre la config d'un run ; sinon les defauts")
    p.add_argument("-o", "--out", default="fig/lab_env_by_seed.png")
    a = p.parse_args()

    if a.config_exp:
        cfg, _ = load_config(a.config_exp)
    else:
        cfg = config_par_defaut()
    cfg = resolve_model(cfg)
    model = build_model(cfg)

    grilles = []
    for s in a.seeds:
        res, murs = grille_de_depart(cfg, model, s)
        grilles.append((s, res, murs))
        print(f"  seed {s:>4} : "
              + "  ".join(f"{label_of(r.id)}={int(res[c].sum())}"
                          for c, r in enumerate(cfg.resources)))
    trace(grilles, cfg, a.out)


if __name__ == "__main__":
    main()
