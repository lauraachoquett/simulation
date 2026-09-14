"""Dessine, montre et sauvegarde des environnements de test figes.

    python -m simulation.tools.make_lab_envs                       # le catalogue
    python -m simulation.tools.make_lab_envs --save patch4x10_s2   # en garder

Les env de lab tirent aujourd'hui leurs ressources au hasard puis les font
pre-croitre : la disposition ET le total dependent de `lab_seed`, et la
STRUCTURE SPATIALE n'est pas pilotable -- un tirage uniforme suivi d'une
croissance par voisinage donne toujours le meme semis faiblement agglomere.

On construit ici une famille a quantite CONSTANTE (40 cases) ou seule la
structure change, du semis uniforme a l'amas unique. Rien ne repoussant pendant
un rollout de lab (les trois lanceurs posent resources_growth=False), la grille
sauvegardee est exactement ce que l'agent voit du premier au dernier pas.

La premiere ligne du catalogue montre les env ACTUELS, lus par le vrai chemin
de code : elle sert d'etalon visuel, et c'est elle qui donne le nombre de cases
sur lequel les candidats low_res sont cales.
"""
import argparse
import hashlib
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import jax
import numpy as np

import jax.numpy as jnp
from jax import random

from EcoEvoJax.source.agent import metaRNNPolicyState_bcppr
from simulation.data_class import color_of, label_of, resolve_model
from simulation.lab_env import (DOSSIER_ENVS, vmap_over_agents_env_lab_high_res,
                                vmap_over_agents_env_lab_low_res)
from simulation.tools.preview_lab_env import grille_de_depart


# Arene et quantite de reference : celles des env de lab actuels, pour que les
# series lab_data deja produites restent comparables.
COTE = 30
TOTAL = 40

# Bande d'apparition de l'agent : init_state_lab tire sa position dans
# [MARGE, L - MARGE). La montrer sur chaque vignette n'est pas decoratif -- dans
# un env a amas unique, la distance entre cette bande et l'amas decide de tout.
MARGE = 10


def config_de(chemin):
    """La Config de reference : celle d'un run si on en donne un, sinon les
    defauts de preview_lab_env -- pas une seconde liste a tenir a jour."""
    if chemin:
        from simulation.utils.utils_sim import load_config
        return load_config(chemin)[0]
    from simulation.tools.preview_lab_env import config_par_defaut
    return config_par_defaut()


def graine_de(nom):
    """Graine stable pour un nom de configuration.

    hash() est sale par processus : deux appels de l'outil ne redonneraient pas
    la meme grille, et --save ne pourrait pas reproduire la vignette choisie.
    """
    return int.from_bytes(hashlib.md5(nom.encode()).digest()[:4], "little")


def cases_libres(cote=COTE):
    """Interieur de la grille : les murs occupent la bordure (init_state_lab)."""
    yy, xx = np.mgrid[0:cote, 0:cote]
    return (yy > 0) & (yy < cote - 1) & (xx > 0) & (xx < cote - 1)


def disque(centre, k, cote=COTE):
    """Les k cases libres les plus proches de `centre`.

    Par distance et non par rayon : un rayon donne un compte qui saute de 5 en
    13 en 21 cases, et la famille ne serait plus a quantite constante.
    """
    yy, xx = np.mgrid[0:cote, 0:cote]
    d2 = (yy - centre[0]) ** 2 + (xx - centre[1]) ** 2
    d2 = np.where(cases_libres(cote), d2, np.inf)
    plats = np.argsort(d2.ravel(), kind="stable")[:k]
    g = np.zeros((cote, cote), dtype=np.int8)
    g.ravel()[plats] = 1
    return g


def centres(n, rng, sep, cote=COTE, marge=3):
    """n centres a l'interieur, separes d'au moins `sep`.

    Sans separation minimale deux disques fusionnent et la configuration ne
    porte plus son nom -- "4 amas de 10" devient "3 amas" sans que rien ne le
    dise. On relache la contrainte plutot que de boucler sans fin si la
    geometrie ne la permet pas, en le signalant.
    """
    for essai in range(400):
        pts = rng.integers(marge, cote - marge, size=(n, 2))
        if n == 1:
            return pts
        d = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
        np.fill_diagonal(d, np.inf)
        if d.min() >= sep:
            return pts
    print(f"  [attention] {n} centres separes de {sep} : non trouve en 400 essais, "
          "derniere tentative gardee")
    return pts


def semis(k, rng, cote=COTE):
    """k cases libres DISTINCTES, tirees uniformement.

    Distinctes : init_state_lab tire k positions avec remise, donc pose en fait
    moins de k cases. Ici le compte annonce est le compte reel.
    """
    libres = np.flatnonzero(cases_libres(cote))
    choix = rng.choice(libres, size=k, replace=False)
    g = np.zeros(cote * cote, dtype=np.int8)
    g[choix] = 1
    return g.reshape(cote, cote)


def amas(n_amas, k, rng, cote=COTE):
    """n_amas disques de k cases chacun."""
    rayon = np.sqrt(k / np.pi)
    pts = centres(n_amas, rng, sep=2 * rayon + 3, cote=cote)
    g = np.zeros((cote, cote), dtype=np.int8)
    for c in pts:
        g |= disque(c, k, cote)
    return g


def organique(n_graines, total, rng, cote=COTE):
    """Amas a contour irregulier : on part de n graines et on ajoute, une par
    une, une case tiree au hasard sur la FRONTIERE de l'ensemble courant.

    Pourquoi ne pas appeler resources_growth : elle fait aussi apparaitre des
    cases spontanement (pop_res_prob s'ajoute meme a zero voisin), ce qui
    ressemerait le fond de la grille, et elle ne tombe sur aucun compte precis.
    Ici l'ensemble reste connexe par graine et le total est exact.
    """
    pts = centres(n_graines, rng, sep=8, cote=cote)
    g = np.zeros((cote, cote), dtype=np.int8)
    for c in pts:
        g[c[0], c[1]] = 1
    libre = cases_libres(cote)
    while g.sum() < total:
        voisins = np.zeros_like(g)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            voisins |= np.roll(np.roll(g, dy, axis=0), dx, axis=1)
        front = np.flatnonzero((voisins > 0) & (g == 0) & libre)
        if not front.size:
            break
        g.ravel()[rng.choice(front)] = 1
    return g


def configurations(n_low):
    """nom -> fabricant. `n_low` vient de la ligne de reference, pas d'un pari."""
    return {
        "scatter40":  lambda r: semis(TOTAL, r),
        "patch8x5":   lambda r: amas(8, TOTAL // 8, r),
        "patch4x10":  lambda r: amas(4, TOTAL // 4, r),
        "patch2x20":  lambda r: amas(2, TOTAL // 2, r),
        "patch1x40":  lambda r: amas(1, TOTAL, r),
        "blob4x10":   lambda r: organique(4, TOTAL, r),
        "blob1x40":   lambda r: organique(1, TOTAL, r),
        "low_scatter": lambda r: semis(n_low, r),
        "low_patch":   lambda r: amas(1, n_low, r),
    }


def fabrique(nom, seed, n_low):
    """La grille (L, L) d'une vignette, reproductible depuis son seul nom."""
    config = configurations(n_low)[nom]
    return config(np.random.default_rng(graine_de(f"{nom}_s{seed}")))


class ModeleFactice:
    """De quoi traverser init_state_lab sans construire de reseau.

    La grille ne depend pas du modele : grille_de_depart lit `out.grid[0, 0]`,
    journalise au pas 0, donc la grille que init_state_lab vient de poser --
    aucun agent n'a encore agi. Le modele n'y sert qu'a dimensionner un tableau
    de poids (num_params) et a remplir l'etat RNN (reset_b), ni l'un ni l'autre
    ne touchant aux ressources.

    On garde donc le VRAI chemin de code -- launch_env_* avec ses remplacements
    de cfg, puis init_state_lab -- qui est ce qui empeche l'etalon de deriver,
    sans trainer EcoEvoJax, flax et la pile de trace derriere un outil qui
    dessine des grilles.
    """
    num_params = 4

    def reset_b(self, x):
        n = x.shape[0]
        return metaRNNPolicyState_bcppr(
            lstm_h=jnp.zeros((n, 2)), lstm_c=jnp.zeros((n, 2)),
            keys=random.split(random.PRNGKey(0), n))

    def get_actions(self, state, params, policy_states):
        # logits nuls : la politique n'a aucun effet sur la grille du pas 0
        return jnp.zeros((state.agents.energy.shape[0], 4)), policy_states


def reference(cfg, graine):
    """Les environnements high_res et low_res d'aujourd'hui.

    Lus par le VRAI chemin de code (`launch_env_*` avec log_grid) et non
    reconstruits : l'etalon suit donc tout changement de ces lanceurs sans
    effort. Les wrappers VMAP, pas les lanceurs bruts -- grille_de_depart
    appelle fn(params, key_env, key_sim, model, cfg), l'ordre des vmap_*, alors
    que launch_env_* prend (..., cfg, model). Se tromper la ne donne pas une
    erreur d'arite mais un AttributeError sur cfg._replace.
    """
    return [(f"{nom} (current)",
             grille_de_depart(fn, cfg, ModeleFactice(), graine)[0].sum(axis=0))
            for nom, fn in (("high_res", vmap_over_agents_env_lab_high_res),
                            ("low_res",  vmap_over_agents_env_lab_low_res))]


def vignette(ax, grille, titre, ident):
    """Une grille, ses murs, et la bande d'apparition de l'agent."""
    cote = grille.shape[0]
    img = np.ones((cote, cote, 3))
    bord = np.zeros((cote, cote), dtype=bool)
    bord[0, :] = bord[-1, :] = bord[:, 0] = bord[:, -1] = True
    img[bord] = (0.25, 0.25, 0.25)
    img[grille > 0] = mcolors.to_rgb(color_of(ident))
    ax.imshow(img, interpolation="nearest")
    ax.add_patch(mpatches.Rectangle(
        (MARGE - .5, MARGE - .5), cote - 2 * MARGE, cote - 2 * MARGE,
        fill=False, ec="#1D3557", lw=1.1, ls=(0, (3, 2)), zorder=3))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{titre}\n{int(grille.sum())} cells", fontsize=8.5)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--save", nargs="*", default=None, metavar="NOM",
                   help="noms de vignettes a ecrire dans lab_envs/ "
                        "(ex. patch4x10_s2). Sans argument, seul le catalogue "
                        "est produit")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="dispositions proposees par configuration (defaut %(default)s)")
    p.add_argument("--from", dest="config_exp", default=None,
                   help="reprendre la config d'un run pour la ligne de reference ; "
                        "sinon les defauts de preview_lab_env")
    p.add_argument("-o", "--out", default="fig/lab_envs_catalogue.png")
    p.add_argument("--envs-dir", default=DOSSIER_ENVS,
                   help="ou ecrire les .npy (defaut %(default)s)")
    a = p.parse_args()

    cfg = resolve_model(config_de(a.config_exp))
    if len(cfg.resources) != 1:
        raise SystemExit(
            f"make_lab_envs : {len(cfg.resources)} ressources dans la config. "
            "Ces environnements figes ne sont definis qu'a UNE ressource "
            "(--from un run a une ressource, ou les defauts).")
    ident = cfg.resources[0].id

    print("Ligne de reference (environnements de lab actuels) :")
    refs = reference(cfg, cfg.lab_seed)
    for nom, g in refs:
        print(f"  {nom:<20} {int(g.sum()):>3} cases")
    n_low = int(refs[1][1].sum())

    noms = list(configurations(n_low))
    lignes, cols = len(noms) + 1, len(a.seeds)
    fig, axes = plt.subplots(lignes, cols, figsize=(3.0 * cols, 3.4 * lignes),
                             squeeze=False)

    for j, (nom, g) in enumerate(refs):
        vignette(axes[0][j], g, nom, ident)
    for ax in axes[0][len(refs):]:
        ax.axis("off")

    grilles = {}
    for i, nom in enumerate(noms, start=1):
        for j, s in enumerate(a.seeds):
            g = fabrique(nom, s, n_low)
            grilles[f"{nom}_s{s}"] = g
            vignette(axes[i][j], g, f"{nom}_s{s}", ident)

    fig.suptitle(
        f"Candidate lab environments — {COTE}×{COTE} arena, "
        f"{TOTAL} resource cells (low_res family: {n_low})\n"
        f"dashed square: agent spawn band · resource identity: {label_of(ident)}",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97], h_pad=2.2)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=130)
    plt.close(fig)
    print(f"Figure saved: {a.out}")

    if not a.save:
        print("\nRegarder la figure, puis : --save <nom_s#> [<nom_s#> ...]")
        return

    os.makedirs(a.envs_dir, exist_ok=True)
    n_ids = ident + 1
    for nom in a.save:
        if nom not in grilles:
            raise SystemExit(f"--save {nom} : nom inconnu. Disponibles : "
                             + ", ".join(sorted(grilles)))
        # (n_ids, L, L) indexe par IDENTITE et non par canal : c'est l'invariant
        # que tient init_state_lab -- une ressource occupe les memes cases quelle
        # que soit la permutation des canaux.
        plan = np.zeros((n_ids, COTE, COTE), dtype=np.int8)
        plan[ident] = grilles[nom]
        chemin = os.path.join(a.envs_dir, f"{nom}.npy")
        np.save(chemin, plan)
        print(f"  {chemin}  {plan.shape}  {int(plan.sum())} cases")


if __name__ == "__main__":
    main()
