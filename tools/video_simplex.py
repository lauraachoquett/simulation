"""Video : le nuage des regimes se deplace dans le simplex, checkpoint apres
checkpoint.

    python -m simulation.tools.video_simplex <exp_dir> -n 50 --fps 6
    python -m simulation.tools.video_simplex <exp_A> <exp_B> --fps 8

Chaque point est un genome REEVALUE dans le lab high_res : meme environnement
pour tous et pour toutes les frames. Le deplacement du nuage est donc un fait
genetique, l'environnement ne bougeant pas.

Ce que la figure porte, et pourquoi :

- le CONTOUR a 50 % (KDE, pointille) suit la forme reelle du nuage, y compris
  en deux paquets si la population se scinde -- ce qu'une ellipse masquerait ;
- le FANTOME, contour de la frame precedente en gris, rend le deplacement
  visible sans avoir a se souvenir de l'image d'avant ;
- la TRAINEE du barycentre remplace la trajectoire d'une lignee, en beaucoup
  moins bruite : une lignee est un echantillon de taille 1 ;
- la LIGNE D'EQUILIBRE separe les regimes qui rapportent de ceux qui coutent ;
- le CERCLE de reference est la composition OFFERTE sur la grille. C'est l'ancre :
  sans lui, un nuage pres de `good` peut n'etre que le reflet de son abondance.
  C'est l'ECART nuage <-> cercle qui est la preference ;
- la FRISE dit ou l'on en est et ou sont tombees les permutations.

Prerequis : trois ressources, et des checkpoints assez denses. Il faut au moins
5 frames par cycle de permutation, sinon on echantillonne toujours la meme phase
et le saut vers le poison tombe entre deux frames -- la video montre alors un
nuage immobile en suggerant qu'il ne se passe rien.
"""
import argparse
import glob
import json
import os
import re

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from jax import random

from simulation.data_class import LABELS, label_of, color_of
from simulation.lab_env import vmap_over_agents_env_lab_high_res
from simulation.run import build_model
from simulation.simulation_data.core import simulation_data
from simulation.tools.replay_lab import (checkpoints_de, resources_au_pas,
                                         par_lots)
from matplotlib.patches import Polygon

from simulation.utils.plots import _bary, _cadre_simplex, ligne_equilibre
from simulation.utils.utils_sim import (load_config, load_checkpoint,
                                        load_shuffle_log)
from simulation.utils.utils_video import VideoWriter


def composition(regime):
    """(M, 3) -> proportions, en ecartant ceux qui n'ont rien mange.

    Un agent qui n'a rien mange n'a pas une composition nulle : il n'en a pas.
    Le garder au barycentre du triangle inventerait un regime equilibre.
    """
    r = np.asarray(regime, dtype=float)
    total = r.sum(axis=1)
    ok = total > 0
    return r[ok] / total[ok, None]


def triangle_patch(ax):
    """Le triangle du simplex, comme patch, pour DECOUPER ce qui en deborde.

    Une densite estimee sur un plan ne connait pas la contrainte p_g+p_m+p_p=1 :
    son contour sort du triangle et suggere des compositions qui n'existent pas.
    """
    coins = np.array([_bary(1, 0, 0), _bary(0, 1, 0), _bary(0, 0, 1)])
    return Polygon(coins, transform=ax.transData, facecolor="none",
                   edgecolor="none")


def _chemins(cs):
    """Les Path d'un contour, quelle que soit la version de matplotlib.

    `ContourSet.collections` est supprime depuis matplotlib 3.10 ; la version
    recente expose get_paths() directement sur le ContourSet.
    """
    if hasattr(cs, "collections"):
        return [p for c in cs.collections for p in c.get_paths()]
    return list(cs.get_paths())


def _decouper(cs, patch):
    """Applique un chemin de decoupe au contour, ancienne et nouvelle API."""
    if hasattr(cs, "collections"):
        for c in cs.collections:
            c.set_clip_path(patch)
    else:
        cs.set_clip_path(patch)


def _segments(cs):
    """Les lobes du contour, separes.

    to_polygons et non `.vertices` : un Path peut contenir plusieurs lobes
    disjoints, et concatener leurs sommets tracerait un trait parasite de l'un a
    l'autre -- precisement le cas que le contour a 50 % doit pouvoir montrer.
    """
    segs = []
    for chemin in _chemins(cs):
        segs.extend(np.asarray(v) for v in chemin.to_polygons(closed_only=False)
                    if len(v) > 1)
    return segs


def contour_kde(ax, x, y, niveau=0.5, **kw):
    """Contour contenant `niveau` de la masse, par estimation de densite.

    Le niveau est pris au percentile des densites AUX POINTS de l'echantillon :
    c'est l'approximation usuelle de la plus petite region a 50 %. Rend les
    segments traces, pour pouvoir les rejouer en fantome.
    """
    if len(x) < 5:
        return None
    from scipy.stats import gaussian_kde
    pts = np.vstack([x, y])
    try:
        k = gaussian_kde(pts)
    except np.linalg.LinAlgError:      # nuage degenere (tous au meme endroit)
        return None
    # maillage sur la boite du triangle, puis coupe DANS le triangle : la
    # densite deborde d'un domaine qui est borne
    gx, gy = np.mgrid[-0.02:1.02:120j, -0.02:0.89:120j]
    d = k(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    seuil = np.percentile(k(pts), 100 * (1 - niveau))
    cs = ax.contour(gx, gy, d, levels=[seuil], **kw)
    _decouper(cs, triangle_patch(ax))
    return _segments(cs)


def frame(fig, p, ages, norm_age, dispo, resources, step, epoques, bornes,
          fantome, trainee, shuffle_actif=False, n_reels=None, taille=110):
    """Compose une frame et rend (contour, barycentre) pour la suivante.

    `p` est deja en PROPORTIONS : l'interpolation entre deux checkpoints se fait
    en amont, dans l'espace des compositions. Une combinaison convexe de deux
    compositions valides en est une -- les points interpoles restent donc dans
    le triangle, ce qu'une interpolation en coordonnees d'ecran ne garantirait
    pas.
    """
    fig.clear()
    ax = fig.add_axes([0.13, 0.13, 0.85, 0.81])
    _cadre_simplex(ax)
    # marge elargie : les etiquettes des sommets sont posees HORS des limites
    # que pose _cadre_simplex, et se faisaient couper
    ax.set_xlim(-0.16, 1.16)
    ax.set_ylim(-0.13, np.sqrt(3) / 2 + 0.11)
    ligne_equilibre(ax, resources, color="0.25", lw=1.6, label=False, zorder=2)

    p = np.asarray(p, dtype=float)
    ages = np.asarray(ages, dtype=float)
    barycentre = None
    contour = None
    if len(p):
        x, y = _bary(p[:, 0], p[:, 1], p[:, 2])

        # fantome d'abord, sous le reste
        if fantome is not None:
            decoupe = triangle_patch(ax)
            for seg in fantome:
                (l,) = ax.plot(seg[:, 0], seg[:, 1], color="0.72", lw=1.4,
                               ls=":", zorder=3)
                l.set_clip_path(decoupe)

        # trainee du barycentre, la plus ancienne la plus pale
        if len(trainee) > 1:
            t = np.array(trainee)
            for i in range(len(t) - 1):
                ax.plot(t[i:i + 2, 0], t[i:i + 2, 1], color="#C1121F",
                        lw=1.1, alpha=0.10 + 0.55 * i / max(len(t) - 2, 1),
                        zorder=4)

        sc = ax.scatter(x, y, s=taille, c=ages, cmap="viridis", norm=norm_age,
                        alpha=.85, edgecolor="white", linewidth=.7, zorder=5)
        contour = contour_kde(ax, x, y, colors="#1D3557", linewidths=2.0,
                              linestyles="--", zorder=6)
        barycentre = (float(x.mean()), float(y.mean()))
        ax.scatter([barycentre[0]], [barycentre[1]], s=120, marker="X",
                   color="#C1121F", edgecolor="white", linewidth=1.2, zorder=7)

    if dispo is not None and np.sum(dispo) > 0:
        d = np.asarray(dispo, float)
        xd, yd = _bary(*(d / d.sum()))
        ax.scatter([xd], [yd], marker="o", s=200, facecolor="none",
                   edgecolor="black", linewidth=2.0, zorder=8)

    ax.set_title(f"Diet composition — step {int(round(step)):,}   "
                 f"({n_reels if n_reels is not None else len(p)} genomes)",
                 fontsize=13)

    # Pas d'encart canal -> identite : la frise du bas porte la meme information,
    # et en continu plutot qu'au seul instant courant.
    if shuffle_actif:
        # coin haut gauche : centre il recouvrait le sommet "good", et au-dessus
        # des axes il recouvrait le titre. Le coin est le seul endroit vide.
        ax.text(0.02, 0.99, "CHANNEL\nPERMUTATION", transform=ax.transAxes,
                ha="left", va="top", fontsize=12, weight="bold",
                color="#C1121F", linespacing=1.15, zorder=9)

    # Frise : une bande par CANAL, coloree selon l'identite qu'il porte. Des
    # traits verticaux diraient qu'une permutation a lieu, pas CE QUI change --
    # or c'est la seule chose que le triangle ne montre pas, ses sommets etant
    # des identites fixes. Ici les couleurs s'echangent, on lit donc la
    # permutation elle-meme.
    n_can = len(epoques[0][1])
    fr = fig.add_axes([0.13, 0.05, 0.80, 0.075])
    fr.set_xlim(bornes)
    fr.set_ylim(-0.5, n_can - 0.5)
    fr.spines[["right", "top"]].set_visible(False)

    for i, (x0, ordre) in enumerate(epoques):
        x1 = epoques[i + 1][0] if i + 1 < len(epoques) else bornes[1]
        if x1 <= bornes[0] or x0 >= bornes[1]:
            continue
        x0, x1 = max(x0, bornes[0]), min(x1, bornes[1])
        for k, ident in enumerate(ordre):
            fr.barh(n_can - 1 - k, x1 - x0, left=x0, height=.82,
                    color=color_of(int(ident)), edgecolor="white", linewidth=.6)

    fr.axvline(step, color="#C1121F", lw=2.4, zorder=5)
    fr.set_yticks(range(n_can))
    fr.set_yticklabels([f"c{n_can - 1 - k}" for k in range(n_can)], fontsize=8)
    fr.set_xlabel("Simulation step", fontsize=9)
    fr.tick_params(labelsize=8)

    if len(p):
        cax = fig.add_axes([0.055, 0.18, 0.018, 0.62])
        barre = fig.colorbar(sc, cax=cax)
        barre.set_label("Lifespan in the lab (steps)", fontsize=9)
        cax.yaxis.set_ticks_position("left")
        cax.yaxis.set_label_position("left")
        cax.tick_params(labelsize=8)

    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    return img, contour, barycentre


def dossier_config(chemin):
    """Ou lire la config : ici, chez le parent (replay/), ou via exp.json (fusion)."""
    if os.path.exists(os.path.join(chemin, "config.json")):
        return chemin
    parent = os.path.dirname(os.path.abspath(chemin.rstrip("/")))
    if os.path.exists(os.path.join(parent, "config.json")):
        return parent
    try:
        with open(os.path.join(chemin, "exp.json")) as fh:
            for e in json.load(fh).get("experiences", []):
                if os.path.exists(os.path.join(e["chemin"], "config.json")):
                    return e["chemin"]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return chemin


def etapes_depuis_lab_data(exp_dirs, cfg):
    """Les nuages deja calcules pendant le run, sans rien reevaluer.

    Les simplex sont traces a chaque evaluation de lab, plus frequente que les
    checkpoints : cette source donne donc PLUS d'images qu'un rejeu, et pour un
    cout nul. Elle porte en revanche les genomes que le run avait retenus
    (les 50 premiers survivants), pas un echantillon qu'on choisit ici.
    """
    par_id = {r.id: r for r in cfg.resources}
    par_step, dispo = {}, None
    for d in exp_dirs:
        for f in glob.glob(os.path.join(d, "lab_data", "simplex_chunk_*.npz")):
            with np.load(f) as z:
                step = int(z["step"])
                eaten, ids, age = z["eaten"], z["ids"], z["age"]
                if dispo is None:
                    dc = z["dispo"]
                    dispo = np.zeros(len(ids))
                    for k, i in enumerate(ids):
                        dispo[int(i)] = dc[k]
            # canal -> identite : les sommets du triangle sont des identites
            par_identite = np.zeros_like(np.asarray(eaten, float))
            for k, i in enumerate(ids):
                par_identite[:, int(i)] = eaten[:, k]
            total = par_identite.sum(axis=1)
            ok = total > 0
            if not ok.any():
                continue
            par_step[step] = dict(
                step=step, p=par_identite[ok] / total[ok, None],
                age=np.asarray(age, float)[ok],
                res=tuple(par_id[int(i)] for i in ids), n=int(ok.sum()))
    return [par_step[s] for s in sorted(par_step)], dispo


def apparier(pa, pb):
    """Affectation de cout minimal entre deux nuages de compositions.

    Les genomes ne sont PAS les memes d'un checkpoint a l'autre : il n'existe
    aucune correspondance naturelle. On en fabrique une qui minimise la distance
    totale parcourue, de sorte que la transition ressemble a un glissement du
    nuage plutot qu'a une pluie de points sans rapport.

    Effectifs differents : le plus petit nuage est complete par tirage avec
    remise. Des points se dedoublent donc, ce qui est le comportement voulu --
    une population qui grossit voit ses paquets se scinder.
    """
    from scipy.optimize import linear_sum_assignment
    na, nb = len(pa), len(pb)
    if na == 0 or nb == 0:
        return pa, pb
    rng = np.random.default_rng(0)
    if na < nb:
        pa = np.vstack([pa, pa[rng.integers(0, na, nb - na)]])
    elif nb < na:
        pb = np.vstack([pb, pb[rng.integers(0, nb, na - nb)]])
    cout = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=-1)
    i, j = linear_sum_assignment(cout)
    return pa[i], pb[j]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("exp_dirs", nargs="+", metavar="EXP_DIR",
                   help="dossier(s) d'experience, dans l'ordre chronologique")
    p.add_argument("-o", "--out", default=None,
                   help="fichier .mp4 (defaut <exp_dir>/videos/simplex.mp4)")
    p.add_argument("-n", type=int, default=50, help="genomes par frame")
    p.add_argument("--batch", type=int, default=25)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--morph", type=int, default=6,
                   help="images intercalaires entre deux checkpoints (defaut "
                        "%(default)s). 0 = saut sec, comme avant")
    p.add_argument("--pause", type=int, default=8,
                   help="frames tenues sur une permutation")
    p.add_argument("--taille", type=float, default=110,
                   help="aire des points, en points^2 (defaut %(default)s)")
    p.add_argument("--trainee", type=int, default=8,
                   help="longueur de la trainee du barycentre, en CHECKPOINTS "
                        "(defaut %(default)s). 0 la supprime")
    p.add_argument("--lab", dest="lab_time_steps", type=int, default=None)
    p.add_argument("--lab-seed", dest="lab_seed", type=int, default=None)
    p.add_argument("--from-lab-data", dest="from_lab_data", action="store_true",
                   help="lire les nuages deja enregistres par le run "
                        "(lab_data/simplex_chunk_*.npz) au lieu de reevaluer "
                        "depuis les checkpoints. Plus d'images, cout nul")
    a = p.parse_args()

    cfg, _ = load_config(dossier_config(a.exp_dirs[0]))
    if a.lab_time_steps:
        cfg = cfg._replace(lab_time_steps=a.lab_time_steps)
    if len(cfg.resources) != 3:
        print(f"video_simplex : {len(cfg.resources)} ressource(s). Le simplex en "
              "demande 3.")
        return
    # ---- passe 1 : d'ou viennent les nuages ? ------------------------------
    if a.from_lab_data:
        etapes, dispo = etapes_depuis_lab_data(a.exp_dirs, cfg)
        if not etapes:
            print("Aucun simplex_chunk_*.npz. Ces runs sont anterieurs a leur "
                  "enregistrement : relancer sans --from-lab-data.")
            return
        print(f"{len(etapes)} frame(s) lues dans lab_data, pas "
              f"{etapes[0]['step']:,} a {etapes[-1]['step']:,}")
    else:
        model = build_model(cfg)
        # checkpoints dedoublonnes par PAS et non par numero de chunk : sur les
        # runs repris avant a7a31a3 les deux divergent
        trouves = {}
        for d in a.exp_dirs:
            for chunk, _ in checkpoints_de(d):
                st = load_checkpoint(d, chunk)
                trouves[int(st.step)] = (d, chunk)
        if not trouves:
            print("Aucun checkpoint.")
            return
        pas_tries = sorted(trouves)
        print(f"{len(pas_tries)} checkpoint(s), pas {pas_tries[0]:,} a "
              f"{pas_tries[-1]:,}")

        graine = a.lab_seed if a.lab_seed is not None else cfg.lab_seed
        key_env = random.PRNGKey(graine)
        cle = random.PRNGKey(graine + 1)
        sd = simulation_data(cfg, 0, 1)
        etapes, dispo = [], None
        for step in pas_tries:
            d, chunk = trouves[step]
            state = load_checkpoint(d, chunk)
            res = resources_au_pas(cfg, d, step)
            cfg_c = cfg._replace(resources=res, log_grid=False)
            sd.cfg, sd.chunk_idx = cfg_c, chunk

            survivants = sd.compute_survivors(state)
            if not survivants:
                print(f"  step {step:>9,} : aucun survivant, saute")
                continue
            rng = np.random.default_rng(step)
            ids = np.array([i for i, _ in survivants])
            if a.n and len(ids) > a.n:
                ids = ids[rng.choice(len(ids), a.n, replace=False)]

            cle, k = random.split(cle)
            out = par_lots(vmap_over_agents_env_lab_high_res,
                           state.agents.params[ids], key_env,
                           random.split(k, len(ids)), model, cfg_c, a.batch)
            mange = sd.eaten_by_type(out)
            age = np.asarray(out.alive).sum(axis=(1, 2)).astype(float)

            par_identite = np.zeros_like(mange)
            for k_canal, r in enumerate(res):
                par_identite[:, r.id] = mange[:, k_canal]
            total = par_identite.sum(axis=1)
            ok = total > 0
            etapes.append(dict(step=step, p=par_identite[ok] / total[ok, None],
                               age=age[ok], res=res, n=int(ok.sum())))

            if dispo is None:
                cle, kg = random.split(cle)
                _, og = vmap_over_agents_env_lab_high_res(
                    state.agents.params[ids[:1]], key_env, random.split(kg, 1),
                    model, cfg_c._replace(log_grid=True))
                dc = sd.available_by_type(og, len(res))
                dispo = np.zeros(len(res))
                for k_canal, r in enumerate(res):
                    dispo[r.id] = dc[k_canal]
            print(f"  step {step:>9,} : {int(ok.sum())} genomes", flush=True)

    if not etapes:
        print("Rien a tracer.")
        return

    shuffle_log = load_shuffle_log(a.exp_dirs[0])
    epoques = [(etapes[0]["step"], [r.id for r in cfg.resources])] + \
              [(e["step"], list(e["order_ids"])) for e in shuffle_log]
    epoques.sort(key=lambda t: t[0])
    bornes = (etapes[0]["step"], etapes[-1]["step"])

    tous_ages = np.concatenate([e["age"] for e in etapes if len(e["age"])])
    norm_age = mcolors.Normalize(vmin=float(tous_ages.min()),
                                 vmax=float(tous_ages.max()))

    # ---- passe 2 : rendre, en interpolant ----------------------------------
    sortie = a.out or os.path.join(a.exp_dirs[0], "videos", "simplex.mp4")
    os.makedirs(os.path.dirname(sortie) or ".", exist_ok=True)
    fig = plt.figure(figsize=(9.0, 7.6), dpi=120)
    fantome, trainee = None, []
    # en IMAGES et non en checkpoints, puisque la trainee suit les intercalaires
    long_trainee = a.trainee * (a.morph + 1)
    n_frames = 0

    with VideoWriter(sortie, fps=a.fps) as vid:
        for i, e in enumerate(etapes):
            precedent = etapes[i - 1] if i else None
            juste_apres = bool(precedent is not None and any(
                precedent["step"] < sh["step"] <= e["step"]
                for sh in shuffle_log))

            # images intercalaires : le nuage GLISSE de l'etape precedente a
            # celle-ci, au lieu de sauter
            inter = []
            if precedent is not None and a.morph:
                pa, pb = apparier(precedent["p"], e["p"])
                aa = np.resize(precedent["age"], len(pa))
                ab = np.resize(e["age"], len(pb))
                for t in np.linspace(0, 1, a.morph + 2)[1:-1]:
                    inter.append(((1 - t) * pa + t * pb,
                                  (1 - t) * aa + t * ab,
                                  precedent["step"] + t * (e["step"] - precedent["step"])))

            # La trainee suit AUSSI les images intercalaires : reliee aux seuls
            # checkpoints elle sautait d'un point a l'autre et se lisait comme un
            # gribouillis, alors que le nuage, lui, glisse.
            for pi, ai, si in inter:
                img, _, b = frame(fig, pi, ai, norm_age, dispo, e["res"], si,
                                  epoques, bornes, fantome, trainee,
                                  shuffle_actif=juste_apres, n_reels=e["n"],
                                  taille=a.taille)
                vid.add(img); n_frames += 1
                if b is not None:
                    trainee.append(b); trainee[:] = trainee[-long_trainee:]

            img, contour, bary = frame(fig, e["p"], e["age"], norm_age, dispo,
                                       e["res"], e["step"], epoques, bornes,
                                       fantome, trainee,
                                       shuffle_actif=juste_apres, n_reels=e["n"],
                                       taille=a.taille)
            vid.add(img); n_frames += 1
            if juste_apres:
                for _ in range(a.pause):
                    vid.add(img); n_frames += 1

            fantome = contour if juste_apres else None
            if bary is not None:
                trainee.append(bary)
                trainee[:] = trainee[-long_trainee:]

    plt.close(fig)
    print(f"\n{n_frames} images, {n_frames / a.fps:.1f} s")
    print(f"Video : {sortie}")


if __name__ == "__main__":
    main()
