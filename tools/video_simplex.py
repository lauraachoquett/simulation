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
import os

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from jax import random

from simulation.data_class import LABELS, label_of
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


def frame(fig, chaine, dispo, resources, step, pas_shuffle, bornes, fantome,
          trainee):
    """Compose une frame et rend (contour, barycentre) pour la suivante."""
    fig.clear()
    ax = fig.add_axes([0.03, 0.13, 0.94, 0.81])
    _cadre_simplex(ax)
    # marge elargie : les etiquettes des sommets sont posees HORS des limites
    # que pose _cadre_simplex, et se faisaient couper
    ax.set_xlim(-0.16, 1.16)
    ax.set_ylim(-0.13, np.sqrt(3) / 2 + 0.11)
    ligne_equilibre(ax, resources, color="0.25", lw=1.6, label=False, zorder=2)

    p = composition(chaine)
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
                        lw=1.6, alpha=0.15 + 0.85 * i / max(len(t) - 2, 1),
                        zorder=4)

        ax.scatter(x, y, s=46, color="#1D3557", alpha=.72, edgecolor="white",
                   linewidth=.5, zorder=5)
        contour = contour_kde(ax, x, y, colors="#1D3557", linewidths=2.0,
                              linestyles="--", zorder=6)
        barycentre = (float(x.mean()), float(y.mean()))
        ax.scatter([barycentre[0]], [barycentre[1]], s=150, marker="X",
                   color="#C1121F", edgecolor="white", linewidth=1.0, zorder=7)

    if dispo is not None and np.sum(dispo) > 0:
        d = np.asarray(dispo, float)
        xd, yd = _bary(*(d / d.sum()))
        ax.scatter([xd], [yd], marker="o", s=200, facecolor="none",
                   edgecolor="black", linewidth=2.0, zorder=8)

    ax.set_title(f"Diet composition — step {step:,}   ({len(p)} genomes)",
                 fontsize=13)

    # frise : ou l'on en est, et ou sont tombees les permutations
    fr = fig.add_axes([0.10, 0.05, 0.80, 0.03])
    fr.set_xlim(bornes); fr.set_ylim(0, 1)
    fr.set_yticks([]); fr.spines[["left", "right", "top"]].set_visible(False)
    fr.axhspan(0, 1, color="0.92")
    for sh in pas_shuffle:
        if bornes[0] <= sh <= bornes[1]:
            fr.axvline(sh, color="#B5651D", lw=1.6)
    fr.axvline(step, color="#C1121F", lw=2.6)
    fr.set_xlabel("Simulation step", fontsize=9)
    fr.tick_params(labelsize=8)

    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    return img, contour, barycentre


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("exp_dirs", nargs="+", metavar="EXP_DIR",
                   help="dossier(s) d'experience, dans l'ordre chronologique")
    p.add_argument("-o", "--out", default=None,
                   help="fichier .mp4 (defaut <exp_dir>/videos/simplex.mp4)")
    p.add_argument("-n", type=int, default=50, help="genomes par frame")
    p.add_argument("--batch", type=int, default=25)
    p.add_argument("--fps", type=int, default=6)
    p.add_argument("--pause", type=int, default=4,
                   help="frames tenues apres une permutation")
    p.add_argument("--trainee", type=int, default=12,
                   help="longueur de la trainee du barycentre")
    p.add_argument("--lab", dest="lab_time_steps", type=int, default=None)
    p.add_argument("--lab-seed", dest="lab_seed", type=int, default=None)
    a = p.parse_args()

    cfg, _ = load_config(a.exp_dirs[0])
    if a.lab_time_steps:
        cfg = cfg._replace(lab_time_steps=a.lab_time_steps)
    if len(cfg.resources) != 3:
        print(f"video_simplex : {len(cfg.resources)} ressource(s). Le simplex en "
              "demande 3.")
        return
    model = build_model(cfg)

    # tous les checkpoints des dossiers donnes, dedoublonnes par PAS et non par
    # numero de chunk : sur les runs repris avant a7a31a3 les deux divergent
    trouves = {}
    for d in a.exp_dirs:
        for chunk, _ in checkpoints_de(d):
            st = load_checkpoint(d, chunk)
            trouves[int(st.step)] = (d, chunk)
    if not trouves:
        print("Aucun checkpoint.")
        return
    pas_tries = sorted(trouves)
    print(f"{len(pas_tries)} frame(s), pas {pas_tries[0]:,} a {pas_tries[-1]:,}")

    graine = a.lab_seed if a.lab_seed is not None else cfg.lab_seed
    key_env = random.PRNGKey(graine)
    cle = random.PRNGKey(graine + 1)

    shuffle_log = load_shuffle_log(a.exp_dirs[0])
    pas_shuffle = [e["step"] for e in shuffle_log]
    bornes = (pas_tries[0], pas_tries[-1])

    sd = simulation_data(cfg, 0, 1)
    sortie = a.out or os.path.join(a.exp_dirs[0], "videos", "simplex.mp4")
    os.makedirs(os.path.dirname(sortie) or ".", exist_ok=True)

    # rapport cale sur celui des limites du triangle elargies : sinon l'aspect
    # egal comprime le dessin et laisse une bande morte
    fig = plt.figure(figsize=(8.2, 7.6), dpi=120)
    fantome, trainee, dispo = None, [], None

    with VideoWriter(sortie, fps=a.fps) as vid:
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
            mange = sd.eaten_by_type(out)              # (M, n_types) par CANAL

            # canal -> identite : les sommets sont des IDENTITES, une permutation
            # ne doit pas faire tourner le triangle
            par_identite = np.zeros_like(mange)
            for k_canal, r in enumerate(res):
                par_identite[:, r.id] = mange[:, k_canal]

            if dispo is None:                          # une seule fois : la
                cle, kg = random.split(cle)            # disponibilite par
                _, og = vmap_over_agents_env_lab_high_res(   # identite ne change
                    state.agents.params[ids[:1]], key_env,   # pas d'une epoque
                    random.split(kg, 1), model,              # a l'autre
                    cfg_c._replace(log_grid=True))
                dc = sd.available_by_type(og, len(res))
                dispo = np.zeros(len(res))
                for k_canal, r in enumerate(res):
                    dispo[r.id] = dc[k_canal]

            img, contour, bary = frame(fig, par_identite, dispo, res, step,
                                       pas_shuffle, bornes, fantome, trainee)
            vid.add(img)

            # une permutation vient-elle de tomber ? on tient la frame, et on
            # garde le contour d'avant en fantome pour la suivante
            juste_apres = any(step - (bornes[1] - bornes[0]) / len(pas_tries)
                              < s <= step for s in pas_shuffle)
            if juste_apres:
                for _ in range(a.pause):
                    vid.add(img)

            fantome = contour if juste_apres else None
            if bary is not None:
                trainee.append(bary)
                trainee[:] = trainee[-a.trainee:]
            print(f"  step {step:>9,} : {len(ids)} genomes"
                  + ("   <- permutation" if juste_apres else ""), flush=True)

    plt.close(fig)
    print(f"\nVideo : {sortie}")


if __name__ == "__main__":
    main()
