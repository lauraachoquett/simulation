"""Rejoue les environnements de lab depuis les checkpoints d'un run.

    python -m simulation.tools.replay_lab exp/2026-09-09/2026-09-09_11-19-39
    python -m simulation.tools.replay_lab <exp_dir> --chunks 100 200 --batch 16

Produit les trois figures d'evolution -- lab_metrics, lab_exploration,
lab_alone_vs_clones -- sur TOUTE la population vivante de CHAQUE checkpoint,
sans relancer la simulation.

Trois points de conception valent d'etre connus.

1. Toute la population, donc par LOTS. `launch_env` vmap sur les n genomes d'un
   coup ; a 1500 survivants la carte n'y suffit pas. On decoupe en lots de
   --batch et on recolle les sorties sur l'axe des genomes.

2. L'ordre des canaux est reconstruit AU PAS DU CHECKPOINT. cfg.resources change
   a chaque permutation, et config.json ne garde que l'ordre initial : rejouer un
   checkpoint tardif avec cet ordre ferait porter chaque metrique sur la mauvaise
   identite, sans que rien ne le signale.

3. Meme `key_env` pour tous les checkpoints, tiree de cfg.lab_seed. C'est ce
   qui rend la serie comparable -- grille de depart identique partout, seuls les
   genomes changent -- et ce qui aligne le rejeu sur les evaluations faites
   pendant le run.

L'env d'ADAPTATION n'est pas rejoue : il empile ses rotations et c'est lui qui
avait fait tomber le GPU en OOM. Les videos non plus, elles dominent le cout.
"""
import argparse
import glob
import os
import re

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from simulation.data_class import label_of
from simulation.lab_env import (env_file_pour,
                                vmap_over_agents_env_lab_high_res,
                                vmap_over_agents_env_lab_low_res,
                                vmap_over_agents_env_lab_high_res_with_clones,
                                vmap_over_agents_env_lab_high_res_with_figurants)
from simulation.run import build_model
from simulation.simulation_data.core import simulation_data
from simulation.utils.plots import (plot_lab_metrics, plot_lab_exploration,
                                    plot_alone_vs_clones)
from simulation.utils.utils_sim import (load_config, load_checkpoint,
                                        load_shuffle_log)


def checkpoints_de(exp_dir):
    """[(chunk, chemin)] tries par chunk."""
    motif = os.path.join(exp_dir, "checkpoints", "state_chunk_*.pkl")
    out = []
    for f in glob.glob(motif):
        m = re.search(r"state_chunk_(\d+)\.pkl$", f)
        if m:
            out.append((int(m.group(1)), f))
    return sorted(out)


def resources_au_pas(cfg, exp_dir, step):
    """cfg.resources tel qu'il etait quand le checkpoint du pas `step` a ete ecrit.

    Comparaison STRICTE (`< step`) et non `<=`, contrairement a
    build_id_timeline : dans run.py le checkpoint est sauve AVANT le bloc de
    shuffle du meme chunk, sur le meme state.step. Un checkpoint dont le pas
    egale celui d'une permutation porte donc l'ANCIEN ordre. Le cas tombe des
    que cycle_period divise checkpoint_freq -- tous les 50 chunks avec les
    valeurs par defaut, pas un cas de bord rare.
    """
    log = load_shuffle_log(exp_dir)
    if not log:
        return cfg.resources
    ordres = [[r.id for r in cfg.resources]] + [e["order_ids"] for e in log]
    i = sum(1 for e in log if e["step"] < int(step))
    par_id = {r.id: r for r in cfg.resources}
    return tuple(par_id[int(k)] for k in ordres[i])


def avertit_env_fige(cfg, exp_dir):
    """Prevenir quand le rejeu va jouer un env que le run d'origine n'a pas vu.

    Les config.json anterieurs aux environnements figes ne nomment aucun
    fichier, et la regle automatique s'applique alors au rejeu. On ne peut pas
    le deviner a leur place -- mais on peut refuser de le faire en silence.
    """
    if len(cfg.resources) != 1:
        return
    nomme = {"high_res": cfg.lab_env_high_res, "low_res": cfg.lab_env_low_res}
    for quel, actif in ((q, env_file_pour(cfg, q)) for q in nomme):
        if actif and not nomme[quel]:
            print(f"  [attention] {exp_dir} ne nomme aucun env {quel} : le rejeu "
                  f"utilisera {actif}, que le run d'origine n'a pas forcement joue.")


def par_lots(fn, params, key_env, cles, model, cfg, batch):
    """`fn` sur tous les genomes, par lots, recolles sur l'axe 0.

    numpy et non jnp pour le recollage : les sorties quittent de toute facon la
    carte pour l'agregation, et les garder en jnp ferait tenir tout le run en
    memoire GPU -- ce qu'on cherche precisement a eviter.
    """
    morceaux = []
    for deb in range(0, len(params), batch):
        _, out = fn(params[deb:deb + batch], key_env, cles[deb:deb + batch],
                    model, cfg)
        morceaux.append(jax.tree_util.tree_map(np.asarray, out))
    if len(morceaux) == 1:
        return morceaux[0]
    return jax.tree_util.tree_map(lambda *xs: np.concatenate(xs, axis=0),
                                  *morceaux)


def lab_data_de(chemin):
    """Le dossier lab_data d'un chemin, qu'on donne l'experience ou son replay.

    replay/ EN PREMIER, et c'est important : un dossier d'experience porte aussi
    son propre lab_data, rempli par les evaluations faites PENDANT le run. Celles-ci
    ne portent que sur les 50 premiers survivants et, sur les runs anterieurs a
    lab_seed, sur un environnement different a chaque graine. Les melanger au
    rejeu donnerait une serie incoherente sans que rien ne le signale.
    """
    for candidat in (os.path.join(chemin, "replay", "lab_data"),
                     os.path.join(chemin, "lab_data")):
        if os.path.isdir(candidat):
            return candidat
    return None


def fusionner(chemins, sortie):
    """Rassemble des lab_data deja produits, sans rien reevaluer.

    Les fichiers portent le numero de chunk dans leur nom : une simple copie
    dedoublonne donc d'elle-meme, et sur un chunk present des deux cotes le
    DERNIER chemin donne l'emporte. D'ou l'ordre chronologique.
    """
    import shutil
    dest = os.path.join(sortie, "lab_data")
    os.makedirs(dest, exist_ok=True)
    total = 0
    for chemin in chemins:
        src = lab_data_de(chemin)
        if src is None:
            print(f"  {chemin} : pas de lab_data/ ni de replay/lab_data/, ignore")
            continue
        n = 0
        for f in sorted(glob.glob(os.path.join(src, "*"))):
            if os.path.isfile(f):
                shutil.copy2(f, os.path.join(dest, os.path.basename(f)))
                n += 1
        print(f"  {src} : {n} fichier(s)")
        total += n
    if not total:
        print("Rien a fusionner.")
        return
    print(f"{len(os.listdir(dest))} fichier(s) apres dedoublonnage")
    tracer(sortie)


def tracer(sortie):
    # les fonctions de trace ecrivent dans <sortie>/fig sans le creer : en mode
    # fusion rien d'autre ne l'a fait avant
    os.makedirs(os.path.join(sortie, "fig"), exist_ok=True)
    plot_lab_metrics(exp_dir=sortie)
    # une figure par geometrie supplementaire. Les suffixes sont decouverts sur
    # disque -- le rejeu n'a ainsi pas besoin de connaitre les noms choisis pour
    # le run -- et dedoublonnes : il y a un fichier par chunk et par geometrie.
    suffixes = set()
    for f in glob.glob(os.path.join(sortie, "lab_data", "chunk_*_env_*_summary.json")):
        m = re.fullmatch(r"chunk_\d+_(env_.+)_summary\.json", os.path.basename(f))
        if m:
            suffixes.add(m.group(1))
    for suf in sorted(suffixes):
        plot_lab_metrics(exp_dir=sortie, suffix=suf)
    plot_lab_exploration(exp_dir=sortie)
    # Familles alone_vs_* decouvertes sur disque : elles portent le suffixe de
    # geometrie, et figurants n'est pas toujours joue.
    etiq = {"clones": "clones (median of peers)", "figurants": "with inert peers"}
    tags = set()
    for f in glob.glob(os.path.join(sortie, "lab_data", "chunk_*_alone_vs_*.json")):
        m = re.fullmatch(r"chunk_\d+_((?:env_.+_)?alone_vs_(\w+))\.json",
                         os.path.basename(f))
        if m:
            tags.add((m.group(1), m.group(2)))
    for tag, cond in sorted(tags):
        plot_alone_vs_clones(
            exp_dir=sortie, tag=tag, prefixes=("alone", cond),
            labels=("alone", etiq.get(cond, cond)),
            titre=f"Focal agent alone vs {cond}"
                  + (f"  [{tag[4:-len('_alone_vs_' + cond)]}]"
                     if tag.startswith("env_") else ""),
            fname=f"lab_{tag}_evolution.png")
    print(f"Figures dans {os.path.join(sortie, 'fig')}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("exp_dirs", nargs="+", metavar="EXP_DIR",
                   help="un ou plusieurs dossiers d'experience a rejouer")
    p.add_argument("--out", default=None,
                   help="dossier de sortie. Defaut <exp_dir>/replay, un par "
                        "experience. Avec --out et plusieurs experiences, un "
                        "sous-dossier par nom d'experience : sans ca elles "
                        "ecriraient toutes dans le meme lab_data et les "
                        "resumes se melangeraient")
    p.add_argument("--chunks", type=int, nargs="*", default=None,
                   help="checkpoints a rejouer (defaut : tous)")
    p.add_argument("--batch", type=int, default=25,
                   help="genomes par vmap (defaut %(default)s, cf. EVO_BATCH)")
    p.add_argument("-n", type=int, default=0,
                   help="genomes par checkpoint, 0 = toute la population")
    p.add_argument("--lab", dest="lab_time_steps", type=int, default=None,
                   help="duree du rollout (defaut : celle de la config)")
    p.add_argument("--merge-only", dest="merge_only", action="store_true",
                   help="ne RIEN reevaluer : fusionner des lab_data deja "
                        "produits et tracer. Les chemins donnes peuvent etre "
                        "les dossiers d'experience (leur replay/ est trouve "
                        "tout seul) ou les replay/ eux-memes")
    p.add_argument("--merge", action="store_true",
                   help="tout ecrire dans UN seul lab_data et ne tracer qu'une "
                        "serie : ce qu'il faut pour un run repris, dont la suite "
                        "vit dans un autre dossier. Passer les experiences dans "
                        "l'ordre CHRONOLOGIQUE -- sur un numero de chunk present "
                        "des deux cotes, la derniere donnee l'emporte")
    p.add_argument("--lab-seed", dest="lab_seed", type=int, default=None,
                   help="graine de l'env de lab (defaut : cfg.lab_seed, pour "
                        "que le rejeu tombe sur le MEME etalon que le run)")
    a = p.parse_args()

    if a.merge_only:
        fusionner(a.exp_dirs, a.out or os.path.join(a.exp_dirs[0], "replay_merge"))
        return

    for exp_dir in a.exp_dirs:
        print(f"\n=== {exp_dir}")
        cfg, _ = load_config(exp_dir)
        avertit_env_fige(cfg, exp_dir)
        if a.lab_time_steps:
            cfg = cfg._replace(lab_time_steps=a.lab_time_steps)
        model = build_model(cfg)
        if a.merge:
            sortie = a.out or os.path.join(a.exp_dirs[0], "replay_merge")
        elif a.out and len(a.exp_dirs) > 1:
            sortie = os.path.join(a.out, os.path.basename(exp_dir.rstrip('/')))
        else:
            sortie = a.out or os.path.join(exp_dir, "replay")
        os.makedirs(sortie, exist_ok=True)

        ckpts = checkpoints_de(exp_dir)
        if a.chunks:
            garde = set(a.chunks)
            ckpts = [c for c in ckpts if c[0] in garde]
        if not ckpts:
            print(f"Aucun checkpoint dans {exp_dir}/checkpoints/")
            return
        print(f"{len(ckpts)} checkpoint(s) : {[c for c, _ in ckpts]}")

        # une seule cle d'env pour toute la serie : voir le point 3 de l'en-tete.
        # Elle vient de cfg.lab_seed, donc le rejeu note les genomes sur exactement
        # le meme etalon que les evaluations faites pendant le run.
        graine_lab = a.lab_seed if a.lab_seed is not None else cfg.lab_seed
        key_env = random.PRNGKey(graine_lab)
        cle = random.PRNGKey(graine_lab + 1)
        print(f"env de lab : graine {graine_lab}")

        sd = simulation_data(cfg, 0, 1)

        for chunk, _ in ckpts:
            state = load_checkpoint(exp_dir, chunk)
            step = int(state.step)
            res = resources_au_pas(cfg, exp_dir, step)
            cfg_c = cfg._replace(resources=res, log_grid=False)
            sd.cfg = cfg_c
            sd.chunk_idx = chunk

            survivants = sd.compute_survivors(state)
            if not survivants:
                print(f"  chunk {chunk:>5} (step {step}) : aucun survivant, saute")
                continue
            ids = np.array([i for i, _ in survivants])
            if a.n:
                ids = ids[:a.n]
            params = state.agents.params[ids]

            cle, k_sim = random.split(cle)
            cles = random.split(k_sim, len(ids))
            canaux = " ".join(label_of(r.id) for r in res)
            print(f"  chunk {chunk:>5} (step {step:>8}) : {len(ids)} genomes, "
                  f"canaux [{canaux}]", flush=True)

            # low_res : hors des geometries, il a sa propre grille
            out_low = par_lots(vmap_over_agents_env_lab_low_res,
                               params, key_env, cles, model, cfg_c, a.batch)
            agg_low, summary_low = sd.data_lab_env_low_res(out_low)
            sd._save_lab_data(agg_low, summary_low, sortie, suffix="lowres")

            # Une condition par geometrie, toutes de meme rang. Liste vide ->
            # une seule condition sans suffixe, c'est-a-dire le comportement
            # d'avant.
            geometries = (tuple(cfg.lab_envs)
                          if len(cfg_c.resources) == 1 and cfg.lab_envs
                          else (None,))
            for geo in geometries:
                stem = os.path.splitext(os.path.basename(geo))[0] if geo else ""
                sfx = f"env_{stem}" if geo else ""
                cfg_g = cfg_c if geo is None else cfg_c._replace(lab_env_high_res=geo)

                out_high = par_lots(vmap_over_agents_env_lab_high_res,
                                    params, key_env, cles, model, cfg_g, a.batch)
                agg, summary = sd.data_lab_env(out_high, resources=res)
                sd._save_lab_data(agg, summary, sortie, suffix=sfx)

                out_clo = par_lots(vmap_over_agents_env_lab_high_res_with_clones,
                                   params, key_env, cles, model, cfg_g, a.batch)
                sd.compare_alone_vs_clones(out_high, out_clo, sortie,
                                           condition="clones", suffix=sfx)

                if cfg.lab_figurants:
                    out_fig = par_lots(vmap_over_agents_env_lab_high_res_with_figurants,
                                       params, key_env, cles, model, cfg_g, a.batch)
                    sd.compare_alone_vs_clones(out_high, out_fig, sortie,
                                               condition="figurants", suffix=sfx)

        if not a.merge:
            tracer(sortie)

    if a.merge:
        # une seule fois, sur le lab_data commun : les fonctions de trace lisent
        # tous les chunk_*_summary.json du dossier et les trient par numero
        tracer(sortie)


if __name__ == "__main__":
    main()
