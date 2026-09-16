"""Evalue la ligne de descendance au lab, chaque ancetre sous la config de sa naissance.

    python -m simulation.tools.lineage_lab <exp_dir>
    python -m simulation.tools.lineage_lab <exp_dir> --lab 2000 --batch 100

lod/ porte les ancetres devenus communs a toute la population, avec leur pas de
naissance et leur genome. On les rejoue ici dans l'env high_res pour en tirer un
comportement -- regime alimentaire, duree de vie, P(manger | ressource en vue) --
plutot que d'accumuler ce qu'ils ont mange dans le monde, qui melange le genome
et la circonstance (ce qui passait a portee, la concurrence, ou ils avaient
marche).

Trois points de conception.

1. Chaque ancetre est evalue sous la configuration de canaux EN VIGUEUR A SA
   NAISSANCE, reconstruite depuis resource_shuffles.jsonl. Le prendre dans
   l'ordre courant ferait porter chaque sommet du simplex sur la mauvaise
   identite pour tous les ancetres anterieurs a la derniere permutation.

2. On groupe par ORDRE DE CANAUX et non par epoque : il n'existe que 3! = 6
   ordres, donc au plus 6 configurations de lab et 6 compilations, quel que soit
   le nombre de permutations traversees.

3. key_env vient de cfg.lab_seed, comme dans tools/replay_lab : les ancetres sont
   notes sur le meme etalon que toutes les autres series de lab du run.

La lignee est UNE chaine, pas un echantillon de lignees tirees au hasard parmi
les vivants : c'est la suite des individus par lesquels l'evolution est
effectivement passee.
"""
import argparse
import glob
import json
import os

import jax.numpy as jnp
import numpy as np
from jax import random

from simulation.genealogy.lod import charge_lignee
from simulation.lab_env import vmap_over_agents_env_lab_high_res
from simulation.run import build_model
from simulation.simulation_data.core import simulation_data
from simulation.simulation_data.lab import EVO_BATCH
from simulation.utils.plots import plot_lineage_simplex, plot_lod_metrics
from simulation.utils.utils_sim import (build_id_timeline, load_config,
                                        load_shuffle_log)


def charge_genomes(exp_dir):
    """{(slot, born): params} depuis les lod/params_chunk_*.npz."""
    genomes = {}
    for f in sorted(glob.glob(os.path.join(exp_dir, "lod", "params_chunk_*.npz"))):
        with np.load(f) as d:
            for s, b, p in zip(d["slot"], d["born"], d["params"]):
                genomes.setdefault((int(s), int(b)), p)
    return genomes


def dernier_chunk_fixe(exp_dir):
    """Chunk de la derniere fixation, pour dater la figure."""
    path = os.path.join(exp_dir, "lod", "lignee.jsonl")
    dernier = 0
    if os.path.exists(path):
        for ligne in open(path):
            if ligne.strip():
                dernier = int(json.loads(ligne)["chunk"])
    return dernier


def evalue(retenus, genomes, cfg, model, sd, key_env, cle, batch, exp_dir):
    """(regime par identite, metriques) pour chaque ancetre.

    Le regroupement par ordre de canaux est ce qui borne le nombre de
    compilations ; sans lui, chaque epoque traversee en declencherait une.
    """
    n_types = len(cfg.resources)
    par_id = {r.id: r for r in cfg.resources}
    naissances = np.array([b for _, b in retenus], dtype=np.int64)
    ordres = build_id_timeline(naissances, load_shuffle_log(exp_dir),
                               [r.id for r in cfg.resources])

    groupes = {}
    for i in range(len(retenus)):
        groupes.setdefault(tuple(int(x) for x in ordres[i]), []).append(i)
    print(f"{len(groupes)} configuration(s) de canaux a compiler")

    regime = np.full((len(retenus), n_types), np.nan)
    mesures = {k: np.full(len(retenus), np.nan)
               for k in ("age", "greediness", "mean_rew")}

    cfg_m = cfg._replace(log_grid=False)
    for ordre, idx in sorted(groupes.items()):
        cfg_g = cfg_m._replace(resources=tuple(par_id[i] for i in ordre))
        sd.cfg = cfg_g                      # _per_agent_metrics lit self.cfg
        noms = " ".join(str(i) for i in ordre)
        print(f"  canaux [{noms}] : {len(idx)} ancetre(s)", flush=True)
        for deb in range(0, len(idx), batch):
            lot = idx[deb:deb + batch]
            X = jnp.asarray(np.stack([genomes[retenus[i]] for i in lot]))
            cle, k = random.split(cle)
            _, out = vmap_over_agents_env_lab_high_res(
                X, key_env, random.split(k, len(lot)), model, cfg_g)

            mange = sd.eaten_by_type(out)            # (B, n_types) par CANAL
            for j, i in enumerate(lot):
                for k_canal, ident in enumerate(ordre):
                    regime[i, ident] = mange[j, k_canal]   # canal -> identite
            par_genome = sd.data_lab_env_grouped(out, resources=cfg_g.resources)
            for cle_m in mesures:
                mesures[cle_m][lot] = par_genome[cle_m]
    return regime, mesures


def disponible_par_identite(genomes, retenus, cfg, model, sd, key_env, cle):
    """Ce que la grille offre, par identite. Invariant d'une epoque a l'autre :
    les parametres de croissance voyagent avec l'identite lors d'une permutation."""
    n_types = len(cfg.resources)
    cle, k = random.split(cle)
    _, out = vmap_over_agents_env_lab_high_res(
        jnp.asarray(genomes[retenus[0]])[None], key_env, random.split(k, 1),
        model, cfg._replace(log_grid=True))
    par_canal = sd.available_by_type(out, n_types)
    dispo = np.zeros(n_types)
    for k_canal, r in enumerate(cfg.resources):
        dispo[r.id] = par_canal[k_canal]
    return dispo


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("exp_dir", help="dossier d'experience portant lod/")
    p.add_argument("--lab", dest="lab_time_steps", type=int, default=None,
                   help="duree du rollout (defaut : celle de la config)")
    p.add_argument("--batch", type=int, default=EVO_BATCH,
                   help="ancetres par vmap (defaut %(default)s)")
    p.add_argument("--lab-seed", dest="lab_seed", type=int, default=None,
                   help="graine de l'env de lab (defaut : cfg.lab_seed, pour "
                        "noter les ancetres sur le meme etalon que le run)")
    p.add_argument("--max", type=int, default=0, metavar="N",
                   help="n'evaluer que N ancetres, repartis regulierement le "
                        "long de la lignee (defaut 0 = tous)")
    p.add_argument("-o", "--out", default=None,
                   help="dossier des figures (defaut <exp_dir>/fig/lod)")
    a = p.parse_args()

    cfg, _ = load_config(a.exp_dir)
    if len(cfg.resources) != 3:
        raise SystemExit(
            f"{len(cfg.resources)} ressource(s) : le simplex des regimes en "
            "demande 3. Cette evaluation n'a d'objet que dans ce cas.")
    if a.lab_time_steps:
        cfg = cfg._replace(lab_time_steps=a.lab_time_steps)

    lignee = charge_lignee(a.exp_dir)
    if not lignee:
        raise SystemExit(
            f"{a.exp_dir} : pas de lod/lignee.jsonl. Ce fichier est ecrit "
            "pendant le run, par une simulation lancee avec le suivi de lignee.")

    genomes = charge_genomes(a.exp_dir)
    retenus = [n for n in lignee if n in genomes]
    couverture = len(retenus) / len(lignee)
    print(f"{len(lignee)} ancetre(s) dans la lignee, {len(retenus)} avec genome "
          f"({100 * couverture:.0f} %)")
    if len(retenus) < 2:
        raise SystemExit("moins de deux genomes retrouves : rien a tracer")

    if a.max and len(retenus) > a.max:
        pris = np.linspace(0, len(retenus) - 1, a.max).astype(int)
        retenus = [retenus[i] for i in sorted(set(pris))]
        print(f"--max {a.max} : {len(retenus)} ancetre(s) evalue(s), "
              "repartis le long de la lignee")

    model = build_model(cfg)
    sd = simulation_data(cfg, 0, 1)
    graine = a.lab_seed if a.lab_seed is not None else cfg.lab_seed
    key_env = random.PRNGKey(graine)
    cle = random.PRNGKey(graine + 1)
    print(f"env de lab : graine {graine}, {cfg.lab_time_steps} pas")

    regime, mesures = evalue(retenus, genomes, cfg, model, sd, key_env, cle,
                             a.batch, a.exp_dir)
    dispo = disponible_par_identite(genomes, retenus, cfg, model, sd, key_env, cle)

    # Une permutation entre la naissance de l'ancetre precedent et la sienne :
    # premiere generation d'une nouvelle epoque. Le precedent RETENU sert de
    # reference -- un ancetre sans genome, ou ecarte par --max, est enjambe.
    pas_shuffle = np.array([e["step"] for e in load_shuffle_log(a.exp_dir)],
                           dtype=np.int64)
    naissances = np.array([b for _, b in retenus], dtype=np.int64)
    post = np.zeros(len(retenus), dtype=bool)
    for i in range(1, len(retenus)):
        if pas_shuffle.size:
            post[i] = ((pas_shuffle > naissances[i - 1])
                       & (pas_shuffle <= naissances[i])).any()

    data_dir = os.path.join(a.exp_dir, "lod", "lab")
    os.makedirs(data_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(data_dir, "evaluation.npz"),
        slot=np.array([s for s, _ in retenus], dtype=np.int32),
        born=naissances, generation=np.arange(len(retenus)),
        regime=regime, post_shuffle=post,
        **{k: v for k, v in mesures.items()})
    print(f"Donnees : {os.path.join(data_dir, 'evaluation.npz')}")

    # Un ancetre n'ayant rien mange au lab est ABSENT de la chaine : sa position
    # dans le simplex est indefinie, pas nulle. Le trait l'enjambe.
    chaine = [dict(regime=regime[i], born=int(naissances[i]),
                   post_shuffle=bool(post[i]))
              for i in range(len(retenus))
              if np.isfinite(regime[i]).all() and regime[i].sum() > 0]
    fig_dir = a.out or os.path.join(a.exp_dir, "fig", "lod")
    if len(chaine) >= 2:
        plot_lineage_simplex([chaine], dispo, a.exp_dir,
                             dernier_chunk_fixe(a.exp_dir),
                             couverture=len(chaine) / len(retenus),
                             fig_dir=fig_dir, titre="line of descent")
    else:
        print("Simplex : moins de deux ancetres ont mange, rien a tracer")

    plot_lod_metrics(np.arange(len(retenus)), naissances, mesures["age"],
                     mesures["greediness"], post, fig_dir=fig_dir)


if __name__ == "__main__":
    main()
