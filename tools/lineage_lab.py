"""Evalue la ligne de descendance au lab, chaque ancetre sous la config de sa naissance.

    python -m simulation.tools.lineage_lab <exp_dir>
    python -m simulation.tools.lineage_lab <exp_dir> --lab 2000 --batch 100
"""
import argparse
import glob
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from simulation.data_class import label_of
from simulation.genealogy.lod import charge_lignee
from simulation.lab_env import vmap_over_agents_env_lab_high_res
from simulation.run import build_model
from simulation.simulation_data.core import simulation_data
from simulation.simulation_data.energy_response import resource_in_view
from simulation.simulation_data.lab import EVO_BATCH, _greediness
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


def proba_poison(out, resources):
    """P(manger du poison | poison en vue), par fenetres de GREED_WINDOW pas.

    Passe par ate_res et non par rewards > 0 : le poison rapporte un gain
    negatif, la greediness ne le verrait donc jamais.
    """
    ch = [k for k, r in enumerate(resources) if label_of(r.id) == "poison"]
    if not ch:
        return None
    n_channels = len(resources) + 2
    G = np.full(out.alive.shape[0], np.nan)
    for b in range(out.alive.shape[0]):
        single = jax.tree_util.tree_map(lambda x: x[b], out)
        alive = np.asarray(single.alive)
        vivants = np.nonzero(alive[0, 1:] == 1)[0]
        lignes = np.nonzero(alive[:, int(vivants[0]) + 1] == 1)[0] if vivants.size else []
        if not len(lignes):
            continue
        slot = int(vivants[0]) + 1
        saw = resource_in_view(single.obs, np.array(ch), n_channels)
        ate = np.asarray(single.ate_res)[..., ch[0]] > 0
        g, _, _ = _greediness(saw, ate, np.array([slot]), np.array([0]),
                              np.array([int(lignes[-1])]))
        G[b] = g[0]
    return G


def evalue(retenus, genomes, cfg, model, sd, key_env, cle, batch, exp_dir):
    """(regime par identite, metriques) pour chaque ancetre."""
    n_types = len(cfg.resources)
    par_id = {r.id: r for r in cfg.resources}
    naissances = np.array([b for _, b in retenus], dtype=np.int64)
    ordres = build_id_timeline(naissances, load_shuffle_log(exp_dir),
                               [r.id for r in cfg.resources])

    # Grouper par ORDRE DE CANAUX et non par epoque : 6 ordres possibles, donc
    # au plus 6 compilations quel que soit le nombre de permutations.
    groupes = {}
    for i in range(len(retenus)):
        groupes.setdefault(tuple(int(x) for x in ordres[i]), []).append(i)
    print(f"{len(groupes)} configuration(s) de canaux a compiler")

    regime = np.full((len(retenus), n_types), np.nan)
    mesures = {k: np.full(len(retenus), np.nan)
               for k in ("age", "greediness", "mean_rew", "p_poison")}

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

            mange = sd.eaten_by_type(out)            # (B, n_types) par canal
            for j, i in enumerate(lot):
                for k_canal, ident in enumerate(ordre):
                    regime[i, ident] = mange[j, k_canal]
            par_genome = sd.data_lab_env_grouped(out, resources=cfg_g.resources)
            for cle_m in ("age", "greediness", "mean_rew"):
                mesures[cle_m][lot] = par_genome[cle_m]
            pp = proba_poison(out, cfg_g.resources)
            if pp is not None:
                mesures["p_poison"][lot] = pp
    return regime, mesures


def disponible_par_identite(genomes, retenus, cfg, model, sd, key_env, cle):
    """Ce que la grille offre, par identite.

    Un seul point de reference : les parametres de croissance voyagent avec
    l'identite, donc l'offre par identite ne change pas d'une epoque a l'autre.
    """
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
                   help="graine de l'env de lab (defaut : cfg.lab_seed, le meme "
                        "etalon que les autres series du run)")
    p.add_argument("--max", type=int, default=0, metavar="N",
                   help="n'evaluer que N ancetres, repartis le long de la "
                        "lignee (defaut 0 = tous)")
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
        print(f"--max {a.max} : {len(retenus)} ancetre(s) evalue(s)")

    model = build_model(cfg)
    sd = simulation_data(cfg, 0, 1)
    graine = a.lab_seed if a.lab_seed is not None else cfg.lab_seed
    key_env = random.PRNGKey(graine)
    cle = random.PRNGKey(graine + 1)
    print(f"env de lab : graine {graine}, {cfg.lab_time_steps} pas")

    regime, mesures = evalue(retenus, genomes, cfg, model, sd, key_env, cle,
                             a.batch, a.exp_dir)
    dispo = disponible_par_identite(genomes, retenus, cfg, model, sd, key_env, cle)

    # Changement REEL de l'affectation canal -> identite entre deux ancetres
    # retenus : deux permutations entre deux naissances peuvent se compenser, et
    # il n'y a alors rien a voir.
    naissances = np.array([b for _, b in retenus], dtype=np.int64)
    ordres = build_id_timeline(naissances, load_shuffle_log(a.exp_dir),
                               [r.id for r in cfg.resources])
    post = np.zeros(len(retenus), dtype=bool)
    post[1:] = (ordres[1:] != ordres[:-1]).any(axis=1)

    data_dir = os.path.join(a.exp_dir, "lod", "lab")
    os.makedirs(data_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(data_dir, "evaluation.npz"),
        slot=np.array([s for s, _ in retenus], dtype=np.int32),
        born=naissances, generation=np.arange(len(retenus)),
        regime=regime, post_shuffle=post,
        **{k: v for k, v in mesures.items()})
    print(f"Donnees : {os.path.join(data_dir, 'evaluation.npz')}")

    # Un ancetre n'ayant rien mange est absent de la chaine : position indefinie,
    # pas nulle. Le trait l'enjambe.
    chaine = [dict(regime=regime[i], born=int(naissances[i]),
                   post_shuffle=bool(post[i]))
              for i in range(len(retenus))
              if np.isfinite(regime[i]).all() and regime[i].sum() > 0]
    fig_dir = a.out or os.path.join(a.exp_dir, "fig", "lod")
    if len(chaine) >= 2:
        plot_lineage_simplex([chaine], dispo, a.exp_dir,
                             dernier_chunk_fixe(a.exp_dir),
                             couverture=len(chaine) / len(retenus),
                             fig_dir=fig_dir, titre="line of descent",
                             shuffle_log=load_shuffle_log(a.exp_dir),
                             ids_initiaux=[r.id for r in cfg.resources],
                             step=int(naissances[-1]))
    else:
        print("Simplex : moins de deux ancetres ont mange, rien a tracer")

    plot_lod_metrics(np.arange(len(retenus)), naissances, mesures["age"], post,
                     poison=mesures["p_poison"], ordres=ordres, fig_dir=fig_dir)


if __name__ == "__main__":
    main()
