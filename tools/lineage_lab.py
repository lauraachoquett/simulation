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
from simulation.lab_env import vmap_mutate, vmap_over_agents_env_lab_high_res
from simulation.run import build_model
from simulation.simulation_data.core import simulation_data
from simulation.simulation_data.energy_response import resource_in_view
from simulation.simulation_data.lab import EVO_BATCH, _greediness
from simulation.utils.plots import (plot_food_simplex, plot_lineage_simplex,
                                    plot_lod_metrics)
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


def evalue_enfants(enfants, parent, ordre, cfg, model, sd, key_env, cle, batch):
    """(bouchees par canal, duree de vie, bouchees du parent) sous l'ordre `ordre`."""
    par_id = {r.id: r for r in cfg.resources}
    cfg_g = cfg._replace(log_grid=False, resources=tuple(par_id[i] for i in ordre))
    sd.cfg = cfg_g
    mange, age = [], []
    for deb in range(0, len(enfants), batch):
        lot = enfants[deb:deb + batch]
        cle, k = random.split(cle)
        _, out = vmap_over_agents_env_lab_high_res(
            lot, key_env, random.split(k, len(lot)), model, cfg_g)
        mange.append(sd.eaten_by_type(out))
        age.append(sd.data_lab_env_grouped(out, resources=cfg_g.resources)["age"])
    cle, k = random.split(cle)
    _, out_p = vmap_over_agents_env_lab_high_res(
        parent[None], key_env, random.split(k, 1), model, cfg_g)
    return np.concatenate(mange), np.concatenate(age), sd.eaten_by_type(out_p)[0]


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
                   help="premiere graine de l'env de lab (defaut : cfg.lab_seed)")
    p.add_argument("--graines", type=int, default=3,
                   help="environnements de lab testes, graines consecutives "
                        "(defaut %(default)s)")
    p.add_argument("--offspring", type=int, default=0, metavar="N",
                   help="pour chaque ancetre ne juste avant une permutation, "
                        "evaluer N mutants avant et apres (defaut 0 = non)")
    p.add_argument("--permutations", type=int, nargs="+", default=None,
                   metavar="K", help="rangs des permutations a traiter avec "
                                     "--offspring (defaut : toutes)")
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
    base = a.lab_seed if a.lab_seed is not None else cfg.lab_seed
    graines = [base + k for k in range(max(1, a.graines))]
    key_env = random.PRNGKey(graines[0])
    cle = random.PRNGKey(base + 1000)
    print(f"env de lab : graines {graines}, {cfg.lab_time_steps} pas")

    par_graine = [evalue(retenus, genomes, cfg, model, sd, random.PRNGKey(g), cle,
                         a.batch, a.exp_dir) for g in graines]
    regime_s = np.stack([r for r, _ in par_graine], axis=1)          # (n, S, 3)
    mesures_s = {k: np.stack([m[k] for _, m in par_graine], axis=1)
                 for k in par_graine[0][1]}                          # (n, S)
    regime = np.nansum(regime_s, axis=1)        # bouchees cumulees sur les graines
    with np.errstate(all="ignore"):
        mesures = {k: np.nanmean(v, axis=1) for k, v in mesures_s.items()}
    dispo_s = np.stack([disponible_par_identite(genomes, retenus, cfg, model, sd,
                                                random.PRNGKey(g), cle)
                        for g in graines])
    dispo = dispo_s.mean(axis=0)

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
        regime=regime, post_shuffle=post, disponible=dispo,
        graines=np.array(graines), regime_par_graine=regime_s,
        disponible_par_graine=dispo_s,
        **{k: v for k, v in mesures.items()},
        **{f"{k}_par_graine": v for k, v in mesures_s.items()})
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

    plot_lod_metrics(naissances, mesures["age"], poison=mesures["p_poison"],
                     journal=load_shuffle_log(a.exp_dir),
                     ids_initiaux=[r.id for r in cfg.resources],
                     generations=np.arange(len(retenus)),
                     duree_vie_s=mesures_s["age"], poison_s=mesures_s["p_poison"],
                     fig_dir=fig_dir)

    if a.offspring > 0:
        journal, ids0 = load_shuffle_log(a.exp_dir), [r.id for r in cfg.resources]
        rangs = list(np.flatnonzero(post))      # premier ancetre d'une nouvelle epoque
        if a.permutations:
            rangs = [rangs[k] for k in a.permutations if k < len(rangs)]
        print(f"descendance : {len(rangs)} permutation(s), {a.offspring} mutants")
        for i in rangs:
            parent = jnp.asarray(genomes[retenus[i - 1]])
            cle, k_mut = random.split(cle)
            enfants = vmap_mutate(parent, random.split(k_mut, a.offspring), cfg)
            # memes enfants, memes cles : avant et apres sont apparies
            for nom, ordre in (("before", ordres[i - 1]), ("after", ordres[i])):
                ordre = [int(x) for x in ordre]
                mange, age, mange_p = evalue_enfants(
                    enfants, parent, ordre, cfg, model, sd, key_env, cle, a.batch)
                plot_food_simplex(
                    mange, ordre, age, [dispo[j] for j in ordre], a.exp_dir,
                    int(naissances[i - 1]) // cfg.chunk_size,
                    suffix=f"_offspring_{nom}",
                    titre=f"offspring, {nom} the permutation",
                    fig_dir=os.path.join(fig_dir, "offspring"), parent=mange_p,
                    age_max=cfg.lab_time_steps, shuffle_log=journal,
                    ids_initiaux=ids0, step=int(naissances[i - 1]))


if __name__ == "__main__":
    main()
