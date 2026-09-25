"""Ligne de descendance sur plusieurs experiences qui se suivent.

    python -m simulation.tools.lineage_global exp/<derniere_reprise>
    python -m simulation.tools.lineage_global exp/A exp/B exp/C -o exp/lignee_globale

Un seul dossier : la chaine est suivie via `resume_from`. Chaque experience doit
avoir ete evaluee par tools/lineage_lab.
"""
import argparse
import json
import os

import numpy as np

from simulation.tools.replot import chaine_de_reprise
from simulation.utils.plots import plot_lineage_simplex, plot_lod_metrics
from simulation.utils.utils_sim import build_id_timeline, load_shuffle_log


def config(exp_dir):
    f = os.path.join(exp_dir, "config.json")
    return json.load(open(f)) if os.path.exists(f) else {}


def ids_initiaux(cfg):
    return [int(r["id"]) for r in cfg.get("resources", [])] or [0, 1, 2]


def _largeur(v, n, S):
    """(n, S) complete par NaN : les experiences n'ont pas toutes le meme nombre de graines."""
    out = np.full((n, S), np.nan)
    if v is not None:
        out[:, :v.shape[1]] = v
    return out


def pas_de_reprise(d):
    cfg = config(d)
    return (int(cfg.get("resume_chunk", 0)) * int(cfg.get("chunk_size", 1000))
            if cfg.get("resume_from") else None)


def fusionne_graines(dossiers, fins):
    """lod/lab/graines_avant.npz des experiences, concatene et tronque de meme.

    Sans ca, la variabilite par graine n'existe pas sur une chaine fusionnee.
    """
    blocs, vus = [], set()
    for k, d in enumerate(dossiers):
        f = os.path.join(d, "lod", "lab", "graines_avant.npz")
        if not os.path.exists(f):
            continue
        with np.load(f) as z:
            e = {c: np.asarray(z[c]) for c in z.files}
        garde = []
        for i, cle in enumerate(zip(e["slot"].tolist(), e["born"].tolist())):
            if fins[k] is not None and e["born"][i] >= fins[k]:
                continue
            if cle not in vus:
                vus.add(cle)
                garde.append(i)
        if garde:
            blocs.append({c: (v[garde] if isinstance(v, np.ndarray)
                              and v.shape[:1] == e["born"].shape else v)
                          for c, v in e.items()})
    if not blocs:
        return None
    S = min(b["regime"].shape[1] for b in blocs)      # meme nombre de graines
    out = {"born": np.concatenate([b["born"] for b in blocs]),
           "slot": np.concatenate([b["slot"] for b in blocs]),
           "regime": np.concatenate([b["regime"][:, :S] for b in blocs]),
           "graines": blocs[0]["graines"][:S]}
    for c in ("age", "disponible"):
        dispo = [b[c] for b in blocs if c in b]
        if dispo:
            out[c] = (np.concatenate([d[:, :S] for d in dispo]) if c == "age"
                      else dispo[0][:S])
    ordre = np.argsort(out["born"])
    return {c: (v[ordre] if isinstance(v, np.ndarray) and v.shape[:1] == out["born"].shape
                else v) for c, v in out.items()}


def fusionne(dossiers):
    """Evaluations concatenees dans l'ordre de la chaine, doublons retires.

    Une experience reprise est TRONQUEE au pas de reprise de la suivante : si
    elle a continue au-dela, ses ancetres d'apres appartiennent a une branche
    divergente, pas a la lignee de la reprise.
    """
    blocs, journal, vus = [], [], set()
    fins = [pas_de_reprise(dossiers[k + 1]) if k + 1 < len(dossiers) else None
            for k in range(len(dossiers))]
    for k, d in enumerate(dossiers):
        f = os.path.join(d, "lod", "lab", "evaluation.npz")
        if not os.path.exists(f):
            raise SystemExit(f"{f} absent : lancer lineage_lab sur {d}")
        cfg, log = config(d), load_shuffle_log(d)
        ids0 = ids_initiaux(cfg)

        # Une reprise repart de l'ordre de SA config : celui du checkpoint depuis
        # la correction de --resume, l'ordre initial avant. Inscrit a la couture.
        if k > 0:
            debut = int(cfg.get("resume_chunk", 0)) * int(cfg.get("chunk_size", 1000))
            journal.append({"step": debut, "order_ids": ids0, "couture": True})
        journal += log

        with np.load(f) as z:
            e = {c: np.asarray(z[c]) for c in z.files}
        e["ordres"] = build_id_timeline(e["born"], log, ids0)
        fin = fins[k]
        n_coupes = int((e["born"] >= fin).sum()) if fin is not None else 0
        if n_coupes:
            print(f"    {n_coupes} ancetre(s) ne(s) apres le pas {fin} : branche "
                  "divergente, ecartee")
        garde = []
        for i, cle in enumerate(zip(e["slot"].tolist(), e["born"].tolist())):
            if fin is not None and e["born"][i] >= fin:
                continue
            if cle not in vus:
                vus.add(cle)
                garde.append(i)
        e = {c: (v[garde] if isinstance(v, np.ndarray) and v.shape[:1] == e["born"].shape
                 else v) for c, v in e.items()}
        e["experience"] = np.full(len(garde), k)
        blocs.append(e)
        print(f"  {os.path.basename(d)} : {len(garde)} ancetre(s)")

    colonnes = ("slot", "born", "regime", "age", "greediness", "mean_rew",
                "p_poison", "ordres", "experience")
    g = {c: np.concatenate([b[c] for b in blocs if c in b]) for c in colonnes
         if all(c in b for b in blocs)}
    for c in ("age_par_graine", "p_poison_par_graine"):
        S = max((b[c].shape[1] for b in blocs if c in b), default=0)
        if S:
            g[c] = np.concatenate([_largeur(b.get(c), len(b["born"]), S)
                                   for b in blocs])
    g["generation"] = np.arange(len(g["born"]))
    g["post_shuffle"] = np.zeros(len(g["born"]), bool)
    g["post_shuffle"][1:] = (g["ordres"][1:] != g["ordres"][:-1]).any(axis=1)
    dispo = next((b["disponible"] for b in blocs if "disponible" in b), None)
    return (g, sorted(journal, key=lambda e: e["step"]), dispo,
            fusionne_graines(dossiers, fins))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("exp_dirs", nargs="+", metavar="EXP_DIR")
    p.add_argument("-o", "--out", default=None,
                   help="dossier de sortie (defaut <derniere>/lod_global)")
    a = p.parse_args()

    dossiers = (chaine_de_reprise(a.exp_dirs[0]) if len(a.exp_dirs) == 1
                else [os.path.abspath(d) for d in a.exp_dirs])
    print(f"{len(dossiers)} experience(s) :")
    g, journal, dispo, graines = fusionne(dossiers)
    coutures = np.flatnonzero(np.diff(g["experience"])) + 1
    print(f"{len(g['born'])} ancetre(s) au total, {len(coutures)} couture(s), "
          f"{int(g['post_shuffle'].sum())} changement(s) de canaux")

    sortie = a.out or os.path.join(dossiers[-1], "lod_global")
    os.makedirs(os.path.join(sortie, "lod", "lab"), exist_ok=True)
    np.savez_compressed(os.path.join(sortie, "lod", "lab", "evaluation.npz"),
                        **g, **({"disponible": dispo} if dispo is not None else {}))
    if graines is not None:
        np.savez_compressed(os.path.join(sortie, "lod", "lab",
                                         "graines_avant.npz"), **graines)
        print(f"{len(graines['born'])} ancetre(s) d'avant permutation, "
              f"{graines['regime'].shape[1]} graines")
    # config.json et journal minimaux : video_lod lit ce dossier sans modification
    cfg0 = config(dossiers[0])
    json.dump({"resources": cfg0.get("resources", []),
               "lab_time_steps": cfg0.get("lab_time_steps"),
               "experiences": dossiers},
              open(os.path.join(sortie, "config.json"), "w"), indent=2)
    with open(os.path.join(sortie, "resource_shuffles.jsonl"), "w") as f:
        for e in journal:
            f.write(json.dumps(e) + "\n")

    fig_dir = os.path.join(sortie, "fig")
    ok = np.isfinite(g["regime"]).all(axis=1) & (g["regime"].sum(axis=1) > 0)
    chaine = [dict(regime=g["regime"][i], born=int(g["born"][i]),
                   post_shuffle=bool(g["post_shuffle"][i]))
              for i in np.flatnonzero(ok)]
    if len(chaine) >= 2:
        plot_lineage_simplex([chaine], dispo, sortie, "all", fig_dir=fig_dir,
                             titre="line of descent, all runs",
                             couverture=len(chaine) / len(g["born"]),
                             shuffle_log=journal, ids_initiaux=ids_initiaux(cfg0),
                             step=int(g["born"].max()))
    plot_lod_metrics(g["born"], g["age"], poison=g.get("p_poison"),
                     journal=journal, ids_initiaux=ids_initiaux(cfg0),
                     coutures=[e["step"] for e in journal if e.get("couture")],
                     generations=g["generation"],
                     duree_vie_s=g.get("age_par_graine"),
                     poison_s=g.get("p_poison_par_graine"), fig_dir=fig_dir)
    print(f"Sorties dans {sortie}")


if __name__ == "__main__":
    main()
