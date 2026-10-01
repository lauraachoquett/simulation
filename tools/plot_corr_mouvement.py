"""Correlation entre P(manger | en vue) d'une identite et le mouvement par pas.

    python -m simulation.tools.plot_corr_mouvement exp/2026-09-17/2026-09-17_17-19-22
    python -m simulation.tools.plot_corr_mouvement <dir> --identite good --bloc 5000

Les deux series viennent de data/chunk_*.npz : n_seen et n_eaten_seen par CANAL,
et mean_movement. Les permutations changent l'identite portee par un canal, donc
on suit resource_shuffles.jsonl pour lire la bonne colonne a chaque pas.
Rien n'est rejoue.
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = ("good", "medium", "poison")
_NUM = re.compile(r"chunk_(\d+)\.npz$")


def chaine_de_reprise(exp_dir):
    """[ancetres..., exp_dir] en remontant resume_from, sans importer jax."""
    chaine, vus = [], set()
    courant = os.path.abspath(exp_dir)
    while courant and courant not in vus:
        vus.add(courant)
        chaine.append(courant)
        f = os.path.join(courant, "config.json")
        if not os.path.exists(f):
            break
        parent = json.load(open(f)).get("resume_from") or ""
        if not parent or not os.path.isdir(parent):
            break
        courant = parent
    return list(reversed(chaine))


def ordre_initial(exp_dir):
    f = os.path.join(exp_dir, "config.json")
    cfg = json.load(open(f)) if os.path.exists(f) else {}
    res = cfg.get("resources") or []
    ids = [r.get("id", k) if isinstance(r, dict) else k for k, r in enumerate(res)]
    return ids or [0, 1, 2], int(cfg.get("chunk_size", 1000))


def permutations(exp_dir):
    """[(step, order_ids)] : identite portee par chaque canal apres la bascule."""
    f = os.path.join(exp_dir, "resource_shuffles.jsonl")
    out = []
    if os.path.exists(f):
        for ligne in open(f):
            if ligne.strip():
                d = json.loads(ligne)
                out.append((int(d["step"]), list(d["order_ids"])))
    return sorted(out)


def series(dossiers, identite, ids0, bascules, taille):
    """(pas, p_manger, mouvement) alignes, un point par pas de simulation.

    mean_movement a un point de MOINS par chunk -- c'est une grandeur de
    transition -- donc l'alignement se fait chunk par chunk, pas apres
    concatenation.
    """
    par_chunk = {}
    for d in dossiers:
        for f in glob.glob(os.path.join(d, "data", "chunk_*.npz")):
            par_chunk[int(_NUM.search(os.path.basename(f)).group(1))] = f

    pas, p, mv = [], [], []
    for c in sorted(par_chunk):
        with np.load(par_chunk[c]) as z:
            if not {"n_seen", "n_eaten_seen", "mean_movement"} <= set(z.files):
                continue
            vu = np.asarray(z["n_seen"], float)
            mange = np.asarray(z["n_eaten_seen"], float)
            mouv = np.asarray(z["mean_movement"], float)
        n = min(len(vu), len(mouv))
        debut = (c - 1) * taille
        steps = debut + np.arange(n)
        # canal portant l'identite voulue, pas par pas
        canal = np.full(n, _canal(ids0, bascules, debut, identite), dtype=int)
        for s, ordre in bascules:
            if debut <= s < debut + n:
                canal[s - debut:] = ordre.index(identite)
        v = vu[np.arange(n), canal]
        m = mange[np.arange(n), canal]
        with np.errstate(invalid="ignore", divide="ignore"):
            pm = np.where(v > 0, m / v, np.nan)
        pas.append(steps), p.append(pm), mv.append(mouv[:n])
    if not pas:
        return None
    return (np.concatenate(pas), np.concatenate(p), np.concatenate(mv))


def _canal(ids0, bascules, step, identite):
    """Canal portant l'identite voulue juste avant `step`."""
    ordre = ids0
    for s, o in bascules:
        if s <= step:
            ordre = o
    return ordre.index(identite)


def blocs(x, y, z, taille):
    """Moyennes par blocs de `taille` pas : un point de simulation est bruite."""
    n = (len(x) // taille) * taille
    if n == 0:
        return x, y, z
    f = lambda a: np.nanmean(a[:n].reshape(-1, taille), axis=1)
    with np.errstate(invalid="ignore"):
        return f(x), f(y), f(z)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("source")
    p.add_argument("--identite", default="poison", choices=list(LABELS))
    p.add_argument("--bloc", type=int, default=1000,
                   help="moyenner par blocs de N pas (defaut %(default)s)")
    p.add_argument("--depuis", type=int, default=0, help="premier pas garde")
    p.add_argument("--jusqu", type=int, default=0, help="dernier pas garde")
    p.add_argument("--no-chaine", dest="no_chaine", action="store_true")
    p.add_argument("--no-titre", dest="no_titre", action="store_true")
    p.add_argument("--fig-format", dest="fig_format", default="png",
                   choices=["png", "pdf"])
    p.add_argument("-o", "--out", default=None)
    a = p.parse_args()

    ids0, taille = ordre_initial(a.source)
    ident = LABELS.index(a.identite)
    if ident not in ids0:
        raise SystemExit(f"identite {a.identite} absente de {ids0}")

    dossiers = ([a.source] if a.no_chaine else chaine_de_reprise(a.source))
    bascules = []
    for d in dossiers:
        bascules += permutations(d)
    s = series(dossiers, ident, ids0, sorted(bascules), taille)
    if s is None:
        raise SystemExit(f"pas de data/chunk_*.npz exploitable sous {a.source}")
    pas, pm, mv = s
    garde = (pas >= a.depuis) & ((pas <= a.jusqu) if a.jusqu else True)
    pas, pm, mv = pas[garde], pm[garde], mv[garde]
    x, y, z = blocs(pas, pm, mv, max(a.bloc, 1))
    ok = np.isfinite(y) & np.isfinite(z)
    x, y, z = x[ok], y[ok], z[ok]
    if len(x) < 3:
        raise SystemExit("moins de 3 points exploitables")

    r = float(np.corrcoef(z, y)[0, 1])
    rg = lambda v: np.argsort(np.argsort(v))
    rho = float(np.corrcoef(rg(z), rg(y))[0, 1])
    print(f"{len(x)} points de {a.bloc} pas | Pearson r = {r:+.3f} | "
          f"Spearman rho = {rho:+.3f}")

    fig, (g, d) = plt.subplots(1, 2, figsize=(12.6, 4.9),
                               gridspec_kw={"width_ratios": [1, 1.25]})
    sc = g.scatter(z, y, c=x / 1e6, cmap="viridis", s=18, alpha=.8,
                   edgecolors="none")
    # droite des moindres carres : elle dit le signe, pas une causalite
    co = np.polyfit(z, y, 1)
    xs = np.linspace(z.min(), z.max(), 50)
    g.plot(xs, np.polyval(co, xs), color="#C1121F", lw=1.6)
    g.set_xlabel("movement per step")
    g.set_ylabel(f"P(eat {a.identite} | {a.identite} in view)")
    g.grid(alpha=.3)
    g.set_title(f"r = {r:+.2f}   ρ = {rho:+.2f}", fontsize=11)
    cb = fig.colorbar(sc, ax=g)
    cb.set_label("simulation step (M)", fontsize=9)

    d.plot(x / 1e6, y, color="#9C27B0", lw=1.2,
           label=f"P(eat {a.identite} | in view)")
    d2 = d.twinx()
    d2.plot(x / 1e6, z, color="#2A9131", lw=1.2, label="movement per step")
    for s_, _ in sorted(bascules):      # permutations : l'identite change de canal
        if x.min() <= s_ <= x.max():
            d.axvline(s_ / 1e6, color="#8A8A8A", lw=.8, ls=(0, (4, 3)), zorder=0)
    d.set_xlabel("simulation step (M)")
    d.set_ylabel(f"P(eat {a.identite} | in view)", color="#9C27B0")
    d2.set_ylabel("movement per step", color="#2A9131")
    d.grid(alpha=.3)
    d.set_title(f"blocks of {a.bloc} steps, dashed = permutations", fontsize=11)

    if not a.no_titre:
        fig.suptitle(f"{os.path.basename(os.path.normpath(a.source))} — "
                     f"{a.identite} intake vs movement", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, .93 if not a.no_titre else 1])
    out = a.out or os.path.join(a.source, "fig",
                                f"corr_{a.identite}_mouvement.{a.fig_format}")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {out}")


if __name__ == "__main__":
    main()
