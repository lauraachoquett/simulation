import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os 
import glob
from functools import reduce
from simulation.genealogy.genealogy import find_root

def make_depth_fn(node_parent):
    """g(node) = nb d'arêtes depuis la racine, mémoïsé. O(N) au total."""
    cache = {}
    def depth(node):
        stack, cur = [], node
        while cur is not None and cur not in cache:
            stack.append(cur)
            cur = node_parent.get(cur)
        d = cache[cur] if cur is not None else -1   # racine -> profondeur 0
        for n in reversed(stack):
            d += 1
            cache[n] = d
        return cache[node]
    return depth

def lca(a, b, node_parent, depth):
    """Plus proche ancêtre commun de deux nœuds (None si racines différentes)."""
    da, db = depth(a), depth(b)
    while da > db: a, da = node_parent.get(a), da - 1   # on égalise les profondeurs
    while db > da: b, db = node_parent.get(b), db - 1
    while a != b:                                        # puis on remonte ensemble
        a, b = node_parent.get(a), node_parent.get(b)
    return a                                             # a == b (le MRCA) ou None

def founder_lineages(leaves, node_parent):
    """Racines (fondateurs) distinctes dont descend la population actuelle."""
    return {find_root(l, node_parent) for l in leaves}

def find_mrca(leaves, node_parent, depth):
    def step(a, b):
        return None if (a is None or b is None) else lca(a, b, node_parent, depth)
    return reduce(step, leaves)

def current_leaves(outputs, node_parent):
    """Population vivante au dernier pas, sous forme de nœuds (idx, born_step)."""
    alive = np.array(outputs.alive)
    born  = np.array(outputs.born_step)
    T, N = alive.shape
    leaves, orphelins = [], []
    for idx in range(1, N):                 # idx 0 = sentinelle
        if alive[T-1, idx] == 1:
            node = (int(idx), int(born[T-1, idx]))
            if node in node_parent:
                leaves.append(node)
            else:
                orphelins.append(node)      # vivant mais sans arête -> anomalie

    if orphelins:
        raise ValueError(
            f"{len(orphelins)} feuille(s) vivante(s) absente(s) de node_parent "
            f"(arête de naissance perdue) : {orphelins[:10]}"
            f"{' …' if len(orphelins) > 10 else ''}"
        )
    return leaves

def coalescence_point(outputs, node_parent):
    depth  = make_depth_fn(node_parent)
    leaves = current_leaves(outputs, node_parent)
    if not leaves:
        return None

    roots        = founder_lineages(leaves, node_parent)
    n_fondateurs = len(roots)

    if n_fondateurs > 1:                       # pas de coalescence complète
        return {
            "coalesced":    False,
            "tmrca_generations":  None,
            "n_fondateurs": n_fondateurs,
            "fondateurs":   sorted(roots),     # les nœuds-racines eux-mêmes
            "n_lineages":   len(leaves),
            "mrca":         None,
        }

    # n_fondateurs == 1  ->  un MRCA existe forcément
    mrca   = find_mrca(leaves, node_parent, depth)
    g_mrca = depth(mrca)
    tmrca  = [depth(l) - g_mrca for l in leaves]
    born   = np.array(outputs.born_step)
    T = born.shape[0]
    return {
        "coalesced":          True,
        "n_fondateurs":       1,
        "mrca":               mrca,
        "n_lineages":         len(leaves),
        "generation_du_mrca": g_mrca,
        "tmrca_generations":  max(tmrca),
        "tmrca_min":          min(tmrca),
        "tmrca_moyen":        sum(tmrca) / len(tmrca),
        "tmrca_pas_de_simu":  int(np.asarray(outputs.step)[-1]) - int(mrca[1]),
    }
    
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_tmrca_gen(pop_history, tmrca_series, exp_dir, t_points=None,
                   filename="tmrca_gen.png", tmrca_pas=None, permutations=None):
    """TMRCA en pas de simulation, et en generations sur l'axe de droite.

    Les deux ne disent pas la meme chose : les pas donnent l'age du MRCA, les
    generations le nombre d'evenements de reproduction enchainés depuis lui,
    donc la vitesse de reproduction de la lignee la plus rapide.
    """
    gen = np.array([np.nan if v is None else v for v in tmrca_series], dtype=float)
    pas = (np.array([np.nan if v is None else v for v in tmrca_pas], dtype=float)
           if tmrca_pas is not None else np.full(len(gen), np.nan))
    n_pas_total = np.asarray(pop_history).shape[0]

    if t_points is not None:
        x = np.asarray(t_points, dtype=float)
    elif len(gen) > 1:
        x = np.linspace(0, n_pas_total - 1, len(gen))
    else:
        x = np.array([n_pas_total - 1], dtype=float)

    mask = np.isfinite(gen) | np.isfinite(pas)
    if not mask.any():
        return None                       # jamais coalescé : rien à tracer
    first = np.argmax(mask)               # premier point ou un TMRCA existe
    x, gen, pas = x[first:], gen[first:], pas[first:]

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    if np.isfinite(pas).any():
        ax1.plot(x, pas, color="tab:blue", marker=".", lw=1.4,
                 label="TMRCA (steps)")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("TMRCA (steps)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(alpha=.3)

    ax2 = ax1.twinx()
    ax2.plot(x, gen, color="tab:red", marker=".", lw=1.2, alpha=.85,
             label="TMRCA (generations)")
    ax2.set_ylabel("TMRCA (generations)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    for k, pas_perm in enumerate(permutations or []):
        if x[0] <= pas_perm <= x[-1]:
            ax1.axvline(pas_perm, color="0.35", ls=(0, (4, 3)), lw=1.1, zorder=0,
                        label="channel permutation" if k == 0 else None)

    lignes = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lignes, [l.get_label() for l in lignes], loc="upper left", fontsize=8)
    fig.tight_layout()
    os.makedirs(exp_dir, exist_ok=True)
    path = os.path.join(exp_dir, 'fig',filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path