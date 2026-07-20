import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os 
import glob
from functools import reduce

def find_root(node, node_parent):
    seen = set()
    while node_parent.get(node) is not None and node not in seen:
        seen.add(node)
        node = node_parent[node]
    return node


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
        "tmrca_pas_de_simu":  (T - 1) - mrca[1],
    }
    
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_tmrca_gen(pop_history, tmrca_series, exp_dir,
                   t_points=None, filename="tmrca_gen.png"):
    pop   = np.asarray(pop_history)
    tmrca = np.array([np.nan if v is None else v for v in tmrca_series], dtype=float)

    t_pop = np.arange(pop.shape[0])
    if t_points is not None:
        t_tmrca = np.asarray(t_points, dtype=float)
    elif len(tmrca) > 1:
        t_tmrca = np.linspace(0, pop.shape[0] - 1, len(tmrca))
    else:
        t_tmrca = np.array([pop.shape[0] - 1], dtype=float)

    # --- commencer au premier TMRCA non-None ---
    mask = np.isfinite(tmrca)
    if not mask.any():
        return None                       # jamais coalescé : rien à tracer
    first   = np.argmax(mask)             # premier indice où tmrca existe
    t_start = t_tmrca[first]
    t_tmrca, tmrca = t_tmrca[first:], tmrca[first:]
    keep = t_pop >= t_start               # on coupe aussi la population avant t_start
    t_pop, pop = t_pop[keep], pop[keep]

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    if pop.ndim == 1:
        ax1.plot(t_pop, pop, color="tab:gray", lw=1, label="N")
    else:
        for s in range(pop.shape[1]):
            ax1.plot(t_pop, pop[:, s], lw=1, label=f"espèce {s}")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("population size N")
    ax1.legend(loc="upper left", fontsize=8)

    ax2 = ax1.twinx()
    ax2.plot(t_tmrca, tmrca, color="tab:red", marker=".", lw=1.2, label="TMRCA")
    ax2.set_ylabel("TMRCA", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    fig.tight_layout()
    os.makedirs(exp_dir, exist_ok=True)
    path = os.path.join(exp_dir, 'fig',filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path