"""Dérive des poids du réseau.

Mesure, une fois la population coalescée, la distance des poids à l'ancêtre
commun, regroupée par bloc fonctionnel (encoder / lstm_input / lstm_recurrent
/ controller), et la compare à une dérive neutre : même généalogie, même
opérateur de mutation, mêmes formes de couches, MÊME chaîne de réduction,
sélection en moins.

--------------------------------------------------------------------------
CONVENTION DE MUTATION (lue depuis l'opérateur réel)

    mutation = cfg.mutation_var * random.normal(...)                # ÉCART-TYPE
    mask     = random.bernoulli(p=cfg.param_mutate, shape=(N, P))   # par coordonnée
    child    = parent + mutation * mask

Donc sigma = cfg.mutation_var = 0.02 et sigma^2 = 4e-4.
`cfg.mutation_var` est mal nommé : c'est un sigma. On ne le passe jamais à une
formule attendant une variance. Pente de la dérive neutre : p*sigma^2 = 3.2e-4
par génération.
--------------------------------------------------------------------------
"""

import warnings

import numpy as np
import jax
from jax.flatten_util import ravel_pytree

from simulation.genealogy.genealogy import find_root
from simulation.utils.plots import (plots_metrics_weight_distance,
                                    plots_weight_selection)


GROUPS = ["encoder", "lstm_input", "lstm_recurrent", "controller"]


# --------------------------------------------------------------------------- #
# Nommage / regroupement des couches                                           #
# --------------------------------------------------------------------------- #
def _group_of(name, include_bias=False):
    if not include_bias and name.endswith("bias"):
        return None
    if "conv" in name:                                    return "encoder"
    if "_lstm/i" in name:                                 return "lstm_input"
    if "_lstm/h" in name:                                 return "lstm_recurrent"
    if "_hiddens" in name or "_output_proj" in name:      return "controller"
    return None


def _iter_named_leaves(tree, prefix=""):
    for name, value in tree.items():
        full = f"{prefix}/{name}" if prefix else name
        if hasattr(value, "items"):
            yield from _iter_named_leaves(value, full)
        else:
            yield full, value


def _group_reduce(leaf_tree, include_bias=False):
    """leaf_tree : chaque feuille réduite à (M,). -> {groupe: (M,)}.

    SOMME des moyennes par couche, pas une seconde moyenne : le bloc vaut le
    cumul de ses couches, donc un bloc à 4 couches pèse 4× un bloc à 1 couche.
    Une conv de 10k poids et une couche de 100 comptent toujours autant l'une que
    l'autre — sommer n'est que multiplier par le nombre de couches, l'erreur
    RELATIVE et la domination par la plus petite couche sont inchangées.

    Conséquence pour le contrôle : l'espérance du neutre devient
    n_couches * p*sigma^2*g, et non plus p*sigma^2*g. Elle diffère donc d'un bloc
    à l'autre, et l'axe y partagé entre panneaux (share_y) les rend d'autant
    moins comparables. Le neutre passe par la même fonction, donc la comparaison
    observé/neutre reste juste.
    """
    buckets = {}
    for name, leaf in _iter_named_leaves(leaf_tree):
        g = _group_of(name, include_bias)
        if g is None:
            continue
        buckets.setdefault(g, []).append(leaf)
    return {g: np.sum(np.stack(v), axis=0) for g, v in buckets.items()}


def _dist_per_layer(struct):
    """pytree de feuilles (M, *shape) -> pytree de feuilles (M,).

    Moyenne du carré sur les paramètres de chaque couche. C'est LA réduction
    appliquée aux données réelles ; le neutre doit passer par la même.
    """
    return jax.tree_util.tree_map(
        lambda w: np.asarray((w ** 2).reshape(w.shape[0], -1).mean(axis=1)),
        struct)


# --------------------------------------------------------------------------- #
# Généalogie                                                                   #
# --------------------------------------------------------------------------- #
def _lineage(node, node_parent):
    """Chemin [node, parent, ..., racine]."""
    path, seen = [node], {node}
    while node in node_parent:
        parent = node_parent[node]
        if parent is None or parent == node or parent in seen:
            break
        path.append(parent)
        seen.add(parent)
        node = parent
    return path


def _gen_depth(node, node_parent):
    return len(_lineage(node, node_parent)) - 1


def fit_generation_rate(steps, gen_depth):
    """Ajuste g(t) = alpha*(t - t0). Renvoie (alpha, t0, r2).

    La linéarité n'est pas acquise : elle suppose un temps de génération
    constant. Si r2 s'écarte de 1, le turnover démographique dérive et
    l'abscisse 'step' confond dérive des poids et changement de démographie.
    """
    steps = np.asarray(steps, dtype=float)
    g     = np.asarray(gen_depth, dtype=float)
    ok    = np.isfinite(steps) & np.isfinite(g)
    steps, g = steps[ok], g[ok]
    if len(steps) < 3:
        return np.nan, np.nan, np.nan
    alpha, b = np.polyfit(steps, g, 1)
    pred   = alpha * steps + b
    ss_res = np.sum((g - pred) ** 2)
    ss_tot = np.sum((g - g.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(alpha), (float(-b / alpha) if alpha else np.nan), float(r2)


# --------------------------------------------------------------------------- #
# Dérive neutre                                                                #
# --------------------------------------------------------------------------- #
def _segment_noise(L, p, sigma, n_flat, rng, dtype):
    """Déplacement accumulé sur L générations consécutives — exact, en O(1).

    Sur L arêtes, une coordonnée reçoit k ~ Binomial(L, p) mutations, et
    conditionnellement à k la somme vaut N(0, k*sigma^2). On tire donc
    directement k puis le gaussien, au lieu de simuler les L pas un par un.
    Indispensable ici : les lignées font des milliers de générations.
    """
    k = rng.binomial(L, p, size=n_flat)
    return (rng.normal(0.0, 1.0, size=n_flat) * sigma * np.sqrt(k)).astype(dtype)


def neutral_drift_groups(alive_nodes, node_parent, unravel_batch,
                         mutation_sigma, param_mutate, n_flat,
                         seed=0, dtype=np.float32, cache=None):
    """Contrôle neutre, réduit EXACTEMENT comme les données réelles.

    On propage un vecteur de poids fictifs de taille `n_flat` (la vraie taille
    aplatie) le long de l'arbre généalogique réel, avec le vrai opérateur de
    mutation et aucune sélection. Puis on applique la même chaîne :
        déplacement -> unravel -> moyenne du carré par couche
                    -> moyenne par groupe -> (mean, std) sur les agents

    Pour un individu neutre, w - w_ancetre EST le déplacement accumulé.

    Deux propriétés utiles pour vérifier la sortie :
      - la MOYENNE neutre vaut p*sigma^2*g pour tous les groupes, indépendamment
        de la taille des couches ;
      - l'ÉCART-TYPE, lui, diffère entre groupes : il dépend de la taille des
        couches et de leur nombre dans le groupe. C'est la raison d'être de
        cette version par groupe.

    Args:
        unravel_batch  : (M, n_flat) -> pytree de feuilles (M, *shape).
                         En pratique jax.vmap(self._unravel).
        mutation_sigma : ÉCART-TYPE de la mutation (= cfg.mutation_var).

    Returns:
        {groupe: (mean, std)}
    """
    rng   = np.random.default_rng(seed)
    sigma = float(mutation_sigma)
    p     = float(param_mutate)

    # --- 1. union des lignées, repérage des branchements ---------------------
    paths, children = {}, {}
    for n in alive_nodes:
        pa = _lineage(n, node_parent)[::-1]          # racine -> feuille
        paths[n] = pa
        for par, ch in zip(pa[:-1], pa[1:]):
            children.setdefault(par, set()).add(ch)

    keep  = {n for n, ch in children.items() if len(ch) > 1}   # branchements
    keep |= set(alive_nodes)                                   # feuilles
    keep |= {pa[0] for pa in paths.values()}                   # racine(s)

    # --- 2. arbre réduit ------------------------------------------------------
    # Entre deux noeuds gardés, la chaîne linéaire est résumée par sa longueur L
    # et traitée en un seul tirage. Mémoire O(M * n_flat) au lieu de
    # O(profondeur * n_flat), ce qui compte sur un run non-épisodique long.
    reduced, order, seen = {}, [], set()
    for n in alive_nodes:
        pa = paths[n]
        if pa[0] not in seen:
            seen.add(pa[0]); reduced[pa[0]] = (None, 0); order.append(pa[0])
        last_kept, L = pa[0], 0
        for node in pa[1:]:
            L += 1
            if node in keep:
                if node not in seen:
                    seen.add(node); reduced[node] = (last_kept, L); order.append(node)
                last_kept, L = node, 0

    # --- 3. propagation (order est déjà racine -> feuilles) -------------------
    # `cache` persiste d'un chunk à l'autre : un nœud déjà simulé conserve son
    # déplacement. Sans lui, chaque chunk serait un tirage indépendant et la
    # courbe neutre sauterait d'un point à l'autre, alors que le processus réel
    # est continu (les poids d'un agent dérivent de ceux du chunk précédent).
    # Seuls les nœuds nouveaux sont tirés ; le tronc partagé reste figé.
    disp = {} if cache is None else cache
    for node in order:
        if node in disp:
            continue
        par, L = reduced[node]
        disp[node] = (np.zeros(n_flat, dtype=dtype) if par is None
                      else disp[par] + _segment_noise(L, p, sigma, n_flat, rng, dtype))

    # --- 4. MÊME chaîne de réduction que les données réelles ------------------
    flat   = np.stack([disp[n] for n in alive_nodes]).astype(np.float64)
    struct = unravel_batch(flat)
    dist_g = _group_reduce(_dist_per_layer(struct))

    # Élagage : on ne garde que les nœuds encore utiles. Les lignées éteintes
    # ne reviendront jamais, donc la mémoire reste en O(M * n_flat) au lieu de
    # croître sur toute la durée du run.
    if cache is not None:
        keep_nodes = set(order)
        for n in [k for k in cache if k not in keep_nodes]:
            del cache[n]

    return {g: (float(v.mean()), float(v.std())) for g, v in dist_g.items()}


# --------------------------------------------------------------------------- #
# Mixin                                                                        #
# --------------------------------------------------------------------------- #
class WeightsMixin:

    def _init_weights(self):
        self.weight_dist    = []     # [{groupe: (mean, std)}]  observé
        self.weight_neutral = []     # [{groupe: (mean, std)}]  neutre
        self.gen_depth      = []
        self.gen_depth_std  = []
        self.metric_steps   = []
        self.n_unresolved   = []     # naissances du dernier pas, non rattachées
        self.founder_params = None
        self._unravel       = None
        self._neutral_cache = {}     # nœud -> déplacement neutre, persiste entre chunks

    def register_founders(self, state, model_policy):
        _, self._unravel = ravel_pytree(model_policy.params['params'])
        self.founder_params = jax.tree_util.tree_map(
            lambda w: np.asarray(w), state.agents.params)          # (N, P)

    # ---------------------------------------------------------------------- #
    def ancestor_ref(self, state):
        """-> (poids_ancetre, noeuds, profondeurs), ou None si non mesurable.

        Appelé sous `if self.coalesced`, donc les vivants ont un MRCA. Sur un
        arbre complet cela implique une racine unique : on remonte du MRCA de
        parent en parent jusqu'à un agent du step 0, à un slot unique, et
        `founder_params[root_slot]` sont exactement les poids de cette racine.

        DÉCALAGE D'UN PAS — structurel et bénin
        ---------------------------------------
        L'arbre est construit depuis `outputs`, la mesure porte sur `state` :

            state, outputs = run_simulation_chunk(...)      # state = carry final
            sim_data.update_genealogy(outputs, state, ...)  # arbre <- outputs

        `outputs` s'arrête un pas avant `state`. Les agents nés à ce dernier pas
        sont dans `state` mais pas encore dans node_parent, et find_root les
        renvoie alors comme leur PROPRE racine — il sort immédiatement quand
        node_parent.get(node) is None. D'où des singletons, en nombre égal aux
        naissances du dernier pas (1, 2, ...). Ce ne sont pas des lignées : on
        les écarte silencieusement, ils seront mesurés au chunk suivant.

        Un fondateur est lui aussi absent de node_parent, mais avec
        born_step == 0 : c'est ce qui sépare les deux cas sans ambiguïté.

        Que `current_leaves` (mrca.py) ne lève jamais — elle exige que TOUT
        vivant de outputs[T-1] soit dans l'arbre — confirme que la généalogie
        elle-même est saine.
        """
        born_last = np.asarray(state.agents.born_step)
        alive     = np.asarray(state.agents.alive).astype(bool)
        alive_nodes = [(int(s), int(born_last[s])) for s in np.nonzero(alive)[0]]

        measurable = [n for n in alive_nodes
                      if n[1] == 0 or n in self.node_parent]
        self.n_unresolved.append(len(alive_nodes) - len(measurable))
        if not measurable:
            return None

        clades = {}
        for n in measurable:
            clades.setdefault(find_root(n, self.node_parent), []).append(n)

        # Ici toutes les racines doivent être fondatrices, et il ne doit y en
        # avoir qu'une : `coalesced` l'impose, et la coalescence est monotone
        # (les descendants d'un MRCA le restent). Sinon c'est un désaccord réel
        # avec coalescence_point, pas un effet de bord.
        if len(clades) > 1:
            warnings.warn(
                f"{len(clades)} racines pour {len(measurable)} vivants résolus "
                f"alors que `coalesced` est vrai — (born_step, effectif) : "
                f"{sorted(((r[1], len(v)) for r, v in clades.items()))[:5]}. "
                f"Désaccord réel avec coalescence_point.")

        root, alive_nodes = max(clades.items(), key=lambda kv: len(kv[1]))
        if root[1] != 0:
            warnings.warn(f"Racine {root} non fondatrice : founder_params ne "
                          f"contient pas ses poids, chunk ignoré.")
            return None

        depths = np.array([_gen_depth(n, self.node_parent) for n in alive_nodes])
        ref = jax.tree_util.tree_map(lambda w: w[root[0]], self.founder_params)
        return ref, alive_nodes, depths

    # ---------------------------------------------------------------------- #
    def update_weight_metrics(self, state, step=None):
        params = np.asarray(state.agents.params)                   # (N, P)

        # Le vrai step de simulation, lu sur l'état. Surtout PAS un compteur de
        # mesures : celles-ci ne démarrent qu'à la coalescence, donc les compter
        # décalerait toute la courbe vers la gauche du nombre de chunks écoulés
        # avant coalescence.
        step = int(state.step) if step is None else step

        res = self.ancestor_ref(state)
        if res is None:
            # Chunk non mesurable : NaN plutôt que rien, pour garder
            # l'alignement des séries. Matplotlib laisse un trou.
            nan = {g: (np.nan, np.nan) for g in GROUPS}
            self.weight_dist.append(nan)
            self.weight_neutral.append(dict(nan))
            self.gen_depth.append(np.nan)
            self.gen_depth_std.append(np.nan)
            self.metric_steps.append(step)
            return

        ref, alive_nodes, depths = res

        # `alive_nodes` exclut les naissances non rattachées : on sélectionne
        # les mêmes agents côté poids, sinon observé et neutre ne porteraient
        # pas sur la même population.
        slots        = np.array([s for s, _ in alive_nodes])
        params_alive = params[slots]                               # (M, P)

        self.gen_depth.append(float(depths.mean()))
        self.gen_depth_std.append(float(depths.std()))

        unravel_batch = jax.vmap(self._unravel)

        # --- observé ---------------------------------------------------------
        delta  = params_alive - ref[None, :]                       # (M, P)
        struct = unravel_batch(delta)                              # (M, *shape)
        dist_g = _group_reduce(_dist_per_layer(struct))            # {g: (M,)}

        missing = [g for g in GROUPS if g not in dist_g]
        if missing:
            raise ValueError(f"Groupes vides : {missing}. Noms vus : "
                             f"{[n for n, _ in _iter_named_leaves(struct)]}")

        self.weight_dist.append({g: (float(v.mean()), float(v.std()))
                                 for g, v in dist_g.items()})
        self.metric_steps.append(step)

        # --- neutre : même arbre, même opérateur, même réduction --------------
        # cfg.mutation_var est un SIGMA (cf. entête) -> passé tel quel.
        self.weight_neutral.append(neutral_drift_groups(
            alive_nodes, self.node_parent, unravel_batch,
            mutation_sigma=self.cfg.mutation_var,
            param_mutate=self.cfg.param_mutate,
            n_flat=params.shape[1],
            seed=len(self.weight_neutral),      # évite de rejouer les mêmes tirages
            cache=self._neutral_cache))         # continuité temporelle

    # ---------------------------------------------------------------------- #
    def plot_weight_metrics(self, exp_dir, x_axis="step"):
        alpha, _, r2 = fit_generation_rate(self.metric_steps, self.gen_depth)
        if np.isfinite(alpha) and alpha:
            print(f"[gen] {alpha:.4g} génération/step "
                  f"(temps de génération {1/alpha:.1f} steps), r2 = {r2:.4f}")
            if r2 < 0.98:
                warnings.warn(
                    f"g(t) non linéaire (r2={r2:.3f}) : le temps de génération "
                    f"dérive. Utiliser x_axis='generation'.")

        plots_metrics_weight_distance(
            self.weight_dist, exp_dir,
            steps=self.metric_steps,
            neutral=self.weight_neutral,
            gen_depth=self.gen_depth,
            x_axis=x_axis)

        # Le rapport observe/neutre : une seule echelle, sans unite, reference a
        # 1. La croissance en p*sigma^2*g commune aux quatre blocs s'annule, et
        # le choix de reduction par couche aussi puisque numerateur et
        # denominateur y passent tous les deux. C'est la figure a montrer.
        plots_weight_selection(
            self.weight_dist, self.weight_neutral, exp_dir,
            steps=self.metric_steps, gen_depth=self.gen_depth, x_axis=x_axis)