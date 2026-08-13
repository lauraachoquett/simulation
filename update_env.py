import jax.numpy as jnp
import jax
import numpy as np

from jax import random
from functools import partial
from jax import jit

def respawn(x, subkey, prob_factor,pop_res_prob):
    conditions = [
        (x == 2) | (x == 1),
        (x == 3),
        (x >= 4),
    ]
    choices = [0.03, 0.08,0.09]
    default = 0.0
    prob = jnp.select(conditions, choices, default) * prob_factor + pop_res_prob
    prob = jnp.clip(prob,0,1)
    return random.bernoulli(subkey, p=prob).astype(jnp.int32)

# Note : On retire le static_argnums ici car cfg.prob_factor est un float standard
respawn_jit = jit(respawn)


@partial(jax.jit, static_argnames=("cfg", "crowd_brake"))
def resources_growth(carry, cfg, crowd_brake=True, step=None):
    """crowd_brake=False -> pas de frein de surpopulation du tout.

    Utilise pour la pre-croissance (init_state / init_state_lab) : elle sert a
    amener la grille a son etat de depart, et le frein l'empecherait d'atteindre
    des densites que la simulation doit pouvoir montrer des le premier pas.

    `step` est le pas GLOBAL de simulation : le frein ne mord qu'a partir du
    premier cycle, soit cfg.cycle_period * cfg.chunk_size steps. Il est tracee
    (contrairement a crowd_brake), donc la condition passe par jnp.where. Requis
    quand crowd_brake=True -- l'oublier desactiverait le frein en silence, d'ou
    l'erreur explicite."""
    grid_resources, key = carry                      # (n_types, L, L)
    if crowd_brake and step is None:
        raise ValueError("resources_growth : `step` est requis quand crowd_brake=True")
    key, key_env, key_tie = jax.random.split(key, 3)

    up    = jnp.roll(grid_resources, 1,  axis=1)
    down  = jnp.roll(grid_resources, -1, axis=1)
    left  = jnp.roll(grid_resources, 1,  axis=2)
    right = jnp.roll(grid_resources, -1, axis=2)
    neighbors = up + down + left + right             # (n_types, L, L)

    occupied = (grid_resources.sum(axis=0, keepdims=True) > 0).astype(grid_resources.dtype)
    res_growth = neighbors * (1 - occupied)          # (n_types, L, L)

    prob_factor  = jnp.array([r.prob_factor  for r in cfg.resources])   # (n_types,)
    pop_res_prob = jnp.array([r.pop_res_prob for r in cfg.resources])   # (n_types,)

    # Frein de surpopulation, par TYPE : un type qui occupe plus de crowd_limit
    # cases retombe sur une croissance lente. Le compte depend de la grille, donc
    # c'est une valeur tracee -> jnp.where et non un if Python. En revanche
    # crowd_brake est statique, d'ou le if Python autour.
    if crowd_brake:
        count = grid_resources.sum(axis=(1, 2))                         # (n_types,)
        # `step` est tracee -> tout passe par jnp.where. `commence` est un
        # scalaire booleen, diffuse sur les n_types.
        commence = step >= cfg.cycle_period * cfg.chunk_size
        trop = (count > cfg.crowd_limit) & commence
        prob_factor  = jnp.where(trop, cfg.crowd_prob_factor,  prob_factor)
        pop_res_prob = jnp.where(trop, cfg.crowd_pop_res_prob, pop_res_prob)

    # L'alea suit l'IDENTITE de la ressource, pas l'indice de canal : sinon une
    # ressource deplacee sur un autre canal (rotation du lab, shuffle_resources)
    # herite d'une autre cle et pousse differemment, ce qui rend deux configs
    # permutees incomparables. cfg est statique -> indexation statique.
    ids = np.array([r.id for r in cfg.resources])                       # canal -> identite
    n_ids = int(ids.max()) + 1
    keys = jax.random.split(key_env, n_ids)[ids]
    new_plants = jax.vmap(respawn)(res_growth, keys, prob_factor, pop_res_prob)  # (n_types, L, L)

    # meme raison pour le departage entre types
    score  = random.uniform(key_tie, (n_ids,) + new_plants.shape[1:])[ids] * new_plants
    winner = (score == score.max(axis=0, keepdims=True)) & (new_plants > 0)
    new_plants = winner.astype(grid_resources.dtype)
    new_plants = new_plants * (1 - occupied)
    
    grid_resources = jnp.clip(grid_resources + new_plants, 0, 1).astype(jnp.int32)
    return (grid_resources, key)

