import jax.numpy as jnp
import jax

from jax import random
from functools import partial
from jax import jit

def respawn(x, subkey, prob_factor):
    conditions = [
        (x == 2) | (x == 1),
        (x == 3),
        (x >= 4),
    ]
    choices = [0.03, 0.06,0.08]
    default = 0.0
    prob = jnp.select(conditions, choices, default) * prob_factor + 0.00008
    prob = jnp.clip(prob,0,1)
    return random.bernoulli(subkey, p=prob).astype(jnp.int32)

# Note : On retire le static_argnums ici car cfg.prob_factor est un float standard
respawn_jit = jit(respawn)


@partial(jax.jit, static_argnames=("cfg"))
def resources_growth(carry, cfg):
    grid_resources, key = carry
    
    key, key_env = jax.random.split(key)
    
    up    = jnp.roll(grid_resources, shift=1,  axis=0)
    down  = jnp.roll(grid_resources, shift=-1, axis=0)
    left  = jnp.roll(grid_resources, shift=1,  axis=1)
    right = jnp.roll(grid_resources, shift=-1, axis=1)
    
    # ---------------------KERNEL USED-----------------------
    # kernel = [
    #     [0, 0, 0, 0, 0],
    #     [0, 0, 1, 0, 0],
    #     [0, 1, 0, 1, 0],
    #     [0, 0, 1, 0, 0],
    #     [0, 0, 0, 0, 0],
    # ]
    # -----------------------------------------------------------
    
    res_growth = grid_resources + up + down + left + right
    
    res_growth = res_growth * (1.0 - grid_resources)
    new_plants = respawn_jit(res_growth, key_env, cfg.prob_factor)
    
    grid_resources = jnp.clip(grid_resources + new_plants, 0.0, 1.0).astype(jnp.int32)
    
    return (grid_resources, key)




# --- Probabilité de repousse -----------
P_LOW    = 0.001     # proba "faible" pour 1..10 voisins (avant prob_factor) — À AJUSTER
BASELINE = 0.00001   # repousse spontanée (ex self.spontaneous_regrow)

def respawn_big(x, subkey, prob_factor):
    conditions = [
        (x >= 1) & (x <= 4),   # bande de repousse : faible
        (x <= 8)& (x > 4),              # surpopulation : coupure explicite
    ]
    choices = [P_LOW, 0.05]
    default = 0.0               # x == 0 : rien autour
    prob = jnp.select(conditions, choices, default) * prob_factor + BASELINE
    prob = jnp.clip(prob, 0.0, 1.0)
    return random.bernoulli(subkey, p=prob).astype(jnp.int32)

respawn_big_jit = jit(respawn_big)

# --- Noyau : diamant L1 de rayon 2 (centre exclu) -> 12 voisins ----------
# [0,0,1,0,0]
# [0,1,1,1,0]
# [1,1,0,1,1]
# [0,1,1,1,0]
# [0,0,1,0,0]
NEIGHBOR_OFFSETS = (
    (-1, 0), (1, 0), (0, -1), (0, 1),     # |dy|+|dx| = 1
    (-2, 0), (2, 0), (0, -2), (0, 2),     # = 2, orthogonaux
    (-1, -1), (-1, 1), (1, -1), (1, 1),   # = 2, diagonaux
)

def count_neighbors(grid):
    total = jnp.zeros_like(grid)
    for dy, dx in NEIGHBOR_OFFSETS:          # bords toriques (comme tes rolls actuels)
        total = total + jnp.roll(jnp.roll(grid, dy, axis=0), dx, axis=1)
    return total

@partial(jax.jit, static_argnames=("cfg"))
def resources_growth_biger(carry, cfg):
    grid_resources, key = carry
    key, key_env = jax.random.split(key)

    neighbor_sum = count_neighbors(grid_resources)        # 0..12 (entiers)
    res_growth   = neighbor_sum * (1.0 - grid_resources)  # masque les cases occupées

    new_plants = respawn_big_jit(res_growth, key_env, cfg.prob_factor)
    grid_resources = jnp.clip(grid_resources + new_plants, 0.0, 1.0).astype(jnp.int32)
    return (grid_resources, key)



# --- Paramètres ----------------------------------------------------------
P_LOW  = 0.01     # étalement pour 1..10 voisins (avant prob_factor)
P_SEED = 1e-6      # nucléation spontanée : TRÈS faible -> patchs distincts
N_MAX  = 7000      # capacité de charge (nb max de ressources)

# --- Noyau : diamant L1 rayon 2 (centre exclu) -> 12 voisins -------------
NEIGHBOR_OFFSETS = (
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-2, 0), (2, 0), (0, -2), (0, 2),
    (-1, -1), (-1, 1), (1, -1), (1, 1),
)

def count_neighbors(grid):
    total = jnp.zeros_like(grid)
    for dy, dx in NEIGHBOR_OFFSETS:                  # bords toriques
        total = total + jnp.roll(jnp.roll(grid, dy, axis=0), dx, axis=1)
    return total

def respawn_max(x, subkey, prob_factor, density_factor):
    conditions = [
        (x >= 1) & (x <= 10),    # étalement : le patch grandit par ses bords
        (x >= 11),               # surpopulation -> coupure
    ]
    choices = [P_LOW, 0.0]
    prob = jnp.select(conditions, choices, 0.0) * prob_factor + P_SEED
    prob = prob * density_factor                     # <-- plafond global
    prob = jnp.clip(prob, 0.0, 1.0)
    return random.bernoulli(subkey, p=prob).astype(jnp.int32)

respawn_max_jit = jit(respawn_max)

@partial(jax.jit, static_argnames=("cfg"))
def resources_growth_max(carry, cfg):
    grid_resources, key = carry
    key, key_env = jax.random.split(key)

    # (a) voisinage -> croissance en patch
    neighbor_sum = count_neighbors(grid_resources)
    res_growth   = neighbor_sum * (1.0 - grid_resources)   # masque l'occupé

    # (b) plafond par la règle : facteur logistique global
    N = grid_resources.sum()
    density_factor = jnp.clip(1.0 - N / N_MAX, 0.0, 1.0)

    # (c) tirage
    new_plants = respawn_max_jit(res_growth, key_env, cfg.prob_factor, density_factor)
    grid_resources = jnp.clip(grid_resources + new_plants, 0.0, 1.0).astype(jnp.int32)
    return (grid_resources, key)



    
