import jax.numpy as jnp
import jax

from jax import random
from functools import partial
from jax import jit

def respawn(x, subkey, prob_factor):
    conditions = [
        x == 0,
        (x >= 1) & (x <= 2),
        (x >= 3) & (x <= 4),
    ]
    choices = [0, 0.01, 0.05]
    default = 0.1
    prob = jnp.select(conditions, choices, default) * prob_factor
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
    
    res_growth = grid_resources + up + down + left + right
    # -----------------------------------------------------------
    
    res_growth = res_growth * (1.0 - grid_resources)
    new_plants = respawn_jit(res_growth, key_env, cfg.prob_factor)
    
    grid_resources = jnp.clip(grid_resources + new_plants, 0.0, 1.0).astype(jnp.int32)
    
    return (grid_resources, key)