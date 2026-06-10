from flax import struct
from typing import Any
import jax.numpy as jnp
from typing import NamedTuple

@struct.dataclass
class AgentState:
    position: jnp.ndarray              # (n_agents, 2)
    orientation: jnp.ndarray           # (n_agents,)
    energy: jnp.ndarray                # (n_agents,)
    time_under_min_energy: jnp.ndarray # (n_agents,)
    time_over_energy_repr: jnp.ndarray # (n_agents,)
    alive: jnp.ndarray                 # (n_agents,)
    parent_id: jnp.ndarray             # (n_agents,)
    born_step: jnp.ndarray             # (n_agents,)
    params: jnp.ndarray                # (n_agents, num_params)
    policy_states: Any                  # État caché du RNN (Pytree)

@struct.dataclass
class SimState:
    grid: jnp.ndarray                  # (2, n, n)
    agents: AgentState
    step: jnp.ndarray
    obs: jnp.ndarray
    last_actions: jnp.ndarray
    rewards: jnp.ndarray
    

class Config(NamedTuple):
    n: int
    prob_init_resources: float
    energy_decay: float
    n_agents_max: int
    n_agents_init: int
    time_to_die: int
    time_above_repr: int
    min_energy_repr:float
    prob_factor :int
    pre_growth_step : int
    mutation_var : float
    starting_energy : int
    agent_view : int
    chunk_size : int
    num_chunks : int
    checkpoint_freq : int
    video_freq : int
    param_mutate : float
    factor_energy_decay_not_moving : float
    pca : int



