from flax import struct
from typing import Any
import jax.numpy as jnp
from typing import NamedTuple
import numpy as np
import matplotlib.colors as mcolors

LABELS = ("good", "medium", "poison")     # id 0, 1, 2
COLOR_BY_ID = {0: "#2A9131", 1: "#3933F35E", 2: "#9C27B0"}

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
    


@struct.dataclass
class ResourceConfig:
    init_number_of_resources: int
    prob_factor: float
    pop_res_prob: float
    delta_energy: float
    id: int = 0            


BASE_RESOURCES = (
    ResourceConfig(init_number_of_resources=30, prob_factor=0.0866, pop_res_prob=5e-5, delta_energy=1.0,  id=0),
    ResourceConfig(init_number_of_resources=20, prob_factor=0.13,  pop_res_prob=5e-5, delta_energy=0.3,  id=1),
    ResourceConfig(init_number_of_resources=15,  prob_factor=0.0065,   pop_res_prob=4e-5, delta_energy=-1.0, id=2),
)




class Config(NamedTuple):
    grid_length : int
    
    ## Simulation computation :
    chunk_size : int
    num_chunks : int
    checkpoint_freq : int
    video_freq : int
    pca_save_freq : int
    
    # AGENTS # 
    n_agents_max: int
    n_agents_init: int
    agent_view : int
    temperature : float #Temperature in the Categorical operation
    
    ## Physiologie
    energy_decay: float
    factor_energy_decay_not_moving : float
    energy_max : float

    time_to_die: int
    time_above_repr: int
    min_energy_repr:float
    starting_energy : int
    random_pos_offspring : bool
    
    ## Mutation parameters
    mutation_var : float
    param_mutate : float
    
    
    # INIT RESOURCES MAP
    pre_growth_step : int
    
    # Random agents action
    dumb_agent : bool = False
    
    resources: tuple  = BASE_RESOURCES
    # Lab env paramaeters
    reproduction_on: bool = True
    resources_growth : bool = True
    letal_wall : bool = True
    energy_to_die : float = 0.0

    log_obs : bool = True

    ablate_memory : bool = False          # coupe les trois canaux ci-dessous
    ablate_recurrence : bool = False      # lstm_h / lstm_c
    ablate_interoception : bool = False   # energie
    ablate_feedback : bool = False        # reward + action precedente

    lab_memory_ablation : bool = True

    video_stride : int = 2

    cycle_period : int = 200

    # Pas global a partir duquel le frein de surpopulation mord. Independant de
    # cycle_period : sinon changer le rythme des permutations deplacerait aussi
    # le demarrage du frein, et deux runs differeraient par deux choses.
    crowd_start : int = 100_000

    lab_after_shuffle : tuple = (5, 10, 20)

    crowd_limit : int = 3000
    crowd_prob_factor : float = 0.0065
    crowd_pop_res_prob : float = 4e-5
    
    lab_time_steps : int = 1000
    



