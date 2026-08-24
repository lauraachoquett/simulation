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
    last_eaten: jnp.ndarray
    


@struct.dataclass
class ResourceConfig:
    init_number_of_resources: int
    prob_factor: float
    pop_res_prob: float
    delta_energy: float
    id: int = 0            


BASE_RESOURCES = (
    ResourceConfig(init_number_of_resources=30, prob_factor= 0.0866, pop_res_prob=5e-5, delta_energy=1.0,  id=0),
    ResourceConfig(init_number_of_resources=20, prob_factor=0.13,  pop_res_prob=5e-5, delta_energy=0.3,  id=1),
    ResourceConfig(init_number_of_resources=15,   prob_factor=0.0065, pop_res_prob=4e-5, delta_energy=-1.0, id=2),
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
    log_grid : bool = True

    ablate_memory : bool = False          # coupe les trois canaux ci-dessous
    ablate_recurrence : bool = False      # lstm_h / lstm_c
    ablate_interoception : bool = False   # energie
    ablate_feedback : bool = False        # reward + action precedente

    lab_memory_ablation : bool = True

    video_stride : int = 2

    cycle_period : int = 200

    crowd_start : int = 100_000

    lab_after_shuffle : tuple = (1, 10)

    crowd_limit : int = 3000
    crowd_prob_factor : float = 0.0065
    crowd_pop_res_prob : float = 4e-5
    
    lab_time_steps : int = 2000

    hidden_dim : int = 8              # taille du carry LSTM (h et c), cf. reset_b
    output_dim : int = 4              # nb d'actions -- doit suivre la table de
                                      # agent_mov.action_depl_theta, qui en compte 4
    hidden_layers : tuple = (32,)     # tete, en aval de [vision, feedback, memoire]
    encoder_layers : tuple = ()      
    encoder : bool = False

    memory_mode : str = "jointe"
    model_version : str = "v1"      

    # evaluabilite : tous les evolvability_freq pas, les N meilleurs genomes
    # sont re-echantillonnes en M enfants mutes et evalues dans l'env high_res
    evolvability_freq : int = 100_000
    evolvability_agents : int = 10
    evolvability_children : int = 100

    # rejeu : les N genomes au plus fort gain de memoire, chacun rejoue avec
    # replay_keys graines, intact et ablate. 0 desactive.
    replay_top_n : int = 5
    replay_keys : int = 50


MODEL_VERSIONS = {
    "v1": dict(memory_mode="jointe",  hidden_dim=4, hidden_layers=(8,)),
    "v2": dict(memory_mode="separee", hidden_dim=8, hidden_layers=(32,)),
}


def resolve_model(cfg):

    if cfg.model_version == "custom":
        return cfg
    if cfg.model_version not in MODEL_VERSIONS:
        raise ValueError(
            f"model_version inconnu : {cfg.model_version!r}. "
            f"Attendu {sorted(MODEL_VERSIONS)} ou 'custom'.")
    return cfg._replace(**MODEL_VERSIONS[cfg.model_version])
