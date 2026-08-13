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

    # obs pese ~84% du log (T,N,11,11,C) et n'est lu QUE par le lab. La sim
    # principale peut donc s'en passer. Defaut True : un oubli redonne le
    # comportement actuel, lent mais correct, jamais un lab casse.
    log_obs : bool = True

    # Coupe TOUTE l'information intra-vie : etat LSTM (h, c) remis a zero a
    # chaque pas, et last_actions / rewards zerotes en entree. Le genome est
    # inchange, seul le flux temporel est coupe -- a activer en LAB sur les
    # memes agents pour une comparaison appariee. Si la baisse de consommation
    # en fin de vie persiste sans memoire, elle ne vient pas d'un apprentissage.
    ablate_memory : bool = False

    # Rejouer l'env adapt une seconde fois avec ablate_memory=True, pour obtenir
    # la comparaison appariee. Double le cout des rollouts adapt.
    lab_memory_ablation : bool = True

    # Une frame sur video_stride est rendue. Le decoupage se fait avant le
    # transfert device->hote, donc ca allege aussi le pickle envoye au worker.
    # Remettre a 1 pour inspecter finement un chunk.
    video_stride : int = 4

    # Periode des cycles, EN CHUNKS (meme unite que checkpoint_freq / video_freq).
    # Sert a deux choses : la frequence des shuffles de ressources, et le moment
    # ou le frein de surpopulation entre en jeu -- soit cycle_period * chunk_size
    # steps. Le premier cycle se deroule donc sans bride, le temps que
    # l'ecosysteme s'installe.
    cycle_period : int = 200

    # Chunks APRES un shuffle ou lancer une analyse de lab, en plus de la grille
    # reguliere pca_save_freq. Celle-ci divise cycle_period, donc elle ne tombe
    # que sur quelques phases du cycle et rate le juste-apres-permutation, ou le
    # genome est le plus inadapte. Remplace l'ancien cas special chunk_idx == 10,
    # qui ne servait que le premier cycle.
    lab_after_shuffle : tuple = (5, 10, 20)

    # Frein de surpopulation : au-dela de crowd_limit cases occupees, un type
    # retombe sur ces taux de croissance lents. Le seuil porte sur le nombre de
    # cases d'un type
    crowd_limit : int = 3000
    crowd_prob_factor : float = 0.0065
    crowd_pop_res_prob : float = 4e-5
    



