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
    is_oracle: jnp.ndarray             # (n_agents,) 1 = politique codee en dur
    croyance: jnp.ndarray              # (n_agents, n_types) delta_energy estime
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
    ResourceConfig(init_number_of_resources=15,   prob_factor=0.0065, pop_res_prob=4e-5, delta_energy=-1.5, id=2),
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

    # politique codee en dur, sans reseau (cf. simulation/oracle.py)
    oracle_agent : bool = False
    # l'oracle se gare a cote d'une ressource au lieu de la manger a satiete
    oracle_wait : bool = True

    # test d'invasion : a partir de invasion_start, les naissances sont
    # converties en oracles jusqu'a ce qu'ils soient invasion_frac de n_agents_max
    invasion_start : int = 0          # 0 = pas d'invasion
    invasion_frac : float = 0.10

    # L'oracle apprend-il la valeur des canaux, ou la connait-il d'avance ?
    # "apprenant" : croyance optimiste a la naissance, corrigee a chaque bouchee.
    # C'est la remise a zero a la naissance qui fait le test intra-vie.
    oracle_apprend : bool = True
    croyance_init : float = 1.0
    
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
    # None -> les taux du poison, lus sur cfg.resources. Un nombre les force,
    # ce qui garde lisibles les config.json anterieurs.
    crowd_prob_factor : float = None
    crowd_pop_res_prob : float = None
    
    lab_time_steps : int = 2000

    hidden_dim : int = 8              # taille du carry LSTM (h et c), cf. reset_b
    output_dim : int = 4              # nb d'actions -- doit suivre la table de
                                      # agent_mov.action_depl_theta, qui en compte 4
    hidden_layers : tuple = (32,)     # tete, en aval de [vision, feedback, memoire]
    encoder_layers : tuple = ()      
    encoder : bool = False

    memory_mode : str = "jointe"
    model_version : str = "v2"      

    # evaluabilite : tous les evolvability_freq pas, les N meilleurs genomes
    # sont re-echantillonnes en M enfants mutes et evalues dans l'env high_res
    evolvability_freq : int = 100_000
    evolvability_agents : int = 10
    evolvability_children : int = 100

    # rejeu : les N genomes au plus fort gain de memoire, chacun rejoue avec
    # replay_keys graines, intact et ablate. 0 desactive.
    replay_top_n : int = 5
    replay_keys : int = 50

    # "v1" : shuffle d'avant 9abdc3d/b0f455d -- permute BASE_RESOURCES avec la
    # cle du chunk, identite exclue. Pour rejouer un run anterieur au 18 aout.
    shuffle_version : str = "v2"

    # suivi des poids : distance a l'ancetre, derive neutre, et la sauvegarde
    # des genomes vivants a chaque chunk (params/, plusieurs Mo par chunk).
    track_weights : bool = False

    # Biais ajoute a la forget gate du LSTM a l'initialisation des genomes.
    # A 0 le biais vaut ~0.01 comme le reste du vecteur, donc sigmoid ~= 0.5 et
    # le carry est divise par deux a chaque pas : la population de depart est
    # amnesique. A 1 la retention passe a ~0.73. 0 retablit l'ancien comportement.
    lstm_forget_bias : float = 1.0

    # Echelle des poids a l'initialisation des genomes.
    #   "constant" : ecart-type 0.01 partout, quelle que soit la couche
    #   "lecun"    : 1/sqrt(fan_in) par couche, biais a zero (la convention)
    # A 0.01 le signal est divise par ~15 a chaque couche : la population de
    # depart ne reagit pas a ce qu'elle voit. Changer ceci demande de remonter
    # temperature, qui compensait des logits minuscules.
    init_scale : str = "constant"

    # fraction de graines positives au-dela de laquelle un genome rejoue est
    # filme, intact et ablate. 1.0 desactive les videos de rejeu.
    replay_video_min_frac : float = 0.8


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
