from flax import struct
from typing import Any
import jax.numpy as jnp
from typing import NamedTuple
import numpy as np
import matplotlib.colors as mcolors

LABELS = ("good", "medium", "poison")     # id 0, 1, 2
COLOR_BY_ID = {0: "#2A9131", 1: "#3933F35E", 2: "#9C27B0"}

# Couleurs des identites au-dela des trois canoniques.
_COULEURS_SUP = ("#B5651D", "#0F6E70", "#8A6D3B", "#C2185B")


def label_of(i):
    """Nom de l'identite i, defini pour TOUT i.

    LABELS n'a que trois entrees : `LABELS[r.id]` levait IndexError des la
    quatrieme ressource. Les identites supplementaires prennent un nom genere.
    """
    i = int(i)
    return LABELS[i] if 0 <= i < len(LABELS) else f"res{i}"


def color_of(i):
    """Couleur de l'identite i, definie pour TOUT i (cf. label_of)."""
    i = int(i)
    if i in COLOR_BY_ID:
        return COLOR_BY_ID[i]
    return _COULEURS_SUP[(i - len(COLOR_BY_ID)) % len(_COULEURS_SUP)]


def ressource_la_plus_lente(resources):
    """La ressource dont la croissance est la plus lente.

    Remplace `next(r for r in resources if LABELS[r.id] == "poison")`, qui
    levait StopIteration des qu'aucune ressource n'etait du poison -- donc a une
    ou deux ressources. L'intention n'a jamais ete "le poison" en tant que tel
    mais "la plus lente" (cf. le commentaire du frein de surpopulation) ; la lire
    sur prob_factor la rend definie quel que soit le nombre de ressources et
    quels que soient leurs noms.
    """
    return min(resources, key=lambda r: (r.prob_factor, r.pop_res_prob))

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
    # 1 des que invasion_frac a ete atteint : l'injection ne se rallume jamais,
    # sinon un declin des envahisseurs serait masque par un re-remplissage
    invasion_faite: jnp.ndarray
    


@struct.dataclass
class ResourceConfig:
    init_number_of_resources: int
    prob_factor: float
    pop_res_prob: float
    delta_energy: float
    id: int = 0            


BASE_RESOURCES = (
    ResourceConfig(init_number_of_resources=30, prob_factor= 0.0866, pop_res_prob=5e-5, delta_energy=1.0,  id=0),
    # ResourceConfig(init_number_of_resources=20, prob_factor=0.13,  pop_res_prob=5e-5, delta_energy=0.3,  id=1),
    # ResourceConfig(init_number_of_resources=15,   prob_factor=0.0866, pop_res_prob=2e-5, delta_energy=-1.3, id=2),
)


# Geometries de ressource testees au lab, a UNE ressource. Les noms sont ceux
# des .npy produits par `python -m simulation.tools.make_lab_envs --save`, ranges
# dans simulation/lab_envs/.
#
# Toutes de meme rang : chacune est jouee dans les trois conditions sociales
# (seul, clones, figurants), et les comparaisons appariees se font A L'INTERIEUR
# d'une geometrie. Toutes portent le meme nombre de cases, donc un ecart entre
# deux series dit ce que la population sait exploiter, pas ce qu'on lui a donne.
#
# Vide -> une seule condition, le tirage aleatoire (ou lab_env_high_res s'il est
# renseigne) : le comportement d'avant, inchange.
#
# Cout : un rollout par geometrie et par condition sociale, a chaque evaluation.
LAB_ENVS = (
    # "scatter40_s0.npy",
    # "patch4x10_s0.npy",
    # "patch1x40_s0.npy",
)



class Config(NamedTuple):
    grid_length : int
    
    ## Simulation computation :
    chunk_size : int
    num_chunks : int
    checkpoint_freq : int
    video_freq : int
    lab_evaluation_freq : int
    
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

    oracle_agent : bool = False
    oracle_wait : bool = True

    # test d'invasion : a partir de invasion_start, les naissances sont
    # converties en oracles jusqu'a ce qu'ils soient invasion_frac de n_agents_max
    invasion_start : int = 0          # 0 = pas d'invasion
    invasion_frac : float = 0.10

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

    crowd_start : int = 1000000

    lab_after_shuffle : tuple = (10, 20)

    crowd_limit : int = 5000
    # None -> les taux du poison, lus sur cfg.resources. Un nombre les force,
    # ce qui garde lisibles les config.json anterieurs.
    crowd_prob_factor : float = None
    crowd_pop_res_prob : float = None
    
    # Graine de l'ENVIRONNEMENT de lab, independante de celle du run. Sans ce
    # decouplage, subkey_env_lab derivait de PRNGKey(args.seed) : deux runs de
    # graines differentes etaient mesures dans deux labs differents, et l'ecart
    # entre eux melangeait la difference des genomes a celle de l'etalon.
    # Congeneres inertes dans un env de test : actions aleatoires, aucune
    # consommation, energie gelee. Ils occupent les DERNIERS slots. A 0 (defaut)
    # rien ne change -- pas meme le flux de cles aleatoires.
    n_figurants : int = 0
    # Jouer l'env de test a figurants a chaque evaluation de lab. Un rollout de
    # plus par evaluation : laisse a False quand la question sociale n'est pas
    # celle qu'on mesure.
    lab_figurants : bool = False

    lab_seed : int = 1        # cf. tools/preview_lab_env : graine retenue

    # Environnements de test figes, a UNE ressource (cf. tools/make_lab_envs).
    # Vides -> le defaut de lab_env.DEFAUTS_ENVS, lui-meme vide tant qu'aucun
    # env n'a ete retenu : le tirage aleatoire reste alors le comportement.
    lab_env_high_res : str = ""
    lab_env_low_res : str = ""
    # Geometries de ressource testees au lab. Toutes de MEME RANG : chacune est
    # jouee dans les trois conditions sociales (seul, clones, figurants) et les
    # comparaisons appariees se font a l'interieur d'une geometrie. Vide -> une
    # seule condition, celle de lab_env_high_res. Voir LAB_ENVS ci-dessous.
    lab_envs : tuple = LAB_ENVS

    # Tracabilite d'une reprise. Une reprise cree un dossier NEUF, donc sans ces
    # deux champs rien dans le config.json ne dit d'ou elle repart -- et les
    # figures d'un run repris sont inlisibles sans savoir quelle moitie vient
    # d'ou. Vides sur un run parti de zero.
    resume_from : str = ""
    resume_chunk : int = 0

    lab_time_steps : int = 2000

    hidden_dim : int = 8              # taille du carry LSTM (h et c), cf. reset_b
    output_dim : int = 4              # nb d'actions -- doit suivre la table de
                                      # agent_mov.action_depl_theta, qui en compte 4
    hidden_layers : tuple = (32,)     # tete, en aval de [vision, feedback, memoire]
    encoder_layers : tuple = ()      
    encoder : bool = False

    memory_mode : str = "jointe"
    model_version : str = "v1"      

    evolvability_freq : int = 100_000
    evolvability_agents : int = 10
    evolvability_children : int = 100

    replay_top_n : int = 5
    replay_keys : int = 50

    shuffle_version : str = "v2"

    track_weights : bool = False

    lstm_forget_bias : float = None

    init_scale : str = "constant"

    replay_video_min_frac : float = 0.8


MODEL_VERSIONS = {
    "v1": dict(memory_mode="jointe",  hidden_dim=4, hidden_layers=(8,)),
    "v2": dict(memory_mode="separee", hidden_dim=8, hidden_layers=(8,)),
}


def resolve_model(cfg):

    if cfg.model_version == "custom":
        return cfg
    if cfg.model_version not in MODEL_VERSIONS:
        raise ValueError(
            f"model_version inconnu : {cfg.model_version!r}. "
            f"Attendu {sorted(MODEL_VERSIONS)} ou 'custom'.")
    return cfg._replace(**MODEL_VERSIONS[cfg.model_version])
