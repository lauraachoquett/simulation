
import os
import numpy as np
import jax
from jax import random
import jax.numpy as jnp
import dataclasses
from typing import NamedTuple


from simulation.data_class import AgentState,SimState, ResourceConfig, LABELS
from simulation.agent_mov import get_obs_vector
from simulation.update_env import resources_growth
from simulation.data_class import Config, MODEL_VERSIONS

import numpy as np
from simulation.utils.utils_video import save_chunk_video
import json
from datetime import datetime
import time
import pickle


    

def masque_forget_bias(model):
    """(num_params,) : 1 sur les entrees du biais de la forget gate, 0 ailleurs.

    L'ordre suit tree_flatten, comme get_params_format_fn qui met en forme le
    vecteur plat -- on le lit donc du meme arbre plutot que de coder des indices
    en dur, qui bougeraient avec hidden_dim ou l'architecture.
    """
    morceaux = []
    for chemin, feuille in jax.tree_util.tree_flatten_with_path(model.params)[0]:
        cles = [str(k.key) for k in chemin if hasattr(k, "key")]
        cible = len(cles) >= 2 and cles[-2] == "hf" and cles[-1] == "bias"
        morceaux.append(jnp.full(feuille.size, 1.0 if cible else 0.0))
    return jnp.concatenate(morceaux)


def stds_fan_in(model):
    """(num_params,) : 1/sqrt(fan_in) pour les poids, 0 pour les biais."""
    morceaux = []
    for _, feuille in jax.tree_util.tree_flatten_with_path(model.params)[0]:
        s = 1.0 / np.sqrt(feuille.shape[0]) if feuille.ndim >= 2 else 0.0
        morceaux.append(jnp.full(feuille.size, s))
    return jnp.concatenate(morceaux)


def init_state(key, cfg, model):
    # 1. Grille de ressources
    key, subkey_grid = random.split(key)
    
    ## Grille mur
    grid_walls = jnp.zeros((cfg.grid_length, cfg.grid_length), dtype=jnp.int32)
    grid_walls = grid_walls.at[0,:].set(1)
    grid_walls = grid_walls.at[:,0].set(1)
    grid_walls = grid_walls.at[-1,:].set(1)
    grid_walls = grid_walls.at[:,-1].set(1)
    
    ## Grille ressources 
    counts = tuple(r.init_number_of_resources for r in cfg.resources) #STATIQUE
    n_types = len(counts)
    total = sum(counts)

    # à quel type appartient chaque ressource : [0,...,0, 1,...,1, 2,...,2]
    type_ids = jnp.repeat(jnp.arange(n_types), jnp.array(counts))       # (total,)

    # une position (ligne, colonne) par ressource
    position_res = random.randint(subkey_grid, (total, 2), 0, cfg.grid_length)

    # grille 3D : un plan par type
    grid_resources = jnp.zeros((n_types, cfg.grid_length, cfg.grid_length), dtype=jnp.int32)
    grid_resources = grid_resources.at[type_ids, position_res[:, 0], position_res[:, 1]].set(1)

    # on éteint les murs (broadcast du plan (L,L) sur l'axe type)
    grid_resources = jnp.where(grid_walls[None] == 1, 0, grid_resources)
    
    # 2. Préparation des agents
    key, *subkeys = random.split(key, 4)
    sk_pos, sk_or, sk_params = subkeys
    
    orientations_pool = jnp.array([0, jnp.pi/2, jnp.pi, -jnp.pi/2])
    idx_or = random.randint(sk_or, (cfg.n_agents_max,), 0, 4)
    
    # Gestion de la survie initiale
    alive_mask = jnp.zeros((cfg.n_agents_max,), dtype=jnp.int32).at[1:cfg.n_agents_init+1].set(1)
    
    # Paramètres réseau et état RNN
    bruit = random.normal(sk_params, (cfg.n_agents_max, model.num_params))
    if cfg.init_scale == "lecun":
        params = bruit * stds_fan_in(model)
    elif cfg.init_scale == "constant":
        params = bruit / 100
    else:
        raise ValueError(f"init_scale inconnu : {cfg.init_scale!r} "
                         "(attendu 'constant' ou 'lecun')")
    # sans ca la forget gate demarre a sigmoid(~0) = 0.5 : le carry est divise
    # par deux a chaque pas et la population de depart n'a aucune memoire.
    if cfg.lstm_forget_bias:
        params = params + cfg.lstm_forget_bias * masque_forget_bias(model)
    policy_states = model.reset_b(jnp.zeros(cfg.n_agents_max))
    
    # Création de l'objet AgentState
    agents = AgentState(
        position=random.randint(sk_pos, (cfg.n_agents_max, 2), 1, cfg.grid_length-1),
        orientation=orientations_pool[idx_or],
        energy=jnp.ones((cfg.n_agents_max,))*cfg.starting_energy,
        time_under_min_energy=jnp.zeros((cfg.n_agents_max,), dtype=jnp.int32),
        time_over_energy_repr=jnp.zeros((cfg.n_agents_max,), dtype=jnp.int32),
        alive=alive_mask,
        parent_id=jnp.zeros((cfg.n_agents_max,), dtype=jnp.int32),
        born_step=jnp.zeros((cfg.n_agents_max,), dtype=jnp.int32),
        params=params,
        is_oracle=jnp.zeros((cfg.n_agents_max,)),
        croyance=jnp.full((cfg.n_agents_max, len(cfg.resources)),
                          cfg.croyance_init),
        policy_states=policy_states
    )


    # 3. Grille d'occupation et observations
    grid_agents = jnp.zeros((cfg.grid_length, cfg.grid_length), dtype=jnp.int32)
    grid_agents = grid_agents.at[agents.position[:, 0], agents.position[:, 1]].add(agents.alive)
    grid = jnp.concatenate([
        grid_resources,          # (n_types, L, L)  -> déjà n canaux
        grid_agents[None],       # (1, L, L)
        grid_walls[None],        # (1, L, L)
    ], axis=0)   
    pos = agents.position

    # 4. État final
    
    key, key_env = jax.random.split(key)

    init_carry = (grid_resources, key_env)

    grid_resources_grown, _ = jax.lax.fori_loop(
        0,
        cfg.pre_growth_step,
        # pas de frein pendant la pre-croissance : elle amene la grille a son
        # etat de depart, le frein la briderait avant meme le premier pas
        lambda i, carry: resources_growth(carry, cfg, crowd_brake=False),
        init_carry
    )
    
    grid_resources_grown_bis = jnp.where(grid_walls[None] == 1, 0, grid_resources_grown)
    
    grid = jnp.concatenate([
        grid_resources_grown_bis,          # (n_types, L, L)  
        grid_agents[None],       # (1, L, L)
        grid_walls[None],        # (1, L, L)
    ], axis=0)   
    
    obs = get_obs_vector(grid, (pos, agents.orientation), cfg.agent_view)

    state = SimState(
        grid=grid,
        agents=agents,
        step=0,
        obs=obs,
        last_actions=jnp.zeros((cfg.n_agents_max, cfg.output_dim)),
        rewards=jnp.zeros((cfg.n_agents_max, 1)),
        last_eaten=jnp.zeros((cfg.n_agents_max, len(cfg.resources))),
    )
    
    return state


def save_checkpoint(state, filepath):
    """Sauvegarde l'état de la simulation sur le disque."""
    # Convertit le PyTree JAX en PyTree NumPy
    state_np = jax.tree_util.tree_map(np.asarray, state)
    with open(filepath, 'wb') as f:
        pickle.dump(state_np, f)

def load_checkpoint(resume_exp,chunk_id):
    """Charge l'état de la simulation depuis le disque."""
    path = os.path.join(resume_exp,f'checkpoints/state_chunk_{chunk_id}.pkl')
    with open(path, 'rb') as f:
        state_np = pickle.load(f)
    # Reconvertit le PyTree NumPy en PyTree JAX
    return jax.tree_util.tree_map(jnp.asarray, state_np)



def sec_to_minutes(secondes):
    minutes, secondes_restantes = divmod(secondes, 60)
    return minutes, secondes_restantes

# --- Sérialiseur : convertit outputs JAX → numpy avant envoi inter-process ---
def outputs_to_numpy(outputs):
    """
    Descend récursivement dans le pytree et convertit chaque feuille en np.array.
    À adapter selon la structure de ton SimOutputs.
    """
    import jax
    return jax.tree_util.tree_map(np.array, outputs)


class VideoPayload(NamedTuple):
    """Le strict nécessaire pour save_chunk_video, et rien d'autre.

    Le pytree complet fait ~5,4 Go par chunk et part par PICKLE dans un pipe vers
    le process worker. save_chunk_video ne lit que ces cinq champs
    (utils_video.py:14, 38-41) ; `grid` domine, le reste est marginal.
    """
    grid:      np.ndarray
    position:  np.ndarray
    born_step: np.ndarray
    alive:     np.ndarray
    step:      np.ndarray


def video_payload(outputs, stride=1):
    """Extrait la charge vidéo, en ne transférant qu'une frame sur `stride`.

    Le découpage est fait AVANT le transfert device->hôte : inutile de rapatrier
    puis de pickler des frames qu'on ne rendra jamais.
    """
    take = lambda x: np.asarray(x[::stride])
    return VideoPayload(
        grid      = take(outputs.grid),
        position  = take(outputs.position),
        born_step = take(outputs.born_step),
        alive     = take(outputs.alive),
        step      = take(outputs.step),
    )


# --- Wrapper picklable pour le worker ---
def _video_worker(outputs_np, vid_path, fps, scale, resources):
    t0 = time.time()
    print(f"  [video | PID {os.getpid()}] START  {vid_path}  @ {time.strftime('%H:%M:%S')}")
    save_chunk_video(outputs_np, vid_path, fps=fps, scale=scale,resources = resources)
    print(f"  [video | PID {os.getpid()}] DONE   {vid_path}  ({time.time()-t0:.2f}s)")
    return vid_path


def create_exp_file(dir):
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join(dir, timestamp)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "videos"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "videos","high"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "videos","low"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "videos","high_res_clones"), exist_ok=True)
    data_dir = os.path.join(exp_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    return exp_dir,data_dir


def save_config(cfg, subkeys, exp_dir):
    cfg_dict = cfg._asdict()
    cfg_dict["resources"] = [dataclasses.asdict(r) for r in cfg.resources]   # <-- ajout
    cfg_dict["seeds"] = [int(k[0]) for k in subkeys]
    cfg_dict["seeds_full"] = [k.tolist() for k in subkeys]
    cfg_dict["env"] = {
        "jax_version": jax.__version__,
        "prng_impl": jax.config.jax_default_prng_impl,
        "threefry_partitionable": jax.config.jax_threefry_partitionable,
    }
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(cfg_dict, f, indent=2)

def load_config(resume_exp):
    with open(os.path.join(resume_exp, 'config.json'), 'r') as f:
        cfg_dict = json.load(f)
    seeds_full = [jnp.array(s, dtype=jnp.uint32) for s in cfg_dict.pop("seeds_full")]
    env = cfg_dict.pop("env", None)
    cfg_dict.pop("seeds", None)
    cfg_dict["resources"] = tuple(ResourceConfig(**r) for r in cfg_dict["resources"])  # <-- ajout
    # JSON ne connait que les listes : sans cette reconversion, Config redevient
    # non hachable et jax.jit(static_argnames=['cfg']) leve a la premiere trace.
    # Les configs anterieures a l'ajout de ces champs n'ont pas les cles : les
    # defauts de Config prennent le relais, et ils valent l'ancien code en dur.
    # JSON ne connait que les listes. Tout champ dont le defaut est un tuple doit
    # etre reconverti, sinon Config n'est plus hachable et jax.jit leve. On le
    # deduit des defauts plutot que d'en tenir une liste, qui se perimerait.
    for champ, defaut in Config._field_defaults.items():
        if isinstance(defaut, tuple) and isinstance(cfg_dict.get(champ), list):
            cfg_dict[champ] = tuple(cfg_dict[champ])
    # Un config.json sans memory_mode precede l'ajout de ces champs, donc aussi
    # 591269d : il ne peut venir que du reseau v1. On l'epingle plutot que de
    # laisser le defaut courant de Config decider a sa place.
    if "memory_mode" not in cfg_dict:
        cfg_dict.update(MODEL_VERSIONS["v1"], model_version="v1")
        print("[load_config] config anterieure aux champs reseau -> v1 "
              "(jointe, hidden_dim=4, hidden_layers=(8,)). Surcharger avec -m.")
    return Config(**cfg_dict), seeds_full

def print_params(params, prefix="", total=0):
    for name, value in params.items():
        full_name = f"{prefix}/{name}" if prefix else name
        if hasattr(value, 'items'):   # duck typing — FrozenDict et dict
            total = print_params(value, prefix=full_name, total=total)
        else:
            n_p = int(np.prod(value.shape))
            total += n_p
            print(f"{full_name:<55} {str(value.shape):<20} {n_p:>10,}")
    return total


def classify_outcome(pop_full, res_full, cfg):
    """Returns 'extinction', 'overpopulation', 'depletion', 'easy' or 'interesting'."""
    n_chunks = 30
    span = n_chunks * cfg.chunk_size

    if pop_full[-1] == 0:
        return 'extinction'

    last = pop_full[-span:]
    if len(last) == span and (last > 0.95 * cfg.n_agents_max).all():
        return 'overpopulation'

    delta_e  = np.array([r.delta_energy for r in cfg.resources])
    good     = np.where(delta_e > 0)[0]
    res_good = res_full[:, good].sum(axis=1)          # (T,) total des bonnes ressources

    if res_good[-1] == 0:
        return 'depletion'

    last_res = res_good[-span:]
    if len(last_res) == span and (last_res > 0.8 * cfg.grid_length ** 2).all():
        return 'easy'

    return 'interesting'

import numpy as np

def shuffle_resources(resources, key):
    """Permutation uniforme des canaux, identite COMPRISE.

    L'identite doit rester tirable. L'exclure rendrait le changement previsible :
    a 2 ressources il n'existe qu'une permutation non triviale, donc l'exclusion
    produirait une alternance deterministe au rythme de cycle_period -- un
    environnement qu'un genome peut anticiper avec une simple horloge interne,
    exactement ce qu'il ne faut pas pour tester l'apprentissage intra-vie.

    En la gardant, l'intervalle entre deux changements REELS est geometrique :
    en moyenne n!/(n!-1) tirages, soit 2 x cycle_period a 2 ressources et
    1.2 x a 3. C'est cet intervalle-la qu'il faut lire sur les figures, pas
    cycle_period, qui n'est que la periode des TIRAGES.
    """
    perm = np.asarray(jax.random.permutation(key, len(resources)))   # concret -> indexe un tuple
    return tuple(resources[int(i)] for i in perm)


def shuffle_resources_v1(resources, key, max_essais=20):
    """Version d'avant b0f455d : identite EXCLUE, et split de la cle DANS la
    boucle. Conservee a l'identique pour rejouer les runs anterieurs -- les deux
    details comptent, la nouvelle version ne tire pas la meme permutation a
    partir de la meme cle.
    """
    n = len(resources)
    ids = [r.id for r in resources]
    for _ in range(max_essais):
        key, sous = jax.random.split(key)
        perm = np.asarray(jax.random.permutation(sous, n))
        tire = tuple(resources[int(i)] for i in perm)
        if n < 2 or [r.id for r in tire] != ids:
            return tire
    return tire


def log_resource_shuffle(exp_dir, chunk_idx, step, old_resources, new_resources):
    record = {
        "chunk_idx": int(chunk_idx),
        "step":      int(step),
        "order_ids": [int(r.id) for r in new_resources],       # canal k -> id  (l'état après shuffle)
        "changes": [
            {"channel": k, "from": LABELS[o.id], "to": LABELS[n.id]}
            for k, (o, n) in enumerate(zip(old_resources, new_resources))
            if o.id != n.id
        ],
    }
    path = os.path.join(exp_dir, "resource_shuffles.jsonl")
    with open(path, "a") as f:                                 # "a" = append, jamais d'écrasement
        f.write(json.dumps(record) + "\n")
        
def build_id_timeline(generations, shuffle_log, initial_order_ids):
    """(T, n_types) : id de la ressource sur chaque canal, à chaque step."""
    steps  = np.array([e["step"] for e in shuffle_log])
    orders = [initial_order_ids] + [e["order_ids"] for e in shuffle_log]
    active = np.searchsorted(steps, generations, side="right")   # 0 = avant le 1er shuffle
    return np.array([orders[a] for a in active])                 # (T, n_types)


def load_shuffle_log(exp_dir):
    path = os.path.join(exp_dir, "resource_shuffles.jsonl")
    if not os.path.exists(path):
        return []                                    # aucun shuffle encore -> log vide
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()] 