import os
from functools import lru_cache

import numpy as np
import jax
from jax import random,vmap
import jax.numpy as jnp

from simulation.data_class import (AgentState, SimState, LABELS, label_of,
                                   ressource_la_plus_lente)
from simulation.agent_mov import get_obs_vector
from simulation.update_env import resources_growth

import numpy as np
import matplotlib.pyplot as plt



# Environnements de test figes. Les .npy vivent a cote du code et non dans le
# repertoire courant : un job de cluster demarre ou il veut, et un chemin
# relatif au cwd se perdrait en silence.
DOSSIER_ENVS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lab_envs")


def resout_chemin(chemin):
    """Un chemin tel quel s'il existe, sinon relatif a DOSSIER_ENVS."""
    if os.path.isabs(chemin) or os.path.exists(chemin):
        return chemin
    return os.path.join(DOSSIER_ENVS, chemin)


@lru_cache(maxsize=None)
def charge_grille(chemin):
    """(n_ids, L, L) int8, indexe par IDENTITE de ressource.

    Cachee pour deux raisons. Le fichier n'est lu qu'une fois, et surtout : la
    fonction est appelee SOUS jit, ou le tableau rendu devient une constante du
    jaxpr. Le chemin etant statique, changer de fichier declenche bien une
    recompilation -- et garder le meme n'en declenche aucune.
    """
    reel = resout_chemin(chemin)
    if not os.path.exists(reel):
        raise FileNotFoundError(
            f"environnement de lab introuvable : {chemin} (cherche a {reel}). "
            "Le produire avec `python -m simulation.tools.make_lab_envs --save`.")
    plan = np.load(reel)
    print(f"[lab] env fige : {reel}  {plan.shape}  {int(plan.sum())} cases",
          flush=True)
    return plan


def grille_figee(chemin, cfg):
    """(n_types, L, L) : le fichier remis dans l'ordre des CANAUX.

    Le fichier est indexe par identite, la grille par canal. Passer par
    `r.id` tient le meme invariant que le tirage aleatoire juste en dessous :
    une ressource occupe les memes cases quelle que soit la permutation des
    canaux, sans quoi baseline et config permutee ne seraient pas comparables.
    """
    plan = charge_grille(chemin)
    attendu = (cfg.grid_length, cfg.grid_length)
    if tuple(plan.shape[1:]) != attendu:
        raise ValueError(
            f"{chemin} : grille {tuple(plan.shape[1:])}, attendu {attendu}. "
            "Ces environnements sont dessines pour une arene precise.")
    ids = [r.id for r in cfg.resources]
    if max(ids) >= plan.shape[0]:
        raise ValueError(
            f"{chemin} : porte {plan.shape[0]} identite(s), mais la config "
            f"demande l'identite {max(ids)}.")
    return jnp.stack([jnp.asarray(plan[i], dtype=jnp.int32) for i in ids])


def env_file_pour(cfg, quel):
    """Le .npy a charger pour l'env `quel`, ou None pour le tirage aleatoire.

    Ne fige que le cas a UNE ressource : au-dela aucun environnement n'a ete
    dessine, et le tirage reste la seule source.
    """
    if len(cfg.resources) != 1:
        return None
    choisi = {"high_res": cfg.lab_env_high_res,
              "low_res":  cfg.lab_env_low_res}[quel]
    # `or None` et non `or ""` : init_state_lab teste `is not None`, et une
    # chaine vide serait prise pour un chemin.
    return choisi or DEFAUTS_ENVS.get(quel) or None


# Env retenus par defaut a UNE ressource : y mettre le nom du .npy choisi dans
# fig/lab_envs_catalogue.png (ex. "patch4x10_s2.npy"), une fois produit par
# `python -m simulation.tools.make_lab_envs --save <nom>`.
#
# Tant qu'ils sont vides, rien ne change : le tirage aleatoire reste le
# comportement, a une ressource comme au-dela.
DEFAUTS_ENVS = {"high_res": "", "low_res": ""}


def init_state_lab(key, cfg, model,agent_params, env_file=None):
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

    # subkey_grid est consommee meme quand la grille vient d'un fichier : le
    # reste du flux de cles (position et orientation de l'agent) doit rester
    # identique, sinon changer d'environnement deplacerait aussi l'agent et on
    # ne saurait plus a quoi attribuer un ecart.
    fige = env_file is not None
    if fige:
        grid_resources = grille_figee(env_file, cfg)
    else:
        # Le tirage des positions est indexe par l'IDENTITE de la ressource, pas par
        # le canal : une meme ressource doit occuper les memes cases quelle que soit
        # la permutation des canaux (rotation du lab, shuffle_resources), sinon
        # baseline et config permutee ne sont pas comparables.
        ids = [r.id for r in cfg.resources]                    # canal -> identite
        keys_pos = random.split(subkey_grid, max(ids) + 1)

        grid_resources = jnp.zeros((n_types, cfg.grid_length, cfg.grid_length), dtype=jnp.int32)
        for k, r in enumerate(cfg.resources):                  # cfg statique -> boucle deroulee
            pos_k = random.randint(keys_pos[r.id], (r.init_number_of_resources, 2),
                                   0, cfg.grid_length)
            grid_resources = grid_resources.at[k, pos_k[:, 0], pos_k[:, 1]].set(1)

    # on éteint les murs (broadcast du plan (L,L) sur l'axe type)
    grid_resources = jnp.where(grid_walls[None] == 1, 0, grid_resources)
    
    # 3. Préparation de l'agents
    key, *subkeys = random.split(key, 3)
    sk_pos, sk_or = subkeys
    
    orientations_pool = jnp.array([0, jnp.pi/2, jnp.pi, -jnp.pi/2])
    idx_or = random.randint(sk_or, (cfg.n_agents_max,), 0, 4)
    
    # Gestion de la survie initiale
    alive_mask = jnp.zeros((cfg.n_agents_max,), dtype=jnp.int32).at[1:cfg.n_agents_init+1].set(1)
    
    # Paramètres réseau et état RNN
    params = jnp.broadcast_to(
        agent_params, (cfg.n_agents_max, model.num_params)
    )

    policy_states = model.reset_b(jnp.zeros(cfg.n_agents_max))
    
    # Création de l'objet AgentState
    agents = AgentState(
        position=random.randint(sk_pos, (cfg.n_agents_max, 2), 10, cfg.grid_length-10),
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

    # 4. Grille d'occupation et observations
    grid_agents = jnp.zeros((cfg.grid_length, cfg.grid_length), dtype=jnp.int32)
    grid_agents = grid_agents.at[agents.position[:, 0], agents.position[:, 1]].add(agents.alive)
    grid = jnp.concatenate([
        grid_resources,          # (n_types, L, L)  -> déjà n canaux
        grid_agents[None],       # (1, L, L)
        grid_walls[None],        # (1, L, L)
    ], axis=0)   
    pos = agents.position
    
    # 5. État final

    key, key_env = jax.random.split(key)

    # Un environnement fige saute la pre-croissance : le fichier EST la grille,
    # et la faire pousser en diluerait justement la structure qu'on a dessinee.
    if fige:
        grid_resources_grown_bis = grid_resources
    else:
        init_carry = (grid_resources, key_env)

        grid_resources_grown, _ = jax.lax.fori_loop(
            0,
            cfg.pre_growth_step,
            # pas de frein pendant la pre-croissance (cf. init_state)
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
        invasion_faite=jnp.zeros(()),
    )
    
    return state


from jax import lax

from jax import random
from functools import partial

from EcoEvoJax.source.agent import metaRNNPolicyState_bcppr
from simulation.update_env import resources_growth
from simulation.agent_mov import vmap_update_agents_position, get_obs_vector
from simulation.data_class import SimState
from simulation.one_simulation import run_simulation_chunk

def rotations_for(resources):
    """Decalages de canaux non triviaux, pour n ressources.

    Le decalage 0 est l'identite, et n aussi : une liste en dur (1,2) donne donc
    un doublon exact de la baseline des qu'on descend a 2 ressources -- une
    condition "poison_to_poison" qui coute la moitie du calcul du lab pour rien.
    """
    return tuple(range(1, len(resources)))



@partial(jax.jit, static_argnames=['cfg','model','env_file'])
def launch_lab_env(agent_params,key_env,key_sim,cfg,model,env_file=None): 
    state = init_state_lab(key_env,cfg, model,agent_params, env_file=env_file)
    # import jax
    # jax.debug.print("inj ok ? {b}", b=jnp.allclose(state.agents.params[1], agent_params))
    key, subkey = jax.random.split(key_sim)
    keys_chunk = jax.random.split(subkey, cfg.lab_time_steps)
    state, outputs = run_simulation_chunk(state,model,keys_chunk, cfg)

    
    return state,outputs

# stocks de l'env high_res, par IDENTITE de ressource. Expose pour que les plots
# puissent afficher le plafond disponible sans redupliquer ces valeurs.
HIGH_RES_COUNTS = {"good": 40, "medium": 40, "poison": 40}
# idem pour l'env pauvre. Expose et non ecrit dans launch_env_low_res : c'est la
# seule source de ces comptes, que tools/make_lab_envs relit pour caler ses
# candidats low_res.
LOW_RES_COUNTS = {"good": 3, "medium": 2, "poison": 10}
# Nombre initial pour une identite absente des tables ci-dessus : les tables
# sont indexees par NOM, et une 4e ressource n'y figure pas.
DEFAUT_COUNT = 40

# Facteur applique aux taux de croissance dans l'env de test.
#
# Il n'y a pas de mortalite des ressources : la croissance ne s'arrete que sur
# les cases occupees, donc le seul equilibre est la grille pleine. Aux taux de
# la sim principale l'env de lab passe de 18% a 39% d'occupation en 1000 pas et
# sature a 97% -- il n'est pas stationnaire, il est croissant, et la
# distribution des observations derive tout au long du rollout.
#
# On ralentit donc la repousse pour qu'elle compense approximativement ce qu'UN
# agent consomme, plutot que de la depasser d'un facteur 5 a 10. A 0.1 la derive
# sans agent tombe a +22 cases sur 1000 pas (+18%), contre +190 (+119%) a 1.0.
#
# Compensation APPROXIMATIVE : la consommation depend de la condition testee
# (un agent a memoire intacte mange plus de poison en debut de vie qu'un agent
# ablate), donc aucune valeur unique ne stabilise les deux exactement. Verifier
# la stationnarite sur les donnees plutot que de s'y fier.
HIGH_RES_GROWTH_SCALE = 0.1


def launch_env_high_res(agent_params, key_env, key_sim, cfg, model, rot=0):
    cfg = cfg._replace(
        grid_length=30,
        n_agents_max=2,
        reproduction_on=True,
        resources_growth=False,
        pre_growth_step=200,
        log_obs=True,          
    )
    count_by_id = HIGH_RES_COUNTS
    poison = ressource_la_plus_lente(cfg.resources)

    cfg = cfg._replace(resources=tuple(
        r.replace(init_number_of_resources=count_by_id.get(label_of(r.id), DEFAUT_COUNT),
                  prob_factor=poison.prob_factor * HIGH_RES_GROWTH_SCALE,
                  pop_res_prob=poison.pop_res_prob * HIGH_RES_GROWTH_SCALE)
        for r in cfg.resources
    ))
    cfg = cfg._replace(resources=rotate_resources(cfg.resources, rot))
    return launch_lab_env(agent_params, key_env, key_sim, cfg, model,
                          env_file=env_file_pour(cfg, "high_res"))
    
def launch_env_high_res_with_clones(agent_params,key_env,key_sim,cfg,model):
    
    ### High resources
    cfg = cfg._replace(
        grid_length=30,
        n_agents_max=5,
        reproduction_on = True,
        resources_growth=False,
        pre_growth_step = 200,
        log_obs=True,          # idem
    )
    count_by_id =HIGH_RES_COUNTS
    poison = ressource_la_plus_lente(cfg.resources)

    cfg = cfg._replace(resources=tuple(
        r.replace(init_number_of_resources=count_by_id.get(label_of(r.id), DEFAUT_COUNT),
                  prob_factor=4*poison.prob_factor * HIGH_RES_GROWTH_SCALE,
                  pop_res_prob=4*poison.pop_res_prob * HIGH_RES_GROWTH_SCALE)
        for r in cfg.resources
    ))
    state,outputs = launch_lab_env(agent_params,key_env,key_sim,cfg,model,
                                   env_file=env_file_pour(cfg, "high_res"))
    return state,outputs
    
def launch_env_low_res(agent_params,key_env,key_sim,cfg,model):
    ### Low resources
    cfg = cfg._replace(
        grid_length=30,
        n_agents_max=2,
        reproduction_on = False,
        resources_growth=False,
        pre_growth_step = 50,
        log_obs=True,          # idem
    )
    count_by_id = LOW_RES_COUNTS

    cfg = cfg._replace(resources=tuple(
        r.replace(init_number_of_resources=count_by_id.get(label_of(r.id), DEFAUT_COUNT))
        for r in cfg.resources
    ))
    state,outputs = launch_lab_env(agent_params,key_env,key_sim,cfg,model,
                                   env_file=env_file_pour(cfg, "low_res"))
    return state,outputs    



def rotate_resources(resources, shift):
    """Rotation cyclique : la ressource du canal k passe au canal (k + shift) % n."""
    n = len(resources)
    return tuple(resources[(k - shift) % n] for k in range(n))

def launch_adaptation_env(agent_params, key_env, key_sim, cfg, model):
    """Le meme genome dans chaque permutation NON TRIVIALE des canaux.

    A une seule ressource il n'en existe aucune : rotations_for rend un tuple
    vide, la boucle ne tourne pas, et `tree_map(f, *[])` levait un TypeError
    illisible sur l'argument `tree` manquant. L'appelant doit tester avant.
    """
    rotations = rotations_for(cfg.resources)
    if not rotations:
        raise ValueError(
            f"launch_adaptation_env : {len(cfg.resources)} ressource(s), donc "
            "aucune permutation non triviale. Tester rotations_for(cfg.resources) "
            "avant d'appeler.")
    states, outputs = [], []
    for rot in rotations:
        s, o = launch_env_high_res(agent_params, key_env, key_sim, cfg, model, rot=rot)
        states.append(s)
        outputs.append(o)
    states  = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *states)
    outputs = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *outputs)
    return states, outputs


def mutate_params(params, key, cfg):
    """Meme operateur que la reproduction dans one_simulation.step."""
    k_masque, k_bruit = random.split(key)
    masque = random.bernoulli(k_masque, p=cfg.param_mutate, shape=params.shape)
    bruit  = cfg.mutation_var * random.normal(k_bruit, shape=params.shape)
    return params + bruit * masque


def vmap_mutate(params, keys, cfg):
    return vmap(mutate_params, in_axes=(None, 0, None))(params, keys, cfg)


def vmap_over_agents_env_lab_high_res(list_agents_param,key_env,key_sim,model,cfg):
    return vmap(launch_env_high_res,in_axes=(0,None,0,None,None))(list_agents_param,key_env,key_sim,cfg,model)

def vmap_over_agents_env_lab_low_res(list_agents_param,key_env,key_sim,model,cfg):
    return vmap(launch_env_low_res,in_axes=(0,None,0,None,None))(list_agents_param,key_env,key_sim,cfg,model)

def vmap_over_agents_env_lab_high_res_with_clones(list_agents_param,key_env,key_sim,model,cfg):
    return vmap(launch_env_high_res_with_clones,in_axes=(0,None,0,None,None))(list_agents_param,key_env,key_sim,cfg,model)

def vmap_over_agents_env_lab_adapt(list_agents_param, key_env, key_sim, model, cfg):
    return vmap(launch_adaptation_env, in_axes=(0, None, 0, None, None))(
        list_agents_param, key_env, key_sim, cfg, model)