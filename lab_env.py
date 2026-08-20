import os
import numpy as np
import jax
from jax import random,vmap
import jax.numpy as jnp

from simulation.data_class import AgentState,SimState,LABELS
from simulation.agent_mov import get_obs_vector
from simulation.update_env import resources_growth

import numpy as np
import matplotlib.pyplot as plt



def init_state_lab(key, cfg, model,agent_params):
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



@partial(jax.jit, static_argnames=['cfg','model'])
def launch_lab_env(agent_params,key_env,key_sim,cfg,model): 
    state = init_state_lab(key_env,cfg, model,agent_params)
    # import jax
    # jax.debug.print("inj ok ? {b}", b=jnp.allclose(state.agents.params[1], agent_params))
    key, subkey = jax.random.split(key_sim)
    keys_chunk = jax.random.split(subkey, cfg.lab_time_steps)
    state, outputs = run_simulation_chunk(state,model,keys_chunk, cfg)

    
    return state,outputs

# stocks de l'env high_res, par IDENTITE de ressource. Expose pour que les plots
# puissent afficher le plafond disponible sans redupliquer ces valeurs.
HIGH_RES_COUNTS = {"good": 40, "medium": 40, "poison": 40}

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
        reproduction_on=False,
        resources_growth=True,
        pre_growth_step=200,
        log_obs=True,          
    )
    count_by_id = HIGH_RES_COUNTS
    # Meme croissance pour les trois types, calee sur celle du poison. En
    # gardant les taux de la sim principale, le medium repousserait 20x plus
    # vite que le poison : une asymetrie de consommation entre types serait
    # alors indissociable d'une asymetrie de REPOUSSE. L'env de test n'a pas a
    # etre ecologiquement realiste, il doit etre symetrique et stationnaire.
    # Lu depuis le poison plutot qu'en dur, pour suivre BASE_RESOURCES.
    poison = next(r for r in cfg.resources if LABELS[r.id] == "poison")
    cfg = cfg._replace(resources=tuple(
        r.replace(init_number_of_resources=count_by_id[LABELS[r.id]],
                  prob_factor=poison.prob_factor * HIGH_RES_GROWTH_SCALE,
                  pop_res_prob=poison.pop_res_prob * HIGH_RES_GROWTH_SCALE)
        for r in cfg.resources
    ))
    cfg = cfg._replace(resources=rotate_resources(cfg.resources, rot))
    return launch_lab_env(agent_params, key_env, key_sim, cfg, model)
    
def launch_env_high_res_with_clones(agent_params,key_env,key_sim,cfg,model):
    
    ### High resources
    cfg = cfg._replace(
        grid_length=30,
        n_agents_max=5,
        reproduction_on = False,
        resources_growth=True,
        pre_growth_step = 200,
        log_obs=True,          # idem
    )
    count_by_id =HIGH_RES_COUNTS
    poison = next(r for r in cfg.resources if LABELS[r.id] == "poison")

    cfg = cfg._replace(resources=tuple(
        r.replace(init_number_of_resources=count_by_id[LABELS[r.id]],
                  prob_factor=4*poison.prob_factor * HIGH_RES_GROWTH_SCALE,
                  pop_res_prob=4*poison.pop_res_prob * HIGH_RES_GROWTH_SCALE)
        for r in cfg.resources
    ))
    state,outputs = launch_lab_env(agent_params,key_env,key_sim,cfg,model)
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
    count_by_id = {"good": 3, "medium": 2, "poison": 10}
    
    
    cfg = cfg._replace(resources=tuple(
        r.replace(init_number_of_resources=count_by_id[LABELS[r.id]]) for r in cfg.resources
    ))
    state,outputs = launch_lab_env(agent_params,key_env,key_sim,cfg,model)
    return state,outputs    



def rotate_resources(resources, shift):
    """Rotation cyclique : la ressource du canal k passe au canal (k + shift) % n."""
    n = len(resources)
    return tuple(resources[(k - shift) % n] for k in range(n))

def launch_adaptation_env(agent_params, key_env, key_sim, cfg, model):
    states, outputs = [], []
    for rot in rotations_for(cfg.resources):
        s, o = launch_env_high_res(agent_params, key_env, key_sim, cfg, model, rot=rot)
        states.append(s)
        outputs.append(o)
    states  = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *states)
    outputs = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *outputs)
    return states, outputs


def vmap_over_agents_env_lab_high_res(list_agents_param,key_env,key_sim,model,cfg):
    return vmap(launch_env_high_res,in_axes=(0,None,0,None,None))(list_agents_param,key_env,key_sim,cfg,model)

def vmap_over_agents_env_lab_low_res(list_agents_param,key_env,key_sim,model,cfg):
    return vmap(launch_env_low_res,in_axes=(0,None,0,None,None))(list_agents_param,key_env,key_sim,cfg,model)

def vmap_over_agents_env_lab_high_res_with_clones(list_agents_param,key_env,key_sim,model,cfg):
    return vmap(launch_env_high_res_with_clones,in_axes=(0,None,0,None,None))(list_agents_param,key_env,key_sim,cfg,model)

def vmap_over_agents_env_lab_adapt(list_agents_param, key_env, key_sim, model, cfg):
    return vmap(launch_adaptation_env, in_axes=(0, None, 0, None, None))(
        list_agents_param, key_env, key_sim, cfg, model)