import os
import numpy as np
import jax
from jax import random,vmap
import jax.numpy as jnp

from simulation.data_class import AgentState,SimState
from simulation.agent_mov import get_obs_vector
from simulation.update_env import resources_growth

import numpy as np
import matplotlib.pyplot as plt

def init_state_lab(key, cfg, model,agent_params):
    # 1. Grille de ressources
    key, subkey_grid = random.split(key)
    
    position_res=random.randint(subkey_grid, (cfg.init_number_of_resources, 2), 0, cfg.grid_length)
    grid_resources = jnp.zeros((cfg.grid_length, cfg.grid_length), dtype=jnp.int32)
    grid_resources = grid_resources.at[position_res[:, 0], position_res[:, 1]].set(1)
    
    # 2. Grille avec les murs
    grid_walls = jnp.zeros((cfg.grid_length, cfg.grid_length), dtype=jnp.int32)
    grid_walls = grid_walls.at[0,:].set(1)
    grid_walls = grid_walls.at[:,0].set(1)
    grid_walls = grid_walls.at[-1,:].set(1)
    grid_walls = grid_walls.at[:,-1].set(1)
    grid_resources = jnp.where(grid_walls==1,0,grid_resources)
    
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
    grid = jnp.stack((grid_resources, grid_agents,grid_walls))
    pos = agents.position
    
    # 5. État final

    key, key_env = jax.random.split(key)

    init_carry = (grid_resources, key_env)

    grid_resources_grown, _ = jax.lax.fori_loop(
        0,
        cfg.pre_growth_step,
        lambda i, carry: resources_growth(carry, cfg),
        init_carry
    )
    grid_resources_grown_bis = jnp.where(grid_walls==1,0,grid_resources_grown)
    grid = jnp.stack((grid_resources_grown_bis, grid_agents,grid_walls))

    obs = get_obs_vector(grid, (pos, agents.orientation), cfg.agent_view)
    state = SimState(
        grid=grid,
        agents=agents,
        step=0,
        obs=obs,
        last_actions=jnp.zeros((cfg.n_agents_max, 4)),
        rewards=jnp.zeros((cfg.n_agents_max, 1))
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




@partial(jax.jit, static_argnames=['cfg','model'])
def launch_lab_env(agent_params,key_env,key_sim,cfg,model): 
    state = init_state_lab(key_env,cfg, model,agent_params)
    # import jax
    # jax.debug.print("inj ok ? {b}", b=jnp.allclose(state.agents.params[1], agent_params))
    key, subkey = jax.random.split(key_sim)
    keys_chunk = jax.random.split(subkey, 500)
    state, outputs = run_simulation_chunk(state,model,keys_chunk, cfg)

    
    return state,outputs


def launch_env_high_res(agent_params,key_env,key_sim,cfg,model):
    
    ### High resources
    cfg = cfg._replace(
        grid_length=30,
        n_agents_max=2,
        reproduction_on = False,
        resources_growth=True,
        pre_growth_step = 200,
        init_number_of_resources=20,
        starting_energy = 1,
        letal_wall=True,
    )
    state,outputs = launch_lab_env(agent_params,key_env,key_sim,cfg,model)
    return state,outputs
    
def launch_env_high_res_with_clones(agent_params,key_env,key_sim,cfg,model):
    
    ### High resources
    cfg = cfg._replace(
        grid_length=30,
        n_agents_max=5,
        reproduction_on = False,
        resources_growth=True,
        pre_growth_step = 200,
        init_number_of_resources=20,
        starting_energy = 1,
        letal_wall=True,
    )
    state,outputs = launch_lab_env(agent_params,key_env,key_sim,cfg,model)
    return state,outputs
    
def launch_env_low_res(agent_params,key_env,key_sim,cfg,model):
    ### Low resources
    cfg = cfg._replace(
        grid_length=30,
        n_agents_max=2,
        reproduction_on = False,
        resources_growth=False, 
        pre_growth_step = 200,
        init_number_of_resources=5,
        starting_energy = 1,
        letal_wall=True,
    )
    state,outputs = launch_lab_env(agent_params,key_env,key_sim,cfg,model)
    return state,outputs    

def vmap_over_agents_env_lab_high_res(list_agents_param,key_env,key_sim,model,cfg):
    return vmap(launch_env_high_res,in_axes=(0,None,0,None,None))(list_agents_param,key_env,key_sim,cfg,model)

def vmap_over_agents_env_lab_low_res(list_agents_param,key_env,key_sim,model,cfg):
    return vmap(launch_env_low_res,in_axes=(0,None,0,None,None))(list_agents_param,key_env,key_sim,cfg,model)

def vmap_over_agents_env_lab_high_res_with_clones(list_agents_param,key_env,key_sim,model,cfg):
    return vmap(launch_env_high_res_with_clones,in_axes=(0,None,0,None,None))(list_agents_param,key_env,key_sim,cfg,model)

