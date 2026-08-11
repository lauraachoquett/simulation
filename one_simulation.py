# JAX's syntax is (for the most part) same as NumPy's!
# There is also a SciPy API support (jax.scipy)
import jax.numpy as jnp
import numpy as np
import jax
# Special transform functions (we'll understand what these are very soon!)

# JAX's low level API 
# (lax is just an anagram for XLA, not completely sure how they came up with name JAX)
from jax import lax

from jax import random
from functools import partial

from EcoEvoJax.source.agent import metaRNNPolicyState_bcppr
from simulation.update_env import resources_growth
from simulation.agent_mov import vmap_update_agents_position, get_obs_vector
from simulation.data_class import SimState

from typing import NamedTuple
import jax

class StepLog(NamedTuple):
    position:  jax.Array   # (N, 2)
    alive:     jax.Array   # (N,)
    time_under_min_energy :     jax.Array   # (N,)
    energy:    jax.Array   # (N,)
    parent_id: jax.Array   # (N,)  -> généalogie
    born_step: jax.Array   # (N,)  -> généalogie / MRCA
    actions:   jax.Array   # (N, 4) -> si besoin pour les vidéos
    rewards:   jax.Array   # (N, 1) -> R0
    grid:      jax.Array   # (3, L, L) -> si tes vidéos affichent la grille
    step :     int
    obs :      jax.Array
    consumed_res: jax.Array   # (n_types,) -> unités retirées de la grille PENDANT ce step
    saw_res:   jax.Array   # (N, n_types) -> ce type est-il dans le champ de vision ?
    ate_res:   jax.Array   # (N, n_types) -> ce type a-t-il été consommé PENDANT ce step ?
    

@partial(jax.jit, static_argnames=['cfg','model'])
def run_simulation_chunk(state,model,keys, cfg):
    

    def step(state, subkey):
        
        grid         = state.grid
        agents       = state.agents
        step_idx     = state.step
        
        n_types = grid.shape[0] - 2          # tout sauf agents + murs
        grid_resources = grid[:n_types]      # (n_types, L, L)  -> slice, garde l'axe type
        grid_agents    = grid[n_types]       # (L, L)           -> index, un seul canal
        grid_walls     = grid[n_types + 1] 

        key_env, key_action, key_respawn, key_mut = random.split(subkey, 4)
        res = jax.tree.map(lambda *xs: jnp.stack(xs), *cfg.resources) # Resource parameters

        # ------- 1. Evaluate alive agents and energy : -------
        energies = agents.energy
        is_alive = agents.alive > 0

        # Incremente time if energy is above or below a set threshold
        new_time_under = jnp.where((energies < 0) & is_alive, agents.time_under_min_energy + 1, 0) #If true a bit closer to death time
        new_time_over = jnp.where((energies > cfg.min_energy_repr) & is_alive, agents.time_over_energy_repr + 1, 0) # If true a bit closer to reproduction time

        survives = (new_time_under < cfg.time_to_die) & is_alive #Surviving agents, already alive and not about to die
        reproduces = new_time_over > cfg.time_above_repr # Check if reproducing time

        new_time_over = jnp.where(reproduces, 0, new_time_over) # If going to reproduce reset reproduction time to zero
        survives_int = jnp.where(survives, 1, 0) #Bool alive array to Int alive array


        # ------- 2. Movement and Physiological update -------
        actions_logit, new_policy_states = model.get_actions(
            state, state.agents.params, state.agents.policy_states
        )
        if cfg.dumb_agent:  
            actions_id = jax.nn.one_hot(
                random.randint(key_action, shape=(cfg.n_agents_max,), minval=0, maxval=4),
                4,
            )
        else:
            
            actions_id = jax.nn.one_hot(
                random.categorical(key_action, actions_logit  / cfg.temperature, axis=-1),
                4,
            )

        acts = jnp.argmax(actions_id, axis=1)
        agents= vmap_update_agents_position(agents,acts,cfg.grid_length) 
        
        pos = agents.position
        
        if cfg.letal_wall :
            survives_int = jnp.where(grid_walls[pos[:, 0], pos[:, 1]]==1,0,survives_int)
            reproduces = jnp.where(grid_walls[pos[:, 0], pos[:, 1]]==1,0,reproduces)
        

        # Compute intern energy : resource consumption - energy decay
        local_resources = grid_resources[:, pos[:, 0], pos[:, 1]].T
        gain = local_resources @ res.delta_energy
        rewards = survives_int * gain
        new_energy = jnp.minimum(energies + rewards - cfg.energy_decay * jnp.where(acts==0, cfg.factor_energy_decay_not_moving, 1) * survives_int, cfg.energy_max)

        # ------- P(manger | en vue) : les deux termes, par agent et par type -------
        # `ate` est exact, contrairement au signe de la recompense : un agent sur une
        # case good+poison a un gain net nul et serait manque par `rewards > 0`.
        ate_res_step = (local_resources > 0) & (survives_int[:, None] > 0)   # (N, n_types)
        # obs est (N, side, side, C), canal en DERNIER ; le padding hors grille vaut
        # -1, ecarte par le test > 0. C'est l'observation sur laquelle l'agent DECIDE.
        saw_res_step = (state.obs[..., :n_types] > 0).any(axis=(1, 2))       # (N, n_types)


        # ------- 3. Environment dynamic -------
        # Agents consume resources on the grid
        # Use max() instead of set() to avoid non-determinism with repeated indices on GPU
        consumed = jnp.zeros((cfg.grid_length, cfg.grid_length), dtype=grid_resources.dtype) # (L, L)
        consumed = consumed.at[pos[:, 0], pos[:, 1]].max(survives_int)     # (L, L)

        # Ce qui disparait de l'environnement, par type. A calculer AVANT la remise a zero.
        # Attention : `consumed` est un masque par CELLULE (max sur les agents), donc une
        # cellule occupee par plusieurs agents n'est vidée qu'une fois alors que chacun d'eux
        # encaisse `gain` -> sum(rewards) != consumed_per_type @ delta_energy.
        consumed_per_type = (grid_resources * consumed[None]).sum(axis=(1, 2))   # (n_types,)

        grid_resources = grid_resources * (1 - consumed[None]) #  consumed[None]: (n_types, L , L)

        # Resources spread via convolution
        if cfg.resources_growth : 
            grid_resources,_ = resources_growth((grid_resources,key_env),cfg)
            grid_resources = jnp.where(grid_walls==1,0,grid_resources)



        # ------- 4. Reproduction -------
        # Compute the number of births (Number of parents VS Number free idx)
        
        if cfg.reproduction_on:
            nb_parents = reproduces.sum()
            nb_free_places = cfg.n_agents_max - survives.sum()
            nb_births = jnp.minimum(nb_parents, nb_free_places)

            free_idx = jnp.nonzero(survives == 0, size=cfg.n_agents_max, fill_value=0)[0]
            sort_free_idx = jnp.sort(free_idx)
            sort_free_idx_ordered = jnp.flip(sort_free_idx)

            spawn_mask = jnp.arange(cfg.n_agents_max) < nb_births
            free_indices = jnp.where(spawn_mask, sort_free_idx_ordered, 0)

            all_potential_parents = jnp.nonzero(reproduces, size=cfg.n_agents_max, fill_value=0)[0]
            parent_indices = jnp.where(spawn_mask, all_potential_parents, 0)
            
            # Draw initial positions for the new borns
            if cfg.random_pos_offspring:
                new_positions = random.randint(key_respawn, (cfg.n_agents_max, 2), minval=1, maxval=cfg.grid_length-1)
            else :
                new_positions = pos[parent_indices,:]


            # ------- 5. Global update -------
            # Put new born in the state with their initial state
            final_pos = pos.at[free_indices].set(new_positions)
            final_energy = new_energy.at[free_indices].set(cfg.starting_energy)
            final_time_under = new_time_under.at[free_indices].set(0)
            final_time_over = new_time_over.at[free_indices].set(0)
            final_alive = survives_int.at[free_indices].set(1)
            final_parent_id = agents.parent_id.at[free_indices].set(parent_indices)
            final_born_step = agents.born_step.at[free_indices].set(step_idx)
            
            final_alive_without_0 = final_alive.at[0].set(0)
            
            key_child_param_mutate,key_params = random.split(key_mut)
            
            parameters_to_mutate = random.bernoulli(key_child_param_mutate, p=cfg.param_mutate, shape=(cfg.n_agents_max, agents.params.shape[1])).astype(jnp.int32)
            number_of_parameters_mutating = parameters_to_mutate.sum(axis=1)
            
            mutation = cfg.mutation_var * random.normal(key_params, shape=(cfg.n_agents_max, agents.params.shape[1]))
            final_params = agents.params.at[free_indices].set(
                        agents.params[parent_indices] +mutation*parameters_to_mutate)
            
            final_policy_states =metaRNNPolicyState_bcppr(
                    lstm_h=new_policy_states.lstm_h.at[free_indices].set(jnp.zeros(new_policy_states.lstm_h.shape[1])),
                    lstm_c=new_policy_states.lstm_c.at[free_indices].set(jnp.zeros(new_policy_states.lstm_c.shape[1])),
                    keys=new_policy_states.keys)

        else : 
            final_pos= pos
            final_energy = new_energy
            final_time_under = new_time_under
            final_time_over = new_time_over
            final_alive = survives_int
            final_parent_id = agents.parent_id
            final_born_step = agents.born_step
            final_alive_without_0 = final_alive.at[0].set(0)
            final_policy_states = new_policy_states
            final_params = agents.params
            
            
        # Update spatial grid with agents positions
        grid_agents = jnp.zeros_like(grid_walls, dtype=jnp.int32)
        grid_agents = grid_agents.at[final_pos[:, 0], final_pos[:, 1]].add(final_alive_without_0)
        
        new_agents = agents.replace(
            position=final_pos,
            energy=final_energy,
            time_under_min_energy=final_time_under,
            time_over_energy_repr=final_time_over,
            alive=final_alive_without_0,
            parent_id=final_parent_id,
            born_step=final_born_step,
            policy_states = final_policy_states,
            params = final_params
        )
        
        new_grid = jnp.concatenate([grid_resources, grid_agents[None], grid_walls[None]], axis=0)
        obs = get_obs_vector(new_grid, (final_pos, new_agents.orientation), cfg.agent_view)

        
        new_state = SimState(
            grid=new_grid,
            agents=new_agents,
            step=step_idx+1,
            obs=obs,
            last_actions=actions_id,
            rewards=jnp.expand_dims(rewards, 1).astype(jnp.float32)
        )
        
        log = StepLog(
            position=state.agents.position,
            alive=state.agents.alive,
            energy=state.agents.energy,
            parent_id=state.agents.parent_id,
            born_step=state.agents.born_step,
            actions=state.last_actions,
            rewards=state.rewards,
            grid=state.grid,    # retire-le si tes vidéos n'en ont pas besoin
            step = step_idx,
            time_under_min_energy = state.agents.time_under_min_energy,
            obs=state.obs,
            # Seuls champs produits PENDANT le step (les autres logguent l'etat de DEBUT
            # de step). C'est l'alignement voulu : consumed_res[t] est preleve sur le
            # stock grid[t], et ate_res[t] suit l'observation saw_res[t] du meme pas.
            consumed_res = consumed_per_type,
            saw_res = saw_res_step,
            ate_res = ate_res_step,
        )
        
        return new_state, log
    state_final, outputs = lax.scan(step, state, keys)

    return state_final, outputs

