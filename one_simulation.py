# JAX's syntax is (for the most part) same as NumPy's!
# There is also a SciPy API support (jax.scipy)
import jax.numpy as jnp
import numpy as np
import jax
# Special transform functions (we'll understand what these are very soon!)
from jax import grad, jit, vmap, pmap

# JAX's low level API 
# (lax is just an anagram for XLA, not completely sure how they came up with name JAX)
from jax import lax

from jax import random
from functools import partial

from EcoEvoJax.source.agent import metaRNNPolicyState_bcppr
from simulation.update_env import resources_growth,resources_growth_biger,resources_growth_max
from simulation.agent_mov import vmap_update_agents_position, get_obs_vector
from simulation.data_class import SimState


@partial(jax.jit, static_argnames=['cfg','model'])
def run_simulation_chunk(state,model,keys, cfg):
    

    def step(state, subkey):
        grid         = state.grid
        agents       = state.agents
        step_idx     = state.step
        grid_resources, grid_agents = grid
        key_env, key_action, key_respawn, key_mut = random.split(subkey, 4)
        
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
        # Update position
        actions_logit, new_policy_states = model.get_actions(state, state.agents.params,
                                                                  state.agents.policy_states)
        
        actions_id= jax.nn.one_hot(random.categorical(key_action, actions_logit * 50, axis=-1), 4)
        
        acts = jnp.argmax(actions_id,axis=1)
        agents= vmap_update_agents_position(agents,acts,cfg.n)
        
        pos = agents.position

        # Compute intern energy : resource consumption - energy decay 
        local_resources = grid_resources[pos[:, 0], pos[:, 1]]
        rewards = local_resources * survives_int
        new_energy = energies + rewards - cfg.energy_decay * jnp.where(acts==0,0.5,1)


        # ------- 3. Environment dynamic -------
        # Agents consume resources on the grid
        # Use max() instead of set() to avoid non-determinism with repeated indices on GPU
        consumed = jnp.zeros_like(grid_resources).at[pos[:, 0], pos[:, 1]].max(survives_int)
        grid_resources = grid_resources * (1 - consumed)

        # Resources spread via convolution
        grid_resources,_ = resources_growth((grid_resources,key_env),cfg)


        # ------- 4. Reproduction -------
        # Compute the number of births (Number of parents VS Number free idx)
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
        #new_positions = random.randint(key_respawn, (cfg.n_agents_max, 2), minval=0, maxval=cfg.n)


        # ------- 5. Global update -------
        # Put new born in the state with their initial state
        final_pos = pos.at[free_indices].set(pos[parent_indices,:])
        final_energy = new_energy.at[free_indices].set(1.0)
        final_time_under = new_time_under.at[free_indices].set(0)
        final_time_over = new_time_over.at[free_indices].set(0)
        final_alive = survives_int.at[free_indices].set(1)
        final_parent_id = agents.parent_id.at[free_indices].set(parent_indices)
        final_born_step = agents.born_step.at[free_indices].set(step_idx)
        
        final_alive_without_0 = final_alive.at[0].set(0)
        
        key_child_param_mutate,key_params = random.split(key_mut)
        
        parameters_to_mutate = random.bernoulli(key_child_param_mutate, p=cfg.param_mutate, shape=(cfg.n_agents_max, agents.params.shape[1])).astype(jnp.int32)
        
        mutation = cfg.mutation_var * random.normal(key_params, shape=(cfg.n_agents_max, agents.params.shape[1]))
        final_params = agents.params.at[free_indices].set(
                    agents.params[parent_indices] +mutation*parameters_to_mutate)
        
        final_policy_states =metaRNNPolicyState_bcppr(
                lstm_h=new_policy_states.lstm_h.at[free_indices].set(jnp.zeros(new_policy_states.lstm_h.shape[1])),
                lstm_c=new_policy_states.lstm_c.at[free_indices].set(jnp.zeros(new_policy_states.lstm_c.shape[1])),
                keys=new_policy_states.keys)

        # Update spatial grid with agents positions
        grid_agents = jnp.zeros_like(grid_resources, dtype=jnp.int32)
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
        
        new_state = SimState(
            grid=jnp.stack((grid_resources, grid_agents)),
            agents=new_agents,
            step=step_idx+1,
            obs=get_obs_vector(grid, final_pos,cfg.agent_view),
            last_actions=actions_id,
            rewards=jnp.expand_dims(rewards, 1).astype(jnp.float32)
        )
        
        return new_state, state

    state_final, outputs = lax.scan(step, state, keys)

    return state_final, outputs


