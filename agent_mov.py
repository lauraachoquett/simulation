import jax.numpy as jnp
from jax import lax,vmap
from simulation.data_class import AgentState

def action_depl_theta(action_id):
    actions = jnp.array([
        jnp.array([0,0]),
        jnp.array([jnp.pi/2,0]),
        jnp.array([-jnp.pi/2,0]),
        jnp.array([0,1]),
        
    ])
    dtheta     = actions[action_id,0]
    deplacement = actions[action_id,1].astype(jnp.int32)
    return dtheta,deplacement


def update_agent_position(agent: AgentState, action_id, n):
    dtheta, dist = action_depl_theta(action_id)
    
    new_ori = agent.orientation + dtheta
    
    move = jnp.array([
        dist * jnp.cos(new_ori),
        dist * jnp.sin(new_ori)
    ]).astype(jnp.int32)
    
    new_pos = lax.clamp(0, agent.position + move, n - 1)
    
    return agent.replace(orientation=new_ori, position=new_pos)



def vmap_update_agents_position(agents_state,actions_id,n):
    agents_state = vmap(update_agent_position,in_axes=(0,0,None))(agents_state,actions_id,n)
    return agents_state
    
    
def get_single_obs(grid, pos, radius):
    side = 2 * radius + 1
    
    padding = ((0, 0), (radius, radius), (radius, radius))
    padded_grid = jnp.pad(grid, padding)
    
    obs = lax.dynamic_slice(padded_grid, (0, pos[0], pos[1]), (2, side, side))
    
    return jnp.transpose(obs, (1, 2, 0))

get_obs_vector = vmap(get_single_obs, in_axes=(None, 0, None))