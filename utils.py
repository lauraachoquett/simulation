
import os
import numpy as np
import matplotlib.colors as mcolors
import jax
from jax import random
import jax.numpy as jnp
from simulation.data_class import AgentState,SimState
from simulation.agent_mov import get_obs_vector

def plot_evolution(outputs, exp_dir,name_fig = ''):
    # Préparation des données
    # Population : somme des agents vivants à chaque pas de temps
    pop_history = np.array(outputs.agents.alive.sum(axis=1))
    # Ressources : somme de la grille (canal 0) à chaque pas de temps
    res_history = np.array(outputs.grid[:, 0, :, :].sum(axis=(1, 2)))
    generations = np.arange(len(pop_history))

    # Configuration de la figure (Evolution à gauche : ratio 3, Initiale à droite : ratio 1)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [3, 1]})
    
    # --- 1. AXE GAUCHE : ÉVOLUTION (axes[0]) ---
    ax_evo_agents = axes[0]
    color_agents = 'tab:red'
    ax_evo_agents.set_xlabel('Générations')
    ax_evo_agents.set_ylabel('Nombre d’agents', color=color_agents)
    ax_evo_agents.plot(generations, pop_history, color=color_agents, linewidth=2, label='Agents')
    ax_evo_agents.tick_params(axis='y', labelcolor=color_agents)
    ax_evo_agents.grid(True, alpha=0.3)

    # Second axe Y pour les ressources
    ax_evo_res = ax_evo_agents.twinx()
    color_res = 'tab:green'
    ax_evo_res.set_ylabel('Quantité de ressources', color=color_res)
    ax_evo_res.plot(generations, res_history, color=color_res, linewidth=2, label='Ressources')
    ax_evo_res.tick_params(axis='y', labelcolor=color_res)
    
    ax_evo_agents.set_title('Dynamique de la simulation')

    # --- 2. AXE DROIT : CONFIGURATION INITIALE (axes[1]) ---
    ax_init = axes[1]
    cmap_res = mcolors.ListedColormap(["white", "#4CAF50"])
    
    # Affichage de la grille de ressources à t=0
    initial_grid = outputs.grid[0, 0, :, :]
    ax_init.imshow(initial_grid, cmap=cmap_res, vmin=0, vmax=1, interpolation="nearest")
    
    # Calcul de la taille des marqueurs basée sur la géométrie de l'axe
    fig.canvas.draw()
    bbox = ax_init.get_window_extent()
    grid_h, grid_w = initial_grid.shape
    cell_px = min(bbox.width / grid_w, bbox.height / grid_h)
    cell_pts = cell_px * 72 / fig.dpi

    # Affichage des agents à t=0
    # On récupère les positions (pos_history attendu en [temps, agent, coord])
    # et l'état de vie à t=0
    pos_init = outputs.agents.position[0] 
    alive_init = outputs.agents.alive[0]
    
    for i in range(len(alive_init)):
        if alive_init[i]:
            # Inversion (y, x) pour imshow/plot : pos[0]=row, pos[1]=col
            ax_init.plot(pos_init[i, 1], pos_init[i, 0], 's', 
                         color="purple" if i == 1 else "red", 
                         markersize=cell_pts)

    ax_init.set_title('Configuration initiale ($t=0$)')
    ax_init.axis("off")

    plt.tight_layout()
    path_save_fig = os.path.join(exp_dir,'fig')
    os.makedirs(path_save_fig,exist_ok=True)
    plt.savefig(os.path.join(path_save_fig,f'plot_sim_{name_fig}.png'))
    plt.close()
    
def plot_several_sim_seeds(output_seeds,cfg,exp_dir):
    n_sims = output_seeds.grid.shape[0]
    for i in range(n_sims):
        outputs_i = jax.tree_util.tree_map(lambda x: x[i], output_seeds)
        plot_evolution(outputs_i,exp_dir,name_fig =f'seed_{i}')
        
        
def init_state(key, cfg, model):
    # 1. Grille de ressources
    key, subkey_grid = random.split(key)
    grid_resources = random.bernoulli(subkey_grid, p=cfg.prob_init_resources, shape=(cfg.n, cfg.n)).astype(jnp.int32)
    
    # 2. Préparation des agents
    key, *subkeys = random.split(key, 4)
    sk_pos, sk_or, sk_params = subkeys
    
    orientations_pool = jnp.array([0, jnp.pi/2, jnp.pi, -jnp.pi/2])
    idx_or = random.randint(sk_or, (cfg.n_agents_max,), 0, 4)
    
    # Gestion de la survie initiale
    alive_mask = jnp.zeros((cfg.n_agents_max,), dtype=jnp.int32).at[1:cfg.n_agents_init+1].set(1)
    
    # Paramètres réseau et état RNN
    params = random.normal(sk_params, (cfg.n_agents_max, model.num_params)) / 100
    policy_states = model.reset_b(jnp.zeros(cfg.n_agents_max))
    
    # Création de l'objet AgentState
    agents = AgentState(
        position=random.randint(sk_pos, (cfg.n_agents_max, 2), 0, cfg.n),
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

    # 3. Grille d'occupation et observations
    grid_agents = jnp.zeros((cfg.n, cfg.n), dtype=jnp.int32)
    grid_agents = grid_agents.at[agents.position[:, 0], agents.position[:, 1]].add(agents.alive)
    grid = jnp.stack((grid_resources, grid_agents))
    pos = agents.position
    obs = get_obs_vector(grid, pos,cfg.agent_view)
    # 4. État final
    state = SimState(
        grid=grid,
        agents=agents,
        step=0,
        obs=obs,
        last_actions=jnp.zeros((cfg.n_agents_max, 4)),
        rewards=jnp.zeros((cfg.n_agents_max, 1))
    )
    
    return state, key

import numpy as np
import graphviz
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_full_genealogy_robust(agents_state_history):
    parent_ids = np.array(agents_state_history.parent_id)
    born_steps = np.array(agents_state_history.born_step)
    alive = np.array(agents_state_history.alive)
    
    T, N = born_steps.shape
    
    nodes_by_step = {}
    edges = set()
    
    def add_node(idx, step):
        if step not in nodes_by_step:
            nodes_by_step[step] = set()
        nodes_by_step[step].add(idx)
        
    # 1. Capture des agents initiaux (t=0)
    for idx in range(1, N):
        if alive[0, idx] == 1:
            add_node(idx, born_steps[0, idx])
            
    # 2. Détection infaillible de toutes les naissances
    for t in range(1, T):
        # Une naissance = le step de naissance a changé par rapport à l'étape précédente
        changed_indices = np.where(born_steps[t] != born_steps[t-1])[0]
        
        for idx in changed_indices:
            if idx == 0:
                continue # On exclut l'agent 0 mort par défaut
                
            c_b_step = born_steps[t, idx]
            p_idx = parent_ids[t, idx]
            
            if p_idx == 0:
                continue
                
            p_b_step = born_steps[t, p_idx]
            
            # Ajouter l'enfant et le parent
            add_node(idx, c_b_step)
            add_node(p_idx, p_b_step)
            
            # Créer le lien généalogique
            edges.add((f"A{p_idx}_T{p_b_step}", f"A{idx}_T{c_b_step}"))

    # 3. Identification des survivants finaux pour le style
    survivors = set()
    last_step = T - 1
    for idx in range(1, N):
        if alive[last_step, idx] == 1:
            survivors.add((idx, born_steps[last_step, idx]))

    # 4. Tracé Graphviz
    dot = graphviz.Digraph(engine='dot', format='pdf')
    dot.attr(rankdir='LR', nodesep='0.15', ranksep='0.4')
    
    if not nodes_by_step:
        return dot
        
    max_step = max(nodes_by_step.keys())
    cmap = plt.cm.Blues_r
    norm = mcolors.Normalize(vmin=0, vmax=max_step if max_step > 0 else 1)

    for step, indices in nodes_by_step.items():
        rgba = cmap(norm(step))
        hex_color = mcolors.to_hex(rgba)
        
        with dot.subgraph() as s:
            s.attr(rank='same')
            for idx in indices:
                node_id = f"A{idx}_T{step}"
                node_label = f"A_{idx}\nStep {step}"
                
                is_survivor = (idx, step) in survivors
                border_color = '#2ca02c' if is_survivor else 'black' # Vert si survivant
                pen_width = '3' if is_survivor else '1'
                
                s.node(node_id, 
                       label=node_label, 
                       style='filled,rounded', 
                       fillcolor=hex_color, 
                       color=border_color,
                       penwidth=pen_width,
                       shape='box', 
                       fontsize='9', 
                       fontcolor='white' if step < max_step/2 else 'black')

    for parent, child in edges:
        dot.edge(parent, child, arrowsize='0.6')

    dot.render('arbre_genealogique_complet', view=True)
    return dot

