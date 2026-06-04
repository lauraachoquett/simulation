import graphviz
import matplotlib.pyplot as plt  # used by plot_full_genealogy_robust for colormap
import matplotlib.colors as mcolors
import numpy as np
import os
import jax
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def plot_evolution(pop_history, res_history, exp_dir, start_step=0, name_fig=''):
    """Plot population and resource dynamics over time."""
    generations = np.arange(start_step, start_step + len(pop_history))

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=generations, y=pop_history, name='Agents', line=dict(color='red', width=2)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=generations, y=res_history, name='Ressources', line=dict(color='green', width=2)),
        secondary_y=True,
    )
    fig.update_xaxes(title_text='Steps')
    fig.update_yaxes(title_text='Population size', secondary_y=False,
                     title_font=dict(color='red'), tickfont=dict(color='red'))
    fig.update_yaxes(title_text='Quantité de ressources', secondary_y=True,
                     title_font=dict(color='green'), tickfont=dict(color='green'))
    fig.update_layout(title='Dynamique de la simulation')

    path_save_fig = os.path.join(exp_dir, 'fig')
    os.makedirs(path_save_fig, exist_ok=True)
    fig.write_html(os.path.join(path_save_fig, f'plot_evo_{name_fig}.html'))


def plot_current_config(current_grid_res, pos, alive, exp_dir, name_fig=''):
    """Plot the initial grid configuration with agent positions."""
    fig, ax = plt.subplots(figsize=(5, 5))

    cmap_res = mcolors.ListedColormap(["white", "#4CAF50"])
    ax.imshow(current_grid_res, cmap=cmap_res, vmin=0, vmax=1, interpolation="nearest")

    fig.canvas.draw()
    bbox = ax.get_window_extent()
    grid_h, grid_w = current_grid_res.shape
    cell_px = min(bbox.width / grid_w, bbox.height / grid_h)
    cell_pts = cell_px * 72 / fig.dpi

    for i in range(len(alive)):
        if alive[i]:
            ax.plot(pos[i, 1], pos[i, 0], 's',
                    color="purple" if i == 1 else "red",
                    markersize=cell_pts)

    ax.set_title('Configuration initiale ($t=0$)')
    ax.axis("off")

    plt.tight_layout()
    path_save_fig = os.path.join(exp_dir, 'fig/sim_grid/')
    os.makedirs(path_save_fig, exist_ok=True)
    plt.savefig(os.path.join(path_save_fig, f'plot_sim_chunk_{name_fig}.png'))
    plt.close()
    
def plot_several_sim_seeds(output_seeds,cfg,exp_dir):
    n_sims = output_seeds.grid.shape[0]
    for i in range(n_sims):
        outputs_i = jax.tree_util.tree_map(lambda x: x[i], output_seeds)
        plot_evolution(outputs_i,exp_dir,name_fig =f'seed_{i}')
        
        




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



def compute_mean_movement_chunk(outputs, n):
    """Compute mean movement magnitude (in cells) per step from a simulation chunk.

    Returns an array of shape (chunk_size - 1,) — one value per step transition.
    Handles toroidal wrap-around for a grid of size n.
    """
    positions = np.array(outputs.agents.position)  # (T, N, 2)
    alive = np.array(outputs.agents.alive)          # (T, N)

    delta = positions[1:] - positions[:-1]  # (T-1, N, 2)
    delta = np.where(delta > n // 2, delta - n, delta)
    delta = np.where(delta < -(n // 2), delta + n, delta)

    magnitude = np.sqrt(delta[:, :, 0] ** 2 + delta[:, :, 1] ** 2)  # (T-1, N)
    alive_mask = alive[1:]  # (T-1, N)
    n_alive = alive_mask.sum(axis=1)
    mean_mov = np.where(n_alive > 0, (magnitude * alive_mask).sum(axis=1) / n_alive, 0.0)
    return mean_mov  # (T-1,)


def compute_resources_consumed_chunk(outputs):
    """Compute total resources consumed per step from a simulation chunk.

    Returns an array of shape (chunk_size,).
    rewards[t] stores consumption from step t-1, so the first value is always 0.
    """
    return np.array(outputs.rewards[:, :, 0].sum(axis=1))  # (T,)


def plot_mean_movement(mov_history, exp_dir, start_step=0, name_fig=''):
    """Plot mean movement of the population over time."""
    steps = np.arange(start_step, start_step + len(mov_history))

    _, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlabel('Steps')
    ax.set_ylabel('Mouvement moyen (cellules)')
    ax.plot(steps, mov_history, color='tab:blue', linewidth=1.5)
    ax.grid(True, alpha=0.3)
    ax.set_title('Mouvement moyen de la population')

    plt.tight_layout()
    path_save_fig = os.path.join(exp_dir, 'fig')
    os.makedirs(path_save_fig, exist_ok=True)
    plt.savefig(os.path.join(path_save_fig, f'plot_movement_{name_fig}.png'))
    plt.close()


def plot_resources_consumed(consumed_history, exp_dir, start_step=0, name_fig=''):
    """Plot total resources consumed by the population over time."""
    steps = np.arange(start_step, start_step + len(consumed_history))

    _, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlabel('Steps')
    ax.set_ylabel('Ressources consommées')
    ax.plot(steps, consumed_history, color='tab:orange', linewidth=1.5)
    ax.grid(True, alpha=0.3)
    ax.set_title('Ressources consommées par la population')

    plt.tight_layout()
    path_save_fig = os.path.join(exp_dir, 'fig')
    os.makedirs(path_save_fig, exist_ok=True)
    plt.savefig(os.path.join(path_save_fig, f'plot_consumed_{name_fig}.png'))
    plt.close()

