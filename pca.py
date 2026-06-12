import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os 
import glob

def update_genealogy(outputs, node_parent, node_children, prev_born=None, prev_parent=None):
    parent_ids = np.asarray(outputs.agents.parent_id)   # (T, N)
    born_steps = np.asarray(outputs.agents.born_step)
    alive      = np.asarray(outputs.agents.alive)
    T, N = born_steps.shape

    first_chunk = prev_born is None
    if first_chunk:
        for idx in np.nonzero(alive[0, 1:] == 1)[0] + 1:
            n = (int(idx), int(born_steps[0, idx]))
            node_children.setdefault(n, set())
            node_parent.setdefault(n, None)
        born_aug, parent_aug = born_steps, parent_ids
    else:
        # on rattache la dernière ligne du chunk précédent
        # -> les naissances à la frontière (ligne 0 de ce chunk) deviennent visibles
        born_aug   = np.vstack([prev_born[None, :],   born_steps])
        parent_aug = np.vstack([prev_parent[None, :], parent_ids])

    ts, idxs = np.where(born_aug[1:] != born_aug[:-1])
    ts += 1                                  # index dans born_aug du pas observé

    p_idxs = parent_aug[ts, idxs]
    keep = (idxs != 0) & (p_idxs != 0)
    ts, idxs, p_idxs = ts[keep], idxs[keep], p_idxs[keep]

    child_born  = born_aug[ts, idxs]
    parent_born = born_aug[ts, p_idxs]

    for i, cb, pi, pb in zip(idxs.tolist(), child_born.tolist(),
                             p_idxs.tolist(), parent_born.tolist()):
        child, parent = (i, cb), (pi, pb)
        node_children.setdefault(child, set())
        node_children.setdefault(parent, set())
        node_parent[child] = parent
        node_children[parent].add(child)

    return born_steps[-1], parent_ids[-1]    # à repasser au prochain chunk

            
def find_root(node, node_parent):
    seen = set()
    while node_parent.get(node) is not None and node not in seen:
        seen.add(node)
        node = node_parent[node]
    return node

def collect_clade(root, node_children):
    clade, stack, seen = [], [root], set()
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        clade.append(n)
        stack.extend(node_children.get(n, ()))
    return clade

def save_alive_snapshot(outputs, chunk_idx, store_dir):
    alive  = np.asarray(outputs.agents.alive[-1])      # (N,)   indexe d'abord, transfère ensuite
    born   = np.asarray(outputs.agents.born_step[-1])  # (N,)
    params = np.asarray(outputs.agents.params[-1])     # (N, P) au lieu de (T, N, P)

    mask = alive == 1
    mask[0] = False
    idx = np.nonzero(mask)[0]

    os.makedirs(store_dir, exist_ok=True)
    np.savez(os.path.join(store_dir, f"alive_chunk_{chunk_idx}.npz"),
             idx=idx.astype(np.int32),
             born=born[idx].astype(np.int64),
             params=params[idx].astype(np.float32))
    
    
def load_clade_snapshots(clade, store_dir, name_save=None):
    """Params des membres du clade présents dans les snapshots de vivants.
    name_save=None -> tous les chunks ; sinon, seulement ceux listés.
    Dédup par individu (idx, born_step)."""
    clade = set(clade)
    if name_save is None:
        files = sorted(glob.glob(os.path.join(store_dir, "alive_chunk_*.npz")))
    else:
        files = [os.path.join(store_dir, f"alive_chunk_{c}.npz") for c in name_save]
        files = [f for f in files if os.path.exists(f)]   # tolère un chunk absent

    node_params = {}
    for f in files:
        data = np.load(f)
        for i, b, p in zip(data['idx'], data['born'], data['params']):
            key = (int(i), int(b))
            if key in clade:
                node_params.setdefault(key, p)            # params constants -> 1ère occurrence
        data.close()
    return node_params

from sklearn.decomposition import PCA

def plot_clade_pca_html(node_params, exp_dir, name_fig='clade'):
    nodes = list(node_params.keys())
    X     = np.stack([node_params[n] for n in nodes])       # (M, P)
    born  = np.array([n[1] for n in nodes], dtype=float)    # born_step

    if X.shape[0] < 4:
        print(f"Sous-ensemble trop petit ({X.shape[0]} agents)"); return

    pca = PCA(n_components=3)
    Y   = pca.fit_transform(X)
    var = pca.explained_variance_ratio_ * 100

    fig = go.Figure(go.Scatter3d(
        x=Y[:, 0], y=Y[:, 1], z=Y[:, 2], mode='markers',
        marker=dict(size=3, color=born, colorscale='Bluered', opacity=0.8,
                    colorbar=dict(title='Step de naissance'), showscale=True),
    ))
    fig.update_layout(
        title=(f"Lineage PCA (agents considered, {X.shape[0]} agents) — "
               f"var. {var[0]:.1f}/{var[1]:.1f}/{var[2]:.1f} %"),
        scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'),
    )
    path = os.path.join(exp_dir, 'fig','pca'); os.makedirs(path, exist_ok=True)
    fig.write_html(os.path.join(path, f'pca_clade_{name_fig}.html'))



def plot_clade_pca_html_res(node_params, resource_history, exp_dir, name_fig='clade'):
    nodes = list(node_params.keys())
    X     = np.stack([node_params[n] for n in nodes])           # (M, P)
    born  = np.array([n[1] for n in nodes], dtype=np.int64)     # born_step (absolu)

    if X.shape[0] < 4:
        print(f"Sous-ensemble trop petit ({X.shape[0]} agents)"); return

    # ressource au step de naissance. resource_history doit être indexable
    # par le step absolu -> forme (T_total,) alignée step 0..T_total-1.
    R = np.asarray(resource_history)

    if np.max(born>40):
        
        res_birth = R[born-40].astype(float) 
    else : # (M,)
        res_birth = R[born].astype(float) 
    if np.max(born)>1000:
        res_birth[born<1000] = np.max(res_birth[born>1000])
    
    pca = PCA(n_components=3)
    Y   = pca.fit_transform(X)
    var = pca.explained_variance_ratio_ * 100

    fig = go.Figure(go.Scatter3d(
        x=Y[:, 0], y=Y[:, 1], z=Y[:, 2], mode='markers',
        marker=dict(size=3, color=res_birth, colorscale='RdBu', opacity=0.8,
                    colorbar=dict(title='Resources at born step'), showscale=True),
        customdata=np.stack([born, res_birth], axis=1),
        hovertemplate=('PC1=%{x:.2f}<br>PC2=%{y:.2f}<br>PC3=%{z:.2f}'
                       '<br>born=%{customdata[0]}<br>res=%{customdata[1]:.3g}'
                       '<extra></extra>'),
    ))
    fig.update_layout(
        title=(f"Lineages PCA (2 chunks, {X.shape[0]} agents) — "
               f"var. {var[0]:.1f}/{var[1]:.1f}/{var[2]:.1f} %"),
        scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'),
    )
    path = os.path.join(exp_dir, 'fig'); os.makedirs(path, exist_ok=True)
    fig.write_html(os.path.join(path, f'pca_clade_{name_fig}.html'))