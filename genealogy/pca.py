import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os 
import glob



def save_alive_snapshot(state, chunk_idx, store_dir):
    alive  = np.asarray(state.agents.alive)      # (N,)   indexe d'abord, transfère ensuite
    born   = np.asarray(state.agents.born_step)  # (N,)
    params = np.asarray(state.agents.params)     # (N, P) au lieu de (T, N, P)

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

