import numpy as np

def update_genealogy(outputs, node_parent, node_children, prev_born=None, prev_parent=None):
    parent_ids = np.asarray(outputs.parent_id)   # (T, N)
    born_steps = np.asarray(outputs.born_step)
    alive      = np.asarray(outputs.alive)
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
        # -> les naissances à la frontière (ligne 0 de ce chunk) deviennent visibles (On veut comparer 
        # le dernier pas de temps du chunk précédent au nouveau pour éviter de louper une naissance qui a lieu au premier pas de temps 
        # d'un chunk)
        born_aug   = np.vstack([prev_born[None, :],   born_steps])
        parent_aug = np.vstack([prev_parent[None, :], parent_ids])

    # Steps et indices de naissance (quand born step change = naissance)
    ts, idxs = np.where(born_aug[1:] != born_aug[:-1]) 
    ts += 1                                  

    # On récupère le slot des parents aux steps de naissance et indices des nouveaux nés 
    p_idxs = parent_aug[ts, idxs]
    keep = (idxs != 0) & (p_idxs != 0) # on retire le slot 0 'indice de gestion pour JAX' et aussi l'initialisation
    ts, idxs, p_idxs = ts[keep], idxs[keep], p_idxs[keep] # Step de naissance, slot de naissance des agents, slot de leurs parents 

    child_born  = born_aug[ts, idxs] # Liste des steps naissance de enfants
    parent_born = born_aug[ts, p_idxs] # Liste des steps où les parents sont nés

    for i, cb, pi, pb in zip(idxs.tolist(), child_born.tolist(),
                             p_idxs.tolist(), parent_born.tolist()):
        child, parent = (i, cb), (pi, pb)
        node_children.setdefault(child, set())
        node_children.setdefault(parent, set())
        node_parent[child] = parent # arbre double un qui descend et l'autre qui monte
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