"""Ligne de descendance : les ancetres fixes, enregistres quand le MRCA avance.

A faire pendant le run : node_parent ne vit qu'en memoire et params/ est
supprime a la fin.
"""
import json
import os

import numpy as np

from simulation.genealogy.genealogy import chaine_ancetres


def segment_fixe(mrca, mrca_prec, node_parent):
    """Noeuds de `mrca_prec` (exclu) a `mrca` (inclus), du plus ancien au plus recent.

    mrca_prec absent de la chaine -> changement de fondateur, signale par `rupture`.
    """
    chaine = chaine_ancetres(mrca, node_parent)      # [mrca, parent, ..., racine]
    if mrca_prec is None or mrca_prec not in chaine:
        return list(reversed(chaine)), mrca_prec is not None
    return list(reversed(chaine[:chaine.index(mrca_prec)])), False


def params_du_segment(segment, params_dir, chunks):
    """Genomes lus dans params/. Incomplet : un instantane par chunk seulement."""
    from simulation.genealogy.pca import load_clade_snapshots
    if not os.path.isdir(params_dir):
        return {}
    return load_clade_snapshots(set(segment), params_dir, name_save=list(chunks))


def enregistre(exp_dir, chunk, step, mrca, mrca_prec, segment,
               params=None, rupture=False):
    """Ajoute un evenement de fixation a lod/lignee.jsonl, en append."""
    d = os.path.join(exp_dir, "lod")
    os.makedirs(d, exist_ok=True)

    rec = {
        "chunk": int(chunk),
        "step": int(step),
        "mrca": [int(mrca[0]), int(mrca[1])],
        "mrca_prec": None if mrca_prec is None else [int(mrca_prec[0]),
                                                     int(mrca_prec[1])],
        "n_ancetres": len(segment),
        "rupture": bool(rupture),
        "ancetres": [[int(s), int(b)] for s, b in segment],
    }
    with open(os.path.join(d, "lignee.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")

    ordre = [n for n in segment if params and n in params]
    if ordre:
        np.savez_compressed(
            os.path.join(d, f"params_chunk_{chunk}.npz"),
            slot=np.array([n[0] for n in ordre], dtype=np.int32),
            born=np.array([n[1] for n in ordre], dtype=np.int64),
            params=np.stack([params[n] for n in ordre]).astype(np.float32),
        )
    return len(ordre)


def capture_vivants(cache, state):
    """Ajoute au cache les genomes des vivants absents. Cle = (slot, born_step).

    Capture en fin de chunk : un individu ne vivant qu'entre deux frontieres est
    manque.
    """
    alive = np.asarray(state.agents.alive)
    born = np.asarray(state.agents.born_step)
    params = np.asarray(state.agents.params)
    for i in np.nonzero(alive == 1)[0]:
        if i == 0:                       # slot 0 : sentinelle
            continue
        cle = (int(i), int(born[i]))
        if cle not in cache:
            cache[cle] = params[i].astype(np.float32)
    return cache


def elague_cache(cache, feuilles, node_parent, mrca=None):
    """Ne garde que les noeuds sur une chaine reliant un vivant au MRCA courant.

    Sans cet elagage le cache accumulerait tous les individus du run.
    """
    garder = set()
    for feuille in feuilles:
        n = feuille
        while n is not None and n not in garder:
            garder.add(n)
            if n == mrca:
                break
            n = node_parent.get(n)
    for cle in list(cache):
        if cle not in garder:
            del cache[cle]
    return cache


def feuilles_vivantes(state):
    """Noeuds (slot, born_step) des agents vivants."""
    alive = np.asarray(state.agents.alive)
    born = np.asarray(state.agents.born_step)
    return [(int(i), int(born[i])) for i in np.nonzero(alive == 1)[0] if i != 0]


def charge_lignee(exp_dir):
    """[(slot, born)] de toute la lignee, du plus ancien au plus recent."""
    path = os.path.join(exp_dir, "lod", "lignee.jsonl")
    if not os.path.exists(path):
        return []
    lignee = []
    for ligne in open(path):
        if ligne.strip():
            rec = json.loads(ligne)
            lignee.extend((int(s), int(b)) for s, b in rec["ancetres"])
    return lignee
