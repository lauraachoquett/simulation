"""Ligne de descendance : les ancetres devenus communs a toute la population.

Quand le MRCA avance, tous les noeuds situes entre l'ancien et le nouveau sont
desormais ancetres de TOUS les vivants. Leur sort ne changera plus : ils sont
fixes. Les enregistrer au fil du run donne la ligne de descendance, c'est-a-dire
la suite des individus par lesquels l'evolution est effectivement passee.

Il faut le faire PENDANT le run : l'arbre (node_parent) ne vit qu'en memoire, et
params/ est supprime a la fin par purge_params. Rien ne permet de reconstituer
la lignee apres coup.
"""
import json
import os

import numpy as np

from simulation.genealogy.genealogy import chaine_ancetres


def segment_fixe(mrca, mrca_prec, node_parent):
    """Noeuds de `mrca_prec` (exclu) a `mrca` (inclus), du plus ancien au plus recent.

    `mrca_prec=None` (premiere coalescence) -> toute la chaine jusqu'a la racine.

    Si `mrca_prec` n'est pas sur la chaine du nouveau MRCA, la population a
    change de fondateur : on rend alors la chaine entiere, et le drapeau
    `rupture` le signale plutot que de laisser un trou silencieux dans la lignee.
    """
    chaine = chaine_ancetres(mrca, node_parent)      # [mrca, parent, ..., racine]
    if mrca_prec is None or mrca_prec not in chaine:
        return list(reversed(chaine)), mrca_prec is not None
    return list(reversed(chaine[:chaine.index(mrca_prec)])), False


def params_du_segment(segment, params_dir, chunks):
    """Genomes des ancetres fixes, lus dans les instantanes de vivants.

    Incomplet par construction : params/ ne garde qu'un instantane par chunk,
    donc un ancetre ne vivant qu'entre deux instantanes n'y figure pas. Le
    nombre reellement retrouve est rendu a l'appelant pour que le trou soit
    visible. Vide si track_weights est False -- rien n'est alors ecrit dans
    params/.
    """
    from simulation.genealogy.pca import load_clade_snapshots
    if not os.path.isdir(params_dir):
        return {}
    return load_clade_snapshots(set(segment), params_dir, name_save=list(chunks))


def enregistre(exp_dir, chunk, step, mrca, mrca_prec, segment,
               params=None, rupture=False):
    """Ajoute un evenement de fixation a lod/lignee.jsonl. Rend le nb de genomes ecrits.

    Un fichier par ligne et en mode append : un run interrompu garde tout ce
    qui precede, et la lignee se lit dans l'ordre chronologique sans rien trier.
    """
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


def charge_lignee(exp_dir):
    """[(slot, born)] de toute la ligne de descendance, du plus ancien au plus recent.

    Les segments sont disjoints et deja dans l'ordre : il suffit de les mettre
    bout a bout. Un segment marque `rupture` signale un changement de fondateur,
    donc un saut dans la suite.
    """
    path = os.path.join(exp_dir, "lod", "lignee.jsonl")
    if not os.path.exists(path):
        return []
    lignee = []
    for ligne in open(path):
        if ligne.strip():
            rec = json.loads(ligne)
            lignee.extend((int(s), int(b)) for s, b in rec["ancetres"])
    return lignee
