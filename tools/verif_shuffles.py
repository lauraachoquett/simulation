"""Le journal des permutations est-il celui d'UN seul run ?

Dans un run coherent, l'etat de depart de chaque ligne est l'order_ids de la
precedente. Une rupture signale que plusieurs runs ecrivent dans le meme fichier.

    python3 verif_shuffles.py <exp_dir>/resource_shuffles.jsonl
"""
import json, sys
from collections import Counter

LABELS = ("good", "medium", "poison")
lignes = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print(f"{len(lignes)} ligne(s)")

doublons = [c for c, n in Counter(r["chunk_idx"] for r in lignes).items() if n > 1]
if doublons:
    print(f"chunk_idx en double : {sorted(doublons)}")

precedent = None                       # order_ids attendu au depart
for i, r in enumerate(lignes):
    apres = r["order_ids"]
    # etat AVANT : le 'from' des canaux modifies, l'apres pour les autres
    avant = [LABELS[i] if i < len(LABELS) else f"res{i}" for i in apres]
    for c in r["changes"]:
        avant[c["channel"]] = c["from"]
    depart = [LABELS.index(n) if n in LABELS else -1 for n in avant]

    etat = "ok"
    if precedent is not None and depart != precedent:
        etat = f"RUPTURE : part de {avant}, or la ligne precedente finissait sur " \
               + str([LABELS[i] for i in precedent])
    print(f"  {i}: chunk {r['chunk_idx']:>5} step {r['step']:>7}  "
          f"{avant} -> {[LABELS[i] for i in apres]}   {etat}")
    precedent = apres
