#!/usr/bin/env bash
# Lance la simulation. Tous les arguments sont transmis tels quels a run.py.
#   ./launcher.sh -m v1 -x r -n 5
#   oarsub -l gpu=1,walltime=12 -S "./launcher.sh -m v2 -c 3"
#
#OAR -l gpu=1,walltime=24:00:00
#OAR -O exp/oar.%jobid%.out
#OAR -E exp/oar.%jobid%.err

set -euo pipefail

# Racine contenant simulation/ et EcoEvoJax/. Surchargeable si le cluster
# range les depots ailleurs : BIOTIC_ROOT=/chemin ./launcher.sh
ICI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${BIOTIC_ROOT:-$(dirname "$ICI")}"

# A ADAPTER : activation de l'environnement du cluster
# source "$ROOT/simulation/.venv/bin/activate"
# module load conda && conda activate biotic

# `import EcoEvoJax` exige que son PARENT soit sur le PYTHONPATH, et celui-ci
# n'est pas toujours ROOT : en local le depot est sous Code/. On cherche plutot
# que de supposer, et on echoue fort si on ne le trouve pas.
ECO=""
for c in "$ROOT" "$ROOT/Code" "$ICI"; do
    [ -d "$c/EcoEvoJax/source" ] && { ECO="$c"; break; }
done
if [ -z "$ECO" ]; then
    echo "[launcher] EcoEvoJax introuvable sous $ROOT, $ROOT/Code ou $ICI." >&2
    echo "[launcher] Poser BIOTIC_ROOT sur le dossier qui le contient." >&2
    exit 1
fi

export PYTHONPATH="$ROOT:$ECO:$ECO/EcoEvoJax/evojax${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1        # sans ca la sortie OAR n'arrive qu'a la fin du job

# Les variables d'allocateur GPU (XLA_PYTHON_CLIENT_PREALLOCATE,
# TF_GPU_ALLOCATOR) sont posees dans run.py AVANT l'import de jax : les mettre
# ici ne servirait a rien, elles sont lues a l'initialisation du backend.

echo "[launcher] $(date '+%F %T')  root=$ROOT  ecoevojax=$ECO"
echo "[launcher] args: $*"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv

cd "$ROOT"
exec python -m simulation.run "$@"
