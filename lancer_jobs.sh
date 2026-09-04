#!/usr/bin/env bash
#
# Produit croise cycle_period x graine, une soumission OAR par combinaison.
#
#   ./lancer_jobs.sh --dry     affiche ce qui serait soumis, ne soumet rien
#   ./lancer_jobs.sh           soumet
#   ./lancer_jobs.sh --force   resoumet meme si le dossier existe deja
#
# Chaque run tourne dans SON PROPRE dossier. C'est necessaire, pas cosmetique :
# run.py ecrit dans "exp/<date>/<HH-MM-SS>" relatif au repertoire courant, avec
# makedirs(exist_ok=True). Deux jobs demarres dans la meme seconde -- ce qui
# arrive des qu'OAR libere plusieurs noeuds d'un coup -- partageraient le meme
# dossier et s'ecraseraient l'un l'autre sans un mot. save_script() resout ses
# chemins depuis __file__, donc changer de repertoire courant ne casse rien.

set -euo pipefail

# ---------------------------------------------------------------- a ajuster --
DEPOT="$HOME/BioTiC"                  # contient simulation/ et Code/EcoEvoJax/
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="ecoevojax"
MODULE="simulation.run"               # ou simulation_meta.run pour la branche meta
SORTIES="$HOME/runs"                  # un sous-dossier par combinaison

WALLTIME="24:00:00"
RESSOURCES="host=1/gpu=1"
QUEUE=()                              # ex: (-q production) ; vide = file par defaut
# NB : l'expansion ${QUEUE[@]+...} plus bas est ce qui permet a ce tableau
# d'etre vide sous `set -u`, y compris en bash 3.2.

# Run de reference : --from en reprend la config ET les graines, puis les flags
# ci-dessous s'appliquent par dessus. CHEMIN ABSOLU obligatoire -- load_config
# ouvre "<dir>/config.json" relatif au repertoire courant, et chaque job tourne
# depuis le sien.
CONFIG_REF="$DEPOT/simulation/exp/2026-09-03/2026-09-03_11-55-10"

# Options identiques pour tous les runs. -c et -s sont ajoutes par la boucle ;
# -s regenere des graines neuves, ce qui est le but (meme config, tirage neuf).
# Tout ce qui n'est pas liste ici vient de CONFIG_REF.
COMMUNES=(--from "$CONFIG_REF" --crowd-limit 6000 -n 3000)

# ------------------------------------------------------------- les tableaux --
CYCLES=(3 10 50)
GRAINES=(101 102 103)
# -----------------------------------------------------------------------------

DRY=0
FORCE=0
for arg in "$@"; do
  case "$arg" in
    --dry)   DRY=1 ;;
    --force) FORCE=1 ;;
    *) echo "argument inconnu : $arg" >&2; exit 2 ;;
  esac
done

[[ -d "$DEPOT/simulation" ]] || { echo "DEPOT introuvable : $DEPOT" >&2; exit 1; }
[[ -f "$CONFIG_REF/config.json" ]] || {
  echo "CONFIG_REF sans config.json : $CONFIG_REF" >&2
  echo "  (--from le lit au demarrage ; sans lui les 9 jobs echouent identiquement)" >&2
  exit 1; }
mkdir -p "$SORTIES"

soumis=0
sautes=0

for cycle in "${CYCLES[@]}"; do
  for graine in "${GRAINES[@]}"; do

    nom="c${cycle}_s${graine}"
    dir="$SORTIES/$nom"

    if [[ -d "$dir" && $FORCE -eq 0 ]]; then
      echo "saute   $nom  (dossier deja present, --force pour resoumettre)"
      sautes=$((sautes + 1))
      continue
    fi
    mkdir -p "$dir"

    # Le job est un script sur disque plutot qu'une commande passee a oarsub :
    # pas de guillemets a echapper, et il reste comme trace exacte de ce qui a
    # tourne, a cote des resultats.
    cat > "$dir/job.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source "$CONDA_SH"
conda activate "$CONDA_ENV"
export PYTHONPATH="$DEPOT:$DEPOT/Code:\${PYTHONPATH:-}"
cd "$dir"
echo "[job] \$(date '+%F %T')  $nom  sur \$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
exec python -m $MODULE $(printf "%q " "${COMMUNES[@]}") -c $cycle -s $graine
EOF
    chmod +x "$dir/job.sh"

    cmd=(oarsub
         -l "$RESSOURCES,walltime=$WALLTIME"
         -n "$nom"
         -O "$dir/oar.%jobid%.out"
         -E "$dir/oar.%jobid%.err"
         ${QUEUE[@]+"${QUEUE[@]}"}
         "$dir/job.sh")

    if [[ $DRY -eq 1 ]]; then
      printf '%q ' "${cmd[@]}"; echo
    else
      "${cmd[@]}"
    fi
    soumis=$((soumis + 1))

  done
done

total=$(( ${#CYCLES[@]} * ${#GRAINES[@]} ))
if [[ $DRY -eq 1 ]]; then
  echo "--- essai a blanc : $soumis job(s) seraient soumis sur $total, $sautes saute(s)"
else
  echo "--- $soumis job(s) soumis sur $total, $sautes saute(s)"
  echo "    suivi : oarstat -u \$USER      resultats : $SORTIES/<nom>/exp/"
fi
