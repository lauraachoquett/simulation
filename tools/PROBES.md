# Les deux tests par gradient

Deux outils qui répondent séparément aux deux moitiés d'une même question :
**un agent peut-il découvrir la valeur des ressources en vivant, puis s'en servir ?**

Ils sortent tous deux de la simulation. Pas d'évolution, pas d'écologie, pas de
mutation — de l'apprentissage supervisé sur le réseau réel. Ce qu'ils établissent
est donc une **capacité**, indépendamment de la question de savoir si la sélection
naturelle la trouve.

```bash
python -m simulation_meta.tools.probe_memory --model v2
python -m simulation_meta.tools.probe_action
```

Les figures atterrissent dans `fig/`, relatif au dossier depuis lequel tu lances
(`--fig-dir` pour changer).

---

## `probe_memory` — le réseau peut-il **apprendre** la valeur des canaux ?

### Le dispositif

Le réseau reçoit exactement le flux qu'il reçoit en simulation — action précédente,
récompense, énergie, ce qu'il vient de manger, et la vision — et on lui demande une
seule chose : **quel canal porte la ressource la plus néfaste ?**

La lecture se fait sur **le carry du LSTM seul** (`self.lecture(h)`), et la vision
n'est ici que du bruit uniforme, indépendant de la permutation. En **v2** le LSTM
ne la reçoit d'ailleurs pas du tout — `mem_in = [last_action, reward, energy,
last_eaten]` — elle passe par le conv, dont la sortie est jetée ; elle n'est
fournie que parce que la signature de `MetaRNN_bcppr` l'exige. En **v1** le LSTM
reçoit la vision encodée mais pas `last_eaten` : il hérite donc du bruit sans
l'information.

Le point qui fait tout : **la permutation canal → identité est retirée à chaque
épisode**. Le canal 2 porte le poison dans un épisode, le good dans le suivant.
Mémoriser « le canal 2 est dangereux » ne sert donc à rien — c'est vrai une fois sur
trois. La seule source d'information est ce que l'agent a goûté depuis le début de
l'épisode.

Une tête linéaire lit le carry du LSTM et sort une classe parmi les *n* canaux. Le
reste du réseau est celui de la simulation, à sa taille réelle. Adam, entropie
croisée, cible tirée à neuf à chaque lot.

### La figure

`fig/probe_memory_<modele>.png` — exactitude en fonction du **nombre de bouchées
déjà prises dans l'épisode**, une couleur par avancement de l'entraînement
(barre de couleur : mises à jour de gradient).

L'abscisse est la quantité de **preuves** dont le réseau dispose au moment où on
l'interroge, et la seule variable qui puisse porter l'information : c'est le
nombre de bouchées qui compte, pas le nombre de pas — un pas sans repas
n'apporte rien. La couleur montre le circuit se former : plate au hasard à zéro
mise à jour, la courbe se redresse et gagne sa forme à mesure que l'entraînement
avance.

Le point **0 bouchée** est le témoin. Sans avoir rien goûté aucune information
n'est disponible, et il doit rester au hasard (1/*n*) **quel que soit
l'entraînement** — toutes les courbes s'y rejoignent. S'il montait, l'exercice
fuirait la réponse quelque part et le reste de la figure ne vaudrait rien.

Le comptage est **inclusif** : `deroule` passe `reward[t]` et `mange[t]` au pas
*t*, donc la bouchée du pas courant est déjà une entrée du réseau. L'exclure
décalait la courbe d'un rang et posait « 0 bouchée » à 0.58, au-dessus du hasard.

### Ce qu'il établit

| architecture | perte après 800 itérations |
|---|---|
| **v1** (`memory_mode="jointe"`) | 1.0986 = ln 3 — le hasard, jamais quitté |
| **v2** (`memory_mode="separee"`) | 0.999 atteint en ~175 itérations |
| v2 privé de `last_eaten` | échoue à l'identique de v1 |

La troisième ligne est le contrôle décisif : **v1 échoue parce que son LSTM ne reçoit
jamais `last_eaten`**. C'est informationnel, pas une défaillance de la recherche.
Aucun réglage de mutation ni d'architecture n'y changerait quoi que ce soit.

### Options

| | |
|---|---|
| `--model v1\|v2` | le câblage testé, lu dans `MODEL_VERSIONS` |
| `--n-res N` | nombre de ressources, la plus néfaste toujours incluse |
| `--steps` | longueur d'un épisode (défaut 40) |
| `--chauffe` | pas ignorés dans la loss — avant, rien n'est déductible |
| `--eval-tous` | période d'évaluation pour la figure |
| `--max-bouchees` | dernière courbe tracée |

---

## `probe_action` — peut-il **s'en servir** pour agir ?

### Le dispositif

La tâche est réduite à la seule décision qui compte. Une ressource est posée **droit
devant l'agent**, `v_pred` lui est fourni, et il doit choisir : avancer ou tourner.
Pas de navigation, pas de murs, un seul pas — on avait établi que se *diriger* vers
une case éloignée est un autre problème.

La classe est tirée d'abord, puis une identité compatible : le hasard vaut donc
exactement **0.50** quels que soient le nombre de ressources et la répartition des
signes. Et comme pour `probe_memory`, la permutation change à chaque échantillon :
un réseau qui ne lit pas `v_pred` ne peut pas dépasser ce plancher.

Trois variantes, qui ne diffèrent que par ce que la tête reçoit :

| variante | `v_pred` |
|---|---|
| `aveugle` | remplacé par des zéros — le plancher, ce qu'on fait sans l'information |
| `concat` | le câblage actuel : `v_pred` concaténé à l'entrée de la tête |
| `carte` | l'option `value_map` : un canal d'observation en plus, où chaque case porte la valeur de ce qui s'y trouve. Le produit vision × valeur est fait **avant** le conv |

`carte` utilise le vrai code de la simulation, pas une imitation.

### La figure

`fig/probe_action.png` — exactitude en fonction des mises à jour, une courbe par
variante, abscisse logarithmique. Un point marque le franchissement des 99 %.

Trois lectures sur la même image :

- **`aveugle` reste à 0.50.** C'est la validation du dispositif : sans `v_pred`, la
  tâche est impossible. Si cette courbe montait, la tâche fuirait l'information
  quelque part et rien de ce qui suit ne vaudrait.
- **`concat` atteint 1.000.** L'architecture, *à sa taille réelle*, sait croiser la
  vision et la valeur. Le blocage de la simulation n'est donc ni l'architecture ni
  l'information disponible.
- **`carte` y arrive ~11× plus vite**, pour +36 paramètres. Le produit vision ×
  valeur est bien le morceau coûteux, et le faire en amont le supprime.

Le tableau imprimé donne les mises à jour nécessaires pour franchir 75 / 90 / 95 /
99 %.

### La réserve, à garder en tête

Une mise à jour de gradient n'est pas une génération. `probe_action` voit **256
exemples étiquetés par pas, avec la dérivée** ; l'évolution avance d'environ un bit
par naissance. Les ~500 mises à jour de `concat` représentent 130 000 exemples.

Le chiffre ne dit donc **pas** « 500 générations suffiront ». Il dit deux choses plus
faibles mais utiles : la fonction est atteignable, le paysage n'est pas pathologique ;
et le **rapport entre variantes** — ×11 — se transpose, lui.

### Options

| | |
|---|---|
| `--hidden` | tête de politique testée (défaut `[8]`, la vraie) |
| `--carry` | taille du carry LSTM |
| `--n-res N` | nombre de ressources |
| `--iters`, `--batch`, `--lr` | entraînement |

---

## Ce que les deux, ensemble, permettent de conclure

`probe_memory` répond oui à « le réseau peut-il apprendre la valeur ».
`probe_action` répond oui à « peut-il s'en servir ».

Les deux capacités existent, séparément et à la taille réelle du réseau. Ce que
l'échec de l'émergence met en cause n'est donc **ni l'une ni l'autre**, mais ce qui
les relie : la sélection ne fixe pas les poids qui les font travailler ensemble, et
le budget d'échantillons d'une vie ne permet pas au gradient de le faire à sa place.

Le détail de cette chaîne est dans [`BILAN_apprentissage_intra_vie.md`](../BILAN_apprentissage_intra_vie.md).
