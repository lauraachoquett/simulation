# Émergence d'une adaptation intra-vie sous sélection par viabilité

**Bilan expérimental** — simulation éco-évolutive ouverte, non épisodique, sans fonction de fitness.
Dépôts : `simulation` (`master`), `EcoEvoJax` (`main`).

---

## 1. Question

Une **adaptation au cours de la vie de l'agent** peut-elle émerger dans une simulation
éco-évolutive ouverte où la sélection ne s'exerce que par un critère de viabilité — les
agents meurent d'épuisement énergétique et se reproduisent au-delà d'un seuil — sans
aucune fonction de fitness explicite ?

La tâche cible : trois ressources de valeurs distinctes (`good` +1.0, `medium` +0.3,
`poison` −1.5) dont l'affectation aux **canaux d'observation est périodiquement permutée**.
Un génome ne peut donc pas encoder « le canal 2 est toxique » : l'information doit être
réacquise à chaque permutation, idéalement pendant la vie de l'agent.

**Réponse courte : non, et la chaîne de mesures ci-dessous localise précisément où ça bloque.**

---

## 2. Dispositif

| | |
|---|---|
| Grille | 200 × 200, murs non létaux |
| Population | 2000 agents max, sélection par viabilité seule |
| Réseau | conv(2) → LSTM(8) → tête(8) → 4 actions, **1519–2892 paramètres** |
| Mutation | `param_mutate` 0.05–0.99, `mutation_var` 0.02 |
| Permutation | tirage tous les `cycle_period` chunks, identité comprise |
| Durée de vie | ~2500 pas (estimée sur la décroissance de `plot_inner_loss_by_age`) |

Architecture `v2` (`memory_mode="separee"`) : **le LSTM ne reçoit pas la vision**, seulement
`[last_action, reward, energy, last_eaten]`. La tête de politique combine vision et sortie
du LSTM.

---

## 3. Chaîne de résultats

### 3.1 L'architecture v1 ne peut pas apprendre — c'est informationnel

Outil : `tools/probe_memory.py`. Apprentissage supervisé isolé — le réseau reçoit le flux
qu'il reçoit en simulation et doit dire **quel canal porte le poison**. La permutation
change à chaque épisode, la réponse n'est donc pas mémorisable dans les poids.

| architecture | perte après 800 itérations |
|---|---|
| **v1** (`jointe`) | **1.0986 = ln 3** — le hasard, jamais quitté |
| **v2** (`separee`) | **0.999 en ~175 itérations** |
| v2 privé de `last_eaten` | échoue à l'identique de v1 |

Le contrôle est décisif : v1 échoue **parce que son LSTM ne reçoit jamais `last_eaten`**,
pas parce que la recherche échoue. Aucun réglage n'y changerait rien.

### 3.2 L'environnement récompense la discrimination

Oracle codé en dur (`simulation/oracle.py`), politique parfaite sans réseau.

| mesure | résultat |
|---|---|
| durée de vie en lab | **754 → 1503 pas** (×2) |
| avec temporisation (`oracle_wait`) | plafond du rollout atteint (2000) |
| test d'invasion | **balayage à 100 %** en ~3000 pas |
| oracle *apprenant* (croyance optimiste) | **1 poison par vie**, médiane au pas 8 |

L'oracle apprenant est le point clé : découvrir la valeur des canaux depuis sa seule
expérience coûte **une bouchée de poison par vie**, et envahit presque aussi vite que
l'oracle omniscient. L'apprentissage intra-vie est donc **écologiquement rentable**.

### 3.3 L'architecture peut exploiter un signal de valeur

Outil : `tools/probe_action.py`. Descente de gradient sur la tâche réduite — une ressource
devant l'agent, `v_pred` fourni, avancer ou tourner. Permutation neuve à chaque échantillon,
donc hasard à 0.50.

| variante | exactitude | mises à jour pour 99 % |
|---|---|---|
| `aveugle` (`v_pred` = 0) | **0.497** | — |
| `concat` (câblage actuel) | **1.000** | **506** |
| `carte` (carte de valeur) | **1.000** | **44** |

L'architecture, **à sa taille réelle** (`hidden_layers=(8,)`, 1519 paramètres), résout la
tâche parfaitement. Le blocage n'est donc ni l'architecture ni l'information : c'est la
recherche.

La carte de valeur — un canal d'observation portant `Σₖ obs[i,j,k]·v_pred[k]`, soit le
produit vision × valeur fait en amont du conv — converge **11× plus vite** pour
**+36 paramètres**.

### 3.4 L'évolution n'exploite pas une information de valeur, même parfaite

Outil : `tools/probe_vpred.py`. Les mêmes génomes rejoués en ne changeant que les trois
nombres de `v_pred`. Run `--vpred-oracle` (valeur imposée exacte, en permanence), chunk 700.

| condition | pas-agents vécus | poison /1000 pas |
|---|---|---|
| `nul` (0, 0, 0) | 475.0 | 17.5 |
| `vrai` (+1, +0.3, −1.5) | 487.0 — Δ +2.5, *p* = 0.148 | 18.2 — *p* = 1.000 |
| `fort` (×10) | **801.0** — Δ +313.5, *p* < 0.001 | **0.0** — *p* < 0.001 |
| `inverse` (×−10) | 525.5 — *p* = 0.130 | 18.5 — *p* = 0.532 |

**À l'échelle naturelle, la population ignore `v_pred`.** Amplifié dix fois, il transforme
le comportement — mais le bras `inverse`, qui annonce le poison comme la meilleure
ressource, ne fait pas manger davantage de poison. Le comportement ne suit donc pas le
**sens** des valeurs.

Deux lectures restent possibles et ce test ne les sépare pas : la tête **inhibe sans
diriger**, ou une entrée dix fois plus grande n'est qu'un choc. Dans les deux cas,
l'évolution n'a pas accordé les poids qui lisent `v_pred`.

### 3.5 Le gradient intra-vie fonctionne, l'évolution le dégrade

Boucle interne (option « interface fixe ») : une tête linéaire sur le carry prédit la valeur
de chaque canal ; sa sortie entre dans la tête de politique. Loss auto-supervisée
`(v_pred[k] − r)²`, cible = la récompense que l'agent perçoit lui-même. BPTT tronqué,
603 paramètres sur 1519 appris pendant la vie, **les poids appris meurent avec l'agent**
(baldwinien, non lamarckien).

**Intra-vie** — l'erreur chute d'un facteur ~4.5 au cours d'une vie (0.9 → 0.2).
Le circuit fonctionne à l'échelle de la population : erreur minimale mesurée **10⁻⁶**.

**Inter-générations** — la prédiction se **dégrade** :

| tranche du run | erreur à l'âge 0 |
|---|---|
| chunks 0–99 | **0.59** |
| chunks 100–199 | 0.73 |
| chunks 200–299 | 0.83 |
| chunks 500–599 | **0.92** |

**Effet Baldwin inversé.** Puisque la politique n'exploite pas `v_pred` (§ 3.4), rien ne
sélectionne la tête de valeur : ses poids dérivent. L'écart entre tranches est maximal à
l'âge 0 et se referme avec l'âge, à mesure que le gradient rattrape la dérive.

*Figure : `plot_inner_loss_by_age.png`*

### 3.6 Boucler la boucle : entraîner aussi la politique

Loss de politique auto-supervisée ajoutée :

$$\mathcal{L} = -\sum_a \pi(a)\,\hat v(a)$$

où $\hat v(a)$ est la valeur que l'agent **croit** trouver sur la case atteinte par l'action
*a*. La cible est sa propre croyance ; aucune information extérieure n'entre. Masques
disjoints : la loss de valeur touche LSTM + tête de valeur (603 poids), celle de politique
le conv + la tête (2212). La croyance passe par `stop_gradient`.

Trois réglages successifs, chacun motivé par une mesure :

| correctif | motif | effet |
|---|---|---|
| écrêtage du gradient | erreur atteignant **10¹⁶** à 500 pas de BPTT | poids max 43 → 1.3 |
| verrou de confiance | la politique suivait une croyance aléatoire dès le pas 1 | plus de mise à jour avant fiabilité |
| fenêtre glissante (`--inner-every`) | **5 mises à jour de valeur par vie** | **50 par vie** |

| | avant | après |
|---|---|---|
| erreur de prédiction | 0.58 | **0.16 – 0.21** |
| agents confiants | 20 % | **63 – 65 %** |
| maj valeur / vie | 5 | **50** |
| maj politique / vie | 110 | **~195** |

La machinerie fonctionne. **Les agents n'esquivent toujours pas le poison.**

---

## 4. Mécanisme de l'échec

La loss de politique est minimisée par l'**immobilité**, et le gradient y mène depuis
n'importe quel départ non discriminant.

| stratégie | valeur de la loss |
|---|---|
| ne jamais avancer | **+0.009** |
| avancer sans discriminer | **+0.076** |
| **discriminer** | **−0.425** |

Discriminer est de loin optimal. Mais depuis un départ non discriminant, le gradient pointe
vers l'immobilité, parce que la **valeur moyenne des ressources est négative** :

$$\frac{1.0 + 0.3 - 1.5}{3} = -0.067$$

Faire prédire le **Δénergie** plutôt que la récompense — ce qui intègre le coût métabolique
et la satiété — a déplacé le plancher de 0 à +0.009, sans changer le signe de la moyenne.
La comparaison qui compte est donc inchangée.

**Résultat observé** : la fraction d'action « rester » croît jusqu'à 55–60 %, « avancer »
s'effondre, et la population s'éteint (1499 → 0 en ~8000 pas).

*Figure : `plot_policy_influence.png`, panneau droit*

### Le déficit d'échantillons

| | mises à jour | échantillons par mise à jour | optimiseur |
|---|---|---|---|
| `probe_action` (100 %) | 506 | **256** | **Adam** |
| boucle interne | ~195 | **1** | SGD, puis Adam |

**130 000 échantillons étiquetés contre ~195 pas bruités — un facteur ≈ 660.** Le réseau
apprend ce qu'il peut apprendre en 195 pas : un décalage uniforme de π(avancer), pas une
règle conditionnelle sur ce qui est devant lui. Adam corrige le conditionnement des
gradients, pas la quantité d'information.

### Conclusion structurelle

> Toute loss gloutonne fondée sur la valeur, dans un environnement dont la ressource
> moyenne est néfaste, converge vers l'inaction **sauf si la discrimination est déjà
> présente**. Et la discrimination est précisément ce que le budget d'échantillons d'une
> vie ne permet pas d'apprendre.

C'est le problème d'amorçage dans sa forme la plus nette : la politique ne peut apprendre à
lire la croyance qu'en discriminant déjà un peu ; l'évolution ne sélectionne la lecture que
si elle sert déjà.

---

## 5. Instruments produits

Réutilisables indépendamment de cette question.

Les deux tests par gradient sont documentés en détail dans
[`tools/PROBES.md`](tools/PROBES.md), avec leurs figures.

| outil | question tranchée |
|---|---|
| `tools/probe_memory.py` | le réseau peut-il **apprendre** la valeur des canaux ? |
| `tools/probe_action.py` | peut-il **s'en servir** pour agir ? Combien de mises à jour ? |
| `tools/probe_vpred.py` | une population donnée **exploite-t-elle** ce signal ? |
| bras à gradient gelé (lab) | ce que l'apprentissage rapporte, à génome et graines égaux |
| `plot_inner_loss_by_age` | sépare l'apprentissage intra-vie de l'a priori génomique |
| `plot_offspring_by_cohort` | intensité de la sélection (dispersion du succès reproducteur) |
| `plot_policy_influence` | ce que le gradient change **aux actions**, pas aux poids |

Drapeaux ajoutés : `--inner`, `--inner-window`, `--inner-every`, `--inner-lr`,
`--inner-clip`, `--inner-optim`, `--inner-target`, `--inner-policy`, `--inner-policy-lr`,
`--inner-policy-seuil`, `--vpred-oracle`, `--vpred-gain`, `--value-map`.

---

## 6. Limites

**Ce qui n'a pas été testé.**
Hérédité lamarckienne (transmission des poids appris) ; ensemencement par un génome cloné
de l'oracle ; régularisation d'entropie contre l'effondrement ; modification des
`delta_energy` pour rendre la moyenne positive. Les trois derniers ont été écartés
délibérément : ils suppriment la question ou donnent la réponse.

**Ce que les mesures ne couvrent pas.**
`probe_action` teste une tâche réduite — une ressource, droit devant, un seul pas, sans
navigation ni murs. `probe_vpred` mesure en lab (2 agents, sans compétition ni repousse),
pas dans l'écologie complète. Le nombre de mises à jour par vie repose sur une durée de vie
estimée à 2500 pas.

**Point non résolu.**
Le bras `inverse` de `probe_vpred` ne sépare pas « la tête inhibe sans diriger » de
« l'amplification n'est qu'un choc ». Trancher demanderait de mesurer la divergence
conditionnellement à la nature de la case devant l'agent.

---

## 7. Ce que le résultat établit

L'adaptation intra-vie n'émerge pas dans ce régime, et **chaque maillon de la chaîne a été
testé séparément** :

1. l'information est disponible dans l'architecture v2 — et absente de v1 *(§ 3.1)* ;
2. l'environnement récompense la discrimination *(§ 3.2)* ;
3. l'architecture sait exploiter un signal de valeur *(§ 3.3)* ;
4. l'évolution ne l'exploite pas, même parfait et gratuit *(§ 3.4)* ;
5. le gradient apprend bien la valeur pendant la vie, mais l'évolution dégrade son point de
   départ faute de pression sélective *(§ 3.5)* ;
6. et une loss auto-supervisée plausible pour la politique mène la population à l'extinction
   en atteignant son optimum *(§ 4)*.

Le dernier point vaut d'être rapporté pour lui-même.
