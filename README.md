# IADATA708 — Projet Groupe
## Algorithmic Fairness, Interpretability and Robustness
**Dataset : Pokec Social Network (Pokec-z)**

---

## 1. Tâche

Classification de noeuds — prédire si un utilisateur **exerce un métier** (`I_am_working_in_field`).

- Tâche binaire : travailleur (1) ou non (0)
- Variable cible très déséquilibrée : environ 85% de non-travailleurs → poids de classe utilisés dans la loss
- Tâche standard sur Pokec-z dans la littérature FairGNN, directement supportée par les fichiers téléchargés

---

## 2. Attributs sensibles

Deux attributs sensibles analysés séparément, tous deux **exclus des features d'entraînement** pour éviter la discrimination directe :

| | Colonne | Encodage | Rôle |
|---|---|---|---|
| **Primaire** | `AGE` | 0 = junior (<30 ans), 1 = senior (≥30 ans) | Ciblé par la méthode d'équité (~17% de seniors) |
| **Secondaire** | `gender` | 1 = homme, 0 = femme | Monitoré uniquement (~51% d'hommes) |

> L'âge est l'attribut primaire car son déséquilibre (17% de seniors) crée un biais plus prononcé et plus mesurable que le genre (51/49%), quasi-équilibré.

---

## 3. Sous-échantillon

**Pokec-z** (~70 000 noeuds) — sous-échantillon officiel issu de [FairGNN (EnyanDai)](https://github.com/EnyanDai/FairGNN).

Fichiers utilisés : `region_job.csv` (noeuds) + `region_job_relationship.txt` (arêtes)

Features retenues : toutes les colonnes du profil utilisateur à l'exception des attributs sensibles (`gender`, `AGE`), de la variable cible (`I_am_working_in_field`) et des colonnes dérivées (`label`, `age_group`, `user_id`).

Graphe : **non-dirigé** (les arêtes follower→followee sont dupliquées dans les deux sens)

Split : **60% train / 20% val / 20% test** (stratifié sur la variable cible)

---

## 4. Modèle baseline

**GraphSAGE** — 2 couches, agrégation mean (PyTorch Geometric), 30 époques

| Hyperparamètre | Valeur |
|---|---|
| Dimension cachée | 64 |
| Dropout | 0.5 |
| Optimiseur | Adam (lr=0.01, weight_decay=5e-4) |

| Métrique | Type | Résultat baseline |
|---|---|---|
| Accuracy | Performance | ~0.52 |
| AUC-ROC | Performance | ~0.80 |
| SPD (âge) | Équité | −0.057 |
| EOD (âge) | Équité | −0.051 |

> SPD et EOD calculés séparément pour `gender` et `age` ; l'attribut primaire pour l'évaluation de l'équité est l'âge.

---

## 5. Méthode d'équité

**Adversarial Debiasing** (in-training), hyperparamètre λ=1

Un adversaire linéaire (`h → 2`) est entraîné en parallèle pour prédire le groupe d'âge à partir des représentations latentes `h`. Le classifieur principal est pénalisé s'il encode cet attribut via la loss :

`L = L_task − λ × L_adv`

Augmenter λ réduit davantage le biais au prix d'une perte de performance. Une analyse du trade-off est réalisée pour λ ∈ {0.0, 0.2, 0.5, 0.8, 1.0, 1.5}.

| Métrique | Baseline | Fair (λ=1) |
|---|---|---|
| AUC-ROC | 0.80 | 0.78 |
| SPD (âge) | −0.057 | −0.037 |
| EOD (âge) | −0.051 | −0.033 |

---

## 6. Interprétabilité

**GNNExplainer** (Ying et al., 2019)

Identifie, pour une prédiction donnée, les features les plus importantes. Appliqué au modèle baseline pour comparer les top-features entre **un noeud junior** et **un noeud senior** du jeu de test.

Résultat : les deux groupes utilisent des ensembles de features entièrement disjoints, révélant une **discrimination indirecte** par features proxy de l'âge (style de vie chez les juniors, niveau d'éducation chez les seniors).

---

## 7. Robustesse

Perturbation contrôlée : **bruit gaussien** sur l'ensemble de la matrice de features X, avec σ ∈ {0, 0.1, 0.3, 0.5} (moyenne sur 5 répétitions).

Résultats : les deux modèles sont robustes en performance (l'agrégation GNN lisse les perturbations individuelles). En revanche, le biais du modèle fair est davantage sensible au bruit car l'Adversarial Debiasing supprime les corrélations fortes entre features et âge.

---

## Pipeline

```
[Données Pokec-z (~70 000 noeuds)]
      │
[EDA : distributions cible, genre, âge]
      │
[Prétraitement : nettoyage, binarisation, split 60/20/20 stratifié]
      │
[Construction du graphe PyG non-dirigé]
      │
[Baseline GraphSAGE 30 époques] ──► métriques perf + équité (SPD, EOD)
      │
[GraphSAGE + Adversarial Debiasing λ=1] ──► trade-off perf/équité (~35% réduction du biais)
      │     └── analyse du trade-off pour λ ∈ {0, 0.2, 0.5, 0.8, 1.0, 1.5}
[GNNExplainer] ──► identification des features proxy de l'âge (junior vs senior)
      │
[Injection de bruit gaussien] ──► robustesse perf + équité sous perturbation
```
