# IADATA708 — Projet Groupe
## Algorithmic Fairness, Interpretability and Robustness
**Dataset : Pokec Social Network (Pokec-z)**

---

## 1. Tâche

Classification de noeuds — prédire la **profession** (`job`) de chaque utilisateur.

- Tâche binaire : appartenir ou non à une catégorie professionnelle donnée (selon le découpage de `region_job.csv`)
- Tâche standard sur Pokec-z dans la littérature FairGNN, directement supportée par les fichiers téléchargés

---

## 2. Attributs sensibles

Deux attributs sensibles, analysés séparément :

| | Colonne | Encodage |
|---|---|---|
| **Primaire** | `gender` | 1 = homme, 0 = femme |
| **Secondaire** | `age` | 0 = jeune (<30), 1 = senior (≥30) |

> `gender` et `age` sont exclus des features d'entraînement lorsqu'ils sont utilisés comme attributs sensibles (éviter la fuite d'information directe).

---

## 3. Sous-échantillon

**Pokec-z** (~10 000 noeuds) — sous-échantillon officiel issu de [FairGNN (EnyanDai)](https://github.com/EnyanDai/FairGNN).

Fichiers utilisés : `region_job.csv` + `region_job_relationship.txt`

Features retenues : `completion_percentage`, `age`, `region` (encodé), `public`, `completed_level_of_education`, `marital_status`

Split : **60% train / 20% val / 20% test**

---

## 4. Modèle baseline

**GraphSAGE** — 2 couches, agrégation mean (PyTorch Geometric)

| Métrique | Type |
|---|---|
| Accuracy, AUC-ROC | Performance |
| Statistical Parity Difference (SPD) | Équité |
| Equal Opportunity Difference (EOD) | Équité |

> SPD et EOD calculés séparément pour `gender` et `age`.

---

## 5. Méthode d'équité

**Adversarial Debiasing** (in-training)

Un adversaire est entraîné en parallèle pour prédire l'attribut sensible à partir des représentations latentes. Le modèle principal est pénalisé s'il encode cet attribut, via un hyperparamètre `lambda` qui contrôle le compromis performance/équité.

---

## 6. Interprétabilité

**GNNExplainer** (Ying et al., 2019)

Identifie, pour une prédiction donnée, le sous-graphe et les features les plus importants. Utilisé pour comparer les explications entre groupes (homme/femme, jeune/senior).

---

## 7. Robustesse

Perturbation contrôlée : **bruit gaussien** sur les features numériques (`completion_percentage`, `age`) avec niveaux croissants σ ∈ {0, 0.1, 0.3, 0.5}.

Objectif : mesurer la dégradation de la performance **et** de l'équité sous bruit.

---

## Pipeline

```
[Données Pokec-z]
      │
[Prétraitement : encodage features, split 60/20/20]
      │
[Baseline GraphSAGE] ──► métriques perf + équité (SPD, EOD)
      │
[GraphSAGE + Adversarial Debiasing] ──► compromis perf/équité
      │
[GNNExplainer] ──► analyse des décisions par groupe
      │
[Injection de bruit] ──► robustesse perf + équité sous perturbation
```
