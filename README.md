# Prédiction du churn Telco avec IA explicable (SHAP & LIME)

Projet Challenge 1 – Data Mining · ENSI · 2026-2027
Cas métier : **Telco Customer Churn** (n° 13 du catalogue) – prédire quels clients vont résilier leur abonnement, et expliquer pourquoi.


## Données
[Telco Customer Churn (IBM, Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) – 7 043 clients, 20 variables, cible `Churn` (~26,5 % de départs).
Le jeu n'est pas inclus dans ce dépôt : il est chargé depuis Kaggle (`/kaggle/input/...`).

## Contenu du notebook
1. Chargement et nettoyage (`TotalCharges` converti en numérique, 11 valeurs vides = clients à `tenure = 0`)
2. Exploration : distributions, taux de churn par modalité, corrélations
3. Préparation : imputation, normalisation, encodage, split stratifié (pipeline sans fuite de données)
4. Modélisation : régression logistique, arbre, KNN, Random Forest, XGBoost (`GridSearchCV`, 5 plis)
5. Évaluation : AUC-ROC, PR-AUC, précision, rappel, F1, matrices de confusion, seuil de décision
6. Explicabilité : importance native et par permutation, SHAP (beeswarm, dépendance, waterfall), LIME, phrase de synthèse, analyse d'équité

## Résultats
| Modèle | AUC CV | AUC test | Rappel | F1 |
|---|---|---|---|---|
| Régression logistique | 0,846 | | | |
| Arbre de décision | 0,832 | | | |
| KNN | 0,835 | | | |
| Random Forest | 0,848 | | | |
| XGBoost | | | | |



## Reproduire
```bash
pip install -r requirements.txt
```
Puis ouvrir `notebooks/telco_churn_xai.ipynb` sur Kaggle (ajouter le jeu *Telco Customer Churn* en entrée) ou en local en adaptant le chemin du CSV.

## Limites
- Plafond des données : AUC ≈ 0,85, quel que soit le modèle.
- Les explications SHAP/LIME décrivent le modèle, pas des relations causales.
- Variables sensibles (`gender`, `SeniorCitizen`, `Partner`, `Dependents`) : voir l'analyse d'équité du notebook.

## Auteur
<Ton nom> – <Filière>, ENSI
