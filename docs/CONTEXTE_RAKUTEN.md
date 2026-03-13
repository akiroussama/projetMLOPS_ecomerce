# Contexte : Pourquoi Rakuten ?

## Historique du projet

Ce repository est base sur un **template MLOps fourni par l'ecole** (structure cookiecutter data science).
L'equipe a choisi de reprendre le **cas d'usage Rakuten** issu de leur projet Data Science precedent,
comme recommande par le professeur Antoine Fradin :

> "Vous etes libres de choisir les donnees, modeles et outils qui vous interessent (reutilisation du projet DS)."

## Le Challenge Rakuten France

Rakuten France gere une marketplace avec des millions de produits. Chaque produit doit etre
classe dans l'une des **27 categories** (`prdtypecode`) pour assurer une navigation coherente.

### Donnees
- **Source** : Challenge Rakuten France (donnees publiques sur S3)
- **Features** : `designation` (titre court), `description` (texte long), `productid`, `imageid`
- **Target** : `prdtypecode` (27 classes de produits)

### Modele baseline
- **Pipeline** : TF-IDF (5000 features, unigrams + bigrams) + SGDClassifier
- **Choix justifie** : Le modele texte-only offre un bon compromis performance/simplicite
  pour la mise en production. Les modeles deep learning (LSTM, VGG16) du projet DS original
  sont conserves comme reference mais non deployes dans le pipeline MLOps.

## Du projet DS au projet MLOps

| Projet DS (avant) | Projet MLOps (maintenant) |
|-------------------|---------------------------|
| Notebooks Jupyter | Scripts Python modulaires |
| Modeles en local | API FastAPI securisee |
| Pas de versioning modele | MLflow tracking |
| Execution manuelle | Airflow orchestration |
| Pas de monitoring | Prometheus + Grafana |
| Pas de CI/CD | GitHub Actions pipeline |
| Pas d'interface | Streamlit dashboard |

L'objectif MLOps est d'**industrialiser** le modele existant avec un pipeline complet,
reproductible et monitorable.
