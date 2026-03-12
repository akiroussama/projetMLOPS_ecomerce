# Projet MLOps E-commerce

Baseline MLOps orientee soutenance pour la classification de produits e-commerce.

Le perimetre obligatoire couvre :

- pipeline de donnees automatisable ;
- feature engineering texte ;
- entrainement d'un modele baseline ;
- API FastAPI d'inference sans reentrainement ;
- suivi des runs avec MLflow ;
- orchestration Airflow ;
- conteneurisation des composants via Docker Compose.

## Architecture

- `src/data/import_raw_data.py` : telecharge les CSV bruts depuis le bucket S3.
- `src/data/make_dataset.py` : nettoie les donnees et cree le split train/validation.
- `src/features/build_features.py` : construit les matrices TF-IDF et le vectorizer serialize.
- `src/models/train_model.py` : entraine le baseline `SGDClassifier`, calcule les metriques et log les runs dans MLflow.
- `src/api/app.py` : expose `GET /health` et `POST /predict`.
- `orchestration/dags/retraining_pipeline.py` : DAG Airflow de retraining hebdomadaire.
- `docker-compose.yml` : assemble `api`, `mlflow`, `airflow` et `postgres`.

## Pipeline locale

Depuis la racine du repo :

```bash
python -m src.main
```

La commande :

1. telecharge les CSV bruts dans `data/raw`
2. prepare les fichiers propres dans `data/preprocessed`
3. cree les features TF-IDF
4. entraine le modele et ecrit les artefacts dans `models/`
5. ecrit les rapports d'evaluation dans `reports/`

Options utiles :

```bash
python -m src.main --skip-download
python -m src.main --tracking-uri http://localhost:5000
```

## API FastAPI

L'API charge les artefacts deja entraines depuis `models/` et ne reentraine jamais au demarrage.

Lancement local :

```bash
set API_AUTH_TOKEN=change-me
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Endpoints :

- `GET /health`
- `POST /predict`

Exemple de payload :

```json
{
  "designation": "robe ete femme",
  "description": "robe rouge legere en coton",
  "productid": 1001,
  "imageid": 2002
}
```

## Stack Docker obligatoire

1. creer un fichier `.env` a partir de `.env.example`
2. initialiser Airflow :

```bash
docker compose up airflow-init
```

3. lancer la stack :

```bash
docker compose up -d api mlflow airflow-webserver airflow-scheduler
```

Services exposes :

- API : `http://localhost:8000/docs`
- MLflow : `http://localhost:5000`
- Airflow : `http://localhost:8080` avec `airflow / airflow`

Pour executer manuellement un entrainement dans le conteneur dedie :

```bash
docker compose run --rm trainer python -m src.main --skip-download
```

## Airflow

Le DAG `rakuten_weekly_retraining` automatise :

1. le telechargement des donnees brutes
2. la preparation des donnees
3. la reconstruction des features
4. le reentrainement du modele
5. la verification des artefacts consommes par l'API

Planification par defaut : tous les lundis a `02:00`.

## MLflow

Le script d'entrainement logge :

- les hyperparametres du baseline ;
- les metriques de validation ;
- le modele serialize ;
- le vectorizer ;
- les rapports JSON de performance.

## Tests

```bash
pytest
```

Les tests couvrent l'API et un smoke test du pipeline baseline jusqu'aux artefacts compatibles avec l'API.
