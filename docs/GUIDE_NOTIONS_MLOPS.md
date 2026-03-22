# Guide de Maitrise MLOps — Toutes les notions expliquees

> Ce guide explique chaque concept, outil et terme technique utilise dans le projet.
> Objectif : repondre a n'importe quelle question du jury avec assurance.

---

## TABLE DES MATIERES

1. [Concepts fondamentaux MLOps](#1-concepts-fondamentaux-mlops)
2. [Le modele ML : TF-IDF + SGDClassifier](#2-le-modele-ml)
3. [FastAPI — L'API de prediction](#3-fastapi)
4. [Docker & Docker Compose — Conteneurisation](#4-docker--docker-compose)
5. [MLflow — Suivi des experiences](#5-mlflow)
6. [Airflow — Orchestration](#6-airflow)
7. [Prometheus — Collecte de metriques](#7-prometheus)
8. [Grafana — Visualisation et dashboards](#8-grafana)
9. [Evidently — Detection de derive](#9-evidently)
10. [GitHub Actions — CI/CD](#10-github-actions--cicd)
11. [Streamlit — Interface utilisateur](#11-streamlit)
12. [DVC — Versioning des donnees](#12-dvc)
13. [Comment tout se connecte](#13-comment-tout-se-connecte)

---

## 1. Concepts fondamentaux MLOps

### Qu'est-ce que MLOps ?

MLOps = **Machine Learning** + **Operations**. C'est l'ensemble des pratiques pour mettre un modele ML en production et le maintenir dans le temps.

**Analogie :** Un data scientist cree un modele dans un notebook. Un ingenieur MLOps le transforme en systeme fiable qui tourne 24h/24, se re-entraine automatiquement, et alerte quand quelque chose ne va pas.

### Les 3 piliers MLOps

```
1. REPRODUCTIBILITE    →  On peut recreer le meme resultat
                           (DVC pour les donnees, MLflow pour les params, Git pour le code)

2. AUTOMATISATION      →  Pas d'action manuelle
                           (Airflow orchestre, CI/CD teste, Docker deploie)

3. MONITORING          →  On sait si ca marche
                           (Prometheus collecte, Grafana affiche, Evidently detecte la derive)
```

### Pipeline vs Workflow vs DAG

| Terme | Definition | Exemple dans notre projet |
|-------|-----------|--------------------------|
| **Pipeline** | Sequence d'etapes qui transforment des donnees | donnees brutes → nettoyage → TF-IDF → modele |
| **Workflow** | Enchainement automatise de taches | Le DAG Airflow qui fait tourner le pipeline chaque lundi |
| **DAG** | Directed Acyclic Graph — graphe oriente sans cycle | Les 6 taches Airflow liees par des fleches, sans retour en arriere |

### Qu'est-ce qu'un artifact ?

Un **artifact** est un fichier produit par une etape du pipeline et consomme par une autre.

Nos artifacts :
- `baseline_model.pkl` — le modele serialise (produit par train_model, consomme par l'API)
- `tfidf_vectorizer.pkl` — le vectorizer TF-IDF (produit par build_features, consomme par l'API)
- `label_mapping.json` — la correspondance code → nom de categorie
- `training_metrics.json` — les metriques d'evaluation
- `classification_report.json` — precision/recall par classe
- `drift_report.html` — le rapport Evidently

### Serialisation (pickle / joblib)

**Serialiser** = sauvegarder un objet Python sur le disque pour le recharger plus tard sans re-entrainer.

```python
# Sauvegarder (dans train_model.py)
joblib.dump(model, "baseline_model.pkl")

# Charger (dans service.py, au demarrage de l'API)
model = joblib.load("baseline_model.pkl")
```

Le format `.pkl` est du **pickle** Python — une representation binaire d'un objet.

---

## 2. Le modele ML

### TF-IDF (Term Frequency — Inverse Document Frequency)

**Objectif :** Transformer du texte en chiffres que le modele peut comprendre.

**Comment ca marche :**

```
TF  = combien de fois un mot apparait dans CE document
IDF = inverse de combien de documents contiennent ce mot (log scale)

TF-IDF = TF × IDF
```

**Exemple concret :**
- Le mot "PlayStation" apparait souvent dans les jeux video → TF eleve pour ces produits
- Le mot "le" apparait dans TOUS les documents → IDF tres bas (mot non discriminant)
- TF-IDF de "PlayStation" sera eleve pour un jeu PS5, bas pour un livre

**Notre configuration :**
- `max_features=5000` → on garde les 5000 mots les plus informatifs
- `ngram_range=(1,2)` → on utilise des mots seuls (unigrams) ET des paires de mots (bigrams)
  - Unigram : "PlayStation"
  - Bigram : "PlayStation 5", "Harry Potter"
- Le texte passe par `normalize_text()` avant vectorisation

### normalize_text() — Le nettoyage du texte

```
Etape 1 : html.unescape()          "caf&eacute;"  →  "cafe"
Etape 2 : Supprimer balises HTML    "<b>Livre</b>" →  "Livre"
Etape 3 : Normalisation Unicode     "café"         →  "cafe" (NFKD)
Etape 4 : Translitteration ASCII    "naïf"         →  "naif"
Etape 5 : Lowercase + alphanum      "PS5 !!!"      →  "ps5"
Etape 6 : Collapse whitespace       "mot   mot"    →  "mot mot"
```

**Pourquoi ?** Sans ca, "café", "Café", "CAFÉ" et "caf&eacute;" seraient 4 tokens differents dans TF-IDF. On veut qu'ils comptent comme un seul.

### SGDClassifier — Le modele

**SGD** = Stochastic Gradient Descent (Descente de Gradient Stochastique).

C'est un algorithme d'optimisation, pas un modele en soi. Avec `loss='log_loss'`, il devient une **regression logistique** — un classifieur lineaire qui produit des probabilites.

**Nos hyperparametres :**

| Parametre | Valeur | Pourquoi |
|-----------|--------|----------|
| `loss='log_loss'` | Regression logistique | Produit des probabilites (predict_proba) — utile pour la jauge de confiance |
| `alpha=1e-5` | Regularisation L2 faible | Avec 5000 features, on ne veut pas trop contraindre le modele |
| `max_iter=1000` | Iterations max | Assez pour converger |
| `tol=1e-3` | Tolerance | Arrete si l'amelioration est < 0.001 |
| `class_weight='balanced'` | Ponderation inverse | Compense le desequilibre des 27 classes |
| `random_state=42` | Graine aleatoire | Reproductibilite |

### class_weight='balanced' — Gestion du desequilibre

Les 27 categories n'ont pas le meme nombre de produits. Sans `balanced` :
- Le modele predit toujours la classe majoritaire (ex: "Livres") car ca maximise l'accuracy
- Les classes rares (ex: "Jeux de societe") sont ignorees

Avec `balanced`, scikit-learn calcule :
```
poids_classe = n_total / (n_classes × n_echantillons_classe)
```
Resultat : une erreur sur une classe rare coute plus cher qu'une erreur sur une classe frequente.

### Metriques d'evaluation

| Metrique | Formule simple | Notre valeur | Pourquoi cette metrique |
|----------|---------------|-------------|------------------------|
| **Accuracy** | predictions correctes / total | ~76% | Facile a comprendre pour le metier |
| **Macro F1** | moyenne des F1 par classe (non ponderee) | ~74% | Penalise si on ignore les classes rares |
| **Precision** | vrais positifs / (vrais positifs + faux positifs) | par classe | "Quand le modele dit Livre, c'est souvent un livre ?" |
| **Recall** | vrais positifs / (vrais positifs + faux negatifs) | par classe | "Parmi les vrais livres, combien sont detectes ?" |

**Pourquoi Macro F1 et pas juste Accuracy ?**
Avec 27 classes desequilibrees, un modele qui predit toujours la classe la plus frequente aurait ~15% d'accuracy mais 0% de F1 sur les classes rares. Le macro F1 traite chaque classe a egalite.

---

## 3. FastAPI

### Qu'est-ce que FastAPI ?

Un **framework Python** pour creer des APIs REST. Plus rapide que Flask, avec validation automatique des donnees (Pydantic) et documentation interactive (Swagger).

### Nos 4 endpoints

```
GET  /health    →  Le service est-il en vie ? Quel modele est charge ?
                   Public (pas de token). Utilise par Docker healthcheck.

POST /predict   →  Envoie un titre + description, recoit la categorie predite.
                   Protege par Bearer token. C'est le coeur de l'API.

GET  /stats     →  Combien de predictions ont ete faites ? Latence moyenne ?
                   Public. Stats en memoire, reset au restart.

GET  /metrics   →  Metriques Prometheus (format texte).
                   Public. Scrape par Prometheus toutes les 5 secondes.
```

### Le flux d'une prediction

```
1. Streamlit envoie :
   POST /predict
   Header: Authorization: Bearer rakuten-soutenance-2024
   Body: { "designation": "FIFA 24 PS5", "description": "Jeu football..." }

2. L'API verifie le token (hmac.compare_digest — timing-safe)

3. PredictionService compose le texte :
   text = designation + " " + description

4. TfidfVectorizer.transform(text) → vecteur sparse de 5000 dimensions

5. SGDClassifier.predict(vecteur) → code predit (ex: 40)

6. label_mapping[40] → "Jeux video / Consoles"

7. SGDClassifier.predict_proba(vecteur) → confiance (ex: 0.36)

8. Retourne :
   { "predicted_label": "Jeux video / Consoles",
     "predicted_code": 40,
     "confidence": 0.36,
     "model_name": "SGDClassifier" }
```

### Securite — Bearer Token

```
Client envoie :    Authorization: Bearer rakuten-soutenance-2024
Serveur compare :  hmac.compare_digest(token_recu, API_AUTH_TOKEN)
```

**hmac.compare_digest** = comparaison en temps constant. Pourquoi ?
- Un `==` classique s'arrete au premier caractere different
- Un attaquant peut mesurer le temps de reponse et deviner le token lettre par lettre (timing attack)
- `compare_digest` prend TOUJOURS le meme temps, quel que soit le nombre de caracteres corrects

### Pydantic — Validation des donnees

```python
class PredictRequest(BaseModel):
    designation: str          # 1 a 512 caracteres, OBLIGATOIRE
    description: str = ""     # 0 a 10000 caracteres, optionnel
    productid: int | None     # >= 0, optionnel
    imageid: int | None       # >= 0, optionnel

    model_config = ConfigDict(extra="forbid")  # Rejette les champs inconnus
```

Si un champ est invalide → erreur **422** avec le detail exact du probleme.

### Lifespan — Chargement du modele au demarrage

```python
@asynccontextmanager
async def lifespan(app):
    # STARTUP : charge le modele UNE SEULE FOIS
    service = PredictionService.from_directory("/app/models")
    app.state.service = service
    yield
    # SHUTDOWN : nettoyage
```

Le modele est charge en memoire au demarrage de l'API. Chaque requete utilise le meme objet — pas de rechargement a chaque prediction.

---

## 4. Docker & Docker Compose

### Qu'est-ce que Docker ?

Docker **isole** une application dans un **conteneur** — un environnement leger qui contient le code, les dependances, et la configuration. Comme une machine virtuelle, mais en beaucoup plus leger.

### Qu'est-ce qu'une image Docker ?

Une **image** = un template immuable (comme un CD d'installation).
Un **conteneur** = une instance en cours d'execution de cette image (comme un programme lance).

```
Image python:3.10-slim  →  Conteneur "api" qui tourne sur le port 8000
                        →  Conteneur "streamlit" qui tourne sur le port 8501
```

### Dockerfile — La recette

```dockerfile
FROM python:3.10-slim           # Image de base (OS + Python)
WORKDIR /app                    # Repertoire de travail
COPY requirements.txt .         # Copie les dependances
RUN pip install -r requirements.txt  # Installe les packages
COPY src/ /app/src/             # Copie le code source
RUN mkdir -p /app/models        # Cree le dossier pour les modeles (vide)
EXPOSE 8000                     # Documente le port utilise
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Pourquoi `mkdir` au lieu de `COPY models/` ?**
Parce que les modeles sont montes via un **volume Docker** au runtime. Si on les copiait dans l'image, ils seraient figes et on ne pourrait pas les mettre a jour sans reconstruire l'image.

### Docker Compose — L'orchestrateur local

Docker Compose permet de definir et lancer **plusieurs conteneurs** qui communiquent entre eux. Un seul fichier `docker-compose.yml`, une seule commande : `docker compose up`.

### Nos 11 services

```
┌─────────────────── RESEAU DOCKER INTERNE ───────────────────┐
│                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐               │
│  │ Streamlit │───▶│   API    │◀───│  MLflow  │               │
│  │   :8501   │HTTP│  :8000   │    │  :5000   │               │
│  └──────────┘    └────┬─────┘    └──────────┘               │
│                       │                                      │
│                  ┌────▼─────┐                                │
│                  │Prometheus│    ┌──────────┐                │
│                  │  :9090   │───▶│ Grafana  │                │
│                  └──────────┘    │  :3000   │                │
│                                  └──────────┘                │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐               │
│  │ Airflow  │    │ Airflow  │    │ Postgres │               │
│  │  Web UI  │    │Scheduler │    │ (BDD)    │               │
│  │  :8080   │    │          │    │  :5432   │               │
│  └──────────┘    └──────────┘    └──────────┘               │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

| Service | Port externe | Port interne | Role |
|---------|-------------|-------------|------|
| `api` | 8200 | 8000 | Prediction FastAPI |
| `mlflow` | 5000 | 5000 | Tracking ML |
| `postgres` | — | 5432 | BDD Airflow |
| `airflow-webserver` | 8280 | 8080 | UI Airflow |
| `airflow-scheduler` | — | — | Execute les DAGs |
| `airflow-init` | — | — | Cree la BDD + admin |
| `streamlit` | 8501 | 8501 | Interface utilisateur |
| `prometheus` | 9090 | 9090 | Collecte metriques |
| `grafana` | 3000 | 3000 | Dashboards |
| `trainer` | — | — | Entrainement standalone |
| `bootstrap` | — | — | Init donnees |

### Volumes — Persistance des donnees

Un **volume** = un dossier partage entre le host et le(s) conteneur(s). Les donnees survivent au restart du conteneur.

```yaml
volumes:
  - ./models:/app/models         # Modeles partages entre api, trainer, airflow
  - ./data:/app/data             # Donnees partages
  - ./reports:/app/reports       # Rapports
  - mlflow-data:/mlflow/data     # BDD MLflow (SQLite)
  - mlflow-artifacts:/mlflow/artifacts  # Artefacts MLflow
  - postgres-db-volume:/var/lib/postgresql/data  # BDD Airflow
```

### Healthcheck — L'API est-elle vivante ?

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 20s      # Verifie toutes les 20 secondes
  timeout: 5s         # Si pas de reponse en 5s → echec
  retries: 5          # 5 echecs d'affilee → conteneur "unhealthy"
```

### Communication inter-services

Dans Docker Compose, chaque service est accessible par **son nom** :
- Streamlit appelle `http://api:8000/predict` (pas localhost)
- Airflow logge dans `http://mlflow:5000`
- Grafana lit `http://prometheus:9090`

C'est le **reseau Docker interne** qui resout les noms en adresses IP.

---

## 5. MLflow

### Qu'est-ce que MLflow ?

Un outil open-source pour **tracker les experiences ML**. Chaque fois qu'on entraine un modele, MLflow enregistre :
- Les **parametres** (hyperparametres utilises)
- Les **metriques** (resultats obtenus)
- Les **artefacts** (fichiers produits — modele, vectorizer, rapports)

### Les concepts cles

| Concept | Definition | Chez nous |
|---------|-----------|-----------|
| **Experiment** | Un groupe de runs lies au meme objectif | `rakuten-text-baseline` |
| **Run** | Un entrainement unique avec ses parametres et resultats | `airflow-retraining` (26 runs) |
| **Parameter** | Un reglage du modele (input) | `alpha=1e-5`, `max_features=5000` |
| **Metric** | Un resultat mesure (output) | `val_accuracy=0.7668`, `val_macro_f1=0.74` |
| **Artifact** | Un fichier produit | `baseline_model.pkl`, `tfidf_vectorizer.pkl` |

### Comment on l'utilise dans le code

```python
import mlflow

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("rakuten-text-baseline")

with mlflow.start_run(run_name="airflow-retraining"):
    # Log des parametres
    mlflow.log_param("model_class", "SGDClassifier")
    mlflow.log_param("alpha", 1e-5)
    mlflow.log_param("max_features", 5000)

    # Entrainement...
    model.fit(X_train, y_train)

    # Log des metriques
    mlflow.log_metric("val_accuracy", 0.7668)
    mlflow.log_metric("val_macro_f1", 0.74)

    # Log des artefacts
    mlflow.log_artifact("models/baseline_model.pkl")
    mlflow.log_artifact("models/tfidf_vectorizer.pkl")
```

### Tracking Server vs Model Registry

| Composant | Ce qu'il fait | On l'utilise ? |
|-----------|-------------|----------------|
| **Tracking** | Logge les runs (params, metriques, artefacts) | OUI — c'est le coeur |
| **Model Registry** | Gere le cycle de vie (Staging → Production → Archived) | NON — c'est une amelioration prevue |

### Notre configuration

- **Serveur** : `http://mlflow:5000` dans le conteneur Docker
- **Backend store** : SQLite (`/mlflow/data/mlflow.db`) — stocke les metadonnees
- **Artifact store** : Volume Docker (`/mlflow/artifacts`) — stocke les fichiers
- **Version** : MLflow 2.20.1

---

## 6. Airflow

### Qu'est-ce que Apache Airflow ?

Un **orchestrateur** de workflows. Il planifie et execute des taches dans un ordre defini, avec retry, alerting, et interface web.

### DAG — Directed Acyclic Graph

Un **DAG** est un graphe qui decrit les dependances entre les taches.

- **Directed** : les fleches ont un sens (A → B signifie "A doit finir avant B")
- **Acyclic** : pas de boucle (A → B → C → A est interdit)
- **Graph** : un ensemble de noeuds (taches) relies par des aretes (dependances)

**Notre DAG :**

```
download_raw_data → prepare_dataset → build_features → train_model → verify_artifacts → generate_drift_report
```

Chaque tache attend que la precedente reussisse. Si `build_features` echoue, `train_model` ne se lance PAS.

### Notre DAG en detail

```python
dag = DAG(
    dag_id="rakuten_weekly_retraining",
    schedule="0 2 * * 1",      # Chaque lundi a 2h UTC
    start_date=datetime(2026, 3, 10),
    catchup=False,              # Ne rattrape pas les executions manquees
    tags=["mlops", "retraining"],
    default_args={
        "owner": "mlops-team",
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    }
)
```

### Les 6 taches

| # | Task ID | Ce qu'elle fait | Commande |
|---|---------|----------------|----------|
| 1 | `download_raw_data` | Telecharge les CSV depuis S3 | `python -m src.data.import_raw_data` |
| 2 | `prepare_dataset` | Nettoie, deduplique, split 80/20 stratifie | `python -m src.data.make_dataset` |
| 3 | `build_features` | Vectorise le texte en TF-IDF (5000 features) | `python -m src.features.build_features` |
| 4 | `train_model` | Entraine SGDClassifier + logge dans MLflow | `python -m src.models.train_model` |
| 5 | `verify_artifacts` | Verifie que les fichiers .pkl existent | Script bash inline |
| 6 | `generate_drift_report` | Genere un rapport Evidently (data drift) | `python -m src.monitoring.drift_report` |

### Expression CRON

```
0 2 * * 1
│ │ │ │ │
│ │ │ │ └── Jour de la semaine (1 = Lundi)
│ │ │ └──── Mois (* = tous)
│ │ └────── Jour du mois (* = tous)
│ └──────── Heure (2h)
└────────── Minute (0)

= "Chaque lundi a 02:00 UTC"
```

### BashOperator

Chaque tache utilise un `BashOperator` — elle execute une commande shell dans le conteneur Airflow.

```python
download_raw_data = BashOperator(
    task_id="download_raw_data",
    bash_command="cd /opt/airflow/project && python -m src.data.import_raw_data --output-dir data/raw",
)
```

### Airflow vs Crontab

| Fonctionnalite | Crontab | Airflow |
|----------------|---------|---------|
| Planification | Oui | Oui |
| Dependances entre taches | Non | Oui (DAG) |
| Retry automatique | Non | Oui (retries=1, retry_delay=5min) |
| Interface web | Non | Oui (port 8280) |
| Historique d'execution | Logs fichier | UI complete avec status par tache |
| Trigger manuel | Non | Oui (bouton dans l'UI) |
| Alerting | Non | Oui (email, Slack) |

### Les 3 conteneurs Airflow

| Conteneur | Role |
|-----------|------|
| `airflow-init` | Cree la BDD PostgreSQL, migre le schema, cree l'utilisateur admin |
| `airflow-webserver` | Sert l'interface web (port 8280) |
| `airflow-scheduler` | Execute les DAGs selon le schedule |

### LocalExecutor vs CeleryExecutor

On utilise **LocalExecutor** — les taches tournent en processus locaux dans le scheduler.

- **LocalExecutor** : Simple, suffisant pour 1 DAG. Pas de broker externe.
- **CeleryExecutor** : Pour du multi-worker distribue. Necessite Redis/RabbitMQ. Overkill pour nous.

---

## 7. Prometheus

### Qu'est-ce que Prometheus ?

Un systeme de **collecte et stockage de metriques** en time series. Il "tire" (scrape) les metriques des applications a intervalles reguliers.

### Le modele Pull vs Push

```
PULL (Prometheus) :
  Prometheus ──demande──▶ API /metrics ──repond──▶ Prometheus stocke

PUSH (ex: Datadog) :
  API ──envoie──▶ Serveur de metriques
```

Avantage du **pull** : si l'API est down, Prometheus le detecte (pas de reponse). Avec du push, le silence pourrait etre un crash ou juste une absence de donnees.

### Notre configuration

```yaml
# docker/monitoring/prometheus.yml
global:
  scrape_interval: 5s      # Toutes les 5 secondes

scrape_configs:
  - job_name: 'fastapi'
    static_configs:
      - targets: ['api:8000']   # Scrape l'API sur le reseau Docker
```

### Les metriques exposees

L'API expose automatiquement `/metrics` grace a `prometheus-fastapi-instrumentator` :

```python
# src/api/app.py
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(application).expose(application)
```

**Zero code metier modifie.** L'instrumentator wrap chaque endpoint automatiquement.

**Metriques collectees :**

| Metrique | Type | Description |
|----------|------|-------------|
| `http_requests_total` | Counter | Nombre total de requetes (par method, handler, status) |
| `http_request_duration_seconds` | Histogram | Distribution de la latence |
| `http_request_size_bytes` | Histogram | Taille des requetes entrantes |
| `http_response_size_bytes` | Histogram | Taille des reponses sortantes |
| `http_requests_in_progress` | Gauge | Requetes actuellement en cours |

### Types de metriques Prometheus

| Type | Comportement | Exemple |
|------|-------------|---------|
| **Counter** | Ne fait que monter (compteur) | `http_requests_total` |
| **Gauge** | Monte et descend | `http_requests_in_progress` |
| **Histogram** | Distribution en buckets | `http_request_duration_seconds` |

### PromQL — Le langage de requete

Grafana interroge Prometheus avec PromQL :

```promql
# Requetes par seconde (par endpoint et status)
rate(http_requests_total[1m])

# Latence P95 (95eme percentile)
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
```

`rate()` calcule le taux de changement par seconde sur une fenetre de temps.

---

## 8. Grafana

### Qu'est-ce que Grafana ?

Un outil de **visualisation** qui cree des dashboards a partir de sources de donnees (Prometheus, ElasticSearch, MySQL, etc.).

### Notre dashboard "Rakuten API Monitoring"

4 panneaux sur notre dashboard :

| Panneau | Metrique PromQL | Ce qu'on voit |
|---------|----------------|---------------|
| **Requests per Second** | `rate(http_requests_total[1m])` | Courbes par endpoint (/health, /metrics, /predict) |
| **Response Latency P95** | `histogram_quantile(0.95, ...)` | Latence au 95eme percentile (~95ms) |
| **Status Code Distribution** | `http_requests_total` par status | 41 233 en 2xx (100%), 14 en 4xx |
| **Active Requests** | `http_requests_in_progress` | Requetes en cours a cet instant |

### Provisioning automatique

On ne configure PAS Grafana a la main. Tout est dans des fichiers YAML montes en volume :

```
docker/monitoring/grafana/
├── provisioning/
│   ├── datasources/
│   │   └── prometheus.yml      # Connecte Prometheus comme source
│   └── dashboards/
│       └── dashboard.yml       # Dit a Grafana ou trouver les dashboards
└── dashboards/
    └── api-monitoring.json     # Le dashboard lui-meme (JSON exporte)
```

Au demarrage, Grafana lit ces fichiers et se configure automatiquement. **Zero clics manuels.**

### Configuration Grafana

```yaml
environment:
  GF_SECURITY_ADMIN_PASSWORD: admin        # Mot de passe admin
  GF_AUTH_ANONYMOUS_ENABLED: "true"        # Acces sans login en lecture
  GF_AUTH_ANONYMOUS_ORG_ROLE: Viewer       # Les anonymes voient mais ne modifient pas
```

---

## 9. Evidently

### Qu'est-ce que le Data Drift ?

Le **data drift** = la distribution des donnees en production change par rapport aux donnees d'entrainement.

**Exemple concret :**
- Notre modele est entraine sur des descriptions en francais
- Rakuten commence a ajouter des produits en anglais
- Les mots TF-IDF ne correspondent plus → le modele se trompe
- L'accuracy sur le dataset de test n'a PAS change (il est fixe !)
- Seul le monitoring Evidently detecte le probleme

### Comment on detecte le drift

On compare 4 features derivees du texte entre le **dataset de reference** (train) et le **dataset courant** (validation) :

| Feature | Calcul | Pourquoi |
|---------|--------|----------|
| `text_length` | `len(designation)` | Si les titres deviennent plus longs/courts |
| `description_length` | `len(description)` | Changement dans les descriptions |
| `has_description` | `1 si non-vide, 0 sinon` | Plus/moins de produits sans description |
| `word_count` | nombre de mots dans designation | Changement de structure des titres |

### Test de Kolmogorov-Smirnov (KS-test)

Le **KS-test** compare deux distributions et donne :
- `ks_stat` : distance maximale entre les deux distributions cumulatives (0 = identiques, 1 = completement differentes)
- `p_value` : probabilite que les deux echantillons viennent de la meme distribution

**Si p_value < 0.05** → drift detecte (les distributions sont significativement differentes).

### Integration dans Airflow

C'est la **6eme et derniere tache** du DAG. Elle produit :
- `drift_report.html` — rapport visuel (consultable dans le navigateur)
- `drift_report.json` — donnees structurees (exploitables par un script d'alerte)

---

## 10. GitHub Actions — CI/CD

### Qu'est-ce que CI/CD ?

- **CI** (Continuous Integration) : a chaque push, on lance automatiquement les tests pour verifier que le code ne casse rien
- **CD** (Continuous Delivery) : si les tests passent, on build et publie automatiquement l'image Docker

### Notre pipeline (3 jobs)

```
Push sur main ou PR vers main
        │
        ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────────┐
│   1. LINT    │───▶│   2. TEST    │───▶│ 3. DOCKER BUILD  │
│   (flake8)   │    │   (pytest)   │    │  (ghcr.io push)  │
└──────────────┘    └──────────────┘    └──────────────────┘
```

| Job | Outil | Commande | Critere de succes |
|-----|-------|---------|-------------------|
| **Lint** | flake8 | `flake8 src/ --max-line-length 127` | Aucune erreur de style |
| **Test** | pytest | `pytest --cov=src/api --cov-fail-under=80` | Tous les tests passent + couverture >= 80% |
| **Docker** | docker build | Build + push vers `ghcr.io` | Image construite avec succes |

### Couverture de code (coverage)

La couverture mesure quel pourcentage du code est execute par les tests :
- **80%** minimum sur `src/api/` — c'est notre seuil
- Si un PR fait baisser en dessous de 80% → le CI est rouge → on ne merge pas

### GHCR — GitHub Container Registry

L'image Docker est publiee sur `ghcr.io` (le registre de conteneurs de GitHub) :
```
ghcr.io/{owner}/projetmlops-ecomerce-api:latest
```
N'importe quel serveur peut ensuite `docker pull` cette image pour deployer.

---

## 11. Streamlit

### Qu'est-ce que Streamlit ?

Un framework Python pour creer des **interfaces web interactives** sans ecrire de HTML/CSS/JS. Ideal pour le prototypage data/ML.

### Nos 4 pages

| Page | Contenu | Ce que voit le jury |
|------|---------|-------------------|
| **Accueil** | Vue d'ensemble + status API | Bandeau vert "API connectee, SGDClassifier charge" |
| **Contexte** | Challenge Rakuten + schema d'architecture + metriques | Schema ASCII de l'infra + 4 KPIs verts |
| **Data Explorer** | 67 932 produits, histogrammes Plotly, exemples | Graphiques interactifs, statistiques |
| **Predictions** | Formulaire + jauge de confiance + historique | Demo live : saisir un produit → categorie predite |

### Architecture : Streamlit ne touche JAMAIS le modele

```
Utilisateur → Streamlit → HTTP POST /predict → API FastAPI → Modele
                                                    ↓
Utilisateur ← Streamlit ← JSON response    ← API FastAPI
```

Streamlit est un **client HTTP**. Il appelle l'API avec `requests.post()` et affiche le resultat. Le modele vit dans le conteneur API, pas dans Streamlit.

### Configuration

```python
API_URL = os.getenv("API_URL", "http://localhost:8000")  # URL de l'API
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN")              # Token Bearer
```

Dans Docker Compose, `API_URL=http://api:8000` (nom du service Docker).

---

## 12. DVC

### Qu'est-ce que DVC ?

**Data Version Control** — comme Git, mais pour les gros fichiers (datasets, modeles).

Git ne peut pas stocker des fichiers de 500 Mo. DVC les stocke sur un remote (S3, Google Drive) et ne met dans Git qu'un petit fichier `.dvc` qui pointe vers le fichier distant.

```
Git tracke :     models.dvc        (2 Ko — hash + metadonnees)
DVC tracke :     models/baseline_model.pkl  (50 Mo — sur Google Drive)
```

### Pourquoi DVC dans un projet MLOps ?

**Reproductibilite** : on peut retrouver exactement quel dataset a produit quel modele.

```bash
git checkout v1.0          # Revient au code de la v1.0
dvc checkout               # Telecharge les donnees de la v1.0
python train.py            # Re-entraine exactement le meme modele
```

---

## 13. Comment tout se connecte

### Le parcours complet d'une prediction

```
1. AIRFLOW (chaque lundi 2h UTC)
   │
   ├── Telecharge les donnees depuis S3
   ├── Nettoie et split le dataset
   ├── Vectorise avec TF-IDF → tfidf_vectorizer.pkl
   ├── Entraine SGDClassifier → baseline_model.pkl
   ├── Logge tout dans MLFLOW (parametres + metriques + artefacts)
   ├── Verifie que les fichiers existent
   └── Genere le rapport EVIDENTLY (drift)
        │
        ▼
2. API FASTAPI (tourne en permanence)
   │
   ├── Au demarrage : charge baseline_model.pkl + tfidf_vectorizer.pkl
   ├── Expose /predict (protege par Bearer token)
   ├── Expose /metrics (pour Prometheus)
   └── Expose /health (pour Docker healthcheck)
        │
        ▼
3. PROMETHEUS (scrape toutes les 5s)
   │
   └── Stocke les metriques en time series
        │
        ▼
4. GRAFANA (affiche en temps reel)
   │
   └── Dashboard "Rakuten API Monitoring"
       (requests/s, latence P95, status codes, active requests)
        │
        ▼
5. STREAMLIT (interface utilisateur)
   │
   ├── Appelle POST /predict via HTTP
   ├── Affiche la categorie + confiance + temps de reponse
   └── Explore les donnees (Data Explorer)
        │
        ▼
6. GITHUB ACTIONS (a chaque push)
   │
   ├── Lint (flake8)
   ├── Test (pytest, couverture >= 80%)
   └── Build + Push image Docker (ghcr.io)
```

### Pourquoi cette architecture est "model-agnostic"

Le modele est un **composant interchangeable**. Pour remplacer SGDClassifier par un DistilBERT :

| Ce qui change | Ce qui ne change PAS |
|--------------|---------------------|
| `train_model.py` (classe du modele) | API FastAPI (endpoints, auth, schemas) |
| Peut-etre `build_features.py` (tokenizer BERT au lieu de TF-IDF) | Airflow (memes 6 taches) |
| | MLflow (meme tracking) |
| | Prometheus/Grafana (memes metriques) |
| | Streamlit (meme formulaire) |
| | CI/CD (meme pipeline) |
| | Docker Compose (memes services) |

**C'est ca la valeur du MLOps : l'infrastructure survit au changement de modele.**

---

## Glossaire rapide

| Terme | Definition en 1 ligne |
|-------|----------------------|
| **API REST** | Interface HTTP pour que des programmes communiquent entre eux |
| **Bearer Token** | Mot de passe envoye dans le header HTTP pour s'authentifier |
| **Container** | Environnement isole qui contient une app + ses dependances |
| **CRON** | Syntaxe pour planifier des taches recurrentes (minute heure jour mois jour_semaine) |
| **DAG** | Graphe oriente sans cycle — modelise les dependances entre taches |
| **Data Drift** | Changement de distribution des donnees en production vs entrainement |
| **Docker Compose** | Outil pour orchestrer plusieurs conteneurs sur une seule machine |
| **Endpoint** | URL specifique d'une API (ex: /predict, /health) |
| **Feature Engineering** | Transformer les donnees brutes en features exploitables par le modele |
| **Healthcheck** | Verification automatique qu'un service est operationnel |
| **Histogram (Prometheus)** | Metrique qui distribue les valeurs en buckets pour calculer les percentiles |
| **Inference** | Utiliser un modele deja entraine pour faire une prediction |
| **KS-test** | Test statistique pour comparer deux distributions |
| **Macro F1** | Moyenne du F1-score par classe, sans ponderation |
| **MLOps** | Pratiques pour mettre un modele ML en production et le maintenir |
| **OAS 3.1** | OpenAPI Specification — standard pour documenter les APIs REST |
| **P95** | 95eme percentile — 95% des requetes sont plus rapides que cette valeur |
| **Pipeline** | Sequence d'etapes de traitement de donnees |
| **PromQL** | Langage de requete pour interroger Prometheus |
| **Provisioning** | Configuration automatique au demarrage (sans intervention manuelle) |
| **Rate** | Taux de changement d'un counter par seconde |
| **Scrape** | Action de Prometheus qui tire les metriques d'une cible |
| **Serialisation** | Sauvegarder un objet en memoire vers un fichier (pickle/joblib) |
| **Sparse matrix** | Matrice ou la plupart des valeurs sont 0 — stockee de maniere compacte |
| **Stratified split** | Split train/val qui conserve la proportion de chaque classe |
| **Time series** | Donnees indexees par le temps (ex: latence a chaque seconde) |
| **Timing attack** | Attaque qui devine un secret en mesurant le temps de reponse |
| **Volume (Docker)** | Dossier partage et persistant entre le host et les conteneurs |
