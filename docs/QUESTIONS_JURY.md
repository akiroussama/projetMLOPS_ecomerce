# Réponses préparées — Questions jury soutenance MLOps

> **Légende responsables :** chaque question indique qui doit répondre en priorité.
> - **Mika** = forgeros1993 (data pipeline, modèle, API métier, DVC, docs)
> - **Akir** = Akir Oussama (Docker, déploiement VPS, CI/CD, monitoring, Airflow, Streamlit)
> - **landroni** = landroni (pipeline ML structuré, MLflow tracking, métriques)
> - **Hery** = Hery Mickael / Hey_Ralai (tests API, qualité code, label mapping)

---
---

# PARTIE 1 — PHASES OBLIGATOIRES (1-2-3) — À MAÎTRISER ABSOLUMENT

> Ces questions portent sur le socle du projet. Le jury s'attend à ce que **chaque membre** puisse répondre sur les bases, mais le responsable doit aller en profondeur.

---

## PHASE 1 : Fondations & Conteneurisation

### Q1 : Quel est le cas d'usage métier ? Quelles sont les attentes fonctionnelles ?
**Responsable : Mika | Backup : landroni**

> "Nous classifions automatiquement les produits e-commerce du catalogue Rakuten France en 27 catégories (`prdtypecode`) à partir du texte : `designation` (titre court) et `description` (texte long). L'objectif métier est d'automatiser le catalogage qui est aujourd'hui manuel — un opérateur doit assigner chaque produit à une catégorie. Cela réduit le temps de mise en ligne et les erreurs humaines de classification.
>
> Les attentes fonctionnelles :
> - Une API qui prend en entrée un titre + description et retourne la catégorie prédite
> - Un temps de réponse < 500ms par requête
> - Un pipeline reproductible de bout en bout (données → modèle → API)"

---

### Q2 : Quelles métriques d'évaluation avez-vous choisies et pourquoi ?
**Responsable : Mika | Backup : landroni**

> "Nous loggons deux métriques principales dans MLflow :
> - **Accuracy** (val_accuracy) : proportion de prédictions correctes — facile à interpréter pour le métier
> - **Macro F1** (val_macro_f1) : moyenne non pondérée du F1 par classe — crucial car nos 27 classes sont **déséquilibrées**. L'accuracy seule masquerait un modèle qui ignore les classes rares.
>
> C'est un problème de **classification multiclasse** (27 classes). Nous avons écarté l'accuracy seule car certaines catégories (ex : `2905` Jeux de société) ont 10x moins d'échantillons que d'autres. Le macro F1 pénalise un modèle qui échoue sur les classes minoritaires."

---

### Q3 : Comment avez-vous collecté et prétraité les données ?
**Responsable : Mika | Backup : landroni**

> "Les données proviennent du challenge Rakuten France (Data Science Competition). Nous les récupérons depuis S3 via le script `import_raw_data.py` :
> - `X_train_update.csv` — features (designation, description, productid, imageid)
> - `Y_train_CVw08PX.csv` — labels (prdtypecode)
> - `X_test_update.csv` — données test
>
> **Prétraitement** (`make_dataset.py`) :
> 1. Strip whitespace sur les colonnes texte
> 2. Suppression des doublons (par designation + description + productid + imageid)
> 3. Split stratifié 80/20 (random_state=42) — stratifié pour conserver la distribution des classes
>
> **Feature engineering** (`build_features.py`) :
> - Normalisation du texte : décodage HTML, suppression des balises, translittération Unicode → ASCII, lowercase, suppression des caractères non-alphanumériques
> - Vectorisation TF-IDF : 5000 features max, unigrammes + bigrammes (ngram_range=(1,2))
> - Output : matrices sparse `.npz` + `tfidf_vectorizer.pkl`"

---

### Q4 : Pourquoi TF-IDF + SGDClassifier et pas du deep learning ?
**Responsable : Mika | Backup : Akir**

**Réponse courte :** Contrainte de ressources computationnelles — CPU only en local et sur VPS (4 vCPU, 8 GB RAM). Entraîner un CamemBERT prend plusieurs heures sur GPU et 24h+ sur CPU.

> "Nous avons fait un choix délibéré de prioriser l'infrastructure MLOps sur la performance brute du modèle. Un pipeline MLflow + Airflow + CI/CD + monitoring autour d'un modèle simple démontre mieux les compétences MLOps qu'un notebook BERT sans pipeline.
>
> **SGDClassifier** avec `loss='log_loss'` est en fait une régression logistique entraînée par descente de gradient stochastique. Il est rapide (quelques secondes d'entraînement), supporte les matrices sparse, et on utilise `class_weight='balanced'` pour gérer le déséquilibre des classes.
>
> Notre architecture est **model-agnostic** : dans `train_model.py`, remplacer `SGDClassifier` par un DistilBERT ne nécessite que de changer la classe. Le pipeline MLflow, Airflow, Prometheus et FastAPI restent identiques."

**Scores de référence (Rakuten challenge) :**
- TF-IDF + SGD (nous) : ~76% accuracy
- TF-IDF + SVM : ~78%
- CamemBERT (texte only) : ~85%
- Fusion texte + image (SOTA) : ~91%

---

### Q5 : Pourquoi vous n'utilisez pas les images ?
**Responsable : Mika | Backup : Akir**

> "Le dataset Rakuten contient des images produits. Une architecture multimodale (BERT + ResNet) atteint ~91% en compétition. Nous avons écarté cette approche :
> 1. Les images ne sont pas dans la version S3 que nous utilisons — seuls les identifiants sont présents
> 2. L'inférence multimodale nécessite un GPU pour rester sous 200ms, incompatible avec notre VPS CPU
>
> Notre API est prête pour l'extension : `PredictRequest` inclut déjà `imageid` et `productid` comme champs."

---

### Q6 : Quels sont les hyperparamètres du modèle et comment les avez-vous choisis ?
**Responsable : Mika | Backup : landroni**

> "Notre `SGDClassifier` utilise :
> - `loss='log_loss'` — régression logistique (probabilités calibrées via `predict_proba`)
> - `alpha=1e-5` — régularisation L2 faible pour ne pas trop contraindre avec 5000 features
> - `max_iter=1000`, `tol=1e-3` — convergence
> - `class_weight='balanced'` — pondère inversement les classes par leur fréquence, essentiel car la catégorie `2583` (Mobilier) a ~10x plus d'exemples que `2905` (Jeux de société)
> - `random_state=42` — reproductibilité
>
> Les hyperparamètres ont été choisis par itération rapide loggée dans MLflow — on peut comparer les runs dans l'UI."

---

### Q7 : Comment fonctionne votre API FastAPI ? Décrivez les endpoints.
**Responsable : Hery | Backup : Mika**

> "Notre API expose 4 endpoints :
>
> | Endpoint | Méthode | Auth | Rôle |
> |----------|---------|------|------|
> | `/health` | GET | Non | Healthcheck — retourne le statut, le nom du modèle chargé, le path |
> | `/predict` | POST | Oui (Bearer) | Prédiction — prend `designation` + `description`, retourne `predicted_label`, `predicted_code`, `confidence`, `model_name` |
> | `/stats` | GET | Non | Statistiques en mémoire — total prédictions, par catégorie, latence avg/min/max |
> | `/metrics` | GET | Non | Métriques Prometheus (auto-instrumenté via `prometheus-fastapi-instrumentator`) |
>
> Le modèle est chargé au **startup** via `PredictionService` (dataclass). Il cherche dans l'ordre : variable d'env `MODEL_FILE`, puis les candidats `baseline_model.pkl`, `classifier.pkl`, `model.pkl` dans le répertoire `/app/models`."

---

### Q8 : Comment le modèle est-il chargé dans l'API ? (sans ré-entraînement)
**Responsable : Hery | Backup : Mika**

> "Au démarrage de l'API (lifespan event FastAPI), `PredictionService` charge 3 artefacts sérialisés avec joblib/pickle :
> 1. **Le modèle** (`baseline_model.pkl`) — SGDClassifier entraîné
> 2. **Le vectorizer** (`tfidf_vectorizer.pkl`) — TfidfVectorizer fitté
> 3. **Le label mapping** (`label_mapping.json`) — correspondance code → nom catégorie
>
> À chaque requête `/predict`, le texte est vectorisé par le TF-IDF chargé, passé au modèle, et le code prédit est mappé au nom lisible. Pas de ré-entraînement à chaque requête. Le modèle est un objet en mémoire."

---

### Q9 : Comment avez-vous testé l'API ?
**Responsable : Hery | Backup : Mika**

> "Nous avons deux niveaux de tests :
> 1. **Tests unitaires** (`test_service.py`) — testent `PredictionService` isolément avec des mocks (mock model, mock vectorizer). Vérifient les cas d'erreur : modèle cassé → `PredictionExecutionError`, token invalide → 403, mauvais schéma auth → 401
> 2. **Tests d'intégration** (`test_app.py`) — utilisent le `TestClient` httpx de FastAPI pour tester les endpoints réels : healthcheck, predict avec auth, erreurs 422/401/500/503
>
> Couverture CI : **≥80%** sur `src/api/` (mesuré par pytest-cov). Le CI échoue si on passe en dessous."

---

### Q10 : Comment l'API gère-t-elle les erreurs ?
**Responsable : Hery | Backup : Mika**

> "Nous avons 3 exception handlers globaux dans FastAPI :
> - `RequestValidationError` → **422** avec le détail des champs invalides (ex : designation trop longue, type incorrect)
> - `HTTPException` → mappé vers `ErrorResponse` avec `error_code` (ex : `AUTHENTICATION_REQUIRED`, `INVALID_TOKEN`, `AUTH_NOT_CONFIGURED`)
> - `Exception` générique → **500 INTERNAL_SERVER_ERROR** (catch-all pour les erreurs inattendues)
>
> La validation Pydantic utilise `ConfigDict(extra='forbid')` — les champs inconnus sont rejetés. Les types sont stricts (`StrictInt`, `StringConstraints` avec min/max length)."

---

## PHASE 2 : Suivi des expériences & Sécurisation

### Q11 : Comment MLflow est-il intégré ? Que loggez-vous ?
**Responsable : landroni | Backup : Akir**

> "MLflow est configuré comme un serveur centralisé (`http://mlflow:5000`) avec backend SQLite et stockage d'artefacts sur volume Docker.
>
> À chaque run d'entraînement, `train_model.py` logge dans l'expérience `rakuten-text-baseline` :
>
> **Paramètres :** model_class, loss, alpha, max_iter, class_weight, max_features
> **Métriques :** val_accuracy, val_macro_f1, train_samples, validation_samples, feature_count
> **Artefacts :** baseline_model.pkl, tfidf_vectorizer.pkl, training_metrics.json, classification_report.json
>
> L'intérêt est de pouvoir **comparer les runs** dans l'UI MLflow : si on change alpha ou max_features, on voit immédiatement l'impact sur le macro F1."

---

### Q12 : Comment comparez-vous les performances entre différents runs MLflow ?
**Responsable : landroni | Backup : Akir**

> "Dans l'UI MLflow (`http://rakuten-mlops.duckdns.org:5000`), on sélectionne plusieurs runs de l'expérience `rakuten-text-baseline` et on les compare :
> - Vue tabulaire : paramètres et métriques côte à côte
> - Vue graphique : scatter plots des métriques vs paramètres
> - Artefacts : on peut télécharger le modèle de n'importe quel run
>
> Par exemple, en changeant `alpha` de 1e-4 à 1e-5, on peut observer l'impact sur val_macro_f1. C'est le principal avantage sur un simple `print()` en notebook."

---

### Q13 : Votre API est-elle sécurisée ? Comment ?
**Responsable : Hery | Backup : Akir**

> "L'endpoint `/predict` est protégé par **Bearer token** :
> - Le token est stocké en variable d'environnement `API_AUTH_TOKEN` (jamais dans le code)
> - La validation utilise `hmac.compare_digest()` — comparaison **timing-safe** qui empêche les timing attacks
> - Le schéma d'auth est vérifié : si le header n'est pas `Bearer <token>` → 401 `INVALID_AUTH_SCHEME`
> - Si le token ne correspond pas → 403 `INVALID_TOKEN`
> - Si `API_AUTH_TOKEN` n'est pas configuré côté serveur → 503 `AUTH_NOT_CONFIGURED`
>
> Les endpoints `/health` et `/metrics` sont publics car ils ne retournent pas de données sensibles — c'est voulu pour que Prometheus et les load balancers puissent y accéder sans token."

---

### Q14 : Qu'est-ce qu'une timing attack et pourquoi `hmac.compare_digest` ?
**Responsable : Hery | Backup : Akir**

> "Une comparaison classique (`==`) s'arrête au premier caractère différent. Un attaquant peut mesurer le temps de réponse et deviner le token caractère par caractère. `hmac.compare_digest()` compare **toujours** tous les caractères — le temps est constant quel que soit le nombre de caractères corrects. C'est une bonne pratique de sécurité standard pour la validation de tokens."

---

### Q15 : Avez-vous des tests unitaires ? Quelle couverture ?
**Responsable : Hery | Backup : Mika**

> "Oui, dans `tests/api/` :
> - `test_service.py` : tests unitaires de la logique de prédiction (mock models, erreurs)
> - `test_app.py` : tests d'intégration des endpoints (httpx TestClient)
>
> Plus dans `tests/pipeline/` :
> - `test_pipeline.py` : tests end-to-end du pipeline (data prep → training → artefacts compatibles API)
>
> Le CI GitHub Actions exige **≥80% de couverture** sur `src/api/`. Si un PR fait baisser la couverture, le CI échoue et bloque le merge."

---

## PHASE 3 : Orchestration & Déploiement

### Q16 : Comment Airflow orchestre-t-il le pipeline ?
**Responsable : Akir | Backup : Mika**

> "Le DAG `rakuten_weekly_retraining` s'exécute **chaque lundi à 2h UTC** (`0 2 * * 1`). Il contient 6 tâches séquentielles :
>
> 1. `download_raw_data` → télécharge les CSV depuis S3
> 2. `prepare_dataset` → nettoyage, déduplication, split stratifié 80/20
> 3. `build_features` → vectorisation TF-IDF (5000 features)
> 4. `train_model` → entraîne SGDClassifier, logge dans MLflow
> 5. `verify_artifacts` → vérifie que `baseline_model.pkl` + `tfidf_vectorizer.pkl` + `training_metrics.json` existent
> 6. `generate_drift_report` → rapport Evidently (data drift + data quality)
>
> Les dépendances sont linéaires : chaque tâche attend la réussite de la précédente. Si `build_features` échoue, `train_model` ne se lance pas."

---

### Q17 : Pourquoi Airflow et pas une simple crontab ?
**Responsable : Akir | Backup : landroni**

> "Une crontab ne donne ni visibilité, ni retry, ni alerting. Airflow apporte :
> - **Historique** : chaque exécution est loggée, on voit quelle tâche a échoué et pourquoi
> - **Retries automatiques** avec backoff exponentiel
> - **Alerting** en cas d'échec
> - **Trigger manuel** depuis l'UI (utile pour forcer un re-entraînement sans attendre lundi)
> - **Dépendances** entre tâches (DAG = Directed Acyclic Graph)
> - **UI Web** pour monitorer en temps réel (http://rakuten-mlops.duckdns.org:8280)"

---

### Q18 : Décrivez votre architecture Docker Compose. Quels services, comment communiquent-ils ?
**Responsable : Akir | Backup : Mika**

> "Nous avons 11 services dans un seul `docker-compose.yml` :
>
> | Service | Port externe | Rôle |
> |---------|-------------|------|
> | `api` | 8200 | FastAPI — inférence |
> | `mlflow` | 5000 | Tracking serveur |
> | `postgres` | — | BDD Airflow (PostgreSQL 16) |
> | `airflow-webserver` | 8280 | UI Airflow |
> | `airflow-scheduler` | — | Exécute les DAGs |
> | `airflow-init` | — | Migration BDD + création user admin |
> | `streamlit` | 8501 | Interface utilisateur |
> | `prometheus` | 9090 | Collecte métriques |
> | `grafana` | 3000 | Dashboards |
> | `trainer` | — | Entraînement standalone (profil Docker) |
> | `bootstrap` | — | Init données (profil Docker) |
>
> **Communication inter-services** : réseau Docker interne. Streamlit appelle `http://api:8000/predict`, Prometheus scrape `http://api:8000/metrics`, Airflow écrit dans MLflow via `http://mlflow:5000`, Grafana lit Prometheus via `http://prometheus:9090`.
>
> **Volumes persistants** : `postgres-db-volume`, `mlflow-data`, `mlflow-artifacts` + bind mounts pour `./models`, `./data`, `./reports`."

---

### Q19 : Pourquoi avoir séparé en microservices plutôt qu'un monolithe ?
**Responsable : Akir | Backup : Mika**

> "Un monolithe mélangerait API, entraînement et monitoring dans un seul process. Les problèmes :
> - Un crash d'Airflow ferait tomber l'API de prédiction
> - Impossible de scaler l'API indépendamment (ex: 3 replicas API, 1 seul scheduler Airflow)
> - Les dépendances Python conflicteraient (Airflow a besoin de centaines de packages, l'API doit rester légère)
>
> Avec Docker Compose, chaque service a son propre Dockerfile, ses propres dépendances, et peut être redémarré ou mis à jour indépendamment. C'est le principe du **découplage des responsabilités**."

---

### Q20 : Comment les services partagent-ils les modèles entraînés ?
**Responsable : Akir | Backup : Mika**

> "Via un **volume partagé** Docker : `./models` est monté dans les containers `api`, `trainer`, `airflow-scheduler`. Quand Airflow entraîne un nouveau modèle et écrit `baseline_model.pkl` dans `/app/models`, l'API peut le charger au prochain restart.
>
> Les artefacts sont aussi sauvegardés dans **MLflow** (`/mlflow/artifacts`) comme backup versionné. Cela permet de revenir à un modèle précédent si le nouveau est moins performant."

---

### Q21 : Quels Dockerfiles avez-vous écrits ? Quels choix techniques ?
**Responsable : Akir | Backup : Mika**

> "4 Dockerfiles custom :
> - **API** (`Dockerfile`) : `python:3.10-slim`, install requirements, copie `src/`, expose 8000, `uvicorn` CMD. On utilise `mkdir /app/models` au lieu de `COPY models/` pour que le volume Docker monte par dessus
> - **MLflow** (`docker/mlflow/Dockerfile`) : `python:3.10-slim`, SQLite backend + artifacts volume
> - **Airflow** (`docker/airflow/Dockerfile`) : basé sur l'image officielle `apache/airflow:2.10.5-python3.10`, ajoute nos requirements pipeline + monitoring
> - **Streamlit** (`docker/streamlit/Dockerfile`) : `python:3.10-slim`, `requirements-ui.txt` séparé pour garder l'image légère
>
> On a séparé les `requirements-*.txt` par service pour minimiser la taille des images et éviter les conflits de dépendances."

---

---

# PARTIE 2 — PHASES OPTIONNELLES (4-5) — Bonus valorisé

---

## PHASE 4 : CI/CD & Monitoring

### Q22 : Décrivez votre pipeline CI/CD GitHub Actions.
**Responsable : Akir | Backup : Mika**

> "Notre workflow `.github/workflows/python-app.yml` se déclenche sur chaque push et PR vers `main` ou `master`. 3 jobs :
>
> 1. **Lint** (flake8) : vérifie le style Python (max line 127, exclusion de fichiers legacy)
> 2. **Test** (pytest + coverage) : installe les dépendances, lance `pytest --cov=src/api --cov-fail-under=80`. Si la couverture tombe sous 80%, le CI est rouge
> 3. **Docker Build & Push** : build l'image API et la push vers `ghcr.io` (GitHub Container Registry). Le push ne se fait que sur merge vers main, pas sur les PRs
>
> L'image est taggée `latest` et publiée sous `ghcr.io/{owner}/projetmlops-ecomerce-api:latest`."

---

### Q23 : Pourquoi 80% de couverture et pas 100% ?
**Responsable : Hery | Backup : Akir**

> "100% donne un faux sentiment de sécurité — on peut avoir 100% de couverture avec des tests qui n'assertent rien. 80% est un seuil pragmatique qui couvre les chemins critiques (prédiction, auth, erreurs) sans forcer des tests triviaux sur du boilerplate. Les 20% restants sont des branches d'erreur rares ou du code de configuration."

---

### Q24 : Comment fonctionne le monitoring Prometheus + Grafana ?
**Responsable : Akir | Backup : landroni**

> "L'API FastAPI est instrumentée avec `prometheus-fastapi-instrumentator` qui expose automatiquement sur `/metrics` :
> - `http_requests_total` — compteur par endpoint et status code
> - `http_request_duration_seconds` — histogramme de latence
> - `http_request_size_bytes` / `http_response_size_bytes`
> - `http_requests_in_progress` — requêtes actives
>
> **Prometheus** scrape `http://api:8000/metrics` toutes les **5 secondes** (config `prometheus.yml`).
> **Grafana** lit Prometheus comme datasource et affiche un dashboard `rakuten-api-monitoring` avec :
> - Requests per second (timeseries)
> - Latence p50/p95/p99
> - Error rate
> - Refresh automatique toutes les 5 secondes, fenêtre de 15 minutes."

---

### Q25 : Comment gérez-vous le drift ?
**Responsable : Akir | Backup : landroni**

> "La dernière tâche du DAG Airflow (`generate_drift_report`) utilise **Evidently** pour comparer les données de référence (train) vs courantes (validation). Nous mesurons 4 features dérivées du texte :
> - Longueur de la designation
> - Longueur de la description
> - Présence d'une description (booléen)
> - Nombre de mots
>
> Le test statistique utilisé est le **KS-test** (Kolmogorov-Smirnov) avec un seuil de p-value < 0.05. Le rapport HTML est loggé dans MLflow comme artefact. En production, si un drift significatif est détecté, on pourrait déclencher un re-entraînement automatique."

---

### Q26 : Qu'est-ce que le data drift concrètement ?
**Responsable : Akir | Backup : landroni**

> "Le data drift, c'est quand la distribution des données en production diverge de celle d'entraînement. Exemple concret : notre modèle est entraîné sur des descriptions en français. Si Rakuten commence à ajouter des produits en anglais, la distribution des features texte change — les bigrammes TF-IDF ne correspondent plus. Le modèle va mal classifier ces produits sans que l'accuracy sur le dataset de test ne bouge. Le monitoring Evidently détecte ce décalage **avant** que les utilisateurs se plaignent."

---

## PHASE 5 : Streamlit

### Q27 : Comment Streamlit interagit-il avec l'API ?
**Responsable : Akir | Backup : Mika**

> "Streamlit n'accède **jamais** directement au modèle. Il communique exclusivement via l'API REST :
> - `utils.py` contient les fonctions `check_api_health()` et `predict(designation, description)` qui appellent `http://api:8000` avec le Bearer token
> - Le token est passé via variable d'environnement `API_AUTH_TOKEN`
>
> 3 pages (onglets) :
> 1. **Contexte** : description du problème, architecture, schéma I/O
> 2. **Data Explorer** : statistiques, distribution des classes, analyse des longueurs de texte
> 3. **Predictions** : formulaire live, 6 exemples pré-remplis (Livre, Jeu PS5, Figurine, Canapé, Perceuse, Carte Pokémon), jauge de confiance Plotly, historique de session"

---

### Q28 : Pourquoi Streamlit et pas React ou Flask ?
**Responsable : Akir | Backup : Mika**

> "Streamlit est conçu pour le prototypage data/ML : pas de HTML/CSS/JS à écrire, composants natifs pour les graphes, tableaux et formulaires. En 200 lignes de Python, on a une interface complète avec jauge de confiance et exploration de données. React aurait pris 10x plus de temps pour un résultat similaire dans le contexte d'une démo MLOps."

---

---

# PARTIE 3 — QUESTIONS TRANSVERSALES & PIÈGES

---

### Q29 : Comment fonctionne le pipeline MLOps de bout en bout ?
**Responsable : Akir | Backup : Tous**

```
[Données S3] → Airflow DAG (hebdo, lundi 2h UTC)
    ├── download_raw_data    : télécharge X_train, X_test, Y_train depuis S3
    ├── prepare_dataset      : nettoyage, déduplication, split 80/20 stratifié
    ├── build_features       : TF-IDF 5000 features, unigrammes + bigrammes
    ├── train_model          : SGDClassifier, log MLflow (params + métriques + artifacts)
    ├── verify_artifacts     : vérifie que baseline_model.pkl + vectorizer sont présents
    └── generate_drift_report: Evidently (DataDrift + DataQuality)

[MLflow]      → tracking expériences, comparaison runs
[FastAPI]     → inférence, auth Bearer, /health, /metrics, /stats
[Prometheus]  → scrape /metrics toutes les 5s
[Grafana]     → dashboards temps réel (latence, throughput, error rate)
[Streamlit]   → interface utilisateur (appelle l'API exclusivement)
[GitHub Actions] → CI : lint + test + docker build/push à chaque push/PR
```

---

### Q30 : Qui a fait quoi dans l'équipe ?
**Responsable : Tous**

> - **Mika** : data pipeline complet (ingestion DVC → nettoyage → feature engineering TF-IDF → modèle SVC/SGD), API FastAPI (logique métier, endpoints, label mapping), documentation (README, data contract)
> - **Akir** : infrastructure Docker Compose (11 services), déploiement VPS, CI/CD GitHub Actions, Airflow DAG, Prometheus/Grafana, Evidently, Streamlit, présentation
> - **landroni** : pipeline ML structuré (make_dataset, build_features, train_model, predict), MLflow tracking, métriques business et performance
> - **Hery** : API FastAPI (fondation + tests), couverture 90%, sécurité Bearer token, label mapping JSON, tests unitaires et d'intégration

---

### Q31 : Qu'est-ce que vous amélioreriez si vous aviez plus de temps ?
**Responsable : Tous (chacun parle de son domaine)**

1. **Modèle** (Mika/landroni) : Fine-tuning CamemBERT/DistilBERT — bloqué par compute CPU
2. **CD pipeline** (Akir) : Déploiement automatique quand un nouveau modèle dépasse le score courant dans MLflow (Model Registry + webhook → restart API)
3. **Drift production** (Akir) : Logger les inputs réels de l'API pour détecter le drift sur des données de production, pas juste train vs val
4. **Tests de charge** (Hery) : Locust pour valider les SLAs sous load
5. **Multimodal** (Mika) : Fusion image (ResNet50) + texte (TF-IDF) en deux étapes d'inférence

---

### Q32 : Pourquoi DVC et pas juste des fichiers sur S3 ?
**Responsable : Mika**

> "DVC (Data Version Control) ajoute le **versioning des données** : chaque version du dataset est trackée comme un commit git (mais pour les gros fichiers). Si le dataset change, on peut revenir à une version précédente et reproduire un entraînement identique. Sans DVC, on n'a aucune traçabilité sur quel dataset a produit quel modèle."

---

### Q33 : Pourquoi PostgreSQL pour Airflow et SQLite pour MLflow ?
**Responsable : Akir | Backup : landroni**

> "Airflow a besoin d'un accès concurrent (webserver + scheduler écrivent en parallèle dans la BDD) — SQLite ne supporte pas bien les écritures concurrentes, donc on utilise PostgreSQL 16.
>
> Pour MLflow, le tracking server est le seul writer (pas de concurrence). SQLite suffit et simplifie l'infrastructure — pas besoin d'un 2ème PostgreSQL. C'est un compromis pragmatique."

---

### Q34 : Votre API peut-elle scaler ? Comment ?
**Responsable : Akir | Backup : Hery**

> "Oui. Avec Docker Compose, on peut ajouter `deploy: replicas: 3` sur le service `api` et mettre un load balancer (Nginx/Traefik) devant. Le modèle est chargé en mémoire à chaque instance — c'est stateless. Prometheus collecterait les métriques de toutes les instances.
>
> Pour aller plus loin : Kubernetes avec un HPA (Horizontal Pod Autoscaler) qui scale selon la latence ou le CPU."

---

### Q35 : Que se passe-t-il si l'API crash en production ?
**Responsable : Akir | Backup : Hery**

> "Docker Compose a `restart: unless-stopped` sur nos services. Si l'API crash, Docker la redémarre automatiquement. Le healthcheck (`/health`) permet à un orchestrateur de détecter un service dégradé. Grafana affiche le error rate en temps réel — on verrait immédiatement un pic d'erreurs 500."

---

### Q36 : Comment avez-vous géré le travail en équipe ?
**Responsable : Tous**

> "Communication via Slack, points de suivi réguliers. Chaque membre a un domaine de responsabilité clair (data/modèle, infra, tests, pipeline ML). On a utilisé GitHub avec des branches et des PRs. Les deadlines du projet (27/02, 06/03, 13/03, 20/03) ont structuré notre avancement par phase."

---

### Q37 : Quelle est la différence entre MLflow Tracking et MLflow Model Registry ?
**Responsable : landroni | Backup : Akir**

> "**Tracking** : logge chaque run (paramètres, métriques, artefacts). C'est ce qu'on utilise — chaque entraînement crée un run dans l'expérience `rakuten-text-baseline`.
>
> **Model Registry** : une couche au dessus pour gérer le cycle de vie des modèles (Staging → Production → Archived). On ne l'utilise pas encore — c'est une amélioration prévue. Ça permettrait de promouvoir un modèle en 'Production' et de faire pointer l'API automatiquement vers le dernier modèle promu."

---

### Q38 : Pourquoi `class_weight='balanced'` dans le modèle ?
**Responsable : Mika | Backup : landroni**

> "Nos 27 catégories sont déséquilibrées — certaines ont 5000 exemples, d'autres 500. Sans `balanced`, le modèle optimise l'accuracy globale et ignore les classes rares (il suffit de prédire la classe majoritaire pour avoir un bon score). Avec `balanced`, scikit-learn pondère chaque classe inversement à sa fréquence : `n_samples / (n_classes * n_samples_per_class)`. Résultat : le modèle est pénalisé s'il ignore les classes rares."

---

### Q39 : Que veut dire `model-agnostic` dans votre architecture ?
**Responsable : Akir | Backup : Mika**

> "Ça signifie que l'infrastructure MLOps ne dépend pas du modèle spécifique. `PredictionService` charge n'importe quel objet qui a une méthode `predict()` (et optionnellement `predict_proba()`). On peut remplacer SGDClassifier par un RandomForest, un XGBoost, ou même un modèle PyTorch wrappé dans une classe scikit-learn compatible — sans toucher à l'API, Airflow, Prometheus ou Streamlit."

---

### Q40 : Pourquoi avez-vous déployé sur un VPS et pas sur le cloud (AWS/GCP) ?
**Responsable : Akir**

> "Un VPS à ~10€/mois nous donne le contrôle total sur l'infrastructure : Docker Compose, volumes, ports, DNS. Sur AWS, l'équivalent (ECS/EKS + RDS + S3 + Load Balancer) coûterait 50-100€/mois et ajouterait une complexité de configuration IAM/VPC disproportionnée pour un projet pédagogique. Le VPS est un compromis pragmatique : on démontre les mêmes compétences MLOps à moindre coût."

---

### Q41 : Qu'est-ce que `normalize_text()` fait exactement et pourquoi ?
**Responsable : Mika | Backup : landroni**

> "C'est notre pipeline de nettoyage texte, 6 étapes :
> 1. `html.unescape()` — décode les entités HTML (`&amp;` → `&`)
> 2. Suppression des balises HTML — les descriptions Rakuten contiennent parfois du HTML brut
> 3. Normalisation Unicode NFKD — décompose les caractères accentués
> 4. Translittération ASCII — `café` → `cafe`, pour uniformiser
> 5. Lowercase + suppression non-alphanumérique — ne garde que lettres et espaces
> 6. Collapse whitespace — `'mot   mot'` → `'mot mot'`
>
> Sans ça, 'Café', 'café', 'CAFÉ' et 'caf&eacute;' seraient 4 tokens différents dans TF-IDF."

---

### Q42 : Comment relancez-vous un entraînement manuellement ?
**Responsable : Akir | Backup : landroni**

> "Deux options :
> 1. **Via Airflow UI** : on va sur le DAG `rakuten_weekly_retraining`, on clique 'Trigger DAG'. Les 6 tâches se lancent séquentiellement
> 2. **Via Docker** : `docker compose run trainer` lance le container d'entraînement standalone
>
> Dans les deux cas, le nouveau modèle est écrit dans le volume `./models` et loggé dans MLflow."

---

### Q43 : Comment votre architecture gère-t-elle la reproductibilité ?
**Responsable : landroni | Backup : Mika**

> "Trois niveaux de reproductibilité :
> 1. **Données** : DVC versionne les datasets — on peut retrouver exactement quelles données ont produit quel modèle
> 2. **Code** : Git — chaque commit est taggable, et le DAG Airflow est lui-même versionné
> 3. **Expériences** : MLflow logge les paramètres et métriques de chaque run — on peut recréer un entraînement identique avec les mêmes hyperparamètres
>
> Plus `random_state=42` partout (split, modèle) pour la reproductibilité numérique."

---

### Q44 : Quelle est la latence de votre API en production ?
**Responsable : Hery | Backup : Akir**

> "Sur notre VPS (4 vCPU, 8 GB RAM), une requête `/predict` prend en moyenne **50-150ms** (TF-IDF vectorization + SGD prediction). L'endpoint `/stats` affiche les latences avg/min/max en temps réel. Grafana montre l'histogramme complet. Le TF-IDF sur 5000 features + SGD est très rapide comparé à un modèle deep learning qui prendrait 500ms-2s sur CPU."

---

### Q45 : Si le jury teste l'API live, que se passe-t-il ?
**Responsable : Hery | Backup : Akir**

> "On peut le montrer en direct :
> 1. **Swagger UI** (`http://rakuten-mlops.duckdns.org:8200/docs`) : interface interactive, on remplit designation + description, on ajoute le Bearer token, on envoie
> 2. **Streamlit** (`http://rakuten-mlops.duckdns.org:8501`) : interface grand public, on tape un titre de produit, on obtient la catégorie + confiance avec une jauge visuelle
> 3. **curl** : `curl -X POST .../predict -H 'Authorization: Bearer ...' -d '{\"designation\": \"Harry Potter tome 1\"}'` → retourne `Livres / Magazines`"
