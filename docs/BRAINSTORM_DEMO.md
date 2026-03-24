# Brainstorm Demo Soutenance — Angle d'expert pour chaque outil

> **Objectif** : Pour chaque outil, expliquer POURQUOI on l'a choisi, quel probleme il resout,
> quelles alternatives existent, et montrer des PREUVES CONCRETES depuis nos serveurs.
> Le jury doit sentir qu'on maitrise, pas qu'on a suivi un tuto.

---

## 1. FastAPI — L'API de prediction

### Probleme resolu
Sans API, le modele est prisonnier d'un notebook. Un data scientist peut l'executer, mais pas un systeme, pas une app mobile, pas un autre service. L'API transforme le modele en **service reutilisable** avec un contrat clair (input/output).

### Pourquoi FastAPI et pas les autres ?

| Alternative | Ce qu'elle fait mieux | Pourquoi FastAPI gagne ici |
|---|---|---|
| **Flask** | Plus mature, plus de tutos | Pas de validation auto (Pydantic), pas de doc Swagger auto, pas d'async natif |
| **Django REST** | Ecosysteme complet (admin, ORM, auth) | Trop lourd pour une API ML — on n'a pas besoin d'ORM |
| **BentoML** | Specialise ML (batching, model packaging) | Lock-in framework, moins de controle sur les endpoints customs (/stats, /health) |
| **TF Serving / TorchServe** | Optimise pour les modeles deep learning | Ne supporte pas scikit-learn nativement, on est sur CPU |
| **Seldon Core** | Inference Kubernetes-native, A/B testing | Necessite Kubernetes, overkill pour notre scope |

### Ce qu'on montre au jury (preuves serveur)

**1. Latence de prediction : 5.49ms en moyenne**
```
Prometheus: http_request_duration_seconds_sum{handler="/predict"} = 7.66s
Prometheus: http_request_duration_seconds_count{handler="/predict"} = 1396
→ 7.66 / 1396 = 5.49ms par prediction
→ 100% des predictions sous 100ms (histogram bucket le=0.1 → count=1395/1396)
```
> "Notre SGDClassifier repond en 5ms. Un BERT prendrait 200ms+ sur CPU. C'est un choix delibere."

**2. Validation Pydantic stricte — on bloque le garbage avant le modele**
```python
# schemas.py — StrictShortText max 512 chars, StrictInt pour productid
# ConfigDict(extra="forbid") → tout champ inconnu est rejete avec 422
```
Montrer dans Swagger : envoyer `{"designation": "test", "unknown_field": "x"}` → 422 VALIDATION_ERROR.
> "Pydantic agit comme un garde-fou. En production, un payload malformed ne touche jamais le modele."

**3. 4 endpoints, chacun avec un role precis**
| Endpoint | Role MLOps |
|---|---|
| `/health` | Probes Docker + monitoring uptime (public) |
| `/predict` | Inference protegee par Bearer token |
| `/metrics` | Exposition Prometheus (auto via instrumentator) |
| `/stats` | Metriques metier temps reel (predictions par categorie, latence) |

**4. Architecture modulaire**
```
app.py    → routes + exception handlers
service.py → logique metier (load model, predict, map labels)
schemas.py → contrats Pydantic (input/output/error)
security.py → auth Bearer avec hmac.compare_digest (timing-safe)
```
> "On peut changer le modele sans toucher aux routes. On peut ajouter un endpoint sans toucher au service."

**5. Securite : hmac.compare_digest**
```python
# security.py ligne 61 — comparaison timing-safe
hmac.compare_digest(credentials.credentials, expected_token)
```
> "On utilise hmac.compare_digest au lieu de `==` pour eviter les timing attacks. C'est un detail mais ca montre qu'on a reflechi a la securite."

---

## 2. Streamlit — L'interface utilisateur

### Probleme resolu
Un modele ML sans interface est invisible pour le metier. Le data scientist voit les metriques, mais le product manager veut taper un titre de produit et voir la categorie. Streamlit est le **"last mile"** du MLOps.

### Pourquoi Streamlit et pas les autres ?

| Alternative | Ce qu'elle fait mieux | Pourquoi Streamlit gagne ici |
|---|---|---|
| **Gradio** | Interface ML zero-config (input → model → output) | Moins flexible pour les pages multiples, pas de custom CSS, oriente demo pas app |
| **Dash (Plotly)** | Dashboards interactifs complexes, callbacks | Plus verbeux, courbe d'apprentissage plus haute, callback hell |
| **Panel** | Integration native avec l'ecosysteme HoloViz | Communaute plus petite, moins de tutos |
| **React / Vue** | Controle total du frontend | Necessite JS/HTML/CSS, hors scope data scientist |

### Ce qu'on montre au jury (preuves serveur)

**1. Streamlit ne charge PAS le modele — tout passe par l'API**
```python
# utils.py — Streamlit appelle uniquement l'API via HTTP
def predict(designation, description):
    return call_api("POST", "/predict", json_body=payload)
```
> "Streamlit est un client HTTP pur. Aucun import sklearn, aucun pickle.load. Si le modele change dans l'API, Streamlit n'a rien a modifier."

**2. 6 exemples pre-remplis — pensee UX**
```python
# 3_Predictions.py — QUICK_EXAMPLES
# Livre, Jeu PS5, Figurine, Canape, Perceuse, Carte Pokemon
```
> "On a choisi des categories tres differentes pour que la demo montre la polyvalence du modele."

**3. Jauge de confiance Plotly — interpretation metier**
```python
# Seuils: < 40% rouge, 40-70% orange, > 70% vert
# Avec 27 classes, un SGD a rarement > 50% de confiance — c'est normal et on l'explique
```
> "Un modele a 27 classes ne peut pas avoir 95% de confiance comme un modele binaire. Le fait qu'il donne la bonne categorie a 36% de confiance est en realite excellent — la baseline random serait 3.7%."

**4. Session history — tracabilite metier**
Chaque prediction est stockee dans `st.session_state` → tableau recapitulatif en bas.
> "En production, cet historique serait persiste en base. Ici c'est un prototype qui montre le concept."

**5. Health check automatique sur chaque page**
```python
# app.py + 3_Predictions.py — check_api_health() au chargement
# Bandeau vert : "API connectee — Modele SGDClassifier charge avec succes"
```
> "L'utilisateur sait immediatement si le systeme est operationnel. Pas de surprise au moment de predire."

---

## 3. MLflow — Suivi des experiences

### Probleme resolu
Sans tracking, le ML est un cauchemar de reproductibilite. "Quel alpha avait donne 76% ?" "C'etait avec quel preprocessing ?" MLflow repond a la question : **quel code + quels parametres + quelles donnees = quel resultat ?**

### Pourquoi MLflow et pas les autres ?

| Alternative | Ce qu'elle fait mieux | Pourquoi MLflow gagne ici |
|---|---|---|
| **Weights & Biases** | UX superieure, collaboration temps reel, sweeps auto | Cloud-only (plan gratuit limité), pas self-hosted facilement |
| **Neptune.ai** | Monitoring en prod, comparaison de runs avancee | SaaS payant, lock-in cloud |
| **ClearML** | Suite complete (tracking + orchestration + serving) | Plus complexe a deployer, communaute plus petite |
| **DVC** | Versioning des donnees (pas juste des params) | DVC tracke les fichiers, pas les metriques de run — complementaire, pas remplacement |
| **Kubeflow** | Pipeline ML complet sur Kubernetes | Necessite K8s, overkill pour notre infra |

### Ce qu'on montre au jury (preuves serveur)

**1. 27 runs, 19 finished, 8 failed — l'iteration est visible**
```
MLflow API: experiment_id=1 → 27 runs total
FINISHED: 19  |  FAILED: 8
```
> "Les 8 runs echecs ne sont pas des erreurs — c'est notre processus d'iteration. On a teste des configs qui ne marchaient pas. MLflow garde tout, meme les echecs."

**2. Sweep d'hyperparametres — top 5 par accuracy**
```
alpha=1e-6                     acc=76.68%  f1=74.25%  loss=log_loss
no_class_weight                acc=76.67%  f1=74.28%  loss=log_loss
alpha=5e-6                     acc=76.57%  f1=74.26%  loss=log_loss
modified_huber alpha=1e-6      acc=76.57%  f1=73.68%  loss=modified_huber
modified_huber alpha=1e-4      acc=76.38%  f1=73.77%  loss=modified_huber
```
> "On a varie alpha (1e-6 a 1e-3), la fonction de perte (log_loss vs modified_huber), et max_iter. Le meilleur: alpha=1e-6, log_loss. La difference est faible (~0.3%) ce qui montre que le modele est stable."

**3. Artefacts stockes par run — reproductibilite garantie**
```
baseline_model.pkl        → 1.08 MB (le classifieur SGD)
tfidf_vectorizer.pkl      → 190 KB (le vectorizer TF-IDF)
classification_report.json → 4.2 KB (metriques detaillees par classe)
training_metrics.json     → 146 B (accuracy + f1 aggreges)
```
> "Chaque run a ses propres artefacts. On peut revenir a n'importe quel run et recharger exactement ce modele."

**4. Drift experiment — detection automatisee**
```
Experiment: rakuten-data-drift (2 runs)
  ks_text_length      = 0.0048  p_value = 0.906  → NO DRIFT
  ks_has_description   = 0.0018  p_value = 1.000  → NO DRIFT
  ks_description_length = 0.0063 p_value = 0.654  → NO DRIFT
  ks_word_count        = 0.0028  p_value = 0.999  → NO DRIFT
  drift_share = 0.0 (0%)
```
> "Le test KS compare la distribution des features entre train et validation. p-value > 0.05 = pas de drift. C'est attendu ici car on utilise le meme dataset, mais en production avec des donnees reelles, cette tache deviendrait critique."

**5. Tracking URI partage entre conteneurs**
```yaml
# docker-compose.yml
MLFLOW_TRACKING_URI: http://mlflow:5000
```
> "Tous les conteneurs (Airflow, trainer, bootstrap) loguent au meme serveur MLflow via le reseau Docker interne. Le nom 'mlflow' est resolu par le DNS Docker Compose."

---

## 4. Airflow — Orchestration

### Probleme resolu
Sans orchestration, le re-entrainement est manuel : "ssh sur le serveur, lancer le script, verifier que ca a marche". Airflow automatise toute la chaine et **detecte les echecs** automatiquement.

### Pourquoi Airflow et pas les autres ?

| Alternative | Ce qu'elle fait mieux | Pourquoi Airflow gagne ici |
|---|---|---|
| **Prefect** | API Python plus moderne, pas besoin de scheduler separe | Moins de plugins, communaute plus petite, moins de docs |
| **Dagster** | Typage des assets, lineage automatique | Plus recent, moins de precedents en production |
| **Luigi** | Simple pour les pipelines lineaires | Pas de UI web native, pas de scheduling integre |
| **Kubeflow Pipelines** | Natif Kubernetes, conteneurisation par tache | Necessite K8s, overkill |
| **Argo Workflows** | Performant, natif K8s | Necessite K8s, YAML-based (pas Python) |
| **cron** | Zero setup, une ligne | Pas de retry, pas de monitoring, pas de dependances entre taches |

### Ce qu'on montre au jury (preuves serveur)

**1. 6 taches avec leurs durees reelles**
```
download_raw_data      →  0.8s   (telechargement S3)
prepare_dataset        →  4.8s   (nettoyage + split stratifie)
build_features         → 39.5s   (TF-IDF 5000 features) ← GOULOT
train_model            → 11.3s   (SGDClassifier + log MLflow)
verify_artifacts       →  0.5s   (check fichiers existent)
generate_drift_report  →  3.0s   (KS-test sur 4 features)
─────────────────────────────────
TOTAL                  → ~60s    (pipeline complet)
```
> "Le bottleneck est build_features a 39s — c'est la vectorisation TF-IDF sur 84k textes. Avec un modele BERT, ce serait le training qui dominerait. Identifier le goulot, c'est la premiere etape pour optimiser."

**2. 5 runs, 1 succes, 4 echecs — iteration reelle**
```
manual__all_green              → SUCCESS  (20 mars 23h10)
manual__perms_fixed            → FAILED   (20 mars 22h56)
manual__final_green            → FAILED   (20 mars 22h45)
manual__soutenance_ok          → FAILED   (20 mars 22h25)
manual__soutenance_final       → FAILED   (20 mars 22h15)
```
> "Les 4 echecs sont des problemes de permissions entre conteneurs — Airflow ecrivait des fichiers que MLflow ne pouvait pas lire. On a du aligner les UID/GID. C'est le genre de probleme qu'on ne rencontre jamais en local."

**3. DAG config — retry automatique**
```python
default_args = {
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}
```
> "Si une tache echoue, Airflow retente automatiquement 5 minutes plus tard. En production, on mettrait 2-3 retries avec backoff exponentiel."

**4. YAML anchor — DRY pour 3 services Airflow**
```yaml
x-airflow-common: &airflow-common
  build: ...
  environment: ...
  volumes: ...

airflow-init:      { <<: *airflow-common, command: "..." }
airflow-webserver: { <<: *airflow-common, command: airflow webserver }
airflow-scheduler: { <<: *airflow-common, command: airflow scheduler }
```
> "Sans le YAML anchor, on dupliquerait 20 lignes de config 3 fois. C'est un pattern Docker Compose professionnel."

**5. Pipeline sequentiel avec verification d'artefacts**
```python
download >> prepare >> build_features >> train_model >> verify_artifacts >> generate_drift_report
```
La tache `verify_artifacts` est un **gate** :
```python
required = [
    Path("models/baseline_model.pkl"),
    Path("models/tfidf_vectorizer.pkl"),
    Path("reports/training_metrics.json"),
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit(f"Missing artifacts: {missing}")
```
> "Si le training rate silencieusement (fichier corrompu, exception avalee), verify_artifacts le detecte AVANT que le drift report ne tourne sur un modele absent."

---

## 5. Prometheus + Grafana — Monitoring

### Probleme resolu
Un modele en production peut degrader **silencieusement**. La precision baisse, la latence augmente, les erreurs se multiplient — et personne ne le voit. Prometheus collecte, Grafana affiche, les alertes previennent.

### Pourquoi Prometheus+Grafana et pas les autres ?

| Alternative | Ce qu'elle fait mieux | Pourquoi Prometheus+Grafana gagne ici |
|---|---|---|
| **Datadog** | Tout-en-un (APM, logs, metrics), SaaS | Payant (~$15/host/mois), cloud-only |
| **New Relic** | APM detaille, distributed tracing | Payant, overkill pour une API simple |
| **ELK Stack** | Logs structures + recherche full-text | Lourd (Elasticsearch gourmand en RAM), complexe a operer |
| **CloudWatch** | Natif AWS, zero setup sur AWS | Cloud-locked, pas portable |
| **InfluxDB+Grafana** | Push-based (meilleur pour IoT, edge) | Prometheus est le standard pour les APIs HTTP |

### Ce qu'on montre au jury (preuves serveur)

**1. API UP depuis 83+ heures**
```
Prometheus: process_start_time → 2026-03-20 20:57 UTC
Uptime: 83.4 heures continues
Memory: 144.8 MB
```
> "L'API tourne depuis plus de 3 jours sans restart. 144 MB de RAM pour servir 1396 predictions — c'est leger."

**2. Zero-code instrumentation**
```python
# app.py — UNE seule ligne ajoute le monitoring complet
Instrumentator().instrument(application).expose(application)
```
> "On n'a pas ecrit une seule ligne de code pour les metriques. prometheus-fastapi-instrumentator genere automatiquement : compteurs de requetes, histogrammes de latence, requetes en cours, taille des reponses."

**3. Dashboard provisionne automatiquement (Infrastructure-as-Code)**
```
grafana/provisioning/
  datasources/prometheus.yml  → configure la source Prometheus
  dashboards/dashboard.yml    → pointe vers le JSON du dashboard
dashboards/
  api-monitoring.json         → 4 panneaux pre-configures
```
> "Au premier docker compose up, Grafana a deja le dashboard. Zero clic dans l'UI. C'est du provisioning-as-code — meme principe que Terraform pour l'infra."

**4. Les 4 panneaux et leurs requetes PromQL**
```
1. Requests per Second    → rate(http_requests_total[1m])
2. Response Latency P95   → histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[1m]))
3. Status Code Distribution → sum by(status) (http_requests_total)
4. Total Predictions      → http_requests_total{handler="/predict",method="POST",status="2xx"}
```
> "rate() calcule le debit instantane. histogram_quantile() donne le percentile 95 — 95% des requetes sont plus rapides que cette valeur. C'est plus utile que la moyenne car elle masque les outliers."

**5. Scrape interval : 5 secondes**
```yaml
# prometheus.yml
global:
  scrape_interval: 5s
scrape_configs:
  - job_name: 'fastapi'
    static_configs:
      - targets: ['api:8000']
```
> "Prometheus pull les metriques toutes les 5 secondes. C'est du pull-based, pas push — l'API n'a pas besoin de connaitre Prometheus. On pourrait ajouter 10 services sans modifier l'API."

**6. Chiffres concrets depuis Prometheus**
```
/predict 2xx  : 1,396 requetes   (95.9%)
/predict 4xx  :    60 requetes   (4.1%) → tentatives sans token
/health  2xx  : 16,121 requetes  → Docker healthcheck toutes les 20s
/metrics 2xx  : 60,026 requetes  → Prometheus scrape toutes les 5s
```
> "Les 60 erreurs 4xx sont intentionnelles — ce sont nos requetes sans Bearer token. Ca prouve que la securite fonctionne. Le ratio 96%/4% est exactement ce qu'on attend."

---

## 6. Docker Compose — Conteneurisation

### Probleme resolu
Le probleme classique du ML : "ca marche sur ma machine". Versions de numpy, scipy, sklearn incompatibles. L'API qui tourne en Python 3.10 mais Airflow en 3.8. Docker isole chaque composant dans son propre environnement.

### Pourquoi Docker Compose et pas les autres ?

| Alternative | Ce qu'elle fait mieux | Pourquoi Docker Compose gagne ici |
|---|---|---|
| **Kubernetes** | Autoscaling, rolling updates, self-healing | Necessite un cluster (min 3 nodes), overkill pour 8 services |
| **Docker Swarm** | Orchestration multi-node simple | Abandonne par Docker, communaute reduite |
| **Podman** | Rootless, pas de daemon | Moins de support pour compose, ecosysteme plus petit |
| **Bare metal** | Pas d'overhead conteneurs | Pas reproductible, dependency hell, pas portable |

### Ce qu'on montre au jury (preuves codebase)

**1. 12 services, 5 Dockerfiles specialises**
```
Dockerfile                    → API (python:3.10-slim, 27 lignes)
docker/streamlit/Dockerfile   → UI (python:3.10-slim, 8 lignes)
docker/mlflow/Dockerfile      → Tracking (python:3.10-slim + mlflow server)
docker/airflow/Dockerfile     → Orchestration (apache/airflow:2.10.5-python3.10)
docker/trainer/Dockerfile     → Training + Bootstrap
```
> "Chaque service a son image optimisee. L'API n'embarque ni Airflow ni MLflow. Le principe de responsabilite unique, applique aux conteneurs."

**2. Volumes partages pour le hot-reload du modele**
```yaml
api:       volumes: [./models:/app/models]
trainer:   volumes: [./models:/app/models]
airflow:   volumes: [./:/opt/airflow/project]  # inclut models/
```
> "Quand Airflow re-entraine le modele, le nouveau fichier .pkl apparait dans le volume. L'API peut le recharger sans rebuild d'image. C'est la cle du deploiement continu ML."

**3. Health checks comme gates de demarrage**
```yaml
api:
  healthcheck:
    test: ["CMD", "python", "-c",
           "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()"]
    interval: 20s
    timeout: 5s
    retries: 5

streamlit:
  depends_on:
    api:
      condition: service_healthy  # attend que l'API soit HEALTHY
```
> "Streamlit ne demarre pas tant que l'API n'a pas repondu 5 fois au healthcheck. Pas de race condition au boot."

**4. Profils pour le bootstrap**
```yaml
bootstrap:
  profiles: ["bootstrap"]
  command: ["bash", "scripts/bootstrap.sh"]
```
> "Le bootstrap ne tourne qu'au premier lancement : `docker compose --profile bootstrap up bootstrap`. Ensuite, `docker compose up -d` suffit. Separation entre initialisation et exploitation."

**5. Grafana anonymous access pour la demo**
```yaml
grafana:
  environment:
    GF_AUTH_ANONYMOUS_ENABLED: "true"
    GF_AUTH_ANONYMOUS_ORG_ROLE: Viewer
```
> "En production, on desactiverait l'acces anonyme. Pour la soutenance, ca permet au jury de voir le dashboard sans login."

---

## 7. GitHub Actions — CI/CD

### Probleme resolu
Sans CI, chaque `git push` est un acte de foi. Le code compile ? Les tests passent ? L'image Docker build ? GitHub Actions automatise les verifications **avant** que le code touche main.

### Pourquoi GitHub Actions et pas les autres ?

| Alternative | Ce qu'elle fait mieux | Pourquoi GitHub Actions gagne ici |
|---|---|---|
| **GitLab CI** | Integrated avec GitLab, runners on-premise | On est sur GitHub, migration inutile |
| **Jenkins** | Extensible, auto-heberge, pipeline-as-code | Lourd a maintenir, interface datee, necessite un serveur |
| **CircleCI** | Rapide, bon caching Docker | Plan gratuit limite, SaaS externe |
| **Travis CI** | Simple, historique open-source | Declin depuis le rachat, moins fiable |

### Ce qu'on montre au jury (3 scenarios d'erreur)

**Pipeline : lint → test → docker-build (sequentiel)**
```yaml
jobs:
  lint:    # flake8 --max-line-length 127
  test:    # pytest --cov=src/api --cov-fail-under=80
    needs: lint
  docker-build:  # docker build + push ghcr.io
    needs: test
```

**Scenario 1 — Import inutilise (flake8 F401)**
```
Commit: feat(api): add structured logging with request context
Erreur: import os, import sys → jamais utilises
Detection: flake8 F401 "imported but unused"
Fix: remove unused imports
Lecon: Le linter attrape le code mort AVANT les tests
```

**Scenario 2 — Contrat API casse (pytest assertion)**
```
Commit: refactor(api): rename health status 'ok' to 'healthy'
Erreur: tests assertent body["status"] == "ok"
Detection: pytest AssertionError
Fix: revert — la valeur 'ok' est utilisee par Docker healthcheck ET les tests
Lecon: Les tests protegent le contrat entre les services
```

**Scenario 3 — Reference indefinie (flake8 F821)**
```
Commit: feat(stats): add median inference latency
Erreur: statistics.median() sans import statistics
Detection: flake8 F821 "undefined name 'statistics'"
Fix: ajouter import statistics
Lecon: Le linter detecte les erreurs de runtime AVANT l'execution
```

> "A chaque fois le pattern est le meme : commit → CI rouge → fix → CI vert. C'est ca l'interet d'une CI — elle transforme les erreurs en feedback rapide au lieu de bugs en production."

---

## 8. Detection de derive (Drift) — Monitoring ML

### Probleme resolu
Un modele entraine sur des donnees 2024 peut devenir obsolete en 2025 si la distribution change (nouveaux produits, nouvelles categories, changement de langue). La detection de drift repond a : **"mes donnees actuelles ressemblent-elles toujours aux donnees d'entrainement ?"**

### Implementation — double approche

**Evidently (si disponible)** : framework dédié, rapport HTML complet
**scipy KS-test (fallback)** : test de Kolmogorov-Smirnov, pur Python, zero dependance lourde

```python
# drift_report.py — 4 features derivees du texte brut
text_length        → longueur du titre (designation)
has_description    → booleen, 1 si description non vide
description_length → longueur de la description
word_count         → nombre de mots dans le titre
```

> "On ne peut pas faire un KS-test directement sur du texte. On derive des features numeriques qui capturent les proprietes statistiques du texte. Si la longueur moyenne des titres change, c'est un signal de drift."

### Resultats concrets
```
text_length:        KS=0.005  p=0.91  → pas de drift
has_description:    KS=0.002  p=1.00  → pas de drift
description_length: KS=0.006  p=0.65  → pas de drift
word_count:         KS=0.003  p=1.00  → pas de drift
```
> "p-value > 0.05 signifie qu'on ne peut pas rejeter l'hypothese que les deux distributions sont identiques. Ici, toutes les p-values sont > 0.65 — aucun drift detecte, ce qui est logique car on compare train/val du meme dataset."

### Integration MLOps
- Derniere tache du DAG Airflow → execute apres chaque re-entrainement
- Resultats logues dans MLflow experiment `rakuten-data-drift`
- En production : si drift_share > 0 → alerte pour declencher un re-entrainement avec nouvelles donnees

---

---

## 9. Pepites d'expert — Phrases qui montrent la maitrise

### Phrases "wow" a placer naturellement pendant le Q&A

**Sur l'architecture globale :**
> "Le modele est le composant le plus simple du systeme. Tout ce qui est autour — l'API, le monitoring, l'orchestration — c'est la la complexite reelle. Et c'est exactement ca le MLOps."

**Sur la latence :**
> "5ms de latence moyenne. Pas parce qu'on l'a optimise, mais parce qu'on a fait un choix delibere : TF-IDF + SGD au lieu de BERT. En production, la latence est une feature, pas un bug."

**Sur les echecs Airflow :**
> "Les 4 runs echecs dans Airflow, c'est des problemes de permissions inter-conteneurs. Quand Airflow ecrit un fichier avec UID 50000 et que MLflow attend UID 1000, le fichier est la mais illisible. C'est le genre de probleme qu'on ne rencontre jamais en local."

**Sur Prometheus pull vs push :**
> "Prometheus utilise le pull : c'est lui qui va chercher les metriques. L'API n'a pas besoin de savoir que Prometheus existe. On pourrait ajouter 10 services a monitorer sans modifier une seule ligne dans l'API."

**Sur la confiance du modele :**
> "36% de confiance semble faible, mais avec 27 classes, la baseline random c'est 3.7%. Notre modele est 10x meilleur que le hasard sur cette prediction."

**Sur le drift :**
> "On ne peut pas faire un test statistique directement sur du texte brut. On derive 4 features numeriques — longueur du titre, presence de description, nombre de mots — et on applique un test de Kolmogorov-Smirnov sur chacune. C'est un proxy, mais en production c'est comme ca qu'on detecte le drift sur du NLP."

**Sur Docker Compose vs Kubernetes :**
> "Docker Compose, c'est un MVP de production. Si on devait scaler a 1000 predictions par seconde avec autoscaling, on migrerait vers Kubernetes. Mais pour 8 services sur un VPS, Compose est le bon outil — ajouter K8s serait de la sur-ingenierie."

**Sur la CI/CD :**
> "Le linter detecte des erreurs de runtime AVANT l'execution. Quand flake8 signale F821 — reference indefinie — c'est une erreur qui aurait crashe l'API en production. Le CI l'attrape au moment du push."

**Sur MLflow :**
> "Chaque run MLflow est atomique : parametres, metriques, artefacts. Si dans 6 mois on se demande 'quel alpha avait donne 76% ?', la reponse est dans MLflow, pas dans un notebook perdu."

**Sur le health check Docker :**
> "Streamlit ne demarre pas tant que l'API n'a pas repondu 5 fois au healthcheck. C'est Docker qui gere ca avec `condition: service_healthy`. Pas de race condition au boot."

**Sur la securite :**
> "On utilise hmac.compare_digest au lieu de == pour comparer les tokens. C'est une comparaison a temps constant qui previent les timing attacks. Le temps de reponse est le meme que le token soit bon ou mauvais."

**Sur Grafana provisioning :**
> "Le dashboard Grafana est un fichier JSON versionne dans Git. Au premier docker compose up, Grafana charge automatiquement le datasource Prometheus et le dashboard. Zero clic dans l'interface. C'est du provisioning-as-code — meme philosophie que Terraform."

---

## Chiffres cles a retenir pour le Q&A

| Metrique | Valeur | Source |
|---|---|---|
| Latence prediction moyenne | **5.49ms** | Prometheus |
| 100% predictions < 100ms | oui | Prometheus histogram |
| Uptime API continu | **83+ heures** | Prometheus process_start_time |
| RAM consommee API | **144.8 MB** | Prometheus process_resident_memory |
| Meilleur accuracy | **76.68%** | MLflow run alpha=1e-6 |
| Meilleur F1-macro | **74.25%** | MLflow run alpha=1e-6 |
| Taille modele | **1.08 MB** | MLflow artifacts |
| Taille vectorizer | **190 KB** | MLflow artifacts |
| Runs MLflow | **27** (19 OK, 8 fail) | MLflow API |
| Pipeline Airflow total | **~60 secondes** | Airflow task instances |
| Goulot pipeline | **build_features 39.5s** | Airflow task instances |
| Predictions servies | **1,396** | Prometheus |
| Taux de succes API | **95.9%** | Prometheus |
| Services Docker | **12** | docker-compose.yml |
| Tests unitaires | **34** (couverture > 80%) | GitHub Actions |
| Drift detecte | **0%** (attendu) | MLflow drift experiment |
| Scrape Prometheus | **5 secondes** | prometheus.yml |
| Categories produits | **27** | Dataset Rakuten |
| Dataset entrainement | **84,916 produits** | Dataset source |
| Descriptions manquantes | **~35%** | Analyse exploratoire |
