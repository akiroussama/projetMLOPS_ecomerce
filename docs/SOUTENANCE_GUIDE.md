# Soutenance MLOps — Guide de Preparation

**Date : 24 mars 2026 | Duree : 20 min presentation + 10 min Q&A**
**Presentation : `reports/PRESENTATION_SOUTENANCE.html`** (ouvrir dans un navigateur, fleches pour naviguer)

---

## Dispatch des interventions

| Ordre | Intervenant | Slides | Duree | Contenu |
|-------|-------------|--------|-------|---------|
| 1 | **Hery** | 0 → 4 | ~4 min | Cover, Sommaire, Contexte Rakuten, Modele Baseline, API FastAPI |
| 2 | **Johan** | 5 → 8 | ~4 min | Docker, Architecture Microservices, MLflow, Securite |
| 3 | **Liviu** | 9 → 13 | ~4.5 min | Airflow, Airflow DAG, CI/CD, Prometheus, Grafana |
| 4 | **Oussama** | 14 → 17 + demo | ~7.5 min | Streamlit (2 min) + Demo live (4 min) + Conclusion (1.5 min) |

> **Regle Q&A** : pendant les 10 min de questions, celui qui a *code* la fonctionnalite repond, meme s'il ne l'a pas presentee. Exemple : question sur MLflow/sweep → Johan repond. Question sur Airflow/DAG → Liviu repond. Question sur /stats ou drift → Oussama. Voir le fichier `docs/QUESTIONS_JURY.md` pour les reponses detaillees prepares.

---

## Speech — Hery (slides 0 → 4, ~4 min)

### Slide 0 — Cover (~30s)

> Bonjour, merci d'etre presents pour cette soutenance. Je suis Hery, et avec Johan, Liviu et Oussama, nous allons vous presenter notre projet MLOps : l'industrialisation d'un pipeline de classification de produits e-commerce pour Rakuten France. Ce projet part de notre travail precedent en Data Science, et l'objectif ici est de transformer un modele de machine learning en un systeme de production complet, reproductible et monitorable.

### Slide 1 — Sommaire (~30s)

> Notre presentation suit les cinq phases du projet. Nous commencerons par les fondations : donnees, modele et API. Puis nous verrons la conteneurisation et l'architecture microservices. Ensuite, le suivi des experiences avec MLflow et la securite. Johan me passera le relais. Liviu couvrira l'orchestration avec Airflow et le monitoring. Et Oussama terminera avec l'application Streamlit et une demonstration en direct. Nous aurons egalement un temps de questions a la fin.

### Slide 2 — Contexte & Donnees (~1 min)

> Rakuten France gere une marketplace avec des millions de produits. Chaque produit doit etre classe dans l'une de vingt-sept categories a partir de son titre, appele « designation », et de sa description textuelle. Notre jeu de donnees provient du challenge Rakuten France, heberge sur Amazon S3 : quatre-vingt-quatre mille neuf cent seize produits d'entrainement. Un point important : environ trente-cinq pourcent des descriptions sont manquantes, ce qui represente un defi reel pour le preprocessing.
>
> La transition vers le MLOps est resume dans ce tableau : nous sommes passes de notebooks Jupyter a des scripts modulaires, d'un modele local a une API FastAPI securisee, et d'une execution manuelle a un pipeline orchestre.

### Slide 3 — Modele Baseline (~1 min)

> Pour la mise en production, nous avons choisi un pipeline texte simple mais performant : TF-IDF avec cinq mille features, unigrams et bigrams, suivi d'un SGDClassifier avec class_weight balance. Ce modele atteint soixante-seize virgule deux pourcent d'accuracy en validation et soixante-treize virgule neuf pourcent de F1-score macro sur un jeu de validation de pres de dix-sept mille echantillons.
>
> Pourquoi ce choix plutot qu'un modele deep learning ? Le compromis performance-simplicite. L'inference prend moins de dix millisecondes sur CPU, le modele tient dans un fichier pickle de quelques megaoctets, et il n'y a aucune dependance GPU en production. Les artefacts — modele, vectorizer et mapping des labels — sont serialises en trois fichiers charges au demarrage de l'API.

### Slide 4 — API FastAPI (~1 min)

> L'API est construite avec FastAPI et expose quatre endpoints principaux. Le endpoint /health permet de verifier l'etat du service et du modele charge. Le endpoint /predict accepte une requete POST avec le titre et la description du produit, et retourne la categorie predite, le code correspondant, le score de confiance et le nom du modele. Le endpoint /metrics expose les metriques Prometheus. Et le nouveau endpoint /stats retourne des statistiques metier en temps reel : nombre de predictions par categorie, temps d'inference moyen, min et max — collectees de facon thread-safe avec un lock Python.
>
> L'architecture est modulaire : app.py gere les routes, service.py contient la logique metier, et schemas.py definit la validation Pydantic. Le endpoint /predict est protege par un Bearer token, ce que Johan va detailler dans la partie securite. Je lui passe la parole.

---

## Speech — Johan (slides 5 → 8, ~4 min)

### Slide 5 — Conteneurisation Docker (~1 min)

> Merci Hery. Je vais maintenant vous montrer comment nous avons conteneurise l'ensemble du projet. Chaque composant a son propre Dockerfile. L'API utilise une image Python 3.10-slim pour minimiser la taille. Les artefacts du modele ne sont pas inclus dans l'image : ils sont montes en volume Docker, ce qui permet de mettre a jour le modele sans reconstruire l'image.
>
> Nous avons cinq Dockerfiles specialises : un pour l'API, un pour le training et le bootstrap, un pour MLflow, un pour Airflow et un pour Streamlit. Le service bootstrap est particulierement important : au premier demarrage, il telecharge automatiquement les donnees depuis S3, effectue le preprocessing, et entraine le modele initial. Cela garantit que le systeme est operationnel sans intervention manuelle.

### Slide 6 — Architecture Microservices (~1 min)

> Voici l'architecture globale. Tous les services sont orchestres par Docker Compose avec un seul fichier docker-compose.yml. L'utilisateur interagit avec Streamlit sur le port huit mille cinq cent un. Streamlit communique avec l'API FastAPI sur le port huit mille deux cents. L'API expose un endpoint /metrics que Prometheus scrape toutes les quinze secondes, et Grafana visualise ces metriques.
>
> En parallele, MLflow sert de serveur de tracking sur le port cinq mille, et Airflow orchestre le re-entrainement hebdomadaire avec son webserver, son scheduler et sa base Postgres. Les volumes partages sont essentiels : le repertoire models/ est accessible par l'API, le trainer et Airflow, ce qui permet a un re-entrainement de mettre a jour le modele en production.

### Slide 7 — MLflow (~1 min)

> Pour le suivi des experiences, nous utilisons MLflow. Vous pouvez voir sur cette capture plus de quinze runs : un run initial bootstrap, des re-entrainements Airflow, et surtout un sweep d'hyperparametres que nous avons lance pour optimiser le modele.
>
> Ce sweep couvre quinze combinaisons : alpha de 1e-6 a 1e-3, la fonction de perte log_loss vs modified_huber, et max_iter de 300 a 2000. Le meilleur run — alpha egal a 1e-6 — atteint 76.68% d'accuracy et 74.25% de F1-score macro. Nous avons aussi une deuxieme experience « rakuten-data-drift » qui logue les resultats du rapport de derive. A chaque run, parametres, metriques et artefacts sont traces automatiquement. Le tracking URI est partage entre les services via la variable MLFLOW_TRACKING_URI.

### Slide 8 — Securite (~30s)

> Cote securite, l'API est protegee par un Bearer token configurable via la variable d'environnement API_AUTH_TOKEN. Le endpoint /health reste public pour permettre les health checks Docker. La validation des entrees est assuree par Pydantic avec des schemas stricts. Aucun secret n'est code en dur : tout passe par des variables d'environnement. Je passe la parole a Liviu pour l'orchestration.

---

## Speech — Liviu (slides 9 → 13, ~4.5 min)

### Slide 9 — Airflow Orchestration (~1 min)

> Merci Johan. Je vais maintenant vous presenter l'orchestration du pipeline avec Apache Airflow. Nous avons cree un DAG appele « rakuten_weekly_retraining » qui automatise l'ensemble du cycle de re-entrainement. Ce DAG est programme pour s'executer chaque lundi a deux heures du matin, via l'expression cron zero deux etoile etoile un.
>
> Le pipeline comprend six taches sequentielles. D'abord, le telechargement des donnees brutes depuis S3. Ensuite, la preparation du dataset : nettoyage et decoupage train/validation. Puis la construction des features TF-IDF. Le train_model entraine le SGDClassifier et logge les resultats dans MLflow. Ensuite, verify_artifacts verifie l'integrite des fichiers produits, et generate_drift_report genere un rapport de derive avec un test de Kolmogorov-Smirnov via scipy.
>
> Et ce n'est pas seulement theorique : nous avons declenche ce DAG manuellement sur notre VPS Hetzner via l'API REST d'Airflow, et les six taches sont passees en vert — succes complet. Vous pouvez le voir sur la capture d'ecran.

### Slide 10 — Airflow DAG Detail (~45s)

> Sur cette capture de l'interface Airflow, vous pouvez voir le DAG et son historique d'execution. L'infrastructure Airflow repose sur trois services : airflow-init qui initialise la base et cree l'utilisateur admin, le webserver qui fournit l'interface web sur le port huit mille deux cent quatre-vingt, et le scheduler qui planifie et execute les DAGs.
>
> Nous utilisons le LocalExecutor, suffisant pour notre pipeline. La configuration est factorisee grace au mecanisme YAML anchor « x-airflow-common » dans le docker-compose, ce qui evite la duplication entre les trois services Airflow.

### Slide 11 — CI/CD GitHub Actions (~45s)

> Pour l'integration continue, nous avons mis en place un pipeline GitHub Actions. A chaque push ou pull request vers la branche main, le pipeline se declenche automatiquement. Il execute le linting avec flake8 et black, lance les tests unitaires et d'integration avec pytest, et verifie les imports du projet.
>
> Ce pipeline garantit que tout code merge dans main respecte les standards de qualite. Si un test echoue, le merge est bloque. A terme, l'objectif est d'ajouter un volet CD avec la construction et la publication automatique des images Docker.

### Slide 12 — Prometheus (~1 min)

> Passons au monitoring. Prometheus collecte les metriques de performance de l'API. L'instrumentation est automatique grace a la librairie prometheus-fastapi-instrumentator, qui expose les metriques sur le endpoint /metrics sans modifier le code metier.
>
> Sur ces captures, vous voyez d'abord les targets Prometheus : notre API FastAPI est detectee comme « UP » avec un temps de scrape de sept millisecondes. En dessous, une requete PromQL — rate de http_requests_total sur une minute — montre le debit de requetes par endpoint : /health, /metrics et /predict, avec des courbes distinctes pour chaque route.
>
> Les metriques collectees incluent le nombre total de requetes, la duree par requete en histogramme, les requetes en cours, et la taille des reponses.

### Slide 13 — Grafana Dashboard (~1 min)

> Et voici le dashboard Grafana qui visualise ces metriques. Quatre panneaux : le debit de requetes par seconde, la latence au percentile quatre-vingt-quinze, la distribution des codes de statut HTTP, et le nombre de requetes en cours.
>
> Les resultats montrent un taux de succes de quatre-vingt-dix-neuf pourcent en codes deux cents, une latence P95 inferieure a cinquante millisecondes, et un taux d'erreurs quatre cents d'environ un pourcent — correspondant aux requetes non authentifiees.
>
> Le provisioning est entierement automatise : le datasource Prometheus et le dashboard sont configures au demarrage via des fichiers YAML dans le repertoire grafana/provisioning. Zero configuration manuelle. Oussama, a toi pour Streamlit et la demo.

---

## Speech — Oussama (slides 14 → 17 + demo, ~7.5 min)

### Slide 14 — Streamlit Application (~1 min)

> Merci Liviu. Je vais conclure notre presentation avec l'application Streamlit, puis je ferai une demonstration en direct. L'application Streamlit est l'interface utilisateur du projet. Elle communique exclusivement avec l'API deployee, comme demande dans le cahier des charges.
>
> L'application propose quatre pages. La page d'accueil donne une vue d'ensemble du projet. La page Contexte affiche le statut de l'API, le modele charge et les categories disponibles — comme vous pouvez le voir sur cette capture, le SGDClassifier est bien charge et operationnel. La page Data Explorer permet de visualiser la distribution du dataset. Et la page Predictions offre un formulaire pour tester le modele en temps reel.

### Slide 15 — Streamlit Detail (~45s)

> Voici les pages en detail. A gauche, le Data Explorer montre la distribution des vingt-sept categories avec des graphiques interactifs. A droite, le formulaire de prediction : l'utilisateur saisit un titre et une description de produit, et obtient en moins d'une seconde la categorie predite avec le score de confiance.
>
> Tout fonctionne via l'API : Streamlit envoie une requete POST a /predict avec le Bearer token, et affiche le resultat. C'est exactement ce que je vais vous montrer en direct.

### Slide 16 — Demo Live (~4 min)

> [OUVRIR LE NAVIGATEUR]
>
> Points a montrer dans la demo :
> 1. **Streamlit** (http://localhost:8501) : faire une prediction live avec un produit reel (ex: "iPhone 15 Pro Max", "256GB Noir Titane"). Montrer le resultat, la confiance, le modele utilise.
> 2. **Grafana** (http://localhost:3000) : montrer l'impact de la requete qu'on vient de faire sur le dashboard temps reel. Les compteurs montent.
> 3. **MLflow** (http://localhost:5000) : ouvrir l'experiment rakuten-text-baseline, montrer les runs, comparer les metriques.
> 4. **Airflow** (http://localhost:8280) : montrer le DAG, son historique, les logs d'un run.
>
> [Adapter selon le temps restant — prioriser Streamlit + Grafana]

### Slide 17 — Conclusion (~1.5 min)

> Pour conclure, nous avons realise un pipeline MLOps complet couvrant les cinq phases du projet. Et nous avons pousse plus loin encore avec cinq ameliorations supplementaires pour cette soutenance.
>
> Premierement, l'endpoint /stats sur l'API pour des statistiques metier en temps reel par categorie. Deuxiemement, un sweep de quinze runs MLflow sur les hyperparametres du SGDClassifier — meilleur resultat : 76.68% d'accuracy. Troisiemement, la detection de derive des donnees par test KS de Kolmogorov-Smirnov, integree comme derniere tache du DAG Airflow et loggee dans MLflow. Quatriemement, le DAG Airflow a ete declenche et valide en conditions reelles sur notre VPS — six taches vertes. Et cinquiemement, nous avons prepare des reponses documentees pour vos questions.
>
> Nos principaux apprentissages portent sur l'orchestration Docker Compose multi-services, la gestion des permissions entre conteneurs, la serialisation pickle et ses pieges, l'instrumentation automatique avec Prometheus, et la compatibilite des bibliotheques en production.
>
> En perspectives : deploiement cloud sur AWS ou GCP, integration de modeles deep learning comme BERT, et mise en place d'A/B testing pour comparer les modeles en production.
>
> Merci pour votre attention. Nous sommes prets pour vos questions.

---

## Guide : Lancer l'application en local

### Prerequis
- Docker Desktop installe et demarre
- Git pour cloner le repo

### Etape 1 : Cloner et configurer

```bash
git clone https://github.com/akiroussama/projetMLOPS_ecomerce.git
cd projetMLOPS_ecomerce

# Copier la config
cp .env.example .env
# Editer .env si besoin (API_AUTH_TOKEN, AIRFLOW_UID=50000)
```

### Etape 2 : Premier demarrage (avec bootstrap)

```bash
# Lancer le bootstrap pour telecharger les donnees et entrainer le modele initial
docker compose --profile bootstrap up bootstrap

# Puis lancer tous les services
docker compose up -d
```

### Etape 3 : Demarrage normal (modele deja entraine)

```bash
docker compose up -d
```

### Etape 4 : Verifier que tout fonctionne

| Service | URL | Login |
|---------|-----|-------|
| API (Swagger) | http://localhost:8200/docs | — |
| API Health | http://localhost:8200/health | — |
| Streamlit | http://localhost:8501 | — |
| MLflow | http://localhost:5000 | — |
| Airflow | http://localhost:8280 | airflow / airflow |
| Grafana | http://localhost:3000 | admin / admin |
| Prometheus | http://localhost:9090 | — |

### Etape 5 : Tester une prediction

```bash
curl -X POST http://localhost:8200/predict \
  -H "Authorization: Bearer change-me" \
  -H "Content-Type: application/json" \
  -d '{"designation": "iPhone 15 Pro Max", "description": "256GB Noir Titane", "productid": 1, "imageid": 1}'
```

### Arreter les services

```bash
docker compose down
```

---

## Preparation pour le Q&A

### Questions probables et qui repond

| Question | Repond |
|----------|--------|
| Pourquoi SGDClassifier et pas un modele deep learning ? | Hery |
| Comment gerez-vous la mise a jour du modele en production ? | Oussama (Airflow) |
| Comment fonctionne le tracking MLflow ? | Johan |
| Comment les conteneurs communiquent entre eux ? | Johan |
| Quelle est la latence de prediction ? | Hery (< 10ms CPU) |
| Comment detectez-vous la derive des donnees ? | Oussama (KS-test scipy dans generate_drift_report, logge dans MLflow experiment rakuten-data-drift) |
| Pourquoi Docker Compose et pas Kubernetes ? | Johan |
| Comment gerez-vous la securite de l'API ? | Hery |
| Le re-entrainement est-il automatique ? | Liviu (DAG cron Airflow 0 2 * * 1, valide en prod VPS) |
| Comment provisionnez-vous Grafana ? | Liviu (YAML provisioning dans docker/monitoring/grafana/provisioning) |
| Qu'est-ce que le sweep MLflow et pourquoi ? | Johan (15 runs SGDClassifier, alpha/loss/max_iter, meilleur : alpha=1e-6, 76.68%) |
| A quoi sert l'endpoint /stats ? | Hery (stats metier : predictions par categorie, temps inference moyen/min/max, thread-safe) |
