# Speech Soutenance MLOps — Rakuten
# 20 min pres + demo | 10 min Q&A | 24 mars 2026

<!-- ============================================================
     RAPPEL ATTENTES JURY (message Antoine) :
     - 5 phases (1-3 obligatoires, 4-5 optionnelles)
     - "comprendre ce que vous avez construit"
     - "conclusions business et scientifiques"
     - format slides + demo
     - pas de rendu, tout se joue a la presentation
     ============================================================ -->

---

## HERY — slides 0 a 4 (~4 min)

### Slide 0 — Cover

Bonjour, merci d'etre la. Je suis Hery, on est quatre sur ce projet : Johan, Liviu, Oussama et moi. On va vous presenter notre projet MLOps — en gros, on a pris le modele de classification de produits qu'on avait fait en projet Data Science, et on l'a industrialise de bout en bout.

L'idee c'est simple : passer d'un notebook qui tourne en local a un systeme complet qu'on peut deployer, monitorer et re-entrainer automatiquement. On a couvert les cinq phases du cahier des charges, y compris les deux optionnelles.

### Slide 1 — Sommaire

Je vais couvrir les fondations — donnees, modele, API. Johan enchainera sur Docker, l'architecture et MLflow. Liviu parlera d'Airflow et du monitoring. Oussama conclura avec Streamlit et fera une demo en direct.

<!-- Pas besoin de s'attarder ici, enchainer vite -->

### Slide 2 — Contexte & Donnees

Le cas d'usage : Rakuten France, marketplace e-commerce, des millions de produits a classifier. On a 27 categories et un dataset de 84 000 produits. L'input c'est du texte — le titre du produit et sa description. Avec un vrai probleme concret : 35% des descriptions sont vides, donc on bosse principalement sur le titre.

La transition ML vers MLOps c'est ce tableau. On est passes de notebooks a des scripts Python modulaires. D'un modele en local a une API. D'une execution a la main a Airflow. C'est ca le coeur du projet.

### Slide 3 — Modele Baseline

Le modele c'est TF-IDF plus SGDClassifier. Cinq mille features, unigrams et bigrams, avec class_weight balance pour gerer le desequilibre des classes. Ca donne 76% d'accuracy, 74% de F1 macro. C'est pas le meilleur score possible — un CamemBERT ferait dans les 85% — mais c'est pas le sujet.

On a fait ce choix volontairement. L'inference prend moins de 10 millisecondes sur CPU, le modele fait quelques megaoctets, et surtout notre architecture est model-agnostic. Si on veut passer a BERT demain, on change une classe dans train_model.py, tout le reste — API, Airflow, MLflow, monitoring — ca reste pareil. C'est la le point important d'une architecture MLOps : le modele c'est un composant interchangeable.

### Slide 4 — API FastAPI

L'API c'est le coeur de la phase 1. FastAPI, quatre endpoints :
- /health pour le health check
- /predict protege par un Bearer token, qui prend un titre et une description et retourne la categorie, le code, le score de confiance
- /metrics pour Prometheus
- /stats qu'on a ajoute recemment — ca donne les statistiques metier en temps reel : combien de predictions par categorie, temps d'inference moyen

L'architecture est modulaire : routes, logique metier et validation sont dans des fichiers separes. Johan va vous montrer comment tout ca est conteneurise.

---

## JOHAN — slides 5 a 8 (~4 min)

### Slide 5 — Docker

Merci Hery. On entre dans la phase 3 en fait — la conteneurisation. Chaque service a son Dockerfile. L'API tourne sur Python 3.10-slim, les artefacts du modele sont montes en volume Docker. C'est important : quand Airflow re-entraine un modele, les fichiers sont mis a jour dans le volume partage, et l'API peut les recharger sans qu'on reconstruise l'image.

On a cinq Dockerfiles. Et un truc pratique : le service bootstrap. Au premier lancement, il telecharge les donnees depuis S3, fait le preprocessing, entraine le premier modele. Du coup quand quelqu'un clone le repo et lance docker compose up, tout est operationnel automatiquement.

### Slide 6 — Architecture

Voila comment les services communiquent. Un docker-compose.yml, huit services. L'utilisateur passe par Streamlit, qui appelle l'API. L'API expose des metriques que Prometheus collecte, et Grafana les affiche. En parallele, MLflow fait le tracking et Airflow orchestre le re-entrainement.

Le point technique qui nous a donne du mal, c'est les volumes partages et les permissions entre conteneurs. Quand Airflow s'execute avec un uid different de celui de MLflow, les fichiers crees par l'un ne sont pas lisibles par l'autre. On a du gerer ca explicitement.

### Slide 7 — MLflow

MLflow c'est notre phase 2 — le suivi des experiences. On a deux experiments : un pour l'entrainement, un pour le drift. Ce qu'on voit sur cette capture c'est nos 15 runs. On a fait un sweep d'hyperparametres : on a varie alpha, la fonction de perte, max_iter, pour trouver la meilleure combinaison.

Le meilleur run donne 76.68% d'accuracy. A chaque run, tout est logge automatiquement — parametres, metriques, artefacts. Ca permet de comparer les runs facilement et de savoir exactement quelle config a produit quel resultat.

### Slide 8 — Securite

La securite c'est aussi dans la phase 2. L'API est protegee par Bearer token, injecte par variable d'environnement — jamais en dur dans le code. Le health check reste public parce que Docker en a besoin pour ses health checks. On valide les entrees avec Pydantic, schemas stricts. Liviu, c'est a toi.

---

## LIVIU — slides 9 a 13 (~4.5 min)

### Slide 9 — Airflow

Merci Johan. L'orchestration c'est la phase 3. On a un DAG Airflow — rakuten_weekly_retraining — programme pour tourner chaque lundi a 2h du matin.

Six taches dans l'ordre : telechargement des donnees S3, preparation du dataset, construction des features TF-IDF, entrainement avec log MLflow, verification des artefacts, et rapport de derive. C'est pas juste un schema : on l'a declenche sur notre VPS chez Hetzner, les six taches passent au vert.

La derniere tache — generate_drift_report — fait un test de Kolmogorov-Smirnov pour detecter si la distribution des donnees a change. Si on detecte du drift, on sait qu'il faut regarder de plus pres.

### Slide 10 — Airflow Detail

Cote infrastructure, Airflow c'est trois conteneurs : l'init pour la base Postgres et l'admin, le webserver pour l'interface, et le scheduler qui execute les DAGs. On utilise le LocalExecutor — pas besoin de Celery pour un seul DAG.

Un detail technique : la config des trois services est factorisee avec un anchor YAML dans le docker-compose. On ecrit une fois, on reutilise trois fois.

### Slide 11 — CI/CD

L'integration continue c'est dans la phase 4 — qui est optionnelle mais qu'on a quand meme implementee. Pipeline GitHub Actions : a chaque push ou PR sur main, ca lance le linting, les tests pytest, et la verification des imports. Si ca casse, on merge pas.

### Slide 12 — Prometheus

Toujours dans la phase 4, le monitoring. Prometheus collecte les metriques de l'API. L'instrumentation est automatique avec prometheus-fastapi-instrumentator — on n'a pas modifie le code de l'API pour ca.

Ce qu'on voit ici : l'API est UP, le scrape prend 7 millisecondes. On collecte le nombre de requetes, la latence en histogramme, les requetes en cours, la taille des reponses.

### Slide 13 — Grafana

Et Grafana affiche tout ca dans un dashboard. Quatre panneaux : requetes par seconde, latence P95, codes de statut HTTP, requetes en cours. On a 99% de 200, une latence sous 50 millisecondes.

Le dashboard et le datasource sont provisiones automatiquement au demarrage — des fichiers YAML dans le dossier grafana/provisioning. Oussama, a toi pour la derniere phase.

---

## OUSSAMA — slides 14 a 17 + demo (~7.5 min)

### Slide 14 — Streamlit

Merci Liviu. La phase 5, l'application Streamlit. C'est l'interface utilisateur. Elle communique uniquement avec l'API deployee — pas d'acces direct aux donnees ou au modele.

Quatre pages : Accueil avec la vue d'ensemble, Contexte qui montre le statut de l'API, Data Explorer pour les distributions des categories, et Predictions pour tester le modele.

### Slide 15 — Streamlit Detail

A gauche le Data Explorer — on voit les 27 categories et leur repartition. A droite le formulaire de prediction. L'utilisateur saisit un titre, une description, et il a la reponse en moins d'une seconde. Tout passe par l'API.

C'est ce que je vais vous montrer maintenant.

### Slide 16 — Demo

<!-- ============================================================
     DEMO — ~4 min
     PRE-REQUIS : docker compose up -d tourne deja
     4 onglets ouverts : Streamlit, Grafana, MLflow, Airflow
     ============================================================ -->

Je vais vous montrer le systeme qui tourne.

<!-- ---- ETAPE 1 : STREAMLIT (~1.5 min) ----
     URL : http://localhost:8501
     Aller sur la page "Predictions"
     Saisir :
       designation = "Console Sony PlayStation 5"
       description = "PS5 edition standard 825GB SSD manette DualSense"
     Cliquer sur Predict
     Montrer : categorie predite, score de confiance, nom du modele
-->

D'abord Streamlit. Je vais classifier un produit — une PlayStation 5. Je mets le titre, la description... On lance la prediction... Et voila, le modele classe ca correctement avec un bon score de confiance. L'appel est passe par l'API en arriere-plan.

<!-- ---- ETAPE 2 : GRAFANA (~1 min) ----
     URL : http://localhost:3000
     Dashboard "FastAPI Monitoring"
     Montrer que le compteur de requetes a bouge
     Montrer la latence
-->

Si on passe sur Grafana, on voit que la requete qu'on vient de faire est tracee. Le compteur a monte, la latence est visible. C'est du temps reel, Prometheus scrape toutes les 15 secondes.

<!-- ---- ETAPE 3 : MLFLOW (~1 min) ----
     URL : http://localhost:5000
     Experiment "rakuten-text-baseline"
     Montrer les 15+ runs
     Cliquer sur le meilleur run — montrer accuracy 76.68%, f1 74.25%
     Optionnel : montrer experiment "rakuten-data-drift"
-->

MLflow maintenant. Dans l'experiment rakuten-text-baseline, nos 15 runs du sweep. Le meilleur run la — accuracy 76.68%. On peut comparer les runs, voir quels parametres ont donne quoi.

<!-- ---- ETAPE 4 : AIRFLOW (~30s) ----
     URL : http://localhost:8280 (login: airflow / airflow)
     Montrer le DAG rakuten_weekly_retraining
     Montrer un run avec les 6 taches vertes
     Si manque de temps : SKIPPER, prioriser Streamlit + Grafana
-->

Et Airflow — notre DAG de re-entrainement. Les six taches, le dernier run tout vert.

<!-- FIN DEMO — revenir sur la presentation -->

### Slide 17 — Conclusion

Pour conclure. On a couvert les cinq phases du cahier des charges — les trois obligatoires et les deux optionnelles. Mais au-dela de la checklist, ce qu'on retient c'est ce qu'on a appris en le construisant.

Gerer huit conteneurs Docker qui communiquent entre eux, c'est un vrai defi d'ingenierie. Les permissions entre conteneurs, les volumes partages, la compatibilite des librairies — on a eu des surprises. Par exemple, Evidently a change son API entre deux versions, on a du adapter notre script de drift pour utiliser scipy en fallback. C'est le genre de probleme qu'on rencontre pas dans un notebook.

L'autre apprentissage c'est que le modele, au final, c'est le composant le plus simple du systeme. Tout ce qui est autour — l'API, le monitoring, l'orchestration, la CI — c'est la ou se trouve la complexite reelle d'un projet MLOps.

En perspectives : deploiement cloud, passage a un modele deep learning, et A/B testing pour comparer les modeles en production.

Merci pour votre attention. On est prets pour vos questions.

---

## Q&A — Qui repond a quoi

| Question probable | Repond |
|---|---|
| Pourquoi SGD et pas deep learning ? | Hery — compromis perf/simplicite, architecture model-agnostic, pas de GPU |
| Comment se met a jour le modele en prod ? | Liviu — DAG Airflow, volume partage, re-entrainement hebdo |
| Comment fonctionne MLflow / le sweep ? | Johan — 15 runs, parametres alpha/loss/max_iter, meilleur alpha=1e-6 |
| Communication entre conteneurs ? | Johan — docker-compose, volumes, health checks, network bridge |
| Latence de prediction ? | Hery — < 10ms CPU, mesure via /stats |
| Detection de derive ? | Oussama — KS-test scipy, 4 features derivees du texte, logge dans MLflow |
| Pourquoi Docker Compose et pas Kubernetes ? | Johan — scope du projet, K8s overkill pour 8 services |
| Securite de l'API ? | Hery — Bearer token, variable d'env, Pydantic |
| Re-entrainement automatique ? | Liviu — cron Airflow 0 2 * * 1, valide sur VPS |
| Qu'amelioreriez-vous ? | Oussama — BERT, CD pipeline, drift sur donnees prod, A/B testing |
