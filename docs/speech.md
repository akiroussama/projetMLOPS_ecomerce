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

## OUSSAMA — slides 14 a 25 + demo live (~9-10 min)

### SLIDE 14 — Transition Demo (~30s)

Merci Liviu. Plutot que de vous montrer des slides sur Streamlit, je vais vous montrer le vrai systeme. Tout ce que vous allez voir tourne en direct sur un VPS Hetzner — quatre vCPU, huit giga de RAM, huit conteneurs Docker.

Mais d'abord, un mot sur pourquoi Streamlit. On avait le choix : Gradio, Dash, React. On a choisi Streamlit parce que c'est du Python pur. Pas de JavaScript, pas de HTML, pas de callbacks. Un data scientist peut construire une interface en une heure. Et surtout — Streamlit ne charge aucun modele. C'est un client HTTP pur. Tout passe par l'API. Si le modele change dans l'API, Streamlit n'a rien a modifier. Zero couplage.

On y va.

### SLIDE 15 — Demo Streamlit Accueil (~40s)

<!-- OUVRIR ONGLET : http://rakuten-mlops.duckdns.org:8501 -->

Voila la page d'accueil. Premiere chose qu'on voit — ce bandeau vert en bas : "API connectee, modele SGDClassifier charge avec succes". C'est pas un texte statique. A chaque chargement de page, Streamlit fait un appel au endpoint /health de l'API. Si l'API tombe, le bandeau passe en orange ou en rouge. L'utilisateur sait immediatement si le systeme est operationnel. Pas de surprise au moment de predire.

On a trois pages : Contexte pour comprendre le projet, Data Explorer pour les donnees, et Predictions pour tester le modele. Je vais d'abord vous montrer le Contexte.

### SLIDE 16 — Demo Page Contexte (~40s)

<!-- CLIC SUR : Page Contexte dans la sidebar -->

Cette page resume le projet pour un nouveau venu — c'est de la documentation vivante. En haut, le schema d'architecture : Airflow orchestre, FastAPI sert, MLflow tracke, Grafana affiche. En dessous, quatre metriques en direct depuis l'API : le statut, le nom du modele charge, si le modele est operationnel, et le nombre de categories. Et tout en bas, les liens vers chaque service. Un developpeur qui decouvre le projet peut comprendre l'architecture et acceder a chaque outil en trente secondes.

On passe aux predictions.

### SLIDE 18 — Demo Prediction 1 : FIFA 24 (~1min)

<!-- CLIC SUR : Page Predictions → Bouton "Jeu PS5" dans les exemples rapides -->

Je vais utiliser un exemple pre-rempli — FIFA 24 sur PS5. On clique... La designation et la description sont remplies automatiquement. On lance la prediction.

<!-- CLIC SUR : Predire la categorie -->

Et voila. Le modele classe ca en "Jeux video / Consoles", code 40. La jauge de confiance affiche environ 36%. Ca semble faible, mais reflechissez : on a 27 categories. La baseline aleatoire, c'est un sur vingt-sept — soit 3.7%. Notre modele est dix fois meilleur que le hasard sur cette prediction. Avec 27 classes, un modele lineaire n'aura jamais 95% de confiance comme un modele binaire. Le fait qu'il donne la bonne categorie a 36%, c'est en realite excellent.

Et le temps de reponse — regardez en bas. C'est Prometheus qui mesure, pas nous : 5 millisecondes en moyenne par prediction. C'est un choix delibere. Un BERT prendrait 200 millisecondes sur CPU. On a privilegie la latence.

### SLIDE 19 — Demo Prediction 2 : Harry Potter (~45s)

<!-- CLIC SUR : Bouton "Livre" dans les exemples rapides, OU saisir manuellement -->
<!-- Designation : "Harry Potter a l'ecole des sorciers - JK Rowling"
     Description : "Premier tome de la saga Harry Potter. Edition de poche." -->

Maintenant, quelque chose de completement different. Un livre — Harry Potter. On lance...

<!-- CLIC SUR : Predire la categorie -->

Categorie predite : Livres. Deux domaines completement differents — jeux video et livres — et le modele gere les deux. C'est le benefice du TF-IDF sur des textes courts. Le vocabulaire de "FIFA PlayStation manette" et celui de "Harry Potter edition poche" sont suffisamment distincts pour que meme un modele lineaire les separe. Et en bas, on voit l'historique de session — les deux predictions sont tracees avec leur confiance et leur temps de reponse.

### SLIDE 20 — Demo Swagger API (~40s)

<!-- OUVRIR ONGLET : http://rakuten-mlops.duckdns.org:8200/docs -->

Passons a l'API directement. Voila la documentation Swagger. Elle est auto-generee par FastAPI — zero effort de notre part. Quatre endpoints : /health public, /predict protege par un cadenas — vous le voyez la, c'est le Bearer token — /metrics pour Prometheus, et /stats pour les metriques metier.

Un developpeur qui veut integrer notre API n'a meme pas besoin de nous contacter. Il ouvre cette page, il voit les schemas d'entree et de sortie, il peut tester directement. Integration en cinq minutes. C'est ca la force de FastAPI par rapport a Flask : la documentation est native, pas un ajout.

### SLIDE 21 — Demo Grafana (~1min)

<!-- OUVRIR ONGLET : http://rakuten-mlops.duckdns.org:3000/d/rakuten-api-monitoring -->

Et maintenant le monitoring. Quatre panneaux Grafana. En haut a gauche, les requetes par seconde — on voit les pics quand j'ai fait les predictions. En haut a droite, la latence P95 — c'est sous 50 millisecondes. Ca veut dire que 95% des requetes sont plus rapides que ca. On utilise le percentile 95 plutot que la moyenne parce que la moyenne masque les outliers.

En bas a gauche, la distribution des codes HTTP — 99% de 200, quelques 400 qui sont nos tentatives sans token. Ca prouve que la securite fonctionne. Et en bas a droite, le total des predictions.

Quelques chiffres concrets. L'API tourne depuis 83 heures sans restart. 144 megaoctets de RAM pour servir 1396 predictions. C'est leger.

Un detail technique : Prometheus utilise le pull — c'est lui qui va chercher les metriques toutes les 5 secondes. L'API n'a pas besoin de connaitre Prometheus. On pourrait ajouter dix services a monitorer sans modifier une seule ligne dans l'API. Ce dashboard est provisionne automatiquement — des fichiers YAML dans Git. Au premier docker compose up, Grafana a deja tout. Zero clic dans l'interface.

### SLIDE 22 — Demo MLflow (~1min)

<!-- OUVRIR ONGLET : http://rakuten-mlops.duckdns.org:5000/#/experiments/1 -->

MLflow maintenant. Nos experiences d'entrainement. On voit ici 27 runs au total. 19 termines, 8 echecs. Les echecs ne sont pas des erreurs — c'est notre processus d'iteration. On a teste des configurations qui ne marchaient pas. MLflow garde tout, meme les runs qui ne marchent pas. C'est ca la reproductibilite : dans six mois, si on se demande "quel alpha avait donne 76%?", la reponse est ici, pas dans un notebook perdu.

Le meilleur run — je clique dessus — alpha egal a un fois dix puissance moins six, 76.68% d'accuracy, 74.25% de F1 macro. Et si on descend dans les artefacts : quatre fichiers. Le modele a 1 megaoctet, le vectorizer TF-IDF a 190 kilooctets, le rapport de classification detaille par classe, et les metriques aggregees. Chaque run est atomique — parametres, metriques, artefacts. On peut revenir a n'importe quel run et recharger exactement ce modele.

### SLIDE 23 — Demo Airflow (~45s)

<!-- OUVRIR ONGLET : http://rakuten-mlops.duckdns.org:8280/dags/rakuten_weekly_retraining/grid -->

Airflow — notre pipeline de re-entrainement. Le DAG rakuten_weekly_retraining, six taches. Telechargement S3, preparation du dataset, construction des features TF-IDF, entrainement avec log MLflow, verification des artefacts, et rapport de derive.

Ce qu'on voit ici — le dernier run, tout vert. Six taches en succes. Le pipeline complet s'execute en 60 secondes. Le goulot c'est la vectorisation TF-IDF a 39 secondes sur 84 000 textes — c'est les deux tiers du temps total. Si on passe a BERT demain, ce sera le training qui domine, pas la vectorisation. Identifier le goulot, c'est la premiere etape pour optimiser. Et les quatre runs en echec au-dessus, c'etait des problemes de permissions entre conteneurs — Airflow ecrivait avec un UID, MLflow attendait un autre. C'est le genre de probleme qu'on ne rencontre jamais en local.

### SLIDE 24 — GitHub Actions CI/CD (~1min30)

<!-- Revenir sur les slides -->

On quitte la demo pour parler d'integration continue. Notre pipeline GitHub Actions — trois etapes sequentielles : lint avec flake8, tests avec pytest, et build Docker. Si une etape echoue, la suivante ne se lance pas. Si ca casse, on ne merge pas.

Et le CI a attrape de vraies erreurs. Je vais vous raconter trois scenarios.

Premier scenario : un import inutilise. Un commit ajoute import os et import sys pour du logging structure. Le code compile, il tourne, mais flake8 detecte F401 — "imported but unused". Le CI passe au rouge. Le linter attrape le code mort avant meme que les tests ne se lancent. C'est de l'hygiene de code automatisee.

Deuxieme scenario, plus subtil : un contrat API casse. Un commit renomme le statut du health check de "ok" vers "healthy". Ca parait logique. Mais nos tests assertent que le body contient status egal "ok". Et surtout, le healthcheck Docker aussi verifie cette valeur. Pytest attrape ca immediatement. Le CI nous force a revert. Les tests ne verifient pas juste le code — ils protegent le contrat entre les services.

Troisieme scenario : une reference indefinie. Un commit ajoute le calcul de la latence mediane avec statistics.median, mais oublie le import statistics. Le code est syntaxiquement correct. Python ne le detecte qu'a l'execution. Mais flake8 detecte F821 — "undefined name statistics" — avant meme que le code ne soit execute. Le linter detecte une erreur de runtime avant l'execution. Sans CI, ca aurait crashe l'API en production.

A chaque fois, le pattern est le meme : commit, CI rouge, fix, CI vert. C'est ca l'interet — transformer les erreurs en feedback rapide au lieu de bugs en production.

### SLIDE 25 — Conclusion (~1min)

Pour conclure. On a couvert les cinq phases du projet — les trois obligatoires et les deux optionnelles. Mais au-dela de la checklist, ce qu'on retient c'est ce qu'on a appris en le construisant.

Gerer huit conteneurs qui communiquent, avec les permissions, les volumes partages, la compatibilite des librairies — c'est un vrai defi d'ingenierie. Evidently a change son API entre deux versions, on a du basculer sur scipy en fallback pour le drift. Le healthcheck Docker qui depend du format exact de la reponse JSON. Les UID Airflow qui bloquent MLflow. C'est le genre de probleme qu'on ne rencontre jamais dans un notebook.

Si on avait plus de temps, quatre pistes. D'abord, deploiement cloud — l'architecture est cloud-agnostic, on pourrait migrer sur AWS ou GCP sans changer le code. Ensuite, passage a BERT — l'architecture est model-agnostic, on change une classe dans train_model.py, tout le reste — API, Airflow, MLflow, monitoring — ca reste pareil. Puis, A/B testing pour comparer les modeles en production, pas juste en validation. Et surtout, logger les inputs reels des utilisateurs pour detecter le drift sur des donnees de production, pas de validation. Parce que c'est la que le drift se produit vraiment.

Le modele, c'est le composant le plus simple du systeme. Tout ce qui est autour — l'API, le monitoring, l'orchestration, la CI, le deploiement — c'est ca le MLOps. Et c'est ce qu'on a appris a construire.

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
