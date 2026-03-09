# Bilan des tickets assignes a akiroussama

Date de reference: 9 mars 2026

Perimetre de ce document:
- Tickets utilisateur assignes a `@akiroussama`
- Issues suivies: `#10`, `#15`, `#19`, `#20`, `#21`, `#25`, `#26`, `#27`, `#30`, `#33`, `#36`, `#39`
- Base d'analyse: contenu des issues GitHub + etat du depot local sur la branche `codex/issue-15-fastapi-endpoints`

Important:
- Ce document decrit ce qui est visible dans le depot local au moment de l'analyse.
- Si du travail existe dans un autre repo, dans des slides, sur un board, ou en local non versionne, il n'apparait pas forcement ici.

## Vue rapide

| Ticket | Statut estime | Resume |
| --- | --- | --- |
| `#10` | Partiel | Le backlog et le plan existent, mais pas de RACI formel ni d'architecture cible explicite. |
| `#15` | Implemente localement | API FastAPI `health/predict` ajoutee sur la branche de travail, avec chargement d'artefacts sans reentrainement. |
| `#19` | Implemente localement | Validation stricte du payload et format d'erreur API coherent ajoutes sur la branche de travail. |
| `#20` | Implemente localement | Authentification Bearer token ajoutee sur `/predict` avec secret en variable d'environnement. |
| `#21` | Implemente localement | Suite de tests API et seuil de couverture `90%` ajoutes sur `src/api`, avec execution automatisee via pytest. |
| `#25` | Non demarre | Le service API existe, mais aucun `Dockerfile` n'est encore present. |
| `#26` | Bloque | Aucun `docker-compose.yml`, pas de stack API/Airflow/MLflow visible. |
| `#27` | Bloque | Aucun scenario E2E ni procedure de recovery documentee dans le repo. |
| `#30` | Partiel | Une CI lint/tests existe, mais pas de build/push d'images Docker. |
| `#33` | Non demarre | Aucune application Streamlit n'est visible dans le repo. |
| `#36` | Non demarre | Le besoin est note dans le plan, mais le script de demo et le plan B ne sont pas encore rediges. |
| `#39` | Non demarre | Rien de concret dans le repo a ce stade, depend de tickets de repetition encore ouverts. |

## Ticket `#10` - Architecture cible + backlog + RACI

**Objectif**

Produire l'architecture MLOps cible, un backlog priorise et une repartition RACI de sprint.

**Ce qui est fait**

- Le document `ROADMAP_PHASE1_PHASE2_TEAM.md` pose une repartition par personne et une sequence anti-blocage pour les phases 1 et 2.
- Le document `PLAN_MLOPS_CHECKPOINTS_SOUTENANCE.md` structure le backlog jusqu'a la soutenance, checkpoint par checkpoint.
- Le document `DATA_CONTRACT.md` couvre une dependance importante de cadrage sur le schema des donnees.

**Reste a faire**

- Formaliser une vraie architecture cible: composants, flux de donnees, services, artefacts et outillage.
- Ajouter une matrice RACI explicite par ticket ou par sprint.
- Eventuellement consolider ces elements dans un document unique de cadrage versionne et facile a presenter.

## Ticket `#15` - FastAPI endpoints health et predict

**Objectif**

Implementer une API FastAPI avec chargement du modele sans reentrainement et un endpoint de prediction.

**Ce qui est fait**

- Une application FastAPI a ete ajoutee dans `src/api/app.py`.
- Les endpoints `GET /health` et `POST /predict` sont exposes.
- Le chargement des artefacts d'inference est isole dans `src/api/service.py`, sans relancer l'entrainement.
- Les schemas d'entree et de sortie sont definis dans `src/api/schemas.py`.
- Des tests API ont ete ajoutes dans `tests/api/test_app.py`.
- Le `README.md` documente le lancement via `uvicorn`.
- `requirements.txt` inclut `fastapi` et `uvicorn`.

**Reste a faire**

- Brancher l'API sur les vrais artefacts du modele baseline final si besoin.
- Verifier le contrat d'entree final si l'inference doit devenir plus riche que le simple texte.
- Committer et merger la branche une fois le ticket valide.

## Ticket `#19` - Validation stricte des payloads API

**Objectif**

Durcir la validation des entrees/sorties et rendre les erreurs 4xx/5xx predictibles.

**Ce qui est fait**

- Le schema `PredictRequest` refuse les champs non prevus.
- `designation` est maintenant une chaine obligatoire, stricte, nettoyee et non vide.
- `description`, `productid` et `imageid` sont valides avec des types et bornes plus stricts.
- L'API retourne un format d'erreur coherent avec `error_code`, `message` et `details`.
- Les erreurs de validation renvoient `422 VALIDATION_ERROR`.
- Les erreurs de service connues renvoient des codes stables comme `503 MODEL_NOT_READY` et `500 PREDICTION_FAILED`.

**Reste a faire**

- Ajouter au besoin des validations metier supplementaires si le contrat produit evolue.
- Verifier si le format d'erreur doit etre aligne avec une convention d'equipe plus large.
- Connecter ensuite ce contrat strict a l'authentification du ticket `#20`.

## Ticket `#20` - Authentification token + securisation API

**Objectif**

Ajouter une authentification par token et proteger les routes sensibles.

**Ce qui est fait**

- La route `/predict` est maintenant protegee par un token Bearer.
- Le secret serveur est lu depuis la variable d'environnement `API_AUTH_TOKEN`.
- La route `/health` reste publique pour les checks d'etat.
- Les erreurs d'authentification sont stabilisees:
  - `401 AUTHENTICATION_REQUIRED` si le header manque
  - `403 INVALID_TOKEN` si le token est faux
  - `503 AUTH_NOT_CONFIGURED` si le serveur n'a pas de token configure
- Des tests automatises couvrent les cas token absent, invalide et valide.

**Reste a faire**

- Etendre cette protection aux futures routes sensibles si l'API s'elargit.
- Prevoir la gestion des secrets en environnement Docker/CI quand les tickets `#25`, `#26` et `#30` seront traites.
- Si besoin, remplacer ce mecanisme simple par une auth plus complete plus tard.

## Ticket `#21` - Tests unitaires API + couverture

**Objectif**

Ajouter une suite de tests unitaires API et un seuil minimum de couverture.

**Ce qui est fait**

- Un premier fichier de tests API existe: `tests/api/test_app.py`.
- Des tests unitaires complementaires existent aussi dans `tests/api/test_service.py`.
- Les cas couverts actuellement sont:
  - `GET /health` avec modele disponible
  - `POST /predict` sur un cas nominal
  - `POST /predict` quand le modele est absent
  - `POST /predict` avec payload invalide: champ manquant, champ en trop, champ vide, type invalide
  - `POST /predict` avec erreur technique stable cote API
  - `POST /predict` avec auth manquante, invalide ou non configuree
- Des branches techniques sont aussi couvertes:
  - chargement de service sans etat preinitialise
  - modele de type pipeline sans vectorizer externe
  - mapping pickle
  - erreur HTTP classique et erreur non geree
- Une CI de base existe deja dans `.github/workflows/python-app.yml` pour lancer `pytest` et `flake8`.
- Un seuil de couverture minimal de `90%` sur `src/api` est maintenant impose via `pytest.ini`.

**Reste a faire**

- Si l'API s'etend, maintenir ce seuil en ajoutant les tests associes aux nouvelles routes.
- Eventuellement affiner les rapports de couverture si l'equipe veut publier `coverage.xml` dans la CI.

## Ticket `#25` - Dockerfile service API

**Objectif**

Conteneuriser le service API avec une image reproductible et un demarrage stable.

**Ce qui est fait**

- L'API existe maintenant et son point d'entree est clair: `src/api/app.py`.
- La commande de demarrage est documentee dans `README.md`.
- Les dependances du service API sont declarees.

**Reste a faire**

- Creer un `Dockerfile` pour l'API.
- Gerer l'installation des dependances, la copie du code et l'exposition du port.
- Injecter proprement les variables d'environnement et les artefacts du modele.
- Verifier le demarrage du conteneur sans erreur.

## Ticket `#26` - Compose inter-services API Airflow MLflow

**Objectif**

Assembler les services via Docker Compose avec reseaux, volumes et variables d'environnement.

**Ce qui est fait**

- Le besoin est mentionne dans `PLAN_MLOPS_CHECKPOINTS_SOUTENANCE.md`.
- Aucune stack multi-service n'est encore visible dans le repo.

**Reste a faire**

- Creer un `docker-compose.yml` ou `compose.yaml`.
- Declarer les services API, Airflow, MLflow et leurs dependances.
- Configurer reseaux, volumes persistants et variables d'environnement.
- Aligner ce travail avec le ticket `#24` sur le DAG Airflow.

## Ticket `#27` - Scenario E2E orchestration + recovery

**Objectif**

Verifier un run complet de bout en bout et documenter une procedure simple de reprise en cas d'echec.

**Ce qui est fait**

- Aucun scenario E2E executable n'est visible dans le repo.
- Aucune procedure de recovery n'est encore documentee.

**Reste a faire**

- Definir le parcours E2E complet: ingestion -> preprocessing -> entrainement -> exposition -> prediction.
- Ajouter une procedure de reprise simple si un service ne demarre pas ou si un run echoue.
- Documenter les checks a lancer avant une demo ou une relance.

## Ticket `#30` - CI build/push images Docker

**Objectif**

Automatiser le build et le push des images Docker dans la CI avec des tags coherents.

**Ce qui est fait**

- Le repo contient deja un workflow GitHub Actions de base dans `.github/workflows/python-app.yml`.
- Ce workflow couvre l'installation des dependances, le lint et les tests.

**Reste a faire**

- Ajouter les etapes de build Docker.
- Ajouter l'authentification au registre et le push des images.
- Definir une strategie de tags coherent.
- Aligner cette CI avec les futurs fichiers de conteneurisation (`#25` et `#26`).

## Ticket `#33` - Streamlit connecte exclusivement a l'API

**Objectif**

Developper une application Streamlit multi-onglets qui passe uniquement par l'API deployee.

**Ce qui est fait**

- Aucune application Streamlit n'est visible dans le depot.
- L'API `health/predict` de `#15` fournit maintenant une base consommable par une UI.

**Reste a faire**

- Creer l'application Streamlit.
- La brancher uniquement en HTTP sur l'API, sans acces direct au modele local.
- Ajouter la gestion du token quand `#20` sera en place.
- Prevoir un flux de demo stable pour la soutenance.

## Ticket `#36` - Script demo live + plan B

**Objectif**

Ecrire un script de demo robuste et un scenario de secours en cas d'incident.

**Ce qui est fait**

- Le besoin apparait dans `PLAN_MLOPS_CHECKPOINTS_SOUTENANCE.md`.
- La base API necessaire a la demo commence a exister via `#15`.

**Reste a faire**

- Rediger un script de demo pas a pas.
- Definir un plan B si l'API, la stack ou l'UI tombent pendant la demo.
- S'appuyer sur `#33` et `#27` pour construire un scenario reproductible.

## Ticket `#39` - Repetition 2 + freeze final

**Objectif**

Realiser la repetition finale et figer la version de demo/presentation.

**Ce qui est fait**

- Rien de concret n'est visible dans le depot pour ce ticket a ce stade.

**Reste a faire**

- Mener la repetition finale avec l'equipe.
- Integrer les derniers ajustements.
- Geler la version de demo et la version de presentation.
- Lever les dependances `#37` et `#38` avant cloture.

## Conclusion courte pour l'oral

Si tu dois le presenter rapidement:

- J'ai pose la partie cadrage de mon scope avec les documents de roadmap et de plan, mais il manque encore une architecture cible et un vrai RACI formel.
- J'ai implemente la base technique de l'API FastAPI avec `health` et `predict`, sans reentrainement, avec une premiere couche de tests.
- J'ai ensuite durci l'API avec une validation stricte du payload et un format d'erreur coherent, et j'ai etendu les tests en consequence.
- J'ai ensuite securise `/predict` avec une authentification Bearer simple basee sur une variable d'environnement.
- J'ai enfin mis en place une vraie suite de tests API avec seuil de couverture a `90%` sur `src/api`.
- Les prochains chantiers logiques sont la conteneurisation `#25`, `#26`, puis la partie UI/demo `#33`, `#36`, `#39`.
