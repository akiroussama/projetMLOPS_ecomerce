# Demo 5 minutes - tickets API akiroussama

Date de preparation: 9 mars 2026
Perimetre de demo: `#15`, `#19`, `#20`, `#21`

## Objectif de la demo

Montrer en 5 minutes que la brique API du projet est maintenant:
- disponible avec `GET /health` et `POST /predict`
- robuste sur les validations
- securisee par token
- testee avec couverture mesuree

## Message cle a faire passer

"Sur mon scope, j'ai transforme une base de projet sans API en un service FastAPI testable, securise et verifiable automatiquement. J'ai traite la chaine logique complete: endpoint d'inference, validation stricte, authentification, puis tests avec couverture."

## Resultat a annoncer d'entree

- `#15` implemente: API FastAPI avec `/health` et `/predict`
- `#19` implemente: validation stricte et erreurs API coherentes
- `#20` implemente: authentification Bearer sur `/predict`
- `#21` implemente: tests automatises + couverture
- Verification locale: `18 passed`, couverture `src/api = 98.67%`

## Plan de demo 5 minutes

### 0:00 -> 0:40 - Contexte

"Au depart, le depot avait des scripts d'entrainement et de prediction, mais pas de vraie API exploitable. Mon objectif etait de livrer la chaine API des checkpoints 1 et 2: exposer le modele, durcir les entrees, securiser l'acces et automatiser les tests."

### 0:40 -> 1:40 - Ticket `#15`

Montre rapidement:
- `src/api/app.py`
- `src/api/service.py`

Speech:

"J'ai d'abord cree une application FastAPI avec deux endpoints. `/health` permet de verifier l'etat du service. `/predict` expose l'inference. Le chargement du modele est isole dans un service dedie, sans reentrainement. Si les artefacts ne sont pas presents, l'API ne plante pas: elle degrade proprement et renvoie un message explicite."

### 1:40 -> 2:30 - Ticket `#19`

Montre rapidement:
- `src/api/schemas.py`
- le format d'erreur dans `src/api/app.py`

Speech:

"Ensuite j'ai durci le contrat d'entree. `designation` est obligatoire et non vide, `description` est bornee, les identifiants sont controles, et tout champ non prevu est refuse. J'ai aussi normalise les erreurs avec un schema unique: `error_code`, `message`, `details`. Ca permet d'avoir des erreurs 4xx et 5xx predictibles."

### 2:30 -> 3:20 - Ticket `#20`

Montre rapidement:
- `src/api/security.py`

Speech:

"Apres validation, j'ai securise `/predict` avec un token Bearer simple. Le token est lu depuis `API_AUTH_TOKEN`. `/health` reste public pour les checks techniques, mais `/predict` exige une authentification. Les cas d'erreur sont differencies: token absent, token invalide, ou configuration serveur manquante."

### 3:20 -> 4:20 - Ticket `#21`

Montre rapidement:
- `tests/api/test_app.py`
- `tests/api/test_service.py`
- `pytest.ini`

Speech:

"Enfin, j'ai ajoute une vraie suite de tests. Elle couvre les cas nominaux, les erreurs de validation, l'authentification, les erreurs techniques et des branches du service d'inference. J'ai ajoute un seuil de couverture a 90% sur `src/api`, et on est actuellement a 98.67%. Le workflow GitHub Actions a aussi ete ajuste pour lancer les tests sur `main` et `master`."

### 4:20 -> 5:00 - Conclusion

Speech:

"Donc sur mes quatre tickets corriges, le resultat concret est une API exploitable, durcie et testee. La suite logique pour moi est maintenant le checkpoint 3: conteneuriser ce service, l'integrer dans une stack Compose avec Airflow et MLflow, puis documenter un scenario E2E avec recovery."

## Script oral compact

Tu peux lire quasiment tel quel:

"J'ai travaille sur quatre tickets qui forment une chaine logique. Le premier etait d'exposer le modele via une API FastAPI. J'ai donc ajoute une application avec `/health` et `/predict`, et un service de chargement d'artefacts qui ne relance jamais l'entrainement. Ensuite, j'ai traite la robustesse du contrat: validation stricte du payload, champs interdits par defaut, et format d'erreur stable pour les cas invalides ou techniques. Puis j'ai securise `/predict` avec un Bearer token lu depuis une variable d'environnement, tout en laissant `/health` public pour les checks de supervision. Enfin, j'ai complete la qualite avec une suite de tests API et service, un seuil de couverture a 90%, et aujourd'hui la couverture du package `src/api` est a 98.67%. Donc ce que j'apporte au projet, c'est une base API propre, securisee et testable, qui peut maintenant etre conteneurisee et branchee au reste de la stack MLOps." 

## Commandes de demo

### Option A - Demo rapide du resultat de test

Commande:

```powershell
python -m pytest
```

Ce que tu dis:

"Ici je montre que la suite passe entierement et que la couverture minimale est enforcee. C'est la preuve la plus rapide que les tickets `#19`, `#20` et `#21` sont reellement integres."

### Option B - Demo live de l'API si les artefacts sont disponibles

Pre-requis:
- avoir les artefacts du modele dans `models/`
- definir un token serveur

Commandes:

```powershell
$env:API_AUTH_TOKEN="demo-token"
uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

Dans un second terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/predict `
  -Headers @{ Authorization = "Bearer demo-token" } `
  -ContentType "application/json" `
  -Body '{"designation":"robe ete femme","description":"robe rouge legere","productid":1001,"imageid":2002}'
```

### Option C - Plan B si les artefacts du modele ne sont pas presents

Tu peux quand meme montrer:
- `GET /health` en etat `degraded`
- un `401` si tu omets le token
- les tests automatises

Message a dire:

"Meme sans les artefacts finaux, l'API se comporte proprement. Elle ne crash pas, elle expose un etat `degraded`, et les comportements critiques sont verifies par les tests."

## What is next pour la semaine CP3

Periode visee: semaine du `9 mars 2026` au `13 mars 2026`
Tes tickets CP3 ouverts:
- `#25` Dockerfile service API
- `#26` Compose inter-services API Airflow MLflow
- `#27` Scenario E2E orchestration + recovery

### Priorite reelle

1. `#25` en premier
2. `#26` ensuite
3. `#27` en dernier

Raison:
- `#25` debloque `#26`
- `#26` debloque `#27`
- `#26` depend aussi de `#24`, qui n'est pas sur ton scope, donc il faut gerer ce risque tres tot

### Plan concret sur la semaine

#### Lundi 9 mars 2026

- Finaliser et stabiliser la branche API actuelle
- Verifier avec l'equipe l'etat de `#24` pour savoir si le service Airflow sera pret a integrer
- Commencer `#25` avec un `Dockerfile` propre pour l'API

#### Mardi 10 mars 2026

- Terminer `#25`
- Valider que le conteneur API demarre et repond sur `/health`
- Si possible, preparer les variables d'environnement a injecter pour le token et le chemin des artefacts

#### Mercredi 11 mars 2026

- Demarrer `#26`
- Poser un `docker-compose.yml` minimal avec au moins l'API et la structure des autres services
- Integrer Airflow et MLflow si `#24` est disponible

#### Jeudi 12 mars 2026

- Stabiliser `#26`
- Documenter comment demarrer la stack
- Commencer `#27` en redigeant le scenario E2E et la procedure de recovery

#### Vendredi 13 mars 2026

- Finaliser `#27`
- Verifier un run de bout en bout
- Preparer une demo CP3 courte et un runbook minimal

### Risque principal CP3

Le risque majeur est `#26`, car il depend de `#24`. Donc pour toi, l'action importante en debut de semaine est:

"je verifie tres vite si le DAG et les artefacts Airflow attendus par `#24` sont prets, sinon je livre quand meme une base Compose exploitable avec l'API et je documente clairement le blocage."

## Phrase de cloture pour ton point oral

"Cette semaine j'ai ferme la chaine API CP1-CP2. La semaine du 9 au 13 mars 2026, mon objectif est de transformer cette API en service conteneurise, puis de l'integrer dans une stack Compose pour ouvrir le checkpoint 3." 
