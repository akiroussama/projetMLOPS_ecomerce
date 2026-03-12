# Plan MLOps vers la soutenance (24/03/2026)

## Objectif final
- Livrer un modele ML pertinent metier, expose via API, consomme par une interface Streamlit.
- Montrer une chaine MLOps complete: tracking, orchestration, conteneurisation, qualite logicielle.
- Defendre des conclusions business et scientifiques claires pendant la soutenance.

## Checkpoint 0 - Cadrage et preparation du call
- Periode: 24/02/2026 -> 25/02/2026 (17h00)
- Objectifs:
  - Verrouiller le cas d'usage metier, la cible ML et le niveau de complexite choisi.
  - Definir 1 metrique metier principale et 2-3 metriques ML de reference.
  - Valider le perimetre du MVP a livrer au 27/02.
- Livrables:
  - Note de cadrage (1 page): probleme, donnees, KPI metier, KPI ML, contraintes.
  - Backlog priorise des taches de phase 1.
- Critere de validation:
  - Equipe alignee sur ce qui est montre au point du 27/02.

## Checkpoint 1 - Fondations et API basique
- Deadline: 27/02/2026
- Objectifs:
  - Disposer d'un premier pipeline data + modele baseline operationnel.
  - Exposer le modele via une API FastAPI testable.
- Livrables:
  - Scripts Python: collecte, nettoyage, feature engineering.
  - Modele baseline entraine + artefact sauvegarde (joblib/pickle).
  - API FastAPI avec `/health` et `/predict`.
  - Test manuel via Swagger ou requete HTTP.
- Critere de validation:
  - Une prediction fonctionne sans reentrainement.
  - Les metriques baseline sont documentees.

## Checkpoint 2 - Tracking et robustesse API
- Deadline: 06/03/2026
- Objectifs:
  - Suivre et comparer les experimentations ML proprement.
  - Renforcer la fiabilite et la securite de l'API.
- Livrables:
  - MLflow active (params, metrics, artefacts, model version candidate).
  - Tableau de comparaison des runs (baseline vs variantes).
  - Tests unitaires API (routes, schemas, erreurs).
  - Authentification simple (token) + validation stricte des entrees.
- Critere de validation:
  - Choix du meilleur modele justifiable sur evidences MLflow.
  - API stable sur les cas normaux et cas d'erreur.

## Checkpoint 3 - Orchestration et microservices
- Deadline: 13/03/2026
- Objectifs:
  - Automatiser la mise a jour data/modele.
  - Isoler les composants en services conteneurises.
- Livrables:
  - DAG Airflow: ingestion -> preprocessing -> entrainement -> enregistrement.
  - Services separes: API, training, MLflow, Airflow.
  - Dockerfiles + `docker-compose.yml` orchestration locale.
- Critere de validation:
  - `docker compose up` demarre l'ensemble.
  - Un run orchestration complet s'execute sans intervention manuelle.

## Checkpoint 4 - Stabilisation finale (optionnels prioritaires)
- Deadline: 20/03/2026
- Objectifs:
  - Verrouiller la qualite logicielle et l'observabilite pour la demo.
  - Finaliser la documentation et l'interface utilisateur.
- Livrables:
  - CI GitHub Actions: tests automatiques + build d'image.
  - Monitoring minimal (Prometheus/Grafana) sur API.
  - Detection de derive (Evidently) sur donnees/predictions.
  - Documentation technique (architecture, runbook, limites).
  - Application Streamlit branchee exclusivement sur l'API.
- Critere de validation:
  - Pipeline CI verte.
  - Dashboard de suivi visible.
  - Demo Streamlit stable sur un scenario metier.

## Checkpoint 5 - Preparation soutenance
- Periode: 21/03/2026 -> 24/03/2026
- Objectifs:
  - Transformer le travail technique en narration claire et convaincante.
  - Securiser la demo et les reponses aux questions du jury.
- Livrables:
  - Slides 20 minutes (probleme -> methode -> resultats -> limites -> suites).
  - Script demo 6-8 minutes + plan B en cas de panne.
  - FAQ Q/R jury (choix metriques, architecture, monitoring, limites).
- Critere de validation:
  - 2 repetitions completes chronometrees.
  - Equipe prete pour 10 minutes de Q/R.

## Rythme de pilotage recommande
- Lundi: point equipe 30 minutes (priorites et risques).
- Jeudi: revue technique 30 minutes (qualite, blocages, avancement).
- Veille de deadline: repetition demo 20 minutes.

## Definition of Done globale (avant soutenance)
- Pipeline reproductible de bout en bout.
- API et interface utilisateur demonstrables en direct.
- Resultats metier et ML traces, compares, et expliques.
- Documentation suffisante pour relancer le projet rapidement.
