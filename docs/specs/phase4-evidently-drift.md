# Phase 4c — Evidently Drift Detection (MVP Demo)

## Objectif
Rapport de drift des données avec Evidently, consultable en HTML.
Wow effect : montrer un rapport visuel qui détecte si les nouvelles données dérivent du training set.

## Architecture

```
data/preprocessed/ ---> script Evidently ---> reports/drift/drift_report.html
                                         ---> reports/drift/drift_report.json
```

## Fichiers à créer

```
src/monitoring/
  __init__.py
  drift_report.py         # Script de génération du rapport
reports/drift/
  .gitkeep                # Le rapport HTML sera généré ici
requirements-monitoring.txt
```

## Script drift_report.py

Le script :
1. Charge le dataset d'entraînement (reference) et de validation (current)
2. Génère un rapport Evidently avec DataDriftPreset
3. Sauve en HTML et JSON

```python
# Pseudo-structure
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

def generate_drift_report(reference_path, current_path, output_dir):
    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)

    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
    ])
    report.run(reference_data=reference, current_data=current)
    report.save_html(output_dir / "drift_report.html")
    report.save_json(output_dir / "drift_report.json")
```

## Intégration Airflow (optionnel bonus)

Ajouter une tâche au DAG existant :
```python
generate_drift_report = BashOperator(
    task_id="generate_drift_report",
    bash_command="python -m src.monitoring.drift_report",
    cwd=PROJECT_ROOT,
    env=COMMON_ENV,
)
# Après verify_artifacts :
verify_artifacts >> generate_drift_report
```

## Dépendances (requirements-monitoring.txt)
```
evidently>=0.4.0
pandas
scikit-learn
```

## Données utilisées
- **Reference** : `data/preprocessed/X_train_clean.csv` (données d'entraînement)
- **Current** : `data/preprocessed/X_val_clean.csv` (données de validation)
- On compare les distributions de longueur de texte, présence de description, répartition des catégories

Note : Evidently travaille sur des features tabulaires. Pour le texte, on compare des features dérivées :
- `text_length` : longueur de la désignation
- `has_description` : booléen si description non vide
- `description_length` : longueur de la description
- `word_count` : nombre de mots

## Scénario de démo
1. Ouvrir `reports/drift/drift_report.html` dans le navigateur
2. Montrer le résumé : "0 features en drift sur 4" (vert)
3. Montrer les distributions comparées
4. "Notre système détecte automatiquement si les nouvelles données dérivent du training set, ce qui déclencherait un ré-entraînement"

## Critères de succès
- [ ] `python -m src.monitoring.drift_report` génère le HTML
- [ ] Le rapport HTML est visuellement riche (graphes Evidently natifs)
- [ ] Le rapport montre la comparaison reference vs current
- [ ] Le script est intégrable dans le DAG Airflow
