# Réponses préparées — Questions jury

## Q1 : Pourquoi TF-IDF + SGD et pas du deep learning ?

**Réponse courte :** Contrainte de ressources computationnelles — CPU only en local et sur VPS (4 vCPU, 8 GB RAM). Entraîner un CamemBERT prend plusieurs heures sur GPU et 24h+ sur CPU.

**Réponse complète :**
> "Nous avons fait un choix délibéré de prioriser l'infrastructure MLOps sur la performance brute du modèle. Un pipeline MLflow + Airflow + CI/CD + monitoring autour d'un modèle faible démontre mieux les compétences MLOps qu'un notebook BERT sans pipeline.
>
> Notre architecture est **model-agnostic** : dans `train_model.py`, remplacer `SGDClassifier` par un fine-tuned DistilBERT ne nécessite que de changer la classe du modèle. Le pipeline MLflow, les DAGs Airflow, les métriques Prometheus et l'API FastAPI restent identiques. C'est précisément l'intérêt d'une architecture MLOps découplée."

**Scores de référence (Rakuten challenge) :**
- TF-IDF + SGD (nous) : ~76% accuracy
- TF-IDF + SVM : ~78%
- CamemBERT (texte only) : ~85%
- Fusion texte + image (SOTA) : ~91%

---

## Q2 : Pourquoi vous n'utilisez pas les images ?

**Réponse :**
> "Le dataset Rakuten contient en effet des images produits. Une architecture multimodale (BERT + ResNet) atteint ~91% en compétition. Nous avons écarté cette approche pour deux raisons :
> 1. Les images ne sont pas disponibles dans la version publique S3 que nous utilisons — seuls les features texte et les identifiants d'images sont présents.
> 2. L'inférence multimodale nécessite une infrastructure GPU pour rester sous 200ms de latence, incompatible avec notre VPS CPU.
>
> Notre API est prête pour l'extension : le `PredictRequest` inclut déjà `imageid` comme champ, et `service.py` peut être étendu pour charger un modèle d'image en parallèle."

---

## Q3 : Comment fonctionne le pipeline MLOps de bout en bout ?

**Réponse :**
```
[Données S3] → Airflow DAG (hebdo, lundi 2h UTC)
    ├── download_raw_data    : télécharge X_train, X_test, Y_train depuis S3
    ├── prepare_dataset      : nettoyage, déduplication, split 80/20 stratifié
    ├── build_features       : TF-IDF 5000 features, unigrammes + bigrammes
    ├── train_model          : SGDClassifier, log MLflow (params + métriques + artifacts)
    ├── verify_artifacts     : vérifie que baseline_model.pkl + vectorizer sont présents
    └── generate_drift_report: Evidently (DataDrift + DataQuality)

[MLflow] → tracking des expériences, comparaison des runs, model registry
[FastAPI] → inference, auth Bearer token, /health, /metrics (Prometheus), /stats
[Prometheus] → scrape /metrics toutes les 15s
[Grafana] → dashboards temps réel (latence, throughput, error rate)
[Streamlit] → interface utilisateur grand public
[GitHub Actions] → CI : lint + tests à chaque push/PR
```

---

## Q4 : Votre API est-elle sécurisée ?

**Réponse :**
> "L'endpoint `/predict` est protégé par Bearer token avec validation HMAC timing-safe (pas de timing attack). Le token est injecté via variable d'environnement (`API_AUTH_TOKEN`) et ne se trouve jamais dans le code. `/health` et `/metrics` sont publics car ils ne retournent pas de données sensibles."

---

## Q5 : Comment gérez-vous le drift ?

**Réponse :**
> "Le DAG Airflow inclut une tâche `generate_drift_report` qui utilise Evidently pour comparer la distribution de données de référence (train) vs courante (validation). Nous mesurons 4 features dérivées du texte : longueur de désignation, longueur de description, présence d'une description, nombre de mots. Le rapport HTML est loggé dans MLflow comme artifact. En production, si le drift score dépasse un seuil, le DAG peut être étendu pour déclencher un re-entraînement automatique."

---

## Q6 : Pourquoi Airflow et pas une simple crontab ?

**Réponse :**
> "Une crontab ne donne ni visibilité, ni retry, ni alerting. Airflow nous permet de :
> - Voir l'historique d'exécution de chaque tâche individuellement (DAG view)
> - Configurer des retries automatiques avec backoff
> - Recevoir des alertes en cas d'échec
> - Trigger manuellement un re-entraînement depuis l'UI sans modifier le code
> - Modéliser les dépendances entre tâches (ex : `build_features` ne tourne que si `prepare_dataset` a réussi)"

---

## Q7 : Qu'est-ce que vous amélioreriez si vous aviez plus de temps ?

**Réponse (honnête) :**
1. **Modèle** : Fine-tuning CamemBERT/DistilBERT (prévu, bloqué par compute)
2. **CD pipeline** : Déploiement automatique quand un nouveau modèle dépasse le score courant en MLflow (MLflow Model Registry + webhook)
3. **Drift en production** : Logguer les inputs réels de l'API pour construire un dataset "production" et détecter le drift sur des données réelles (pas juste train vs val)
4. **Tests de charge** : Locust pour valider les SLAs sous load
5. **Multimodal** : Fusion image (ResNet50) + texte (TF-IDF) sans GPU, en deux étapes d'inférence séparées
