import pandas as pd
import joblib
import numpy as np
import os
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, recall_score
import sys
import mlflow


def train_model(X_train=None, X_val=None, y_train=None, y_val=None, feats=None):
    # Define mlflow tracking_uri
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    # Define experiment name, run name and artifact_path name
    rakuten_experiment = mlflow.set_experiment("Rakuten Models")
    RUN_NAME = "SVC-C0.5"
    ARTIFACT_PATH = "rf_rakuten"

    if any(e is None for e in [X_train, X_val, y_train, y_val, feats]):
        sys.path.append('src/features/')
        import build_features

        print('Features not supplied. Building features...')
        X_train, X_val, y_train, y_val, feats = build_features.main(preprocessed_path = "data/preprocessed")

    params = {
        "C": 0.5
    }

    clf = LinearSVC(**params)

    model = Pipeline([
        ("feats", feats),
        ("clf", clf),
    ])

    # Train model
    print("Entrainement du modele final...")
    model.fit(X_train, y_train)
    
    # Evaluate model
    print("Evaluate model...")
    y_pred = model.predict(X_val)
    mod_f1_score = f1_score(y_val, y_pred, average='weighted')
    mod_recall_score = recall_score(y_val, y_pred, average='weighted')
    metrics = {"f1_score": mod_f1_score, "recall_score": mod_recall_score}
    print(metrics)

    # Store information in tracking server
    print("Storing information in MLflow tracking server...")
    with mlflow.start_run(run_name=RUN_NAME) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            sk_model=model, 
            input_example=pd.DataFrame(X_val), 
            artifact_path=ARTIFACT_PATH
        )

    print(f"Finished MLflow run ({RUN_NAME}).")

    os.makedirs("models/artifacts", exist_ok=True)
    joblib.dump(model, "models/artifacts/model_final.joblib")
    print("Model saved as joblib.")

if __name__ == "__main__":
    train_model()
