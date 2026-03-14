import json
import pandas as pd
import mlflow
from pathlib import Path
import os
from features.build_features import build_features


def display_artifacts(client, run_id):
    """
    Display all artifacts in a run.
    
    Args:
        client: MLflow client
        run_id: ID of the run to inspect
    """
    artifacts = client.list_artifacts(run_id)
    print("\nAvailable artifacts:")
    for idx, artifact in enumerate(artifacts, 1):
        print(f"{idx}. {artifact.path} {'(dir)' if artifact.is_dir else '(file)'}")
        if artifact.is_dir:
            nested_artifacts = client.list_artifacts(run_id, artifact.path)
            for nested in nested_artifacts:
                print(f"   - {nested.path}")
    return artifacts


def select_model_path(artifacts):
    """
    Let user select which artifact directory to use for model registration.
    
    Args:
        artifacts: List of artifacts
    Returns:
        str: Selected artifact path
    """
    # Filter only directories
    dirs = [art for art in artifacts if art.is_dir]
    
    if not dirs:
        raise Exception("No directories found in artifacts")
    
    if len(dirs) == 1:
        return dirs[0].path
        
    print("\nMultiple model directories found. Please select one:")
    for idx, dir_artifact in enumerate(dirs, 1):
        print(f"{idx}. {dir_artifact.path}")
        
    while True:
        try:
            choice = int(input("\nEnter the number of your choice: "))
            if 1 <= choice <= len(dirs):
                return dirs[choice-1].path
            print(f"Please enter a number between 1 and {len(dirs)}")
        except ValueError:
            print("Please enter a valid number")


def get_model_uri(tracking_uri, experiment_name, run_id=None):
    """
    Get model URI either from a specific run_id or the latest successful run in an experiment.
    """
    mlflow.set_tracking_uri(tracking_uri)
    print(f"Using tracking URI: {tracking_uri}")
    
    # Get experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    if experiment is None:
        experiments = mlflow.search_experiments()
        available_experiments = [exp.name for exp in experiments]
        raise Exception(f"Experiment '{experiment_name}' not found. Available experiments: {available_experiments}")
    
    if run_id:
        print(f"Loading model from run ID: {run_id}")
    else:
        print(f"Loading latest successful model from experiment: {experiment_name}")
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="status = 'FINISHED'",
            order_by=["start_time DESC"],
            max_results=1
        )
        if runs.empty:
            raise Exception(f"No successful runs found in experiment '{experiment_name}'")
        run_id = runs.iloc[0].run_id
        print(f"Found latest run ID: {run_id}")
    
    # Get run information and artifacts
    client = mlflow.tracking.MlflowClient()
    artifacts = display_artifacts(client, run_id)
    
    # Select model path
    model_path = select_model_path(artifacts)
    model_uri = f"runs:/{run_id}/{model_path}"
    return model_uri, run_id, experiment_id


def predict_pipeline(preprocessed_path = "data/preprocessed", 
                     output_path = "data/output", 
                     tracking_uri = "http://127.0.0.1:8080", 
                     experiment_name = "Rakuten Models",
                     artifact_path = "rf_rakuten"):
    # 1. Loading data for predict
    print("Loading data...")
    X_train, X_val, y_train, y_val, feats = build_features(preprocessed_path = preprocessed_path)

    # 2. Identify latest model from mlflow server
    # Get model URI
    TRACKING_URI = tracking_uri
    EXPERIMENT_NAME = experiment_name 
    ARTIFACT_PATH = artifact_path
    model_uri, run_id, experiment_id = get_model_uri(TRACKING_URI, EXPERIMENT_NAME)
    print(model_uri)
    print(run_id)

    model_path = f"mlruns/{experiment_id}/{run_id}/artifacts/{ARTIFACT_PATH}"
    print(f"Using MLflow server: {TRACKING_URI}")
    print(f"Using Experiment: {EXPERIMENT_NAME}")
    print(f"Using model: {model_path}")

    # 3. Load model from mlflow server
    print("Loading model...")
    model = mlflow.sklearn.load_model(model_path)

    # 4. Make predictions 
    print("Calculating predictions for 'X_val'...")
    predictions = model.predict(X_val)
    predictions_df = pd.DataFrame(predictions)

    # 5. Save predictions
    print("Saving predictions...")
    output_path = Path(output_path)
    os.makedirs(output_path, exist_ok=True)
    predictions_path = output_path / "predictions.csv"
    predictions_df.to_csv(predictions_path)

    print("Predictions saved.")
    return predictions_df


if __name__ == "__main__":
    predict_pipeline(preprocessed_path = "data/preprocessed", 
                     output_path = "data/output", 
                     tracking_uri = "http://127.0.0.1:8080", 
                     experiment_name = "Rakuten Models",
                     artifact_path = "rf_rakuten")