import pandas as pd

from src.api.schemas import PredictRequest
from src.api.service import PredictionService
from src.data.make_dataset import prepare_datasets
from src.features.build_features import build_feature_matrices
from src.models.train_model import train_baseline_model


def _write_raw_dataset(base_dir):
    raw_dir = base_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    x_train = pd.DataFrame(
        [
            {
                "Unnamed: 0": 1,
                "designation": "robe rouge",
                "description": "robe coton femme",
                "productid": 11,
                "imageid": 101,
            },
            {
                "Unnamed: 0": 2,
                "designation": "robe ete",
                "description": "robe legere rouge",
                "productid": 12,
                "imageid": 102,
            },
            {
                "Unnamed: 0": 3,
                "designation": "sandales femme",
                "description": "chaussures ete",
                "productid": 13,
                "imageid": 103,
            },
            {
                "Unnamed: 0": 4,
                "designation": "puzzle bois",
                "description": "jeu enfant puzzle",
                "productid": 21,
                "imageid": 201,
            },
            {
                "Unnamed: 0": 5,
                "designation": "voiture miniature",
                "description": "jouet collection metal",
                "productid": 22,
                "imageid": 202,
            },
            {
                "Unnamed: 0": 6,
                "designation": "jeu societe",
                "description": "jeu famille cartes",
                "productid": 23,
                "imageid": 203,
            },
            {
                "Unnamed: 0": 7,
                "designation": "robe rouge",
                "description": "robe coton femme",
                "productid": 11,
                "imageid": 101,
            },
        ]
    )
    y_train = pd.DataFrame(
        {
            "Unnamed: 0": [1, 2, 3, 4, 5, 6, 7],
            "prdtypecode": [10, 10, 10, 20, 20, 20, 10],
        }
    )
    x_test = pd.DataFrame(
        [
            {
                "Unnamed: 0": 8,
                "designation": "robe chic",
                "description": "robe femme ville",
                "productid": 31,
                "imageid": 301,
            },
            {
                "Unnamed: 0": 9,
                "designation": "puzzle geant",
                "description": "jeu enfant",
                "productid": 32,
                "imageid": 302,
            },
        ]
    )

    x_train.to_csv(raw_dir / "X_train_update.csv", index=False)
    y_train.to_csv(raw_dir / "Y_train_CVw08PX.csv", index=False)
    x_test.to_csv(raw_dir / "X_test_update.csv", index=False)
    return raw_dir


def test_prepare_datasets_creates_clean_split(tmp_path):
    raw_dir = _write_raw_dataset(tmp_path)
    output_dir = tmp_path / "preprocessed"

    summary = prepare_datasets(raw_dir, output_dir, test_size=0.33, random_state=42)

    assert summary["duplicates_removed"] == 1
    assert (output_dir / "X_train_clean.csv").exists()
    assert (output_dir / "X_val_clean.csv").exists()
    assert (output_dir / "Y_train_clean.csv").exists()
    assert (output_dir / "Y_val_clean.csv").exists()
    assert (output_dir / "X_test_clean.csv").exists()


def test_training_pipeline_outputs_api_compatible_artifacts(tmp_path, monkeypatch):
    raw_dir = _write_raw_dataset(tmp_path)
    preprocessed_dir = tmp_path / "preprocessed"
    model_dir = tmp_path / "models"
    report_dir = tmp_path / "reports"

    prepare_datasets(raw_dir, preprocessed_dir, test_size=0.33, random_state=42)
    feature_summary = build_feature_matrices(
        input_dir=preprocessed_dir,
        output_dir=preprocessed_dir,
        model_dir=model_dir,
        max_features=256,
    )
    training_summary = train_baseline_model(
        input_dir=preprocessed_dir,
        model_dir=model_dir,
        report_dir=report_dir,
        max_features=256,
    )

    assert feature_summary["feature_count"] > 0
    assert training_summary.model_path.exists()
    assert training_summary.vectorizer_path.exists()
    assert training_summary.metrics_path.exists()
    assert training_summary.report_path.exists()

    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    monkeypatch.delenv("MODEL_FILE", raising=False)
    monkeypatch.delenv("VECTORIZER_FILE", raising=False)
    monkeypatch.delenv("LABEL_MAPPING_FILE", raising=False)

    service = PredictionService(project_root=tmp_path)
    assert service.load() is True

    prediction = service.predict(
        PredictRequest(
            designation="robe ete",
            description="robe rouge legere femme",
            productid=999,
            imageid=888,
        )
    )

    assert prediction["predicted_code"] in {10, 20}
    assert prediction["model_name"] == "SGDClassifier"
