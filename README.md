Rakuten Ecommerce Classification
==============================

This project is a starting Pack for MLOps projects based on the subject "Rakuten Product Classification". It's not perfect so feel free to make some modifications on it.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── .dvc/              <- DVC configuration for data and model versioning.
    ├── data
    │   ├── external       <- Data from third party sources -> the external data you want to make a prediction on
    │   ├── preprocessed   <- The final, canonical data sets for modeling.
    |   |  ├── image_train <- Where you put the images of the train set
    |   |  ├── image_test  <- Where you put the images of the predict set
    |   |  ├── X_train_update.csv <- Tabular training data
    |   |  ├── X_test_update.csv  <- Tabular testing data
    │   └── raw            <- The original, immutable data dump.
    |   |  ├── image_train <- Original train images
    |   |  ├── image_test  <- Original predict images
    │
    ├── logs               <- Logs from training and predicting
    │
    ├── models             <- Trained and serialized models or label mappings
    │   ├── artifacts/     <- model_final.joblib (The primary Scikit-Learn Pipeline)
    │   └── label_mapping.json <- Mapping between prdtypecode and category names
    │
    ├── notebooks          <- Jupyter notebooks for data exploration and experimentation.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment.
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   ├── main.py        <- Scripts to train models 
    │   ├── predict.py     <- Scripts to make prediction on the files in ../data/preprocessed
    │   │
    │   ├── api            <- FastAPI implementation
    │   │   └── main.py    <- Main API script (Production and Test compatible)
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── check_structure.py    
    │   │   ├── import_raw_data.py 
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models                
    │   │   └── train_model.py
    │   └── config         <- Describe the parameters used in train_model.py and predict.py

--------

Once you have downloaded the github repo, open the anaconda powershell on the root of the project and follow those instructions :

> `conda create -n "Rakuten-project" python=3.10`    <- It will create your conda environement

> `conda activate Rakuten-project`       <- It will activate your environment

> `conda install pip`                    <- May be optionnal

> `pip install -r requirements.txt`      <- It will install the required packages

> `pip install dvc-gdrive`               <- Install DVC Google Drive support

> `dvc pull`                             <- Download models and data from remote storage

> `python src/data/import_raw_data.py`   <- It will import the tabular data on data/raw/

> Upload the image data folder set directly on local from https://challengedata.ens.fr/participants/challenges/35/, you should save the folders image_train and image_test respecting the following structure

    ├── data
    │   └── raw           
    |   |  ├── image_train 
    |   |  ├── image_test 

> `python src/data/make_dataset.py data/raw data/preprocessed`      <- It will copy the raw dataset and paste it on data/preprocessed/

> `python src/main.py`                   <- It will train the models on the dataset and save them in models.

> `python src/predict.py`                <- It will use the trained models to make a prediction on the desired data.
>
    Exemple : python src/predict.py --dataset_path "data/preprocessed/X_test_update.csv" --images_path "data/preprocessed/image_test"
                                        
                                         The predictions are saved in data/preprocessed as 'predictions.json'

## FastAPI service

The project now exposes a FastAPI application for online inference.

1. Install dependencies:

   `pip install -r requirements.txt`

2. Place the serialized inference artifacts in the `models/` folder.

   Expected defaults:
   - `models/artifacts/model_final.joblib` (Scikit-Learn Pipeline)
   - `models/label_mapping.json` (Category mapping)

3. Start the API from the repository root:

   `uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload`

4. Check the service:

   - `GET /health`
   - `POST /predict`

Authentication:

- `GET /health` is public
- `POST /predict` requires `Authorization: Bearer <token>` (If security is enabled)
- The server token must be provided through the `API_AUTH_TOKEN` environment variable

Tests and coverage:

- Run the API test suite with coverage from the repository root: `pytest`
- The project now enforces a minimum coverage threshold of `90%` on `src/api`
- Coverage is configured in `pytest.ini`

Example request body:

```json
{
  "text": "Lot de 4 chaises scandinaves pour salle à manger",
  "image_path": "string"
}
Validation rules for POST /predict:

text: required string for text classification (Scikit-Learn)

model_type: optional string ("lstm" or "vgg16"), defaults to "lstm"

image_path: required if classification is image-based

Any unexpected field is handled by standard FastAPI validation

Error responses are normalized:

401 AUTHENTICATION_REQUIRED when the Bearer token is missing

422 VALIDATION_ERROR for invalid payloads or missing required fields

503 MODEL_NOT_READY when artifacts are missing or not loaded

500 INTERNAL_SERVER_ERROR for technical failures during prediction

Url drive johan: https://drive.google.com/drive/folders/1vYf7JAkDylxW53viUhayQODOK_1kuzc9

You can download the trained models loaded here : https://drive.google.com/drive/folders/1fjWd-NKTE-RZxYOOElrkTdOw2fGftf5M?usp=drive_link and insert them in the models folder

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

🐳 Lancer l'API avec Docker (Mise en production)
L'API est entièrement conteneurisée. Pour la démarrer sur n'importe quel environnement :

Assurez-vous d'avoir rapatrié le modèle final via DVC :

Bash
dvc pull
Construisez l'image Docker en local :

Bash
docker build -t rakuten-api:latest .
Lancez le conteneur :

Bash
docker run -p 8000:8000 rakuten-api:latest
L'API sera alors accessible sur http://localhost:8000/docs.

Note on Data Preparation: To correctly set up the data structure, run:
python src/data/make_dataset.py data/raw data/preprocessed