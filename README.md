Project Name
==============================

This project is a starting Pack for MLOps projects based on the subject "movie_recommandation". It's not perfect so feel free to make some modifications on it.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources -> the external data you want to make a prediction on
    │   ├── preprocessed      <- The final, canonical data sets for modeling.
    |   |  ├── image_train <- Where you put the images of the train set
    |   |  ├── image_test <- Where you put the images of the predict set
    |   |  ├── X_train_update.csv    <- The csv file with te columns designation, description, productid, imageid like in X_train_update.csv
    |   |  ├── X_test_update.csv    <- The csv file with te columns designation, description, productid, imageid like in X_train_update.csv
    │   └── raw            <- The original, immutable data dump.
    |   |  ├── image_train <- Where you put the images of the train set
    |   |  ├── image_test <- Where you put the images of the predict set
    │
    ├── logs               <- Logs from training and predicting
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   ├── main.py        <- Scripts to train models 
    │   ├── predict.py     <- Scripts to use trained models to make prediction on the files put in ../data/preprocessed
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── check_structure.py    
    │   │   ├── import_raw_data.py 
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models                
    │   │   └── train_model.py
    │   └── config         <- Describe the parameters used in train_model.py and predict_model.py

--------

Once you have downloaded the github repo, open the anaconda powershell on the root of the project and follow those instructions :

> `conda create -n "Rakuten-project"`    <- It will create your conda environement

> `conda activate Rakuten-project`       <- It will activate your environment

> `conda install pip`                    <- May be optionnal

> `pip install -r requirements.txt`      <- It will install the required packages

> `python src/data/import_raw_data.py`   <- It will import the tabular data on data/raw/

> Upload the image data folder set directly on local from https://challengedata.ens.fr/participants/challenges/35/, you should save the folders image_train and image_test respecting the following structure

    ├── data
    │   └── raw           
    |   |  ├── image_train 
    |   |  ├── image_test 

> `python src/data/make_dataset.py data/raw data/preprocessed`      <- It will copy the raw dataset and paste it on data/preprocessed/

> `python src/main.py`                   <- It will train the models on the dataset and save them in models. By default, the number of epochs = 1

> `python src/predict.py`                <- It will use the trained models to make a prediction (of the prdtypecode) on the desired data, by default, it will predict on the train. You can pass the path to data and images as arguments if you want to change it
>
    Exemple : python src/predict_1.py --dataset_path "data/preprocessed/X_test_update.csv" --images_path "data/preprocessed/image_test"
                                        
                                         The predictions are saved in data/preprocessed as 'predictions.json'

## FastAPI service

The project now exposes a FastAPI application for online inference.

1. Install dependencies:

   `pip install -r requirements.txt`

2. Place the serialized inference artifacts in the `models/` folder.

   Expected defaults:
   - `models/baseline_model.pkl` or `models/classifier.pkl` or `models/model.pkl`
   - `models/tfidf_vectorizer.pkl` (optional if the model is already a pipeline)
   - `models/label_mapping.json` (optional)

3. Start the API from the repository root:

   `uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload`

4. Check the service:

   - `GET /health`
   - `POST /predict`

Authentication:

- `GET /health` is public
- `POST /predict` requires `Authorization: Bearer <token>`
- The server token must be provided through the `API_AUTH_TOKEN` environment variable

Tests and coverage:

- Run the API test suite with coverage from the repository root: `pytest`
- The project now enforces a minimum coverage threshold of `90%` on `src/api`
- Coverage is configured in `pytest.ini`

Example request body:

```json
{
  "designation": "robe ete femme",
  "description": "robe rouge legere en coton",
  "productid": 1001,
  "imageid": 2002
}
```

Validation rules for `POST /predict`:
- `designation`: required string, trimmed, 1 to 512 characters
- `description`: optional string, trimmed, up to 10000 characters, defaults to `""`
- `productid`: optional integer, must be greater than or equal to `0`
- `imageid`: optional integer, must be greater than or equal to `0`
- Any unexpected field is rejected

Error responses are normalized:
- `401 AUTHENTICATION_REQUIRED` when the Bearer token is missing
- `401 INVALID_AUTH_SCHEME` when the authorization scheme is not `Bearer`
- `403 INVALID_TOKEN` when the token is present but invalid
- `422 VALIDATION_ERROR` for invalid payloads
- `503 AUTH_NOT_CONFIGURED` when the server token is missing
- `503 MODEL_NOT_READY` when artifacts are missing or not loaded
- `500 PREDICTION_FAILED` or `500 INTERNAL_SERVER_ERROR` for technical failures

If the model artifacts are missing, `/health` returns a `degraded` status and `/predict` returns `503` with an explicit error message instead of retraining the model.

> You can download the trained models loaded here : https://drive.google.com/drive/folders/1fjWd-NKTE-RZxYOOElrkTdOw2fGftf5M?usp=drive_link and insert them in the models folder
> 
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
python make_dataset.py "../../data/raw" "../../data/preprocessed"
## Run API locally

Start API server

uvicorn src.api.main:app --reload

Swagger documentation

http://127.0.0.1:8000/docs

Health test

curl http://127.0.0.1:8000/health

Predict test

curl -X POST http://127.0.0.1:8000/predict \
-H "Content-Type: application/json" \
-d '{"text":"smartphone samsung"}'

