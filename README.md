conda activate mlops_ray

MLOps ML with mlflow, dvc, evidently, model retraining, orchestration prefect, docker,CI/CD (cml) github actions on huggingface

Done:
    Model and FastAPI
    Script 1: new data comes, check validity (deepchecks), data drift and model drift with evidently
    Script 2: track mlflow, train a new model - XGB, on best model apply hyperparameter tuning create

Next:
    Script 3: Wrap script 2 in dvc the data loading, data preprocess and model training
    Script 4: choose best model from mlflow and store in model directory
    Script 5: Prefect with Docker
    Script 6: Add new data push with streamlit/some storage - add in prefect flow (Script 1)
    Script 7: CI/CD on huggingface