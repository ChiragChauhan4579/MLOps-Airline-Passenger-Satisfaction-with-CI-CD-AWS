import pandas as pd
from deepchecks.tabular import Dataset, Suite
from deepchecks.tabular.suites import full_suite
from evidently.report import Report
from evidently.metrics import DataDriftTable
import subprocess
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score


def validate_data():
    # Load new data
    new_data = pd.read_csv('path/to/new_data.csv')
    
    # Deepchecks for data validation
    dataset = Dataset(new_data, label='satisfaction')
    suite = full_suite()
    suite.run(dataset).save_as_html('reports/deepchecks_report.html')
    
    # Evidently for data drift detection
    reference_data = pd.read_csv('path/to/reference_data.csv')
    report = Report(metrics=[DataDriftTable()])
    report.run(reference_data=reference_data, current_data=new_data)
    report.save_html('reports/evidently_drift_report.html')
    
    # Check if validation passes
    if suite.run(dataset).status.value == 'PASS':
        return True
    else:
        raise ValueError("Data validation failed.")

def run_dvc_pipeline():
    # Pull the latest data and run the pipeline
    subprocess.run(['dvc', 'pull'], check=True)
    subprocess.run(['dvc', 'repro'], check=True)

def train_models():
    data = pd.read_csv('path/to/processed_data.csv')
    X = data.drop('satisfaction', axis=1)
    y = data['satisfaction']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "RandomForest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(),
        "LightGBM": lgb.LGBMClassifier()
    }
    
    mlflow.set_experiment("satisfaction_prediction")
    
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(model, name)

def evaluate_and_save_best_model():
    best_run = mlflow.search_runs(order_by=["metrics.accuracy DESC"]).iloc[0]
    best_model_uri = best_run.artifact_uri + f"/{best_run.tags['mlflow.runName']}"
    mlflow.sklearn.load_model(best_model_uri).save("model_directory/best_model.pkl")
