import os
import subprocess
from prefect import task, flow
from prefect.client.schemas.schedules import IntervalSchedule
import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity
from datetime import datetime
from evidently.report import Report
from evidently.metrics import DataDriftTable
import json
import mlflow
import numpy as np 
from joblib import dump , load
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report ,confusion_matrix , precision_score, recall_score, f1_score, classification_report

os.environ['PREFECT_API_URL'] = "http://127.0.0.1:4200/api"
ROOT_URL = "C:/Users/Chirag/Desktop/MLOps/MLOps Airline Passenger Satisfaction/"

# Function to validate passenger data using Deepchecks
@task
def validate_passenger_data(df):
    try:
        # df = df.iloc[:,1:]
        # Define categorical and numerical columns
        categorical_features = ['Gender', 'Type of Travel', 'Customer Type', 'Class']
        # Create a Deepchecks Dataset object
        dataset = Dataset(df, cat_features=categorical_features)
        # Run the data integrity suite
        suite = data_integrity()
        result = suite.run(dataset)
        # Save or print the result
        result.save_as_html(ROOT_URL + 'reports/validation_report.html')
        print("Validation completed! Report saved as 'validation_report.html'.")
        # Return success flag
        return True
    except Exception as e:
        print(f"An error occurred during validation: {e}")
        # Return failure flag
        return False

@task
def datadrift(baseline_data,new_data):
    try:
        # Define the Data Drift Report
        data_drift_report = Report(metrics=[DataDriftTable()])
        # Generate the report
        data_drift_report.run(reference_data=baseline_data, current_data=new_data)
        # Visualize the report (opens in the browser or can be saved as HTML)
        data_drift_report.save_html(ROOT_URL + 'reports/data_drift_report.html')
        report_json = json.loads(data_drift_report.json())
        # print(report_json['metrics'][0]['result']['number_of_drifted_columns'])
        # print(report_json['metrics'][0]['result']['dataset_drift'])
        # print(report_json['timestamp'])

        if not report_json['metrics'][0]['result']['dataset_drift']:
            return True
        else:
            return False
    except Exception as e:
        print(f"An error occurred during validation: {e}")
        return False

@task
def model_retrain():

    try:
        result = subprocess.run(["dvc", "repro"], capture_output=True, text=True)

        if result.returncode == 0:
            print("DVC repro executed successfully!")
            print(result.stdout)
            return True
        else:
            print("Error executing DVC repro:")
            print(result.stderr)
            return False
        
    except Exception as e:
        print(f"An error occurred during Model Retraining: {e}")
        return False

@flow
def workflow():

    file_path = ROOT_URL + 'data/test.csv'
    df = pd.read_csv(file_path)
    try:
        df.drop(columns=['id'],inplace=True)
    except Exception as e:
        print(f"An error occurred during dropping columns: {e}")
    data_validity = validate_passenger_data(df)

    if data_validity:

        print("Data Validation Passed! Checking for Data Drift")

        baseline_data = pd.read_csv(ROOT_URL + 'data/train.csv')
        try:
            baseline_data.drop(columns=['id'],inplace=True)
        except Exception as e:
            print(f"An error occurred during validating data: {e}")
        datadrift_check = datadrift(baseline_data,df)

        if datadrift_check:
            print("Data Drift not Detected! Training the model.")

            model_retrain_status = model_retrain()
        else:
            print("Data Drift Detected!!!")
    else:
        print("Data Validation Failed")

if __name__ == "__main__":
    workflow.serve(
        name="test",
        cron="*/1 * * * *",
    )

# prefect server start
# prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"
# prefect config view
# python orchestrator.py