import os
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
        # Load dataset
        df = pd.read_csv(ROOT_URL + "data/train.csv")

        # Define categorical and target columns
        categorical_columns = ['Gender', 'Type of Travel', 'Customer Type', 'Class']
        target_column = 'satisfaction'

        # Split data into features and target
        X = df.drop(target_column, axis=1)
        y = df[target_column]

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Create and apply the OrdinalEncoder on training data only
        encoder = OrdinalEncoder()
        X_train[categorical_columns] = encoder.fit_transform(X_train[categorical_columns])
        X_test[categorical_columns] = encoder.transform(X_test[categorical_columns])

        # Encode labels
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

        # Standardize selected columns on training data only
        columns_to_scale = ['Flight Distance', 'Age', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
        scaler = StandardScaler()
        X_train[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
        X_test[columns_to_scale] = scaler.transform(X_test[columns_to_scale])

        # MLflow experiment tracking
        current_date = datetime.now().strftime("%Y-%m-%d")
        mlflow.set_experiment(f"airline_satisfaction_prediction_{current_date}")

        # Train and evaluate the XGBoost classifier
        with mlflow.start_run():
            xgb_clf = XGBClassifier(random_state=42)
            xgb_clf.fit(X_train, y_train)

            y_pred_xgb = xgb_clf.predict(X_test)

            # Calculate metrics
            accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
            precision = precision_score(y_test, y_pred_xgb)
            recall = recall_score(y_test, y_pred_xgb)
            f1 = f1_score(y_test, y_pred_xgb)

            # Log metrics to MLflow
            mlflow.log_metric("accuracy", accuracy_xgb)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            print(f'XGBoost Accuracy: {accuracy_xgb * 100:.2f}%')
            print(classification_report(y_test, y_pred_xgb))
            print("Confusion Matrix:")
            print(confusion_matrix(y_test, y_pred_xgb))
            print(f"Precision: {precision:.2f}")
            print(f"Recall: {recall:.2f}")
            print(f"F1-Score: {f1:.2f}")

            # Log model artifacts
            mlflow.sklearn.log_model(xgb_clf, "xgboost_model")
            mlflow.log_artifact(ROOT_URL + "reports/data_drift_report.html","Data Drift report")
            mlflow.log_artifact(ROOT_URL + "reports/validation_report.html","Data Validation report")

            # Save the best model based on F1 score
            if not os.path.exists("best_f1_score.txt") or f1 > float(open("best_f1_score.txt").read()):
                with open("best_f1_score.txt", "w") as f:
                    f.write(str(f1))
                dump(xgb_clf, ROOT_URL + 'model/XGBoost_model.joblib')
                dump(encoder, ROOT_URL + 'model/encoder.joblib')
                dump(scaler, ROOT_URL + 'model/scaler.joblib')
                dump(le, ROOT_URL + 'model/label_encoder.joblib')
                print("New best model saved based on F1 score.")
            else:
                print("Model not better than the last one!")

        mlflow.end_run()
        return True
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