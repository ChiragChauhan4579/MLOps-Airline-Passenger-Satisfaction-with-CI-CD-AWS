import os
import mlflow
import numpy as np
import pandas as pd
from joblib import dump
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report , confusion_matrix , precision_score, recall_score, f1_score, classification_report

docker_flag = 1

if docker_flag == 1:
    ROOT_URL = "/app/"
else:
    ROOT_URL = "C:/Users/Chirag/Desktop/MLOps/MLOps Airline Passenger Satisfaction/"

X_train = pd.read_csv(ROOT_URL + "data/processed/X_train.csv")
X_test = pd.read_csv(ROOT_URL + "data/processed/X_test.csv")
y_train = pd.read_csv(ROOT_URL + "data/processed/y_train.csv")
y_test = pd.read_csv(ROOT_URL + "data/processed/y_test.csv")

current_date = datetime.now().strftime("%Y-%m-%d")

# MLflow experiment tracking
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
    
    # Save the best model based on F1 score
    if not os.path.exists("best_f1_score.txt") or f1 > float(open("best_f1_score.txt").read()):
        with open("best_f1_score.txt", "w") as f:
            f.write(str(f1))
        dump(xgb_clf, ROOT_URL + 'model/XGBoost_model.joblib')
        # dump(encoder, 'encoder.joblib')
        # dump(scaler, 'scaler.joblib')
        # dump(le, 'label_encoder.joblib')
        print("New best model saved based on F1 score.")
    else:
        print("Model not better than the last one!")

mlflow.end_run()