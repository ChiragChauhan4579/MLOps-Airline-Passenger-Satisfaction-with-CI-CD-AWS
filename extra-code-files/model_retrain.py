import os
import mlflow
import numpy as np
import pandas as pd
from joblib import dump , load
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report , confusion_matrix , precision_score, recall_score, f1_score, classification_report

# Load dataset
df = pd.read_csv("C:/Users/Chirag/Desktop/MLOps/MLOps Airline Passenger Satisfaction/data/train.csv")

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
mlflow.set_experiment("airline_satisfaction_prediction")

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
        dump(xgb_clf, 'best_XGBoost_model.joblib')
        dump(encoder, 'encoder.joblib')
        dump(scaler, 'scaler.joblib')
        dump(le, 'label_encoder.joblib')
        print("New best model saved based on F1 score.")
    else:
        print("Model not better than the last one!")

mlflow.end_run()