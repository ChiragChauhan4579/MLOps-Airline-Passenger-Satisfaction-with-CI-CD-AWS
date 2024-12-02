# dvc init --no-scm

"""
dvc stage add --run --force -n prepare_data -d "C:/Users/Chirag/Desktop/MLOps/MLOps Airline Passenger Satisfaction/data/raw/train.csv" -d "C:/Users/Chirag/Desktop/MLOps/MLOps Airline Passenger Satisfaction/src/prepare_data.py" -o "C:/Users/Chirag/Desktop/MLOps/MLOps Airline Passenger Satisfaction/data/processed/X_train.csv" -o "C:/Users/Chirag/Desktop/MLOps/MLOps Airline Passenger Satisfaction/data/processed/X_test.csv" -o "C:/Users/Chirag/Desktop/MLOps/MLOps Airline Passenger Satisfaction/data/processed/y_train.csv" -o "C:/Users/Chirag/Desktop/MLOps/MLOps Airline Passenger Satisfaction/data/processed/y_test.csv" -o "C:/Users/Chirag/Desktop/MLOps/MLOps Airline Passenger Satisfaction/model/encoder.joblib" -o "C:/Users/Chirag/Desktop/MLOps/MLOps Airline Passenger Satisfaction/model/label_encoder.joblib" -o "C:/Users/Chirag/Desktop/MLOps/MLOps Airline Passenger Satisfaction/model/scaler.joblib" python "C:/Users/Chirag/Desktop/MLOps/MLOps Airline Passenger Satisfaction/src/prepare_data.py"
"""

"""
dvc stage add --run --force -n train_model -d "C:/Users/Chirag/Desktop/MLOps/MLOps Airline Passenger Satisfaction/data/processed/X_train.csv" -d "C:/Users/Chirag/Desktop/MLOps/MLOps Airline Passenger Satisfaction/data/processed/X_test.csv" -d "C:/Users/Chirag/Desktop/MLOps/MLOps Airline Passenger Satisfaction/data/processed/y_train.csv" -d "C:/Users/Chirag/Desktop/MLOps/MLOps Airline Passenger Satisfaction/data/processed/y_test.csv" -d "C:/Users/Chirag/Desktop/MLOps/MLOps Airline Passenger Satisfaction/src/train_model.py" -d "C:/Users/Chirag/Desktop/MLOps/MLOps Airline Passenger Satisfaction/src/best_f1_score.txt" -o "C:/Users/Chirag/Desktop/MLOps/MLOps Airline Passenger Satisfaction/model/XGBoost_model.joblib" python "C:/Users/Chirag/Desktop/MLOps/MLOps Airline Passenger Satisfaction/src/train_model.py"
"""

# stages:
#   prepare_data:
#     cmd: python "C:/Users/Chirag/Desktop/MLOps/MLOps Airline Passenger Satisfaction/src/prepare_data.py"
#     deps:
#       - "C:/Users/Chirag/Desktop/MLOps/MLOps Airline Passenger Satisfaction/data/raw/train.csv"
#       - "C:/Users/Chirag/Desktop/MLOps/MLOps Airline Passenger Satisfaction/src/prepare_data.py"
#     outs:
#       - "C:/Users/Chirag/Desktop/MLOps/MLOps Airline Passenger Satisfaction/data/processed/X_train.csv"
#       - "C:/Users/Chirag/Desktop/MLOps/MLOps Airline Passenger Satisfaction/data/processed/X_test.csv"
#       - "C:/Users/Chirag/Desktop/MLOps/MLOps Airline Passenger Satisfaction/data/processed/y_train.csv"
#       - "C:/Users/Chirag/Desktop/MLOps/MLOps Airline Passenger Satisfaction/data/processed/y_test.csv"
#       - "C:/Users/Chirag/Desktop/MLOps/MLOps Airline Passenger Satisfaction/model/encoder.joblib"
#       - "C:/Users/Chirag/Desktop/MLOps/MLOps Airline Passenger Satisfaction/model/label_encoder.joblib"
#       - "C:/Users/Chirag/Desktop/MLOps/MLOps Airline Passenger Satisfaction/model/scaler.joblib"

#   train_model:
#     cmd: python "C:/Users/Chirag/Desktop/MLOps/MLOps Airline Passenger Satisfaction/src/train_model.py"
#     deps:
#       - "C:/Users/Chirag/Desktop/MLOps/MLOps Airline Passenger Satisfaction/data/processed/X_train.csv"
#       - "C:/Users/Chirag/Desktop/MLOps/MLOps Airline Passenger Satisfaction/data/processed/X_test.csv"
#       - "C:/Users/Chirag/Desktop/MLOps/MLOps Airline Passenger Satisfaction/data/processed/y_train.csv"
#       - "C:/Users/Chirag/Desktop/MLOps/MLOps Airline Passenger Satisfaction/data/processed/y_test.csv"
#       - "C:/Users/Chirag/Desktop/MLOps/MLOps Airline Passenger Satisfaction/src/train_model.py"
#       - "C:/Users/Chirag/Desktop/MLOps/MLOps Airline Passenger Satisfaction/src/best_f1_score.txt"
#     outs:
#       - "C:/Users/Chirag/Desktop/MLOps/MLOps Airline Passenger Satisfaction/model/XGBoost_model.joblib"