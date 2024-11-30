from joblib import load
import pandas as pd

categorical_columns = ['Gender', 'Type of Travel', 'Customer Type', 'Class']
columns_to_scale = ['Flight Distance', 'Age', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']

df = pd.read_csv("C:/Users/Chirag/Desktop/MLOps/MLOps Airline Passenger Satisfaction/data/test.csv")
df = df.iloc[:5,2:-1]
print(df.columns)

encoder = load('./model/encoder.joblib')
scaler = load('./model/scaler.joblib')
model = load('./model/XGBoost_model.joblib')

df[categorical_columns] = encoder.transform(df[categorical_columns])
df[columns_to_scale] = scaler.transform(df[columns_to_scale])

print(model.predict(df))