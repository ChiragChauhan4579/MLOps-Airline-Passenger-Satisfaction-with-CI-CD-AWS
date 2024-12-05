import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

docker_flag = 1

if docker_flag == 1:
    ROOT_URL = "/app/"
else:
    ROOT_URL = "C:/Users/Chirag/Desktop/MLOps/MLOps Airline Passenger Satisfaction/"

# Load dataset
df = pd.read_csv(ROOT_URL + "data/raw/train_latest.csv")

categorical_columns = ['Gender', 'Type of Travel', 'Customer Type', 'Class']
target_column = 'satisfaction'
columns_to_scale = ['Flight Distance', 'Age', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']

# Split data
X = df.drop(target_column, axis=1)
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Encode categorical columns
encoder = OrdinalEncoder()
X_train[categorical_columns] = encoder.fit_transform(X_train[categorical_columns])
X_test[categorical_columns] = encoder.transform(X_test[categorical_columns])
joblib.dump(encoder, ROOT_URL + 'model/encoder.joblib')

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
joblib.dump(le, ROOT_URL + 'model/label_encoder.joblib')

# Standardize columns
scaler = StandardScaler()
X_train[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
X_test[columns_to_scale] = scaler.transform(X_test[columns_to_scale])
joblib.dump(scaler, ROOT_URL + 'model/scaler.joblib')

# Save processed data
X_train.to_csv(ROOT_URL + "data/processed/X_train.csv", index=False)
X_test.to_csv(ROOT_URL + "data/processed/X_test.csv", index=False)
pd.DataFrame(y_train).to_csv(ROOT_URL + "data/processed/y_train.csv", index=False)
pd.DataFrame(y_test).to_csv(ROOT_URL + "data/processed/y_test.csv", index=False)