from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
from joblib import load

# Initialize FastAPI app
app = FastAPI()

docker_flag = 1

if docker_flag == 1:
    ROOT_URL = "/app/"
else:
    ROOT_URL = "C:/Users/Chirag/Desktop/MLOps/MLOps Airline Passenger Satisfaction/"
    
# Load the models and transformers during startup
encoder = load(ROOT_URL + 'model/encoder.joblib')
scaler = load(ROOT_URL + 'model/scaler.joblib')
model = load(ROOT_URL + 'model/XGBoost_model.joblib')

# Define categorical and numerical columns
categorical_columns = ['Gender', 'Type of Travel', 'Customer Type', 'Class']
columns_to_scale = ['Flight Distance', 'Age', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']

# Define input data schema using Pydantic
class PassengerData(BaseModel):
    Gender: str
    Customer_Type: str
    Age: int
    Type_of_Travel: str
    Class: str
    Flight_Distance: float
    Inflight_wifi_service: int
    Departure_Arrival_time_convenient: int
    Ease_of_Online_booking: int
    Gate_location: int
    Food_and_drink: int
    Online_boarding: int
    Seat_comfort: int
    Inflight_entertainment: int
    On_board_service: int
    Leg_room_service: int
    Baggage_handling: int
    Checkin_service: int
    Inflight_service: int
    Cleanliness: int
    Departure_Delay_in_Minutes: float
    Arrival_Delay_in_Minutes: float

# Endpoint to predict satisfaction
@app.post("/predict")
async def predict_satisfaction(passenger_data: List[PassengerData]):
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([data.dict() for data in passenger_data])
        input_df.columns = ['Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class',
            'Flight Distance', 'Inflight wifi service',
            'Departure/Arrival time convenient', 'Ease of Online booking',
            'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
            'Inflight entertainment', 'On-board service', 'Leg room service',
            'Baggage handling', 'Checkin service', 'Inflight service',
            'Cleanliness', 'Departure Delay in Minutes',
            'Arrival Delay in Minutes']

        # Encode categorical variables and scale numerical variables
        input_df[categorical_columns] = encoder.transform(input_df[categorical_columns])
        input_df[columns_to_scale] = scaler.transform(input_df[columns_to_scale])

        # Make predictions
        predictions = model.predict(input_df)
        return {"satisfaction": predictions.tolist()[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)