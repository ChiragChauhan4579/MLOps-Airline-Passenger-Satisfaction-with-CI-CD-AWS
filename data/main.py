from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse
import pandas as pd
import shutil
import os

app = FastAPI()

# Directory where the train.csv file is stored

docker_flag = 1

if docker_flag == 1:
    DATA_DIR = "/app/data/raw"
    CSV_FILE = f"{DATA_DIR}/train.csv"
    LATEST_CSV = f"{DATA_DIR}/train_latest.csv"
    PREVIOUS_CSV = f"{DATA_DIR}/train_previous.csv"
else:
    DATA_DIR = "C:/Users/Chirag/Desktop/MLOps/MLOps Airline Passenger Satisfaction/data/raw"
    CSV_FILE = f"{DATA_DIR}/train.csv"
    LATEST_CSV = f"{DATA_DIR}/train_latest.csv"
    PREVIOUS_CSV = f"{DATA_DIR}/train_previous.csv"

# Ensure the data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

@app.get("/get_train_latest")
async def get_train_latest():
    # Check if the latest CSV exists
    if not os.path.exists(LATEST_CSV):
        raise HTTPException(status_code=404, detail="train_latest.csv not found.")
    return FileResponse(LATEST_CSV, media_type="text/csv", filename="train_latest.csv")

@app.get("/get_train_previous")
async def get_train_previous():
    # Check if the previous CSV exists
    if not os.path.exists(PREVIOUS_CSV):
        raise HTTPException(status_code=404, detail="train_previous.csv not found.")
    return FileResponse(PREVIOUS_CSV, media_type="text/csv", filename="train_previous.csv")


@app.post("/add_train_data")
async def add_train_data(file: UploadFile):
    # Validate CSV file type
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")
    
    # Read the uploaded file into a DataFrame
    uploaded_df = pd.read_csv(file.file)
    
    if os.path.exists(LATEST_CSV):
        # Backup the current train_latest.csv as train_previous.csv
        if os.path.exists(PREVIOUS_CSV):
            os.remove(PREVIOUS_CSV)  # Remove the older backup
        shutil.copy(LATEST_CSV, PREVIOUS_CSV)
        
        # Read the existing train_latest.csv
        existing_df = pd.read_csv(LATEST_CSV)
        # Append new data to the existing DataFrame
        combined_df = pd.concat([existing_df, uploaded_df], ignore_index=True)
    else:
        # If train_latest.csv doesn't exist, the new data becomes the train_latest.csv
        combined_df = uploaded_df
    
    # Save the combined DataFrame back to train_latest.csv
    combined_df.to_csv(LATEST_CSV, index=False)

    return {"message": "Data appended successfully to train_latest.csv, and previous version saved as train_previous.csv."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)