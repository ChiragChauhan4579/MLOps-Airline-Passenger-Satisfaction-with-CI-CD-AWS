from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse
import pandas as pd
import shutil
import os

app = FastAPI()

# Directory where the train.csv file is stored
DATA_DIR = "C:/Users/Chirag/Desktop/MLOps/MLOps Airline Passenger Satisfaction/data-storage"
CSV_FILE = f"{DATA_DIR}/train.csv"

# Ensure the data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

@app.get("/get_train_data")
async def get_train_data():
    if not os.path.exists(CSV_FILE):
        raise HTTPException(status_code=404, detail="train.csv not found.")
    return FileResponse(CSV_FILE, media_type="text/csv", filename="train.csv")

@app.post("/add_train_data")
async def add_train_data(file: UploadFile):
    # Validate CSV file type
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")
    
    # Read the uploaded file into a DataFrame
    uploaded_df = pd.read_csv(file.file)
    
    if os.path.exists(CSV_FILE):
        # Read the existing train.csv
        existing_df = pd.read_csv(CSV_FILE)
        # Append new data to the existing DataFrame
        combined_df = pd.concat([existing_df, uploaded_df], ignore_index=True)
    else:
        # If train.csv doesn't exist, the new data becomes the train.csv
        combined_df = uploaded_df
    
    # Save the combined DataFrame back to train.csv
    combined_df.to_csv(CSV_FILE, index=False)
    
    return {"message": "Data appended successfully to train.csv."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)