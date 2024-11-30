import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity
from datetime import datetime
import os

# Function to validate passenger data using Deepchecks
def validate_passenger_data(file_path: str):
    # Load data

    try:
        df = pd.read_csv(file_path)

        df = df.iloc[:,1:]

        # Define categorical and numerical columns
        categorical_features = ['Gender', 'Type of Travel', 'Customer Type', 'Class']

        # Create a Deepchecks Dataset object
        dataset = Dataset(df, cat_features=categorical_features)

        # Run the data integrity suite
        suite = data_integrity()
        result = suite.run(dataset)

        # Save or print the result
        result.save_as_html('C:/Users/Chirag/Desktop/MLOps/MLOps-Airline-Passeneger-Satisfaction/reports/validation_report.html')
        print("Validation completed! Report saved as 'validation_report.html'.")
        # Return success flag
        return True
    
    except Exception as e:
        print(f"An error occurred during validation: {e}")
        
        # Return failure flag
        return False

# Example usage
# if __name__ == "__main__":
#     file_path = 'C:/Users/Chirag/Desktop/MLOps/MLOps-Airline-Passeneger-Satisfaction/data/test.csv'  # Replace with your CSV file path
#     success = validate_passenger_data(file_path)

#     if not success:
#         print("Validation failed. Please check the logs for more details.")
#     else:
#         print("Validation succeeded.")
