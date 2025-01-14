import pandas as pd
from evidently.report import Report
from evidently.metrics import DataDriftTable
import json


def datadrift(baseline_data,new_data):

    # Define the Data Drift Report
    data_drift_report = Report(metrics=[DataDriftTable()])

    # Generate the report
    data_drift_report.run(reference_data=baseline_data, current_data=new_data)

    # Visualize the report (opens in the browser or can be saved as HTML)
    data_drift_report.save_html('C:/Users/Chirag/Desktop/MLOps/MLOps Airline Passenger Satisfaction/reports/data_drift_report.html')

    report_json = json.loads(data_drift_report.json())

    print(report_json['metrics'][0]['result']['number_of_drifted_columns'])
    print(report_json['metrics'][0]['result']['dataset_drift'])
    print(report_json['timestamp'])

    if report_json['metrics'][0]['result']['dataset_drift']:
        return True
    else:
        return False

if __name__ == "__main__":
    baseline_data = pd.read_csv('C:/Users/Chirag/Desktop/MLOps/MLOps Airline Passenger Satisfaction/data/train.csv')
    new_data = pd.read_csv('C:/Users/Chirag/Desktop/MLOps/MLOps Airline Passenger Satisfaction/data/test.csv')
    print(datadrift(baseline_data,new_data))