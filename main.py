from data_processing import *
from training_algorithms import *
from visualise_algorithms import *
from sklearn.preprocessing import StandardScaler
import torch
import pandas as pd
import numpy as np
import os
import time
import warnings
warnings.filterwarnings('always') 

def get_df(selected_columns):
    filename = "dec_2023_data"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, f'data/{filename}_pivot.csv')
    if os.path.exists(csv_path) == False:
        pivot_and_store(filename)
    df = pd.read_csv(csv_path, index_col='timestamp')

    df = df[selected_columns]
    df['anomaly'] = 0
    df = df.replace(["NaN", "NULL", ""], np.nan)
    df = df.dropna(how='any')
    unique_firmware_names = df.columns.unique().tolist()
    return df

# firmware names is an array of all involved firmware names, and inference point is a numpy array of values corresponding to those firmware names
def single_point_inference(algorithm, inference_point, selected_columns):
    if inference_point == None:
        inference_point = [-0.5824,-0.0612,-0.7458,-0.7021,20000,0]
    lstm_df = get_df(selected_columns)
    inference_point.append(0)
    # inference_point = [-0.5824, -0.0612, -0.7458, -0.7021, 20000, 0] # Test Value
    print(inference_point)
    X_train_scaled, X_test_scaled, X_train_tensor, X_val_tensor, X_test_tensor, X_test, X_test_lstm = data_processing(lstm_df, num_of_anomalies=0, inference_point=inference_point)
    match algorithm:
        case 'LSTM':
            simple_inference_test = torch.tensor([inference_point], dtype=torch.float32)
            y_pred = infer_lstm_ae_model(X_train_tensor, X_val_tensor, simple_inference_test, firmware_names, num_of_std=3.5) # Change this figure to affect recall
        case 'GMM':
            y_pred = infer_gmm_model(X_train_scaled, X_test_scaled, selected_columns)
        case _:
            raise Exception("Invalid algorithm")
    prediction = "Anomaly" if y_pred['anomaly_predicted'].values == 1 else "Normal"
    print(f"Prediction: {prediction}")
    # return y_pred['anomaly_predicted'].values
    return prediction

def multi_point_visualisation(algorithm, selected_columns, num_of_anomalies=2):
    df = get_df(selected_columns)
    X_train_scaled, X_test_scaled, X_train_tensor, X_val_tensor, X_test_tensor, X_test, X_test_lstm = data_processing(df, num_of_anomalies)

    match algorithm:
        case 'LSTM':
            y_pred = infer_lstm_ae_model(X_train_tensor, X_val_tensor, X_test_tensor, firmware_name=f'{selected_columns}', num_of_std=3.5, plot_error=True) # Change this figure to affect recall
        case 'GMM':
            y_pred = infer_gmm_model(X_train_scaled, X_test_scaled, selected_columns)
        case _:
            raise Exception("Invalid algorithm")
    
    X_test_lstm['anomaly_predicted'] = y_pred['anomaly_predicted'].values
    inliers = X_test_lstm[X_test_lstm['anomaly_predicted'] == 0]
    outliers = X_test_lstm[X_test_lstm['anomaly_predicted'] == 1]
    print(f"{algorithm}")
    evaluate_algorithm(X_test_lstm, num_of_anomalies=num_of_anomalies)
    plot_scatter(X_test_lstm)
            
    # plot_pca(X_test_lstm, ['anomaly', 'anomaly_predicted'])

if __name__ == "__main__":
    start = time.time()
    TS_temp_6 = ['ADC_TS-1 (XP) Temperature', 'ADC_TS-2 (XN) Temperature', 'ADC_TS-3 (YP) Temperature', 'ADC_TS-4 (YN) Temperature', 'ADC_TS-5 (ZP) Temperature', 'ADC_TS-6 (ZN) Temperature']
    TS_temp_5 = ['ADC_TS-1 (XP) Temperature', 'ADC_TS-2 (XN) Temperature', 'ADC_TS-3 (YP) Temperature', 'ADC_TS-4 (YN) Temperature', 'ADC_TS-5 (ZP) Temperature']

    selected_columns = TS_temp_5

    # multi_point_visualisation(selected_columns)
    # algorithms = ['LSTM', 'GMM']
    # multi_point_visualisation_gmm('LSTM', selected_columns, 2)

    # Delete existing models to retrain on the entire dataset, including the original "test" data
    test_anomaly = [-0.5824, -0.0612, -0.7458, -0.7021, 0]
    single_point_inference('GMM', test_anomaly, selected_columns)
    # test_normal = [-0.5824, -0.0612, -0.7458, -0.7021, 6.0000]
    # single_point_inference(test_normal, selected_columns)
    end = time.time()
    print(f'Execution time: {end - start}s')
