from data_processing import *
from training_algorithms import *
from visualise_algorithms import *
import torch
import pandas as pd
import numpy as np

def get_df():
    filename = "dec_2023_data"
    pivot_and_store(filename)
    df = pd.read_csv(f'data/{filename}_pivot.csv', index_col='timestamp')

    df = df[selected_columns]
    df['anomaly'] = 0
    df = df.replace(["NaN", "NULL", ""], np.nan)
    df = df.dropna(how='any')
    unique_firmware_names = df.columns.unique().tolist()
    return df

# firmware name is an array of all involved firmware names, and inference point is a numpy array of values corresponding to those firmware names
def single_point_inference(inference_point, firmware_name):
    lstm_df = get_df()
    inference_point.append(0)
    # inference_point = [-0.5824, -0.0612, -0.7458, -0.7021, 20000, 0] # Test Value
    print(inference_point)
    X_train_scaled, X_test_scaled, X_train_tensor, X_val_tensor, X_test_tensor, X_test, X_test_lstm = data_processing(lstm_df, num_of_anomalies=0, single_inference=True)
    simple_inference_test = torch.tensor([inference_point], dtype=torch.float32)
    y_pred = infer_lstm_ae_model(X_train_tensor, X_val_tensor, simple_inference_test, firmware_name, num_of_std=3.5) # Change this figure to affect recall
    prediction = "Anomaly" if y_pred['anomaly_predicted'].values == 1 else "Normal"
    print(f"Prediction: {prediction}")
    # return y_pred['anomaly_predicted'].values
    return prediction

def multi_point_visualisation():
    lstm_df = get_df()
    num_of_anomalies = 3
    X_train_scaled, X_test_scaled, X_train_tensor, X_val_tensor, X_test_tensor, X_test, X_test_lstm = data_processing(lstm_df, num_of_anomalies)
    y_pred = infer_lstm_ae_model(X_train_tensor, X_val_tensor, X_test_tensor, firmware_name=f'{selected_columns}', num_of_std=3.5, plot_error=True) # Change this figure to affect recall
    X_test_lstm['anomaly_predicted'] = y_pred['anomaly_predicted'].values
    inliers = X_test_lstm[X_test_lstm['anomaly_predicted'] == 0]
    outliers = X_test_lstm[X_test_lstm['anomaly_predicted'] == 1]
    print("LSTM Autoencoder")
    evaluate_algorithm(X_test_lstm, num_of_anomalies=num_of_anomalies)
    plot_scatter(X_test_lstm)
    plot_pca(X_test_lstm, ['anomaly', 'anomaly_predicted'])

if __name__ == "__main__":
    TS_temp_6 = ['ADC_TS-1 (XP) Temperature', 'ADC_TS-2 (XN) Temperature', 'ADC_TS-3 (YP) Temperature', 'ADC_TS-4 (YN) Temperature', 'ADC_TS-5 (ZP) Temperature', 'ADC_TS-6 (ZN) Temperature']
    TS_temp_5 = ['ADC_TS-1 (XP) Temperature', 'ADC_TS-2 (XN) Temperature', 'ADC_TS-3 (YP) Temperature', 'ADC_TS-4 (YN) Temperature', 'ADC_TS-5 (ZP) Temperature']

    selected_columns = TS_temp_5

    # multi_point_visualisation()

    # Delete existing models to retrain on the entire dataset, including the original "test" data
    # test_anomaly = [-0.5824, -0.0612, -0.7458, -0.7021, 20000]
    # single_point_inference(test_anomaly, selected_columns)
    test_normal = [-0.5824, -0.0612, -0.7458, -0.7021, 0.0000]
    single_point_inference(test_normal, selected_columns)