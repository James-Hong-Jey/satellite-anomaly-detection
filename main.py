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

# firmware names is an array of all involved firmware names, and inference point is a numpy array of values corresponding to those firmware names
def single_point_inference(df, algorithm, inference_point, selected_columns):
    if df is None:
        single_df = get_df("dec_2023_data", selected_columns)
    else:
        single_df = df.copy(deep=True)
        
    if inference_point == None:
        inference_point = [-0.5824,-0.0612,-0.7458,-0.7021,20000,0]
    inference_point.append(0)
    # inference_point = [-0.5824, -0.0612, -0.7458, -0.7021, 20000, 0] # Test Value
    print(inference_point)
    X_train, X_test, X_train_scaled, X_test_scaled, X_train_tensor, X_val_tensor, X_test_tensor, X_test_lstm = data_processing(single_df, num_of_anomalies=0, inference_point=inference_point)
    match algorithm:
        case 'LSTM':
            simple_inference_test = torch.tensor([inference_point], dtype=torch.float32)
            y_pred = infer_lstm_ae_model(X_train_tensor, X_val_tensor, simple_inference_test, firmware_names, num_of_std=3.5) # Change this figure to affect recall
        case 'GMM':
            y_pred = infer_gmm_model(X_train_scaled, X_test_scaled, selected_columns)
        case _:
            raise Exception("Invalid algorithm")
    prediction = y_pred['anomaly_predicted'].values
    print(f"Prediction: {"Anomaly" if prediction == 1 else "Normal"}")
    # return y_pred['anomaly_predicted'].values
    return prediction


def multi_point_visualisation(df, algorithm, selected_columns, num_of_anomalies=2, random_state=None, plot=True):
    multi_df = df.copy(deep=True)
    X_train, X_test, X_train_scaled, X_test_scaled, X_train_tensor, X_val_tensor, X_test_tensor, X_test_lstm = data_processing(multi_df, num_of_anomalies, None, random_state)
    y_test = X_test[['anomaly']]
    match algorithm:
        case 'IF':
            y_pred = infer_if_model(X_train_scaled, X_test_scaled, y_test, selected_columns)
            visual_X_test = X_test
        case 'OCSVM':
            y_pred = infer_ocsvm_model(X_train_scaled, X_test_scaled, y_test, selected_columns)
            visual_X_test = X_test
        case 'LSTM':
            y_pred = infer_lstm_ae_model(X_train_tensor, X_val_tensor, X_test_tensor, selected_columns, num_of_std=3, plot_error=False) # Change this figure to affect recall
            visual_X_test = X_test_lstm
        case 'GMM':
            y_pred = infer_gmm_model(X_train_scaled, X_test_scaled, selected_columns) # "distance" based probabilistic models benefit from scaling
            visual_X_test = X_test
        case 'XGB':
            y_pred = infer_xgb_model(X_train, X_test, selected_columns) # decision trees insensitive to feature scaling - easier to work with df
            visual_X_test = X_test
        case _:
            raise Exception("Invalid algorithm")
    
    visual_X_test['anomaly_predicted'] = y_pred['anomaly_predicted'].values
    inliers = visual_X_test[visual_X_test['anomaly_predicted'] == 0]
    outliers = visual_X_test[visual_X_test['anomaly_predicted'] == 1]
    # print(f"{algorithm}")
    if plot:
        plot_scatter(algorithm, visual_X_test)
        # plot_pca(visual_X_test, ['anomaly', 'anomaly_predicted'])
        # plot_tsne(visual_X_test, ['anomaly', 'anomaly_predicted'])
        return evaluate_algorithm(visual_X_test, num_of_anomalies=num_of_anomalies, verbose=True)
    else:
        return evaluate_algorithm(visual_X_test, num_of_anomalies=num_of_anomalies, verbose=False)

def benchmark_algorithm(df, algorithm, selected_columns, n=100, num_of_anomalies=5):
    results = []
    for i in range (0, num_of_anomalies + 1): 
        f1_score, precision, recall = 0, 0, 0
        for j in range (1, n+1):
            score = multi_point_visualisation(df, algorithm, selected_columns, num_of_anomalies=i, random_state=j, plot=False)
            f1_score += score['f1_score']
            precision += score['precision']
            recall += score['recall']
        print(f"Algorithm: {algorithm}, Num of Anomalies: {i}, F1 Score: {f1_score/n}, Precision: {precision/n}, Recall: {recall/n}")
        results.append({
            'Algorithm': algorithm,
            'Num of Anomalies': i,
            'F1 Score': f1_score/n,
            'Precision': precision/n,
            'Recall': recall/n
        })
    results_df = pd.DataFrame(results)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    selected_columns = hash_firmware_names(selected_columns)
    csv_path = os.path.join(current_dir, f"results/results_{algorithm}_{selected_columns}.csv")
    results_df.to_csv(csv_path)
    return results_df
        

if __name__ == "__main__":
    start = time.time()

    all_adcs_params = ['ADC_TS-1 (XP) Temperature', 'ADC_TS-2 (XN) Temperature', 'ADC_TS-3 (YP) Temperature', 'ADC_TS-4 (YN) Temperature', 'ADC_TS-5 (ZP) Temperature', 'ADC_TS-6 (ZN) Temperature', 'ADC_MCU Temperature', 'IMU.A_Temperature', 'IMU.B_Temperature', 'FSS.PX Temperature', 'FSS.NX Temperature', 'FSS.NZ Temperature', 'RW.X Temperature', 'RW.Y Temperature', 'RW.Z Temperature']
    TS_temp_6 = ['ADC_TS-1 (XP) Temperature', 'ADC_TS-2 (XN) Temperature', 'ADC_TS-3 (YP) Temperature', 'ADC_TS-4 (YN) Temperature', 'ADC_TS-5 (ZP) Temperature', 'ADC_TS-6 (ZN) Temperature']
    TS_temp_5 = ['ADC_TS-1 (XP) Temperature', 'ADC_TS-2 (XN) Temperature', 'ADC_TS-3 (YP) Temperature', 'ADC_TS-4 (YN) Temperature', 'ADC_TS-5 (ZP) Temperature']
    TS_temp_3 = ['ADC_TS-1 (XP) Temperature', 'ADC_TS-2 (XN) Temperature', 'ADC_TS-3 (YP) Temperature']
    RX_temp = ['RW.X Temperature', 'RW.Y Temperature', 'RW.Z Temperature']

    # 1) Fill in filename, choose which list of columns to use 
    filename = "dec_2023_data"
    selected_columns = all_adcs_params 
    hashed_columns = hash_firmware_names(selected_columns) # hashed > 200 chars, untouched otherwise
    df = get_df(filename, selected_columns)   

    # plot_scatter(selected_columns, df)

    random_state = 42
    algorithms = ['IF', 'OCSVM', 'LSTM', 'GMM', 'XGB']

    # 2) Uncomment either multi_point_visualisation or benchmark_algorithm, don't do both
    # Results from benchmark_algorithm will be saved in ./results, rename to prevent overwrite
    # Alternatively, comment out the for loop and specify algorithm manually

    # delete_models()
    # multi_point_visualisation(df, 'LSTM', hashed_columns, 4, random_state)
    # benchmark_algorithm(df, 'LSTM', hashed_columns, n=1000, num_of_anomalies=10)

    for algorithm in algorithms:
        # multi_point_visualisation(df, algorithm, hashed_columns, 4, random_state)
        delete_models(algorithm=algorithm, firmware_names=selected_columns)
        benchmark_algorithm(df, algorithm, hashed_columns, n=300, num_of_anomalies=5)

    # 3) Uncomment single_point_inference to test a single point
    # When switching between single_point_inference and multi_point_visualisation, 
    # remember to delete the corresponding models in ./models to retrain it
    delete_models(algorithm='GMM', firmware_names=selected_columns)
    test_normal = [1, 4, 8, 8, 0]
    test_anomaly = [1, 4, 8, 8, 3]
    single_point_inference(df, 'GMM', test_normal, selected_columns)
    single_point_inference(df, 'GMM', test_anomaly, selected_columns)

    end = time.time()
    print(f'Execution time: {end - start}s')
