# Multivariate version of data_processing.py
import pandas as pd
import numpy as np
import torch
import csv, warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
from sklearn import metrics
import os
from visualise_algorithms import plot_scatter

# Ignore warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

def get_csv_vitals(filename='FF_Vitals.csv'):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        params = next(reader)
    return params

def pivot_and_store(filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    destination = os.path.join(current_dir, f"data/{filename}.csv")
    read_df = pd.read_csv(destination)
    pivot_df = read_df.set_index('PARAMETER').T # Check the CSV itself
    pivot_df.reset_index(inplace=True)
    pivot_df.rename(columns={'index': 'timestamp'}, inplace=True)
    print(f'saving csv: data/{filename}_pivot.csv')
    pivot_destination = os.path.join(current_dir, f'data/{filename}_pivot.csv')
    pivot_df.to_csv(pivot_destination, index=False)
    return pivot_df

# Depreciated
def read_data(file_path):
    df = pd.read_csv(file_path)
    df = df.loc[df['timestamp'] >= 1700000000] # Just to remove erroneous entries
    df['value'] = pd.to_numeric(df['value'], errors="coerce")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[['value', 'timestamp', 'firmware_name']]
    df = df.drop_duplicates(subset=['timestamp', 'firmware_name'])
    # df = df.applymap(lambda x: pd.to_numeric(x, errors='coerce'))
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    return df

# Depreciated
def pivot_data(df):
    unique_firmware_names = df['firmware_name'].unique()
    pivot_df = df.pivot(index='timestamp', columns='firmware_name', values='value').reset_index()
    # pivot_df.dropna(inplace=True)
    # pivot_df.reset_index(drop=True, inplace=True)
    return pivot_df, unique_firmware_names

def split_train_test(df, test_proportion=0.2):
    test_size = int(len(df) * test_proportion) 
    X_test = df[:test_size]
    X_train = df[test_size:] 
    
    return X_train, X_test

def split_train_val_test(df, test_prop=0.2, val_prop=0.15):
    test_size = int(len(df) * test_prop)
    val_size = int(len(df) * val_prop)
    train_size = len(df) - test_size - val_size
    X_train = df[test_size:test_size+train_size]
    X_val = df[test_size+train_size:]
    
    return X_train, X_val

# This function standardize the features
def standardize_features(X_train_data, X_val_data, X_test_data):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_data)
    X_test_scaled = scaler.transform(X_test_data)

    if X_val_data is not None:
        X_val_scaled = scaler.transform(X_val_data)
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    else:
        return X_train_scaled, X_test_scaled

# This function converts the data to PyTorch tensors specifically for LSTM Autoencoder model
def convert_to_tensor(X_train_scaled, X_val_scaled, X_test_scaled):
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    return X_train_tensor, X_val_tensor, X_test_tensor

# This function extracts time-related features
def extract_time_features(df):
    df['day'] = df['timestamp'].dt.day
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['second'] = df['timestamp'].dt.second
    df = df.drop(columns=['timestamp'])

    return df

# Legacy Code 
def label_test_data_anomalies(X_test, firmware_name):
    print(f"Now operating on {firmware_name}")
    
    match firmware_name:
        case "CIM_ant_Atemp" | "CIM_ant_Btemp":
            X_test['anomaly'] = X_test[firmware_name].apply(lambda x : 1 if int(x) > 20 else 0)
            # X_test['anomaly'] = np.where(X_test[firmware_name] > 20, '1', '-1')
    
        case "ADCS_ADC_SDC.voltage":
            X_test['anomaly'] = X_test[firmware_name].apply(lambda x : 0 if int(x) >= 3135 and int(x) <= 3465 else 1)
        
        case "ADCS_ADC_MCU.temp":
            X_test['anomaly'] = X_test[firmware_name].apply(lambda x : 0 if int(x) >= -25 and int(x) <= 55 else 1)

        case "CIM_CPU_Temp":
            X_test['anomaly'] = X_test[firmware_name].apply(lambda x : 0 if int(x) >= -100 and int(x) <= 100 else 1)

        case "CIM_SBAND_BBTemp":
            X_test['anomaly'] = X_test[firmware_name].apply(lambda x : 0 if int(x) >= -100 and int(x) <= 100 else 1)
    
    y_test = X_test[['timestamp', 'anomaly']].copy()
    X_test = X_test.drop(columns=['anomaly'])
    return y_test, X_test

def insert_anomalies(df, num_of_anomalies=3):
    random_indexes = random.sample([int(i) for i in df.index.tolist()], num_of_anomalies)
    # firmware_names = ['ADCS_ACS_RW.X.temp', 'ADCS_ACS_RW.Y.temp', 'ADCS_ACS_RW.Z.temp', 'ADCS_ADC_MCU.temp', 'ADCS_ADC_TS.temp[0]', 'ADCS_ADC_TS.temp[1]']
    firmware_names = df.drop(columns=['anomaly']).columns.tolist()
    for random_index in random_indexes:
        near_zero_increment = 8
        random_firmware = random.choice(firmware_names)
        multiplier = random.randrange(1, 30)/10
        difference = max(abs(df.loc[random_index, random_firmware] * multiplier - df.loc[random_index, random_firmware]), 8)
        df.loc[random_index, random_firmware] = df.loc[random_index, random_firmware] + int(difference)
        df.loc[random_index, 'anomaly'] = 1
    return df

def evaluate_algorithm(df, num_of_anomalies=2):
    total_rows = df.shape[0]
    print(f'total_rows = {total_rows}')
    print(f'Num of Anomalies planted: {num_of_anomalies}')
    true_positive = df[(df['anomaly'] == 1) & (df['anomaly_predicted'] == 1)].shape[0]
    print(f'true_positive (red) = {true_positive}')
    false_negative = df[(df['anomaly'] == 1) & (df['anomaly_predicted'] == 0)].shape[0]
    print(f'false_negative (purple) = {false_negative}')
    true_negative = df[(df['anomaly'] == 0) & (df['anomaly_predicted'] == 0)].shape[0]
    print(f'true_negative (blue)  = {true_negative}')
    false_positive = df[(df['anomaly'] == 0) & (df['anomaly_predicted'] == 1)].shape[0]
    print(f'false_positive (green)  = {false_positive}')

    precision = metrics.precision_score(df['anomaly'].values, df['anomaly_predicted'], pos_label=1)
    recall = metrics.recall_score(df['anomaly'].values, df['anomaly_predicted'], pos_label=1)
    f1_score = metrics.f1_score(df['anomaly'].values, df['anomaly_predicted'], pos_label=1)
    print(f'precision = {precision}') # True positive / True + False Positives 
    print(f'recall = {recall}') # True positive / True Positive + False Negative 
    print(f'f1_score = {f1_score}')



# Accepts pre-pivoted data 
def data_processing(df, num_of_anomalies=2, inference_point=None):
    # unique_firmware_names = pivot_df.columns.unique().tolist()
    # random_state = 42
    random_state = random.randint(0, 1000)

    # For LSTM Autoencoder - split the data into 20% test, 65% train, and 15% validation
    if inference_point:
        # Non-LSTM
        X_train = df.copy(deep=True)
        X_test = pd.DataFrame([inference_point], columns=X_train.columns)

        # LSTM
        X_train_lstm, X_val_lstm = train_test_split(df, test_size=0.2, random_state=random_state)
        X_test_lstm = X_train_lstm.copy(deep=True)
        X_test_lstm['anomaly'] = 0
    else:
        # Non-LSTM
        X_train, X_test = train_test_split(df, test_size=0.2, random_state=random_state)
        X_test['anomaly'] = 0
        X_test = insert_anomalies(X_test, num_of_anomalies)

        # LSTM
        train_val_df, X_test_lstm = train_test_split(df, test_size=0.2, random_state=random_state)
        X_train_lstm, X_val_lstm = train_test_split(train_val_df, test_size=0.18, random_state=random_state)
        X_test_lstm['anomaly'] = 0
        X_test_lstm = insert_anomalies(X_test_lstm, num_of_anomalies)

    # print('Before Anomalies added:')
    # plot_scatter(X_test_lstm)
    # print('After Anomalies added:')
    # plot_scatter(X_test_lstm)
    # print(f'Anomaly indices = {X_test_lstm.index[X_test_lstm['anomaly'] == 1].tolist()}')
    # Standardize the features (X_train_scaled_2 & X_val_scaled_2 is for LSTM Autoencoder)
    X_train_scaled, X_test_scaled = standardize_features(X_train, None, X_test)
    X_train_lstm_scaled, X_val_lstm_scaled, X_test_lstm_scaled = standardize_features(X_train_lstm, X_val_lstm, X_test_lstm)

    # Convert to tensor (for LSTM Autoencoder)
    X_train_tensor, X_val_tensor, X_test_tensor = convert_to_tensor(X_train_lstm_scaled, X_val_lstm_scaled, X_test_lstm_scaled)

    return X_train_scaled, X_test_scaled, X_train_tensor, X_val_tensor, X_test_tensor, X_test, X_test_lstm