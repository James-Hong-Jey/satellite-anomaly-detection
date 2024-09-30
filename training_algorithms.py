import pandas as pd
import numpy as np
from sklearn import metrics
from pyod.models.iforest import IForest
from pyod.models.ocsvm import OCSVM
import torch
import torch.nn as nn
import itertools
import joblib 
import ast
import matplotlib.pyplot as plt
import os

# TODO: Clean up the current directory stuff, not everything is consistent right now

# This function trains the Isolation Forest model and predicts the anomalies for the test data
def train_iforest_model(X_train_scaled, X_test_scaled, y_test, firmware_name):
    """ This portion of the code trains the model with the training data and applies the model on the test data """
    
    # Load the best model and its hyperparameters for the specified TM Dataset
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_folder = "models"
    model_path = os.path.join(current_dir, f'{model_folder}/best_isolation_forest_model_{firmware_name}.pkl')
    try: 
        best_model = joblib.load(model_path)
    except:
        # Uncomment the following line to do hyperparameter tuning and save the best model and its hyperparameters
        hyperparameter_tuning_iforest(X_train_scaled, X_test_scaled, y_test, firmware_name)
        best_model = joblib.load(model_path)

    # Predict the anomaly for the test data
    # y_pred = best_model.predict(X_test_scaled).astype(str)
    y_pred = best_model.predict(X_test_scaled)

    return y_pred

# This function does hyperparameter tuning for the Isolation Forest model
def hyperparameter_tuning_iforest(X_train_scaled, X_test_scaled, y_test, firmware_name):
    # Define hyperparameter space
    contamination_values = [0.1, 0.001, 1e-6]
    # contamination_values = np.linspace(0.001, 0.400, 10)
    max_features_list = [2, 3, 4]
    n_estimators_list = [100, 150, 200]
    max_samples_list = [0.7, 0.8, 0.9]
    
    # Initialize variables to track the best model
    best_score = -1
    best_params = {}
    best_model = None

    # Generate all combinations of hyperparameters
    for contamination, max_feature, n_estimator, max_sample in itertools.product(contamination_values, max_features_list, n_estimators_list, max_samples_list):
        # Print the current set of hyperparameters
        print(f"Hyperparameters: contamination={contamination}, max_features={max_feature}, n_estimators={n_estimator}, max_samples={max_sample}")

        # Initialize the OCSVM model with the current set of hyperparameters
        model = IForest(contamination=contamination, max_features=max_feature, n_estimators=n_estimator, max_samples=max_sample)

        # Fit the OCSVM model
        model.fit(X_train_scaled)

        # Predict the anomaly for the test data
        # y_pred = model.predict(X_test_scaled).astype(str)
        y_pred = model.predict(X_test_scaled)
        # print(y_pred)

        # Evaluate model using precision
        # score = metrics.precision_score(y_test['anomaly'].values, y_pred, pos_label='1')
        y_test_values = y_test['anomaly'].to_numpy()
        # print(f'y_test[\'anomaly\'] is {y_test_values.dtype}')
        # print(f'y_pred[\'anomaly\'] is {y_pred.dtype}')
        score = metrics.precision_score(y_test['anomaly'].to_numpy(), y_pred, pos_label=1)

        # Update the best model if the current model is better
        if score > best_score:
            best_score = score
            best_params = {'contamination': contamination, 'max_features': max_feature, 'n_estimators': n_estimator, 'max_samples': max_sample}
            best_model = model

    # Save the best model and its hyperparameters
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_folder = "models"
    txt_path = os.path.join(current_dir, f'{model_folder}/best_hyperparameters_isolation_forest_{firmware_name}.txt')  
    pkl_path = os.path.join(current_dir, f'{model_folder}/best_hyperparameters_isolation_forest_{firmware_name}.pkl')  
    joblib.dump(best_model, pkl_path)
    print("Model saved")
    with open(txt_path, 'w') as f:
        f.write(str(best_params))

    return

# This function trains the One-Class SVM model and predicts the anomalies for the test data
def train_ocsvm_model(X_train_scaled, X_test_scaled, y_test, firmware_name):
    """ This portion of the code trains the model with the training data and applies the model on the test data """
    # Uncomment the following line to do hyperparameter tuning and save the best model and its hyperparameters
    
    # Load the best model and its hyperparameters for the specified TM Dataset
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_folder = "models"
    model_path = os.path.join(current_dir, f'{model_folder}/best_OCSVM_model_{firmware_name}.pkl')
    try: 
        best_model = joblib.load(model_path)
    except:
        # Uncomment the following line to do hyperparameter tuning and save the best model and its hyperparameters
        hyperparameter_tuning_ocsvm(X_train_scaled, X_test_scaled, y_test, firmware_name)
        best_model = joblib.load(model_path)

    # Predict the anomaly for the test data
    y_pred = best_model.predict(X_test_scaled)

    return y_pred

# This function does hyperparameter tuning for the OC-SVM model
def hyperparameter_tuning_ocsvm(X_train_scaled, X_test_scaled, y_test, firmware_name):
    # Define hyperparameter space
    contamination_values = [0.01, 0.001, 1e-6]
    kernel_values = ['rbf', 'poly', 'sigmoid']
    gamma_values = ['auto', 'scale']
    nu_values = [0.3, 0.5, 0.7]
    
    # Initialize variables to track the best model
    best_score = -1
    best_params = {}
    best_model = None

    # Generate all combinations of hyperparameters
    for contamination, kernel, gamma, nu in itertools.product(contamination_values, kernel_values, gamma_values, nu_values):
        # Print the current set of hyperparameters
        print(f"Hyperparameters: contamination={contamination}, kernel={kernel}, gamma={gamma}, nu={nu}")

        # Initialize the OCSVM model with the current set of hyperparameters
        model = OCSVM(contamination=contamination, kernel=kernel, gamma=gamma, nu=nu)

        # Fit the OCSVM model
        model.fit(X_train_scaled)

        # Predict the anomaly for the test data
        y_pred = model.predict(X_test_scaled)

        # Evaluate model using precision
        score = metrics.precision_score(y_test['anomaly'].values, y_pred, pos_label=1)

        # Update the best model if the current model is better
        if score > best_score:
            best_score = score
            best_params = {'contamination': contamination, 'kernel': kernel, 'gamma': gamma, 'nu': nu}
            best_model = model

    # Save the best model and its hyperparameters
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_folder = "models"
    txt_path = os.path.join(current_dir, f'{model_folder}/best_hyperparameters_OCSVM_{firmware_name}.txt')  
    pkl_path = os.path.join(current_dir, f'{model_folder}/best_OCSVM_model_{firmware_name}.pkl')  
    joblib.dump(best_model, pkl_path)
    print("Model saved")
    with open(txt_path, 'w') as f:
        f.write(str(best_params))

    return

# Define the LSTM Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(Autoencoder, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.LSTM(hidden_size, input_size, num_layers, batch_first=True)


    def forward(self, x):
        encoded, _ = self.encoder(x)
        encoded = self.dropout(encoded)  # Apply dropout after encoder
        decoded, _ = self.decoder(encoded)
        decoded = self.dropout(decoded)  # Apply dropout after decoder
        return decoded

# This function does hyperparameter tuning for the LSTM Autoencoder model
def hyperparameter_tuning_lstm_ae(X_train_tensor, X_val_tensor, firmware_name):
    input_size = X_train_tensor.shape[1] # No. columns/input features
    hidden_size_list = [8, 16, 32, 64, 128, 256] # No. of layers in each LSTM layer
    learning_rate_list = [0.01, 0.1]
    num_epochs = 250
    num_layers_list = [2, 3, 4, 5] # depth of the LSTM 
    dropout_list = [0.2, 0.3, 0.4]

    # Initialize variables to track the best model
    best_val_loss = 1
    best_params = {}
    best_model = None

    # Generate all combinations of hyperparameters
    for hs, lr, nl, dropout in itertools.product(hidden_size_list, learning_rate_list, num_layers_list, dropout_list):
        # Print the current set of hyperparameters
        print(f"Hyperparameters: hidden_size={hs}, learning_rate={lr}, num_layer={nl}, dropout={dropout}")

        # Create an instance of the Autoencoder model
        model = Autoencoder(input_size, hs, nl, dropout)

        # Define the loss function, optimizer and learning rate schedular
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4) 

        loss_values = []

        # Train the model based on the parameters defined
        best_loss = np.inf
        epochs_no_improve = 0
        n_epochs_stop = 15

        for epoch in range(num_epochs):
            # Forward pass
            outputs = model(X_train_tensor)
            loss = criterion(outputs, X_train_tensor)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Early stopping (when validation loss stops decreasing)
            if loss < best_loss:
                epochs_no_improve = 0
                best_loss = loss
            else:
                epochs_no_improve += 1
                if epochs_no_improve == n_epochs_stop:
                    print('Early stopping!')
                    break

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
                loss_values.append(loss.item())

        # Evaluate the model using the validation data
        model.eval()
        with torch.no_grad():
            # Calculate error metrics on validation data
            val_output = model(X_val_tensor)
            val_mse = criterion(val_output, X_val_tensor)
            print(f'Mean Squared Error (Validation Data): {val_mse.item():.4f}')

            # Update the best model if the current model is better
            if val_mse < best_val_loss:
                best_val_loss = val_mse
                best_params = {'input_size': input_size, 'hidden_size': hs,'learning_rate': lr,'num_layers': nl,'dropout': dropout}
                best_model = model
            
    # Save the best model and its hyperparameters
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_folder = "models"
    txt_path = os.path.join(current_dir, f'{model_folder}/best_hyperparameters_lstm_ae_{firmware_name}.txt')  
    pth_path = os.path.join(current_dir, f'{model_folder}/best_model_lstm_ae_{firmware_name}.pth')
    torch.save(best_model.state_dict(), pth_path)
    with open(txt_path, 'w') as f:
        f.write(str(best_params))
    
    return


# Helper function for infer_lstm_ae_model
# to get the train error statistics
def get_train_error_std(X_train_tensor, model):
    model.eval()
    with torch.no_grad():
        # Apply model to test data
        train_predictions = model(X_train_tensor)

        # Calculate train error for each data point (difference between original value and predicted value generated by model)
        train_error = torch.mean(torch.square(train_predictions - X_train_tensor), dim=1)
        return train_error.median().item(), train_error.mean().item(), train_error.std().item()

# This function trains the LSTM Autoencoder model and predicts the anomalies for the test data
def infer_lstm_ae_model(X_train_tensor, X_val_tensor, X_test_tensor, firmware_name, num_of_std=3, plot_error=False):
    """ This portion of the code sets hyperparameters and trains the model using the training data. """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_folder = "models"
    model_path = os.path.join(current_dir, f'{model_folder}/best_hyperparameters_lstm_ae_{firmware_name}.txt')    
    try: 
        with open(model_path, 'r') as file:
            hyperparameters = file.read()
    except:
        # Uncomment the following line to do hyperparameter tuning and save the best model and its hyperparameters
        hyperparameter_tuning_lstm_ae(X_train_tensor, X_val_tensor, firmware_name)
        with open(model_path, 'r') as file:
            hyperparameters = file.read()

    hyperparameters = ast.literal_eval(hyperparameters)
    input_size = hyperparameters['input_size']
    hidden_size = hyperparameters['hidden_size']
    learning_rate = hyperparameters['learning_rate']
    num_layers = hyperparameters['num_layers']
    dropout = hyperparameters['dropout']
    
    # Create an instance and load the saved model
    pth_path = os.path.join(current_dir, f'{model_folder}/best_model_lstm_ae_{firmware_name}.pth')    
    model = Autoencoder(input_size, hidden_size, num_layers, dropout)
    model.load_state_dict(torch.load(pth_path, weights_only=True))
            
    model.eval()
    with torch.no_grad():
        # Apply model to test data
        test_predictions = model(X_test_tensor)

        # Calculate test error for each data point (difference between original value and predicted value generated by model)
        test_error = torch.mean(torch.square(test_predictions - X_test_tensor), dim=1)
        test_error_df = pd.DataFrame(test_error, columns=['error'])
        #pd.concat([X_test, test_error_df], axis=1).to_csv("test_error_df.csv", index=False) 

        # Apply model to training data
        train_predictions = model(X_train_tensor)

        # Calculate train error for each data point (difference between original value and predicted value generated by model)
        train_error = torch.mean(torch.square(train_predictions - X_train_tensor), dim=1)
        train_median, train_mean, train_std = train_error.median().item(), train_error.mean().item(), train_error.std().item()

    """ This portion of the code sets the threshold for classifying anomalies and identifying them in the test data. """
    # Percentile-based threshold for classifying anomalies. Qn: how to determine threshold?
    # threshold_percentile = 0.92
    # threshold = test_error_df['error'].quantile(threshold_percentile) 
    test_error_df['z_score'] = (test_error_df['error'] - train_median) / train_std
    outliers = test_error_df[test_error_df['z_score'] > num_of_std]
    inliers = test_error_df[test_error_df['z_score'] <= num_of_std]

    if plot_error:
        # Plotting the histogram for the second column
        print(f'train_mean = {train_mean}')
        print(f'train_median = {train_median}')
        print(f'train_std = {train_std}')
        print(f'test_error_df.median() = {test_error_df["error"].median()}')
        print(f'test_error_df.mean() = {test_error_df["error"].mean()}')
        print(f'test_error_df.std() = {test_error_df["error"].std()}')
        plt.scatter(inliers.index, inliers['error'], c='blue', label='inliers')
        plt.scatter(outliers.index, outliers['error'], c='red', label='out')
        plt.title('Scatter Plot of Values')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.grid(True)
        plt.show()
        # print("threshold: ",threshold)

    # Get indices of anomalous data (based on threshold)
    # anomaly_indices = torch.where(test_error > threshold)[0]
    anomaly_indices = torch.where((test_error - train_median) / train_std > num_of_std)[0]
    # anomaly_indices = torch.where((test_error - test_error.median().item()) / train_std > num_of_std)[0]
    
    # Retrieve data rows from the test data using above indices
    anomaly_indices_list = anomaly_indices.detach().numpy().tolist()
    y_pred = pd.DataFrame({'anomaly_predicted': [0] * len(test_error_df)})
    y_pred.loc[anomaly_indices_list, 'anomaly_predicted'] = 1

    return y_pred
