# Anomaly Detection
## Overview
Study of AI/ML techniques to autonomously detect, isolate and recover from spacecraft anomalies. Completed by DSO Intern as part of STAR's Research Thrust 2.

The dataset originally studied was Lumelite 4's Whole Orbit Data (WOD) from December 2023, and study typically revolved around ADCS Temperature Sensors. It can be configured in two modes, single point inference (for use on incoming housekeeping data) and multi point visualisation mode (for large amounts of data). Furthermore it can be configured to use different algorithms. 

The algorithms available are Isolation Forest (IF), One-Class Support Vector Machine (OCSVM), and Long-Short-Term Memory Autoencoder (LSTM Autoencoder) from Scikitlearn & Pytorch. By default, LSTM Autoencoder is used.

## Setup
Project was developed in ```Python 3.12.3```
1. ```pip install -r requirements.txt```
2. Configure main.py
    1. Configure the data source "filename"
    2. Configure the firmware names in an array
    3. Choose main.py behaviour (see below section)
3. ```python main.py```

### For use in MCS:
```
ANOMALY_DETECTION_DEST = r"C:\Users\John\Anomaly Detection"
sys.path.append(ANOMALY_DETECTION_DEST)
from main import single_point_inference

# Take note that the data set uses alias names, Log Params uses firmware names
LOG_PARAMS = ['ADCS_ADC_TSXP_temp', 'ADCS_ADC_TSXN_temp', 'ADCS_ADC_TSYP_temp', 'ADCS_ADC_TSYN_temp', 'ADCS_ADC_TSZP_temp']

def anomaly_detection():
    firmware_name = ['ADC_TS-1 (XP) Temperature', 'ADC_TS-2 (XN) Temperature',
    'ADC_TS-3 (YP) Temperature', 'ADC_TS-4 (YN) Temperature', 'ADC_TS-5 (ZP) Temperature']
    inference_point = np.array([])
    logged_params = mcs.read() # Specifically for MCS
    for param in firmware_name:
        np.array.append(int(logged_params[param]))
    prediction = single_point_inference(array, firmware_name)
    # print(f"Prediction: {"Anomaly" if prediction == 1 else "Normal"}")
    return prediction # Either 1 or 0
```

## Data Requirements
The CSV required in /data/{filename}.csv would ideally have a similar format to below where each row represents a different parameter, and each column represents a timestamp. This format is later pivoted to every row being a timestamp and each column being a parameter, saved as /data/{filename}_pivot.csv. 
```
"PARAMETER","1702388586","1702388637","1702388693","1702388755","1702388814","1702388884","1702388939","1702388993","1702389049","1702389103","1702389170","1702389228","1702389289","1702389339","1702389393","1702389457","1702389558","1702389605","1702389673","1702395518","1702391868","1702391917","1702391970","1702392029","1702392080","1702392143","1702392197","1702392253","1702392358","1702392411","1702392467","1702392530","1702392588","1702392639","1702392700","1702392748","1702392810","1702392874","1702392938","1702392988","1702393093","1702393154","1702393209","1702393258","1702393321","1702393370","1702393429","1702393489","1702393541","1702393594","1702393645","1702393708","1702393762","1702393808","1702393867","1702393918","1702393975","1702394031","1702394096","1702394151","1702394191","1702394256","1702394308","1702394358","1702394419","1702394488","1702394545","1702394639","1702394702","1702394765","1702394819","1702394876","1702394919","1702394981","1702395047","1702395109","1702395158","1702395218","1702395278","1702395379","1702395443","1702395499","1702395548","1702395603","1702395661","1702394755","1702394821","1702394900","1702394971","1702395040","1702395111","1702395171","1702395227","1702395288","1702395342","1702395397"
"YYYY-mm-ddTHH:MM:SS (Satellite)","2023-12-12T13:43:06","2023-12-12T13:43:57","2023-12-12T13:44:53","2023-12-12T13:45:55","2023-12-12T13:46:54","2023-12-12T13:48:04","2023-12-12T13:48:59","2023-12-12T13:49:53","2023-12-12T13:50:49","2023-12-12T13:51:43","2023-12-12T13:52:50","2023-12-12T13:53:48","2023-12-12T13:54:49","2023-12-12T13:55:39","2023-12-12T13:56:33","2023-12-12T13:57:37","2023-12-12T13:59:18","2023-12-12T14:00:05","2023-12-12T14:01:13","2023-12-12T15:38:38","2023-12-12T14:37:48","2023-12-12T14:38:37","2023-12-12T14:39:30","2023-12-12T14:40:29","2023-12-12T14:41:20","2023-12-12T14:42:23","2023-12-12T14:43:17","2023-12-12T14:44:13","2023-12-12T14:45:58","2023-12-12T14:46:51","2023-12-12T14:47:47","2023-12-12T14:48:50","2023-12-12T14:49:48","2023-12-12T14:50:39","2023-12-12T14:51:40","2023-12-12T14:52:28","2023-12-12T14:53:30","2023-12-12T14:54:34","2023-12-12T14:55:38","2023-12-12T14:56:28","2023-12-12T14:58:13","2023-12-12T14:59:14","2023-12-12T15:00:09","2023-12-12T15:00:58","2023-12-12T15:02:01","2023-12-12T15:02:50","2023-12-12T15:03:49","2023-12-12T15:04:49","2023-12-12T15:05:41","2023-12-12T15:06:34","2023-12-12T15:07:25","2023-12-12T15:08:28","2023-12-12T15:09:22","2023-12-12T15:10:08","2023-12-12T15:11:07","2023-12-12T15:11:58","2023-12-12T15:12:55","2023-12-12T15:13:51","2023-12-12T15:14:56","2023-12-12T15:15:51","2023-12-12T15:16:31","2023-12-12T15:17:36","2023-12-12T15:18:28","2023-12-12T15:19:18","2023-12-12T15:20:19","2023-12-12T15:21:28","2023-12-12T15:22:25","2023-12-12T15:23:59","2023-12-12T15:25:02","2023-12-12T15:26:05","2023-12-12T15:26:59","2023-12-12T15:27:56","2023-12-12T15:28:39","2023-12-12T15:29:41","2023-12-12T15:30:47","2023-12-12T15:31:49","2023-12-12T15:32:38","2023-12-12T15:33:38","2023-12-12T15:34:38","2023-12-12T15:36:19","2023-12-12T15:37:23","2023-12-12T15:38:19","2023-12-12T15:39:08","2023-12-12T15:40:03","2023-12-12T15:41:01","2023-12-12T15:25:55","2023-12-12T15:27:01","2023-12-12T15:28:20","2023-12-12T15:29:31","2023-12-12T15:30:40","2023-12-12T15:31:51","2023-12-12T15:32:51","2023-12-12T15:33:47","2023-12-12T15:34:48","2023-12-12T15:35:42","2023-12-12T15:36:37"
"YYYY-mm-ddTHH:MM:SS (MCS)","2023-12-12T23:26:20","2023-12-12T23:26:24","2023-12-12T23:26:28","2023-12-13T09:43:11","2023-12-13T09:43:15","2023-12-13T09:43:19","2023-12-13T09:43:23","2023-12-13T09:43:27","2023-12-15T11:25:42","2023-12-13T09:43:35","2023-12-15T11:28:13","2023-12-13T13:10:56","2023-12-13T09:43:47","2023-12-13T13:11:38","2023-12-13T09:43:55","2023-12-13T09:44:06","2023-12-13T09:44:10","2023-12-13T13:12:22","2023-12-13T13:12:26","2023-12-13T13:12:30","2023-12-13T11:26:16","2023-12-13T11:26:20","2023-12-15T14:50:55","2023-12-13T11:26:28","2023-12-13T18:16:20","2023-12-13T11:26:36","2023-12-13T11:26:40","2023-12-13T11:26:44","2023-12-13T11:26:56","2023-12-13T11:26:59","2023-12-13T11:27:02","2023-12-13T11:27:05","2023-12-13T11:27:09","2023-12-13T18:16:58","2023-12-13T11:27:15","2023-12-13T18:17:11","2023-12-15T11:27:00","2023-12-13T11:27:26","2023-12-13T11:27:29","2023-12-13T11:27:33","2023-12-13T11:27:37","2023-12-13T11:27:41","2023-12-13T11:27:45","2023-12-12T23:29:13","2023-12-13T11:27:53","2023-12-13T11:28:04","2023-12-13T18:17:43","2023-12-13T11:28:11","2023-12-13T11:28:14","2023-12-13T11:28:18","2023-12-13T18:18:05","2023-12-13T18:18:09","2023-12-13T11:28:28","2023-12-13T11:28:31","2023-12-13T11:28:34","2023-12-13T11:28:38","2023-12-16T21:44:16","2023-12-13T11:28:46","2023-12-13T11:28:50","2023-12-13T11:28:54","2023-12-15T20:00:37","2023-12-13T14:55:04","2023-12-13T11:29:13","2023-12-13T11:29:16","2023-12-13T11:29:19","2023-12-13T11:29:23","2023-12-13T14:51:06","2023-12-13T14:51:10","2023-12-13T11:29:33","2023-12-13T11:29:36","2023-12-13T11:29:40","2023-12-13T11:29:43","2023-12-13T11:29:47","2023-12-13T11:29:50","2023-12-13T11:29:54","2023-12-13T18:18:26","2023-12-13T18:18:43","2023-12-13T14:51:51","2023-12-13T14:51:55","2023-12-13T14:51:59","2023-12-13T14:52:03","2023-12-13T14:52:07","2023-12-13T14:52:11","2023-12-13T14:52:15","2023-12-13T14:52:19","2023-12-13T18:19:12","2023-12-17T11:26:41","2023-12-13T14:52:36","2023-12-13T14:52:40","2023-12-13T18:19:32","2023-12-13T14:52:47","2023-12-13T14:52:50","2023-12-13T14:53:43","2023-12-13T14:53:47","2023-12-13T14:53:51","2023-12-13T14:53:55"
```

## Configuring main.py
You can configure it to 

1. Perform Multi-Point Visualisation 
    * This will plant a certain number of anomalies in the test dataset and visualise the model's predictions
2. Benchmark Algorithm Performance
    * This will run multi-point inference n (default=1000) times across a range of planted anomaly numbers, and return the average Precision, Recall and F1-Score in ```./results/results_{algorithm}.csv```
    * Usually takes ~30 mins at n=1000
3. Perform Single Point Inference

single_point_inference(inference_point, firmware_name) can be imported elsewhere where 
firmware name is an array of all involved firmware names, and inference point is a numpy array of values corresponding to those firmware names. Designed for use with MCS

The first time that a specific series of firmware names is run for a certain algorithm, it will train the data and create a new model. If you wish to use the same firmware names but create a new model i.e. swapped the dataset or changed from multi to single inference, please **delete the existing model in ```./models```** before continuing. The training sets will be different and therefore should be retrained.

## Acknowledgements 
- James Hong, the DSO Intern in charge of this project
- Muhammed Anas S/O Tariq, James' supervisor for his internship
- Evangeline Ng, a DSO Intern whose assistance in the project was invaluable
- Tan Zhi Hao, Evangeline's supervisor who also provided assistance
  
## References
- [Relaty Tutorial](https://www.relataly.com/multivariate-outlier-detection-using-isolation-forests-in-python-detecting-credit-card-fraud/4233/#google_vignette)
- [Pytorch LSTM Documentation](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)