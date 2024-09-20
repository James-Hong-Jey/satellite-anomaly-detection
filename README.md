# Anomaly Detection
## Overview
Study of AI/ML techniques to autonomously detect, isolate and recover from spacecraft anomalies. Completed by DSO Intern as part of STAR's Research Thrust 2.

The dataset originally studied was Lumelite 4's Whole Orbit Data (WOD) from December 2023, and study typically revolved around ADCS Temperature Sensors. It can be configured in two modes, single point inference (for use on incoming housekeeping data) and multi point visualisation mode (for large amounts of data). Furthermore it can be configured to use different algorithms. 

The algorithms available are Isolation Forest (IF), One-Class Support Vector Machine (OCSVM), and Long-Short-Term Memory Autoencoder (LSTM Autoencoder) from Scikitlearn & Pytorch. By default, LSTM Autoencoder is used.

## Setup
Project was developed in Python 3.12.3 
1. '''pip install -r requirements.txt'''

## Configuring main.py
You can configure it to perform single point datapoint inference, or multi-point inference with visualisation over a randomised test set under main.py

single_point_inference(inference_point, firmware_name) can be imported elsewhere where 
firmware name is an array of all involved firmware names, and inference point is a numpy array of values corresponding to those firmware names.

## Acknowledgements
- Evangeline Ng 