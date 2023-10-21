# AI_Midterm_12204505
# README

## LSTM-Based Predictive Maintenance for Conveyor Belts

This repository contains code for a Long Short-Term Memory (LSTM) neural network model designed for predictive maintenance for conveyor belts. The model is implemented using the Keras library and is trained on synthetic data generated with varying failure probabilities. The goal is to detect anomalies or failures in the data sequences.

### Table of Contents
1. [Introduction](#introduction)
2. [Data Generation](#data-generation)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Architecture](#model-architecture)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Real-Time Simulation](#real-time-simulation)
8. [Usage](#usage)
9. [Dependencies](#dependencies)

### Introduction
Anomaly detection is a critical task in various domains, including predictive maintenance, quality control, and fault detection. This LSTM-based model is designed to detect anomalies in sequential data, such as sensor readings over time. It generates synthetic data with varying failure probabilities and trains the LSTM model to learn patterns that indicate failures. The model is then evaluated on a test dataset and used for real-time anomaly detection.

### Data Generation
The `generate_one_sequence` and `generate_sequential_data` functions generate synthetic sequential data for training and testing the model. Each sequence represents a time series of temperature, vibration, and conveyor speed values. Failure events are introduced with a given probability, resulting in distinct data patterns for normal and failure cases.

### Data Preprocessing
Before training the model, the data is preprocessed to ensure consistency and stability. StandardScaler is used to normalize the data by subtracting the mean and dividing by the variance. This step ensures that all input features have similar scales and helps the model converge faster during training.

### Model Architecture
The LSTM-based model architecture consists of the following layers:
- A single LSTM layer with 8 units, taking input sequences of shape `(time_steps, 3)` (representing temperature, vibration, and speed).
- Three dense layers with ReLU activation functions.
- Dropout layers to prevent overfitting.
- The output layer with a sigmoid activation function to predict failure probabilities.

### Training
The model is trained using Mean Squared Error (MSE) loss and the Adam optimizer. Training is performed for 25 epochs with a batch size of 5. The model learns to predict the probability of failure based on the input sequences.

### Evaluation
After training, the model is evaluated on a test dataset using accuracy as the evaluation metric. The accuracy score indicates the model's ability to correctly predict failures in the test data.

### Real-Time Simulation
The code includes a real-time simulation section that runs for 10 iterations. It generates new data sequences with a specified failure probability and uses the trained model to detect failures. If the model predicts a failure with a probability greater than 0.5, it prints a corresponding message.
