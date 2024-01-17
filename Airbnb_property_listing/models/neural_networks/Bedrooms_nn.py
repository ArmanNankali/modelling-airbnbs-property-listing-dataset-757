import numpy as np
import pickle
import pandas as pd
import torch
import os
os.chdir(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)))
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split, ShuffleSplit, validation_curve
from torch.utils.data import DataLoader
import torch.nn.functional as F
import yaml
from sklearn.metrics import r2_score
from math import sqrt
import time
from itertools import product
import json
from joblib import dump
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from utilities import load_dataframe, load_airbnb, AirbnbNightlyPriceRegressionDataset, MLP, RMSELoss, train, save_model, test_set_test, custom_tune_NN_model_hyperparameters

# First the transformed nuermical data is loaded.
numerical_data = load_dataframe(r"..\tabular_data\YJH_transformed_no_bedrooms_numerical_data.pkl")

numerical_data.describe()

# Now the features are seperated from the labels ("Price_night").
features, labels = load_airbnb(numerical_data, "bedrooms")

# Random seed set for reproducibility and comparison.
np.random.seed(99)

# We instantiate the class with numerical_data dataframe creating the object "abb1" (Airbnb1).
abb1 = AirbnbNightlyPriceRegressionDataset(features, labels)
abb1.__len__()

# Features and labels are first split into training and test batches(70% and 30% respectively) 
features_train_val, features_test, labels_train_val, labels_test = train_test_split(features, labels, test_size=0.3)
# Training features and labels are then further split into training and validation (50%/50%).
features_train, features_val, labels_train, labels_val = train_test_split(features_train_val, labels_train_val)

# The standard scaler class is instantiated with the "scaler" object
scaler = StandardScaler()
# Fit the scaler on the training data and transform it
features_train = scaler.fit_transform(features_train)

# Transform the validation and test data using the same transformer first fit to training data to prevent data leakage.
features_val = scaler.transform(features_val)
features_test = scaler.transform(features_test)

# The AirbnbNightlyPriceRegressionDataset class is instantiated with the following three objects to convert training, validation and test set features and labels to torch.Tensor format.
train_dataset = AirbnbNightlyPriceRegressionDataset(features_train, labels_train)
val_dataset = AirbnbNightlyPriceRegressionDataset(features_val, labels_val)
test_dataset = AirbnbNightlyPriceRegressionDataset(features_test, labels_test)

# Create DataLoader instances for training, validation, and test sets with shuffling only available for training for reproducibility for validation and test metrics.
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# This is the hyperparameter config dictionary taken from the best performing price per night predictor.
hyperparam_config = {
   'input_size': 11,
   'optimiser': 'SGD',
   'learning_rate': 0.001,
   'model_depth': 2,
   'hidden_layer_width': 5,
   'output_size': 1,
   'weight_decay': 0.001,
   'activation_function': 'ELU',
   'drop-out': 0.2}
model1 = MLP(hyperparam_config)
metrics, hyperparams, bestr2__model_state = train(model1, train_dataloader, val_dataloader, hyperparam_config, epochs=700)

print(metrics, hyperparams)
# 'Average Train RMSE loss': 0.4800379148551396, 'Average Train R2': 0.7529801756892792
# 'Avg_val_r2_rmse': 0.48202103972434995, 'Avg_val_r2': 0.7705742267639637
# Here it is apparent that the training and validation performance is similar, indicating no overfitting.
# A grid search will be carried out to improve performance.

hyperparam_grid = {
    'input_size': [11],
    'optimiser': ["Adam"],
    'learning_rate': [0.001],
    'model_depth': [2],
    'hidden_layer_width': [3, 5],
    'output_size': [1],
    'weight_decay': [0.001, 0.0001],
    'activation_function': ["ReLU", "SELU"],
    'drop-out': [0.2, 0.03]}

best_model, best_hyperarameters, best_model_best_state, best_metrics, config_grid, _ = custom_tune_NN_model_hyperparameters(MLP, train_dataloader, val_dataloader, hyperparam_grid, "neural_networks/regression/Bedrooms")
print(f" Best hyperparameters: {best_hyperarameters} Metrics: {best_metrics} Hyperparameter grid tested: {config_grid}")

# Best hyperparameters
# 'configs': {'input_size': 11, 'optimiser': 'Adam', 'learning_rate': 0.001, 'model_depth': 2, 'hidden_layer_width': 3, 'output_size': 1, 'weight_decay': 0.0001, 'activation_function': 'SELU', 'drop-out': 0.03

# "Average Train RMSE loss": 0.42694179500852314, "Average Train R2": 0.8003577724405531
# "Avg_val_r2_rmse": 0.4596353590488434, "Avg_val_r2": 0.7847118149337675
# Based on how close both training and validation RMSE and R^2 are the model does not yet appear to be suffering from overfitting.
# Taking the best model state and hyperparameter configurations, the model's performance on unseen test data will be evaluated.

best_params = best_hyperarameters["configs"]
model2 = MLP(best_params)
model2.load_state_dict(best_model_best_state)
test_set_test(model2, test_dataloader)

# Average RMSE on the test data: 0.4151860848069191
# Average R^2 score on the test data: 0.7732703752921032
# Batch with best r2: 0.9307619136876811, best r2 run rmse: 0.29590851068496704

# Average R^2 relatively close to training and validation. An average of 77.33% of the variance in the unseen number of bedrooms can be explained by the model. 
# The RMSE is quite close as well, indicating the model generalises well and doesn't overfit.
# Considering the average RMSE is 0.41 on test data, as long as the bedrooms are rounded to the nearest whole integer, the predictions can still be quite useful.
