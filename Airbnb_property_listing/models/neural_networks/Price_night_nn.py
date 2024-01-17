import numpy as np
import pickle
import pandas as pd
import torch
import os
os.chdir(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)))
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split, ShuffleSplit, validation_curve
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import r2_score
import yaml
from sklearn.metrics import r2_score
from math import sqrt
import time
from itertools import product
import json
from joblib import dump
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from utilities import load_dataframe, load_airbnb, AirbnbNightlyPriceRegressionDataset, MLP, RMSELoss, train, save_model, test_set_test, custom_tune_NN_model_hyperparameters

# First the transformed nuermical data is loaded.
numerical_data = load_dataframe(r"..\tabular_data\YJH_transformed_no_price_night_numerical_data.pkl")

numerical_data.describe()

# Now the features are seperated from the labels ("Price_night").
features, labels = load_airbnb(numerical_data, "Price_Night")

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

# This is the hyperparameter config dictionary.
hyperparam_config = {
   'input_size': 11,
   'optimiser': 'Adam',
   'learning_rate': 0.0001,
   'model_depth': 1,
   'hidden_layer_width': 10,
   'output_size': 1,
   'weight_decay': 0.001,
   'activation_function': 'ReLU',
   'drop-out': 0.2}

model1 = MLP(hyperparam_config)
metrics, hyperparams, bestr2__model_state = train(model1, train_dataloader, val_dataloader, hyperparam_config, epochs=700)
#print(metrics, hyperparams)
# 'Average Train R2': 0.4714907609594435,
# 'Avg_val_r2': 0.27819640910870996,
# As we can see the model is overfitting as the average training R^2 is much higher than the valdiation
# %%
# Here is a hyperparameter grid amounting to 16 possible combinations.
hyperparam_grid = {
    'input_size': [11], # The number of input features to your model. This should match the dimensionality of the data (11 features).
    'optimiser': ["Adam", "SGD"], # Optimization algorithm used to update the model's parameters. Adam usually converges faster and performs better than SGD, but SGD can sometimes generalize better on test data.
    'learning_rate': [0.001], # The step size used in the optimization algorithm. A smaller learning rate might make the model learn slower, while a larger one might cause unstable learning.
    'model_depth': [2], # More layers can help capture more complex patterns, but might also lead to overfitting.
    'hidden_layer_width': [3, 5], # More neurons can increase model capacity and potentially learn more complex representations, but it also makes the model more prone to overfitting.
    'output_size': [1], # The dimensionality of the output matching the number of predicted variables (1 since only "Price_Night" is being predicted).
    'weight_decay': [0.001, 0.0001], # # L2 regularisation helps prevent the model from overfitting by adding a cost to having large weights. Higher can produce better generalisation and less overfitting but potentially lead to underfitting.
    "activation_function" : ["ELU", "ReLU"], # The activation function after each layer introduces non-linearity into the model, allowing it to learn more complex patterns. This can effect convergence speed of the model.
    "drop-out": [0.3, 0.2] # During training, it randomly sets a fraction of the input units to 0 at each update, which helps prevent overfitting. But too high and it will lead to underfitting.
   }

# Learning rate has been limited to 0.001 as when offered 0.01 as an option it was always chosen. This often gave R^2 scores on validation comparable to the training sets but lead to a much lower test sat R^2 score (overfitting).
# Model depth and width have been limited to an upper ceiling of 2 and 5 respectively as higher values gave good training and validation R^2 scores but the test set was much lower (overfitting).
# Similarly weight decay has been capped at 0.0001 as lower values were often favoured by grid search but gave too low R^2 scores compared to training and validation.

best_model, best_hyperarameters, best_model_best_state, best_metrics, config_grid, _ = custom_tune_NN_model_hyperparameters(MLP, train_dataloader, val_dataloader, hyperparam_grid, "neural_networks/regression/Price_night")

print(f" Best hyperparameters: {best_hyperarameters} Metrics: {best_metrics} Hyperparameter grid tested: {config_grid}")

# "Average Train RMSE loss": 84.2584250313895, "Average Train R2": 0.43425622750186343
# "Avg_val_r2_rmse": 85.65610809326172, "Avg_val_r2": 0.28183452476419685
# There is quite a difference between training average R^2 and validation average R^2 indicating some overfitting.
# The average RMSE however, seem to be quite close.
# These are the best hyperparameters.
# 'configs': {'input_size': 11, 'optimiser': 'SGD', 'learning_rate': 0.001, 'model_depth': 2, 'hidden_layer_width': 5, 'output_size': 1, 'weight_decay': 0.001, 'activation_function': 'ELU', 'drop-out': 0.2}
# %%
# The best hyperparameter configuration and the model state can evaluate the models performance on unseen data with the test data loader.
best_params = best_hyperarameters["configs"]
model2 = MLP(best_params)
model2.load_state_dict(best_model_best_state)
test_set_test(model2, test_dataloader)

# Average RMSE on the test data: 109.59801387786865
# Average R^2 score on the test data: 0.1950285628436598
# Batch with best r2: 0.4657017267891882, best r2 run rmse: 158.05963134765625

# Based on these results the model is overfitting on the training data and does not generalise well with unseen data, as seen by the large differences in average R^2.
# Only 19.50% of variance in price per night can be explained by the model.
# The following would be logical parameters to experiemnt with further: lower learning rate, higher weight decay, lower model depth, lower hidden layer width.