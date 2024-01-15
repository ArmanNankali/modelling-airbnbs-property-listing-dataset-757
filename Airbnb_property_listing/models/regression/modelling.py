import os
os.chdir(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)))
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split, ShuffleSplit, validation_curve, GridSearchCV
from sklearn.pipeline import Pipeline
from utilities import load_dataframe, load_airbnb, save_model, custom_tune_regression_model_hyperparameters, GridCV_tune_regression_model_hyperparameters, evaluate_all_models, find_best_model
import json
import joblib
from joblib import dump, load
import os
from itertools import product
from math import sqrt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import datasets, ensemble

# Load in the Yeo-Johnson transformed numerical data (excluding price night transformation)
numerical_data = load_dataframe(r"..\tabular_data\YJH_transformed_no_price_night_numerical_data.pkl")

#YJH_transformed_no_price_night_numerical_data.pkl
# Features and labels are seperated.
features, price_night = load_airbnb(numerical_data, "Price_Night")

print(type((features, price_night)))

# Random seed set for reproducibility and comparison.
np.random.seed(99)

# Features and labels are first split into training and test batches(70% and 30% respectively) 
X_train, X_test, y_train, y_test = train_test_split(features, price_night, test_size=0.3)

# Training features and labels are then further split into training and validation (50%/50%).
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train)

# Stochastic Gradient Descent regressor class is instantiated and fit on the training features and labels.
sgd = SGDRegressor(random_state=42)
sgd.fit(X_train, y_train)

# Mean absolute error (MAE): measures the average absolute difference between the predicted values and the actual values.
train_mae = mean_absolute_error(y_train, sgd.predict(X_train))
val_mae = mean_absolute_error(y_validation, sgd.predict(X_validation))
print(f"train_mae: {train_mae}")
print(f"val_mae: {val_mae}")
# train_mae: 63.455053855215695
# val_mae: 65.33802364275718

# Mean squared error: measures the average squared difference between predicted and actual values.
# Roose mean squared error (RMSE): measures the root of the average squared difference between predicted and actual values.
# Here the function mean_squared_error is used, but since "squared=False" the result is the square root of the MSE.
train_rmse = mean_squared_error(y_train, sgd.predict(X_train), squared=False)
val_rmse = mean_squared_error(y_validation, sgd.predict(X_validation), squared=False)
print(f"train_rmse: {train_rmse}")
print(f"val_rmse: {val_rmse}")
# train_rmse: 99.0796794009564
# val_rmse: 91.79705274394469

# R^2: measures the proportion of the variance in the dependent variable (target) that is explained by the independent variables (features) between 0 and 1 (representing 0% and 100% respectively).
train_r2_score = sgd.score(X_train, y_train)
print(f"The train R^2 score is {train_r2_score}")
# The train R^2 score is 0.3906685247280842

validation_r2_score = sgd.score(X_validation, y_validation)
print(f"The validation R^2 score is {validation_r2_score}")
# The validation R^2 score is 0.2433980129507164
# The R^2 of the validation set is lower than the training R^2, indicating overfitting of the model on training data.
print(sgd.coef_)

train_cv_scores = cross_val_score(sgd, X_train, y_train, cv=5)
print(f"train cross_val_scores: {train_cv_scores}")

val_cv_scores = cross_val_score(sgd, X_validation, y_validation, cv=5)
print(f"train cross_val_scores: {val_cv_scores}")

# hyperparameters: tol, warm_start, max_iter, alpha
hyperparameters = {
    "tol" : [1e-2, 1e-3, 1e-4],
    "warm_start" : [True, False],
    "max_iter" : [1000, 1500, 3000],
    "alpha" : [0.1, 0.2, 0.3, 0.4, 0.5]
}

# Now we implement the custome hyperparameter grid search and retrieve the model with the best RMSE
best_model, best_hyperparameters, best_metrics_val = custom_tune_regression_model_hyperparameters(SGDRegressor, X_train, y_train, X_validation, y_validation, X_test, y_test, hyperparameters)

print(f"Custom tune results: {best_model} {best_hyperparameters} {best_metrics_val}")

model = SGDRegressor
SGD_param_grid = {
    
    "tol" : [1e-2, 1e-3, 1e-4],
    "warm_start" : [True, False],
    "max_iter" : [1000, 1500, 3000],
    "alpha" : [0.1, 0.2, 0.3, 0.4, 0.5]
}

SGD_best_model, SGD_best_hyperparams, SGD_best_metrics = GridCV_tune_regression_model_hyperparameters(model, X_train, y_train, X_test, y_test, SGD_param_grid)

print(f"GridSearchCV results: {SGD_best_hyperparams} {SGD_best_metrics}")

# The model, best hyperarameters and metrics are saved.
save_model(SGD_best_model, SGD_best_hyperparams, SGD_best_metrics, r"regression\linear_regression", "ABB_SGD_regression_model")

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import datasets, ensemble
Decision_tree_param_grid = {
    'max_depth': [1, 5, 10, None],
    "splitter" : ["best", "random"],
    'min_samples_split': [2, 5, 10]
}

# Hyperparamater grid search is performed for Decision tree regressor
GridCV_tune_regression_model_hyperparameters(DecisionTreeRegressor, X_train, y_train, X_test, y_test, Decision_tree_param_grid)

RandomForest_param_grid = {
    'n_estimators': [10, 50, 100],
    "max_depth" : [1,2, 5, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Hyperparamater grid search is performed for Random forest regressor
GridCV_tune_regression_model_hyperparameters(RandomForestRegressor, X_train, y_train, X_test, y_test, RandomForest_param_grid)

GradientBoosting_param_grid = {
    "n_estimators": [10, 50, 100, 500],
    "max_depth": [1, 2, 4],
    "min_samples_split": [2, 5, 10],
    "learning_rate": [0.001, 0.01, 0.1]
    }

# Hyperparamater grid search is performed for Gradient boosting regressor
GridCV_tune_regression_model_hyperparameters(ensemble.GradientBoostingRegressor,  X_train, y_train, X_test, y_test, GradientBoosting_param_grid)

# This dictionary contains three different model classes and a grid of their possible hyperparameter combinations
all_models_dict = {
    "Decisiontree": {"model_class" : DecisionTreeRegressor,
                     "param_grid": {
                        'max_depth': [1, 5, 10, None],
                        "splitter" : ["best", "random"],
                        'min_samples_split': [2, 5, 10]
                        },
                     "model_name": "ABB_DT_regression_model"
                    },
    "RandomForest": {"model_class" : RandomForestRegressor,
                     "param_grid": {
                        'n_estimators': [10, 50, 100],
                        "max_depth" : [1,2, 5, 10, 20],
                        'min_samples_split': [2, 5, 10]
                        },
                     "model_name": "ABB_RF_regression_model"
                    },
    "GradientBoosting": {"model_class" : ensemble.GradientBoostingRegressor,
                     "param_grid": {
                        "n_estimators": [10, 50, 100, 500],
                        "max_depth": [1, 2, 4],
                        "min_samples_split": [2, 5, 10],
                        "learning_rate": [0.001, 0.01, 0.1]
                        },
                     "model_name": "ABB_GB_regression_model"
                    }
}

# This function tests each model with their possible hyperarameter combinations and saves the best combination, the model and the metrics
evaluate_all_models(all_models_dict, X_train, y_train, X_test, y_test, "regression\linear_regression")

# Here the best of the three models is returned with its hyperparameters, metrics and the filename prefix for identification.
best_model, best_model_hperparameters, best_metrics, best_model_prefix =find_best_model("regression\linear_regression")
