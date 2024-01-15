import pickle
import json
from joblib import dump
import joblib
import os
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from itertools import product

def load_dataframe(file_path):
    """
    Load a Pandas DataFrame from a pickle file.

    Parameters:
    - file_path (str): The path to the pickle file.

    Returns:
    - pandas.DataFrame: The loaded DataFrame.
    """
    with open(file_path, "rb") as file:
        dataframe = pickle.load(file)
    return dataframe

def load_airbnb(df, label):
    """
    Separate features and labels from an Airbnb DataFrame.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame containing features and labels.
    - label (str): The column name representing the label.

    Returns:
    - tuple: A tuple containing features (DataFrame) and labels (Series).
    """
    features = df.drop(columns=[label])
    labels = df[label]
    return (features, labels)

def save_model(model, hyperparameters, metrics, file_path, model_name):
    """
    Save a regression model, hyperparameters, and metrics to files.

    Parameters:
    - model: The regression model to be saved.
    - hyperparameters: Dictionary containing hyperparameters.
    - metrics: Dictionary containing metrics.
    - file_path: The directory path to save the model files.
    - model_name: The name to use for the model files.
    """
    dump(model, os.path.join(file_path, f"{model_name}.joblib"))
    with open(os.path.join(file_path, f"{model_name}_hyperparameters.json"), "w") as file:
        json.dump(hyperparameters, file)
    with open(os.path.join(file_path, f"{model_name}_metrics.json"), "w") as file:
        json.dump(metrics, file)

def custom_tune_regression_model_hyperparameters(model_class, X_train, y_train, X_val, y_val, X_test, y_test, hyperparameters):
    """
    Perform hyperparameter tuning for a regression model.

    Parameters:
    - model_class (class): The regression model class to be used for hyperparameter tuning.
    - X_train: The features of the training set.
    - y_train: The target values of the training set.
    - X_val: The features of the validation set.
    - y_val: The target values of the validation set.
    - X_test: The features of the test set.
    - y_test: The target values of the test set.
    - hyperparameters (dict): A dictionary containing hyperparameter names as keys and lists of possible values as values.

    Returns:
    - tuple: A tuple containing the best-trained model, the best hyperparameters, and a dictionary of evaluation metrics.
    """
    best_model = None
    best_hyperparameters = None
    best_RMSE = float("inf")

    # Generate all combinations of hyperparameter values
    hyperparameter_combinations = list(product(*hyperparameters.values()))

    for combination in hyperparameter_combinations:
        # A dictionary of a combination of the hyperparameters
        hyperparam_dict = dict(zip(hyperparameters.keys(), combination))
        # Instantiate regression model with hyperparameters
        model = model_class(**hyperparam_dict)
        # Fit the model on the training data
        model.fit(X_train, y_train)
        # Predict the labels for the validation set using validation features
        y_val_pred = model.predict(X_val)
        # Calculate the validation set RMSE
        score = mean_squared_error(y_val, y_val_pred, squared=False)
        # Calculate R^2 for validation set
        r2 = model.score(X_val, y_val)

        # If current model's RMSE is better than best_RMSE then it is stored as best model    
        if score < best_RMSE:
            best_model = model
            best_hyperparameters = hyperparam_dict
            best_RMSE = score
            best_r2 = r2

    # Best model predicts the labels of the test set
    y_test_pred = best_model.predict(X_test)
    # RMSE of test set
    test_score = mean_squared_error(y_test, y_test_pred)
    # R^2 of test set
    test_r2 = model.score(X_test, y_test)

    best_metrics_val = {
        "Best validation RMSE score": best_RMSE,
        "Best validation R^2": best_r2,
        "Test RMSE score": test_score,
        "Test R^2 score": test_r2
    }
    
    return best_model, best_hyperparameters, best_metrics_val

def GridCV_tune_regression_model_hyperparameters(model_class, X_train, y_train, X_test, y_test, hyperparameters):
    """
    Perform hyperparameter tuning using GridSearchCV for a regression model.
    
    Parameters:
    - model_class: The class of the regression model to be tuned.
    - X_train: The training features.
    - y_train: The training labels.
    - X_test: The test features.
    - y_test: The test labels.
    - hyperparameters: The grid of hyperparameters to search over.

    Returns:
    - dictionary: Best hyperparameters and best estimator.
    - dictionary: Best RMSE score on the validation set.
    - dictionary: RMSE score and R^2 score on the test set.
    """
    # Create GridSearchCV object
    grid_search = GridSearchCV(model_class(), hyperparameters, cv=5, scoring='neg_root_mean_squared_error')
    # Fit the model on the training set
    grid_search.fit(X_train, y_train)
    
    # Retrieve results for validation set
    best_parameters = grid_search.best_params_
    best_estimator = grid_search.best_estimator_
    best_rmse_val = (grid_search.best_score_)*-1

    # Best model predicts the labels of the test set
    y_test_pred = grid_search.best_estimator_.predict(X_test)
    # RMSE of test set
    test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
    # R^2 of test set
    test_r2 = r2_score(y_test, y_test_pred)

    best_metrics_val = {
        "Best validation RMSE score": best_rmse_val,
        "Test RMSE score": test_rmse,
        "Test R^2 score": test_r2
    }

    return best_estimator, best_parameters, best_metrics_val

def evaluate_all_models(all_models_dict, X_train, y_train, X_test, y_test, model_save_file_path):
    """
    Evaluate and tune regression models based on a predefined dictionary of models.

    Parameters:
    - all_models_dict (dict): A dictionary where keys are model names and values are dictionaries containing:
    - X_train: The training features.
    - y_train: The training labels.
    - X_test: The test features.
    - y_test: The test labels.
    - model_save_file_path: The file path to the folder where the best models are to be saved
    Returns:
    None
    """
    # Iterates through each model_name and model_dict combination in all_models_dict to retrieve model class, hyperparameters and model name
    for model_name, model_dict in all_models_dict.items():
        model_class = model_dict["model_class"]
        param_grid = model_dict["param_grid"]
        model_name = model_dict["model_name"]
        best_model, best_hyperparams, best_metrics = GridCV_tune_regression_model_hyperparameters(model_class, X_train, y_train, X_test, y_test, param_grid)
        # Best model and hyperparameters saved for each model class as well as metrics
        save_model(best_model, best_hyperparams, best_metrics, model_save_file_path, model_name)

def find_best_model(directory):
    """
    Find the best model based on validation RMSE from a directory containing metrics files.

    Parameters:
    - directory (str): The directory path where metrics files are stored. Default is "regression\linear_regression".

    Returns:
    - tuple: The best model, its hyperparameters, metrics, and the model prefix.
    """
    
    # Best RMSE and metric file are set to baseline to be beaten by models
    best_RMSE = float("inf")
    best_metric_file = None

    # All file names are iterated through in the directory and only those ending with _metrics.json are opened
    for filename in os.listdir(directory):
        if filename.endswith("_metrics.json"):
            with open(os.path.join(directory, filename), "r") as file:
                metrics_data = json.load(file)
            # The RMSE metric is retrieved and compared against the current best
            rmse = metrics_data["Best validation RMSE score"]
            if rmse < best_RMSE:
                # If the model RMSE is better than current, that value replaces the current best RMSE and takes the place of best metrics data as well as file name
                best_metric_file = filename
                best_RMSE = rmse
                best_metrics = metrics_data
    
    # The best performing model's metrics file is split to obtain the prefix for that model class
    best_model_prefix = best_metric_file.replace("_metrics.json", "")

    # This prefix is used to obtain the relevant model's .joblib file and its hyperparameter dictionary
    best_model = joblib.load(os.path.join(directory, f"{best_model_prefix}.joblib"))
    with open(os.path.join(directory, f"{best_model_prefix}_hyperparameters.json"), 'r') as f:
        best_model_hperparameters = json.load(f)
    
    
    print(f"Best model: {best_model}, best hyperparameters: {best_model_hperparameters}, best metrics: {best_metrics}")
    # The best model, hyperparameteres and metrics are returned
    return best_model, best_model_hperparameters, best_metrics, best_model_prefix


    