import pickle
import json
from joblib import dump
import joblib
import os
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from itertools import product
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import torch.optim
from datetime import datetime

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
    Save a machine learning model along with hyperparameters and metrics to a specified file path.

    Parameters:
    - model: The machine learning model to be saved.
    - hyperparameters: Dictionary containing hyperparameters used for training the model.
    - metrics: Dictionary containing evaluation metrics for the model.
    - file_path: The directory where the model and associated information will be saved.
    """
    if model_name is None:
            # Generate a timestamped model name if not provided
            model_name = datetime.now().isoformat().replace(":", "_")
        # Create a folder for the model
    model_folder = os.path.join(file_path, model_name)
    os.makedirs(model_folder, exist_ok=True)
    # Save the model's state dictionary or joblib file depending on the type of model
    if isinstance(model, torch.nn.Module):
        if model_name is None:
            # Generate a timestamped model name if not provided
            model_name = datetime.now().isoformat().replace(":", "_")
        # Create a folder for the model
        model_folder = os.path.join(file_path, model_name)
        os.makedirs(model_folder, exist_ok=True)
        # Save the model's state dictionary
        torch.save(model.state_dict(), os.path.join(model_folder, f"{model_name}.pt"))
    else:
        dump(model, os.path.join(model_folder, f"{model_name}.joblib"))

    # Save hyperparameters as JSON file
    with open(os.path.join(model_folder, f"{model_name}_hyperparameters.json"), "w") as file:
        json.dump(hyperparameters, file)

    # Save metrics as JSON file
    with open(os.path.join(model_folder, f"{model_name}_metrics.json"), "w") as file:
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

def GridCV_tune_classification_model_hyperparameters(model_class, X_train, y_train, X_test, y_test, hyperparameters):
    """
    Perform hyperparameter tuning using GridSearchCV for a classification model.
    
    Parameters:
    - model_class: The class of the classification model to be tuned.
    - X_train: The training features.
    - y_train: The training labels.
    - X_test: The test features.
    - y_test: The test labels.
    - hyperparameters: The grid of hyperparameters to search over.

    Returns:
    - dictionary: Best hyperparameters and best estimator.
    - dictionary: Best F1 score on the validation set.
    - dictionary: F1 score, accuracy, precision, and recall on the test set.
    """
    # Create GridSearchCV object
    grid_search = GridSearchCV(model_class(), hyperparameters, cv=5, scoring='f1_macro')
    # Fit the model on the training set
    grid_search.fit(X_train, y_train)
    
    # Retrieve results for validation set
    best_parameters = grid_search.best_params_
    best_estimator = grid_search.best_estimator_
    best_f1_val = grid_search.best_score_

    # Best model predicts the labels of the test set
    y_test_pred = grid_search.best_estimator_.predict(X_test)
    # F1 score of test set
    test_f1 = f1_score(y_test, y_test_pred, average='macro')
    # Accuracy of test set
    test_accuracy = accuracy_score(y_test, y_test_pred)
    # Precision of test set
    test_precision = precision_score(y_test, y_test_pred, average='macro')
    # Recall of the test set
    test_recall = recall_score(y_test, y_test_pred, average='macro')

    best_metrics_val = {
        "Best validation F1 score": best_f1_val,
        "Test F1 score": test_f1,
        "Test accuracy": test_accuracy,
        "Test precision": test_precision,
        "Test recall": test_recall
    }

    return best_estimator, best_parameters, best_metrics_val

def evaluate_all_classifier_models(all_models_dict, X_train, y_train, X_test, y_test, model_save_file_path):
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
        best_model, best_hyperparams, best_metrics = GridCV_tune_classification_model_hyperparameters(model_class, X_train, y_train, X_test, y_test, param_grid)
        # Best model and hyperparameters saved for each model class as well as metrics
        save_model(best_model, best_hyperparams, best_metrics, model_save_file_path, model_name)

def find_best_classifier_model(directory):
    """
    Find the best model based on validation RMSE from a directory containing metrics files.

    Parameters:
    - directory (str): The directory path where metrics files are stored. Default is "regression\linear_regression".

    Returns:
    - tuple: The best model, its hyperparameters, metrics, and the model prefix.
    """
    
    # Best RMSE and metric file are set to baseline to be beaten by models
    best_f1 = 0.0
    best_metric_file = None

    # All file names are iterated through in the directory and only those ending with _metrics.json are opened
    for filename in os.listdir(directory):
        if filename.endswith("_metrics.json"):
            with open(os.path.join(directory, filename), "r") as file:
                metrics_data = json.load(file)
            # The RMSE metric is retrieved and compared against the current best
            f1 = metrics_data["Best validation F1 score"]
            if f1 > best_f1:
                # If the model RMSE is better than current, that value replaces the current best RMSE and takes the place of best metrics data as well as file name
                best_metric_file = filename
                best_f1 = f1
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

class AirbnbNightlyPriceRegressionDataset(Dataset):
    """
    PyTorch Dataset for Airbnb price per night regression.

    Parameters:
    - features: pandas DataFrame or numpy array, input features.
    - labels: pandas Series or numpy array, labels.

    Attributes:
    - features: torch.Tensor, input features as float.
    - labels: torch.Tensor, corresponding labels as float.
    """
    def __init__(self, features, labels):
        """
        Initializes the dataset with features and labels.

        Converts features and labels to torch.Tensor.

        Parameters:
        - features: pandas DataFrame or numpy array, features.
        - labels: pandas Series or numpy array, corresponding labels.
        """
        if isinstance(features, pd.DataFrame):
            features = features.values
        if isinstance(labels, pd.Series):
            labels = labels.values
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).float()
        
    def __getitem__(self, idx):
        """
        Retrieves a single pair (features, labels) from the dataset.

        Parameters:
        - idx: int, index of the item to retrieve.

        Returns:
        - tuple: (torch.Tensor, torch.Tensor), features and corresponding labels.
        """
        return self.features[idx], self.labels[idx]
    
    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
        - int: Length of the dataset.
        """
        return len(self.features)
    
class MLP(torch.nn.Module):
    def __init__(self, config):
        """
        Multilayer Perceptron (MLP) model.

        Parameters:
        - config (dict): Configuration dictionary containing model hyperparameters.
        """
        # Calls the __init__ method of the superclass (torch.nn.Module).
        super(MLP, self).__init__()
        # An initially empty list of layers.
        layers = []
        # Add first hidden layer.
        layers.append(torch.nn.Linear(config["input_size"], config["hidden_layer_width"]))
        # Add batch normalization to first hidden layer.
        layers.append(torch.nn.BatchNorm1d(config["hidden_layer_width"]))
        # Add activation function to first hidden layer.
        layers.append((getattr(torch.nn, config["activation_function"]))())
        # Add dropout to first hidden layer.
        layers.append(torch.nn.Dropout(config["drop-out"]))
        # Add additional hidden layers based on model depth.
        for _ in range(config["model_depth"] -1):
            layers.append(torch.nn.Linear(config["hidden_layer_width"],config["hidden_layer_width"]))
            layers.append(torch.nn.BatchNorm1d(config["hidden_layer_width"]))
            layers.append((getattr(torch.nn, config["activation_function"]))())
            layers.append(torch.nn.Dropout(config["drop-out"]))
        layers.append(torch.nn.Linear(config["hidden_layer_width"], config["output_size"]))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor.
        """
        return self.layers(x)

class RMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self,x,y):
        """
        Forward pass of the RMSE loss.

        Parameters:
        - x (torch.Tensor): Predicted values.
        - y (torch.Tensor): Ground truth values.

        Returns:
        - torch.Tensor: RMSE loss.
        """
        return torch.sqrt(self.mse(x,y))

def train(model, data_loader, val_dataloader, config, epochs=700):
    """
    Train the given model using the specified configuration.

    Parameters:
    - model (torch.nn.Module): The neural network model.
    - data_loader (torch.utils.data.DataLoader): Training data loader.
    - val_dataloader (torch.utils.data.DataLoader): Validation data loader.
    - config (dict): Configuration dictionary containing model hyperparameters.
    - epochs (int): Number of training epochs.

    Returns:
    - tuple: Metrics, hyperparameters, best RMSE model state, and best R^2 model state.
    """
    # Optimiser is constructed by getting the activation function attribute from the torch.nn module. 
    optimiser_class = getattr(torch.optim, config["optimiser"])
    optimiser = optimiser_class(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    
    # The RMSEloss class is instantiated.
    rmse_loss = RMSELoss()
    # The SummaryWriter class is instantiated.
    writer = SummaryWriter()
    batch_idx = 0
    # Timestamp is obtained at the start of model training.
    start_time = time.time()
    best_rmse_model_state = None 
    bestr2__model_state = None
    best_val_r2 = 0.0
    best_val_r2_loss = float("inf")
    
    # Patience =  number of epochs the model can be trained with no improvement before training is terminated.
    patience = 100
    epochs_without_improvement = 0

    for epoch in range(epochs):
        total_inference_latency = 0
        num_batches = 0
        total_train_loss = 0.0
        total_train_r2 = 0.0
        for batch in data_loader:
            features, labels = batch
            # Timestamp just before prediction is made.
            pred_start_time = time.time()
            prediction = model(features)
            # Timestamp just after prediction is made.
            pred_end_time = time.time()
            total_inference_latency += pred_end_time - pred_start_time
            num_batches += 1
            # Label reshape for correct broadcasting
            labels = labels.view(-1, 1)
            # Training rmse loss is calculated for this batch and added to the total.
            train_loss = rmse_loss(prediction, labels)
            total_train_loss += train_loss.item()
            # Training R^2 is calculted for this batch and added to the total.
            trainr2 = r2_score(labels.detach().numpy(), prediction.detach().numpy())
            total_train_r2 += trainr2
            # Backpropagation: compute gradients of the loss with respect to model parameters.
            train_loss.backward()
            print(train_loss.item())
            # Perform a single optimization step (parameter update)
            optimiser.step()
            # Clear the gradients for the next iteration
            optimiser.zero_grad()
            batch_idx += 1
        
        # Average training RMSE loss and R^2 for the current epoch is calculated
        average_train_loss = total_train_loss / num_batches
        average_train_r2 = total_train_r2 / num_batches
        average_inference_latency = total_inference_latency / num_batches
        # Average training RMSE loss and R^2 for the current epoch is plotted on the Tensorboard graph
        writer.add_scalar("average Train RMSE loss", average_train_loss, epoch)
        writer.add_scalar("average Train r2", average_train_r2, epoch)

        # The model is in evaluation mode so validation predictions won't influence it's learning (data leakage) and for deactivating dropout and batch normalization layers.
        model.eval()
        # Gradients are not tracked and intermediate values are not stored for gradient computation.
        with torch.no_grad():
            val_losses = []
            val_r2_scores = []
            # Validation data is loaded
            for features, labels in val_dataloader:
                prediction = model(features)
                #label reshape for broadcasting
                labels = labels.view(-1, 1)
                val_losses.append(rmse_loss(prediction, labels).item())
                valr2 = r2_score(labels.detach().numpy(), prediction.detach().numpy())
                val_r2_scores.append(valr2)
            # Average validation RMSE loss and R^2 are calculated for this epoch
            val_loss = sum(val_losses) / len(val_losses)
            val_r2 = sum(val_r2_scores) / len(val_r2_scores)
        
        # Tensorboard will also graph the validation RMSE and R^2 and training RMSE for each epoch
        writer.add_scalar("Validation RMSE loss", val_loss, epoch)
        writer.add_scalar("Validation R^2 score", val_r2, epoch)
        writer.add_scalar("Train RMSE loss", train_loss.item(), epoch)

        # Calculate the average validation RMSE loss for the current epoch.
        average_val_loss = sum(val_losses) / len(val_losses)
        # Calculate the average validation R^2 for the current epoch.
        average_val_r2 = sum(val_r2_scores) / len(val_r2_scores)

        # Current average R^2 is compared to best and determines when to reset epochs without improvement.
        if average_val_r2 > best_val_r2:
            # If current average R^2 for validation is best, "best_val_r2" and "best_val_r2_loss" is replaced.
            epochs_without_improvement = 0
            best_val_r2 = average_val_r2
            # best_val_r2_loss represents the RMSE loss at the time of the best R^2.
            best_val_r2_loss = average_val_loss
            # Store the current state of the model as it has produced the best validation R^2 so far.
            bestr2__model_state = model.state_dict()

        else:
            # If the current best R^2 is not beaten the epoch counter without improvement increases by 1.
            epochs_without_improvement += 1
            if epochs_without_improvement == patience:
                # epochs without improvement reaching patience limit will break the training loop.
                print(f"Early stopping due to no improvement after {patience} epochs at epoch:{epoch}.")
                break
   
    # Timestamp following ermination of training.
    end_time = time.time()
    training_duration = end_time - start_time
    
    metrics = {
        "Training RMSE loss" : train_loss.item(), # The current RMSE loss for the training set.
        "Training R2" : trainr2, # The current R^2 score for the training set.
        "Validation RMSE loss" : val_loss, # The current RMSE loss for the validation set.
        "Validation R2" : valr2, # The current R^2 score for the validation set.
        
        "Average Train RMSE loss": average_train_loss, # The average RMSE loss over all batches in the current training epoch.
        "Average Train R2": average_train_r2, # The average R^2 score over all batches in the current training epoch.

        "Avg_val_r2_rmse" : best_val_r2_loss, # The RMSE loss corresponding to the best validation R^2 over all epochs.
        "Avg_val_r2" : best_val_r2, # The best average R^2 score over all epochs.

        "training_duration" : training_duration, # Time taken from start to end of training.
        "inference_latency" : average_inference_latency # Time taken to make predictions.
    }

    hyperparams = {
        "epochs" : epochs,
        "patience" : patience,
        "configs" : config
    }
    
    return metrics, hyperparams, bestr2__model_state

def test_set_test(model, test_dataloader):
    """
    Evaluate the given model on the test set and print average RMSE, R^2 score, and additional information.

    Parameters:
    - model: The PyTorch model to be evaluated.
    """
    model.eval()  # Set the model to evaluation mode
    rmse_losses = []
    r2_scores = []
    best_r2 = 0.0
    best_r2_rmse = float("inf")

    with torch.no_grad():  # No need to track gradients
        for features, labels in test_dataloader:
            predictions = model(features)
            # Reshape labels for correct broadcasting
            labels = labels.view(-1, 1)
            # Compute RMSE and R^2 score for test data
            rmse_loss = torch.sqrt(torch.nn.functional.mse_loss(predictions, labels)).item()
            r2 = r2_score(labels.detach().numpy(), predictions.detach().numpy())
            rmse_losses.append(rmse_loss)
            r2_scores.append(r2)
            
            if r2 > best_r2:
                best_r2_rmse = rmse_loss
                best_r2 = r2

    # Compute average RMSE and R^2 score over all batches
    average_rmse = sum(rmse_losses) / len(rmse_losses)
    average_r2 = sum(r2_scores) / len(r2_scores)

    print(f'Average RMSE on the test data: {average_rmse}')
    print(f'Average R^2 score on the test data: {average_r2}')
    print(f"Batch with best r2: {best_r2}, best r2 run rmse: {best_r2_rmse}")

def custom_tune_NN_model_hyperparameters(model_class, data_loader, val_dataloader, config_grid, best_model_folder):
    """
    Custom hyperparameter tuning for a neural net model.

    Parameters:
    - model_class: The class of the regression model to be tuned.
    - data_loader: Training data loader.
    - val_dataloader: Validation data loader.
    - config_grid: A dictionary containing hyperparameter names as keys and lists of values to be tested as values.

    Returns:
    - best_model: The best performing regression model.
    - best_hyperparameters: The hyperparameters of the best performing model.
    - best_model_best_state: The state of the best model with the highest validation R^2.
    - best_metrics: Metrics (including R^2 and RMSE) of the best-tuned model.
    - config_grid: The original hyperparameter grid.
    - additional_info: Dictionary containing additional information, such as validation RMSE and best R^2.
    """
    best_model = None
    best_hyperparameters = None
    best_RMSE = float("inf")
    best_r2 = 0.0

    hyperparameter_combinations = list(product(*config_grid.values()))

    for combination in hyperparameter_combinations:
        hyperparam_dict = dict(zip(config_grid.keys(), combination))
        model = model_class(hyperparam_dict)
        metrics, hyperparams, best_r2_model_state = train(model, data_loader, val_dataloader, hyperparam_dict)
    
        if metrics["Avg_val_r2"] > best_r2:
            best_model = model
            best_hyperparameters = hyperparams
            best_RMSE = metrics["Avg_val_r2_rmse"]
            best_r2 = metrics["Avg_val_r2"]
            best_metrics = metrics
            best_model_best_state = best_r2_model_state
    # Best model is saved.        
    save_model(best_model, best_hyperparameters, metrics, best_model_folder, None)
    return best_model, best_hyperparameters, best_model_best_state, best_metrics, config_grid, {'validation_RMSE': best_RMSE, "best_r2": best_r2} 

