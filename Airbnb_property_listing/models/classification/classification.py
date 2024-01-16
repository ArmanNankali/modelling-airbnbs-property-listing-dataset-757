import os
os.chdir(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)))
import pickle
import pickle
import numpy as np
from utilities import load_dataframe, load_airbnb, save_model, GridCV_tune_classification_model_hyperparameters, evaluate_all_classifier_models, find_best_classifier_model
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Load in the cleaned dataframe with all columns remaining.
listings = load_dataframe("../airbnb_listings_data.pkl")

# Here this function will take the original cleaned dataframe and remove all non-numerical columns except for "Category".
def drop_column_list(df, col_list):
    for col in col_list:
        df.drop(col, axis=1, inplace=True)
    return df

col_list = ["ID", "Title", "Description", "Amenities", "Location", "url"]
numerical_data = drop_column_list(listings, col_list)

# Here the dataframe is checked to see it is structured as intended
numerical_data.info()

# Random seed set for reproducibility and comparison.
np.random.seed(99)

# Features and labels are seperated.
features, category = load_airbnb(numerical_data, "Category")

# Features and labels are first split into training and test batches(70% and 30% respectively) 
X_train, X_test, y_train, y_test = train_test_split(features, category, test_size=0.3)
# Training features and labels are then further split into training and validation (50%/50%).
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train)

# We instantiate the RandomForestClassifier class object.
clf = RandomForestClassifier(max_depth=2, random_state=0)
# The model is fit on x and y training data
clf.fit(X_train, y_train)

# The fitted model is used to make predictions on the validation features.
predictions = clf.predict(X_validation)

# F1 score, the precision, the recall, and the accuracy for both the training and test sets
def clf_metrics(x, y):
    predictions = clf.predict(x)
    # We obtain an accuracy score for the training data: total number of correct predictions/ total number of predictions.
    accuracy_train = accuracy_score(y, predictions)
    # Pecision score is calculated: number of true positive predictions/ (true positives + flase positives)
    precision_train = precision_score(y, predictions, average='macro')
    # Recall/ sensitivty is calculated: number of true positive predictions/ (number of true positives + false negatives)
    recall_train = recall_score(y, predictions, average='macro')
    # F1 is calculated: (precision * recall)/(precision + recall)
    f1_train = f1_score(y, predictions, average='macro')
    print(f'Accuracy={accuracy_train}, Precision={precision_train}, Recall={recall_train}, F1={f1_train}')
    
clf_metrics(X_validation, y_validation)
# Training set: Accuracy=0.3493150684931507, Precision=0.21288888888888885, Recall=0.32151802656546485, F1=0.2554633536490077
clf_metrics(X_test, y_test)
# Test set: Accuracy=0.3534136546184739, Precision=0.42196254416082, Recall=0.3468689953212615, F1=0.31152614596376094

# All model classes and their possible hyperparameter combinations are represented here.
all_models_dict = {
    "RandomForest": {"model_class" : RandomForestClassifier,
                    "param_grid": {
                        "n_estimators": [10, 50, 100],
                        "max_depth": [None, 10, 30], 
                        "min_samples_split": [2, 5],  
                        "max_features": ['sqrt', 'log2', None]},
                    "model_name": "ABB_RF_classifier_model"},
    
    "DecisionTree": {"model_class" : DecisionTreeClassifier,
                    "param_grid": {
                    "criterion": ["gini", "entropy"],
                    "max_depth": [None, 10, 20, 30], 
                    "min_samples_split": [2, 5, 10],  
                    "max_features": ['sqrt', 'log2', None],
                    "min_samples_leaf": [1, 5, 10]},
                    "model_name": "ABB_DT_classifier_model"},
    
    "GradientBoosting": {"model_class" : GradientBoostingClassifier,
                        "param_grid": {
                        "n_estimators": [100, 200],
                        "max_depth": [2, 10], 
                        "min_samples_split": [2, 5],  
                        "max_features": ['sqrt', None],
                        "learning_rate": [1, 5]},
                    "model_name": "ABB_GB_classifier_model"}
}

# All model classes and their possible hyperparameter combinations are tested here and the best for each class is saved.
evaluate_all_classifier_models(all_models_dict, X_train, y_train, X_test, y_test, "classification/logistic_regression")

# Here we compare the metrics files and find the best performing model class and return: the model, hyperparameters, metrics and file prefix.
best_model, best_model_hperparameters, best_metrics, best_model_prefix = find_best_classifier_model("classification\logistic_regression")
print(f"Best model: {best_model}, best hyperparameters: {best_model_hperparameters}, best metrics: {best_metrics}, best model file prefix: {best_model_prefix}")