# modelling-airbnbs-property-listing-dataset-757
ArmanNankali/modelling-airbnbs-property-listing-dataset-757

## Table of Contents
1. [Project Description](#project-description)
2. [Installation Instructions](#installation-instructions)
3. [Usage Instructions](#usage-instructions)
4. [File Structure of the Project](#file-structure-of-the-project)
5. [License Information](#license-information)

## Project Description
This project aims to build a framework for systematically training, tuning, and evaluating models on tasks tackled by the Airbnb team.

The first task involves cleaning the Airbnb data in `tabular_data.py`. Default values are specified, erroneous values are removed or replaced, and remaining null rows are dropped. The Descriptions are also formatted correctly, with some repeated prefix statements removed. A cleaned dataframe containing only numerical variables is produced: `numerical_data.pkl`.

Next, the data is pre-processed for the machine learning models in `Data_cleaning_and_transformation.py`. The skew, normality, and correlation of the variables are measured. Skew can negatively impact the reliability of predictions by regression models. Some models explicitly assume a normal distribution in the data to make calculations. Multicollinearity can lead to poor generalisation and more overfitting. Yeo-Johnson transformations were applied to reduce skewness for each variable, excluding the label variable.

`modelling.py` used functions imported from `utilities.py` to create several regressor machine learning models and use both custom and Sklearn's built in hyperparameter gridsearch functions to determine the best hyperparameter combinations resulting in the lowest root mean squared error. Once the best parameter combination for each model class is obtained, the model, hyperparameter dictionary and performance metrics are saved. This collection of files is evaluated to obtain the best performing of all the four models, returning the best model and its best hyperparameters with the relevant performance metrics (RMSE and R^2).

`classification.py` imports functions from `utilities.py` to create several classification logistic regression models on a new version of the airbnb data where "Category" is the label and numerical variables are the features. Both a custom gridsearch and Sklearn's GridSearchCV are used to find the best hyperparameter combinations for the following model classes: RandomForest, DecisionTree and GradientBoosting. The best performance is measured by the highest F1 score and the best model with its hyperparameters and metrics are saved in the `classification` folder. Then the metrics files are evaluated and the best of the three optimised models is returned with its hyperparameters and metrics.

A final update to `utilities.py` enables the creation of a Multi-Layered-Perceptron (MLP) neural network to predict the price per night in `Price_night_nn.py` and the number of bedrooms in `Bedrooms_nn.py`. A new custom grid search function allows a hyperparameter grid of the following to be iterated through: optimiser, learning_rate, model_depth, hidden_layer_width, output_size, weight_decay, activation_function, drop-out. The best model is evaluated on it's average valdiation R^2 score but the accompanying RMSE is also logged. This neural network features the use of standard scaler, batch normalisation, L2 regulariser, drop-out layers, activation functions, learning rates and optimisers. Standard scaling of the features was implemented to help the model converge faster for potentially higher learning rates. A batch normaliser was used to standardise the input to each layer to improve generalisation and to imrpove the speed of training. The best models, their metrics and hyperparameters are all saved. Then the best model state is loaded and tested on the unseen test data to provide a final evaluation where average training RMSE and R^2 as measured.

## Installation instructions
To run this project locally, follow these steps:
1. Clone the repository: `git clone https://github.com/ArmanNankali/modelling-airbnbs-property-listing-dataset-757.git`
2. Install dependencies: `pip install -r requirements.txt`

## Usage instructions
1. Navigate to the project directory: `cd modelling-airbnbs-property-listing-dataset-757`
2. Run the `tabular_data.py` script for data cleaning and `Data_cleaning_and_transformation.py` for numerical data pre-processing.
3. Run the `modelling.py` script for finding the best hyperparameters for each of the following regression model classes: Stochastic Gradient Descent, Decision tree, Random forest, Gradient boosting. Then the best of these models is selected based on the lowest root mean squared error metric.
4. Run the `classification.py` script for finding the best hyperparameters for each of the following regression model classes: RandomForest, DecisionTree and GradientBoosting. Then the best of these models is selected based on the highest F1 score metric.
5. Run the `Price_night_nn.py` script to find the best hyperparameter configuration for the Multi-Layered-Perceptron (MLP) neural network to predict price per night for the Airbnb properties. The best model is chosen based on average validation R^2 score and the metrics, hyperparameters and model state are saved in the folder `Price_night`. The best model is then tested on the unseen test dataloader. This neural network features the use of standard scaler, batch normalisation, L2 regulariser, drop-out layers, activation functions, learning rates and optimisers.
6. Run the `Bedrooms_nn.py` script to find the best hyperparameter configuration for the MLP neural network to predict number of bedrooms for the Airbnb properties. The best model is chosen based on average validation R^2 score and the metrics, hyperparameters and model state are saved in the folder `Bedrooms`. The best model is then tested on the unseen test dataloader.

## File structure of the project
- `requirements.txt`: List of project dependencies.
- `tabular_data/`: Directory containing scripts for data cleaning and transformation.
    - `tabular_data.py`: Python scripts for data cleaning.
    - `listing.csv`: The original Airbnb property listing dataset in CSV format.
    - `airbnb_listings_data.pkl`: Cleaned and transformed data saved as a pickle file.
    - `numerical_data.pkl`: Numerical data for training machine learning models.
    - `Data_cleaning_and_transformation.py`: Python scripts for data transformation.
    - `YJH_transformed_no_price_night_numerical_data.pkl`: Yeo-Johnson transformed variables, excluding price_night label.
    - `YJH_transformed_no_bedrooms_numerical_data.pkl`: Yeo-Johnson transformed variables, excluding bedrooms label.
- `models/`: Directory containing scripts for tuning machine learning models.
    - `regression/`: Folder containing linear regression models for predicting price per night.
    - `utilities.py`: Python scripts for: loading pandas dataframe, splitting dataframe into features and labels, saving machine learning model with hyperparameters and performance metrics, custom hyperparameter gridsearch function, Sklearn GridSearchCV hyperparameter gridsearch function, function to evaluate all machine learning models and a function to retrieve the best model, hyperprameters and metrics.
        - `modelling.py`: Python scripts for creating Stochastic Gradient Descent, Decision tree, Random forest and Gradient boosting regression machine learning models to predict price per nigth for Airbnb properties. Custom hyperparameter grid search  as well as Sklearn's GridSearchCV is used to find the best hyperparameters. All model-hyperparameter combinations are evaluated to find the model with the lowest root mean squared error.
        - `linear_regression`: Folder containing all the model (.joblib), hyperparameter and metrics files (both .json).
            - `ABB_DT_regression_model.joblib`: Trained Decision Tree (DT) model.
            - `ABB_DT_regression_model_hyperparameters.json`: Hyperparameters used for the Decision Tree model.
            - `ABB_DT_regression_model_metrics.json`: Metrics for the Decision Tree model.
            - `ABB_GB_regression_model.joblib`: Trained Gradient Boosting (GB) model.
            - `ABB_GB_regression_model_hyperparameters.json`: Hyperparameters used for the Gradient Boosting model.
            - `ABB_GB_regression_model_metrics.json`: Metrics for the Gradient Boosting model.
            - `ABB_RF_regression_model.joblib`: Trained Random Forest (RF) model.
            - `ABB_RF_regression_model_hyperparameters.json`: Hyperparameters used for the Random Forest model.
            - `ABB_RF_regression_model_metrics.json`: Metrics for the Random Forest model.
            - `ABB_SGD_regression_model.joblib`: Trained Stochastic Gradient Descent (SGD) model.
            - `ABB_SGD_regression_model_hyperparameters.json`: Hyperparameters used for the Stochastic Gradient Descent model.
            - `ABB_SGD_regression_model_metrics.json`: Metrics for the Stochastic Gradient Descent model.
    - `classification/` Folder containing logistic regression classification models for predicting category.
    - `classification.py`: Python scripts for creating RandomForest, DecisionTree and GradientBoosting logistic regression classification machine learning models to category for Airbnb properties. Custom hyperparameter grid search  as well as Sklearn's GridSearchCV is used to find the best hyperparameters. All model-hyperparameter combinations are evaluated to find the model with the lowest root mean squared error.
    - `logistic_regression/`: Folder containing all the model (.joblib), hyperparameter and metrics files (both .json).
        - `ABB_DT_classifier_model.joblib`: Trained Decision Tree classifier (DT) model.
        - `ABB_DT_classifier_model_hyperparameters.json`: Hyperparameters used for the Decision Tree model.
        - `ABB_DT_classifier_model_metrics.json`: Metrics for the Decision Tree model.
        - `ABB_GB_classifier_model.joblib`: Trained Gradient Boosting classifier (GB) model.
        - `ABB_GB_classifier_model_hyperparameters.json`: Hyperparameters used for the Gradient Boosting model.
        - `ABB_GB_classifier_model_metrics.json`: Metrics for the Gradient Boosting model.
        - `ABB_RF_classifier_model.joblib`: Trained Random Forest classifier (RF) model.
        - `ABB_RF_classifier_model_hyperparameters.json`: Hyperparameters used for the Random Forest model.
        - `ABB_RF_classifier_model_metrics.json`: Metrics for the Random Forest model.
    - `neural_networks/`: Folder containing python scripts for tuning neural networks.
        - `regression/`: Subfolder containing python scripts for tuning neural nets for regression and the models produced with their performance metrics, model state and hyperparameters.
            - `Price_night_nn.py`: Python scripts for creating and tuning a neural network for predicting price per night for Airbnb properties.
            - `Price_night/`: Folder containing metrics and hyperparameter .json files for the best performing model.
                - `2024-01-17T16_48_54.156604/`: Folder containing relevant files for model trained on 2024-01-17.
                    - `2024-01-17T16_48_54.156604_hyperparameters.json`: Hyperparameter json.
                    - `2024-01-17T16_48_54.156604_metrics.json`: Metrics json.
                    - `2024-01-17T16_48_54.156604.pt`: Model state PyTorch.
            - `Bedrooms.py`: Python scripts for creating and tuning a neural network for predicting number of bedrooms for Airbnb properties.
            - `Bedrooms/`: Folder containing metrics and hyperparameter .json files for the best performing model.
                - `2024-01-17T21_01_23.257566/`: Folder containing relevant files for model trained on 2024-01-17.
                    - `2024-01-17T21_01_23.257566_hyperparameters.json`: Hyperparameter json.
                    - `2024-01-17T21_01_23.257566_metrics.json`: Metrics json.
                    - `2024-01-17T21_01_23.257566.pt`: Model state PyTorch.

## License information
This project is licensed under the [MIT License](LICENSE).