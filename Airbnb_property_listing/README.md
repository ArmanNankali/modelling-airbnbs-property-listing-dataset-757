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

Next, the data is pre-processed for the machine learning models in `Data_cleaning_and_transformation.py`. The skew, normality, and correlation of the variables are measured. Skew can negatively impact the reliability of predictions by regression models. Some models explicitly assume a normal distribution in the data to make calculations. Multicollinearity can lead to poor generalisation and more overfitting. Yeo-Johnson transformations reduced the skew for every variable, leaving out the label variable.

`modelling.py` used code imported from `utilities.py` to create several regressor machine learning models and use both custom and Sklearn's built in hyperparameter gridsearch functions to determine the best hyperparameter combinations resulting in the lowest root mean squared error. Once the best parameter combination for each model class is obtained, the model, hyperparameter dictionary and performance metrics are saved. This collection of files is evaluated to obtain the best performing of all the four models, returning the best model and its best hyperparameters with the relevant performance metrics (RMSE and R^2).


## Installation instructions
To run this project locally, follow these steps:
1. Clone the repository: `git clone https://github.com/ArmanNankali/modelling-airbnbs-property-listing-dataset-757.git`
2. Install dependencies: `pip install -r requirements.txt`

## Usage instructions
1. Navigate to the project directory: `cd modelling-airbnbs-property-listing-dataset-757`
2. Run the `tabular_data.py` script for data cleaning and `Data_cleaning_and_transformation.py` for numerical data pre-processing.
3. Run the `modelling.py` script for finding the best hyperparameters for each of the following regression model classes: Stochastic Gradient Descent, Decision tree, Random forest, Gradient boosting. Then the best of these models is selected based on the lowest root mean squared error metric.

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
- `models/`: Directory containing scripts for tuning machine learning models predicting price per night.
    - `modelling.py`: Python scripts for creating Stochastic Gradient Descent, Decision tree, Random forest and Gradient boosting regression machine learning models to repdict price per nigth for Airbnb properties. Custom hyperparameter grid search  as well as Sklearn's GridSearchCV is used to find the best hyperparameters. All model-hyperparameter combinations are evaluated to find the model with the lowest root mean squared error.
    - `utilities.py`: Python scripts for: loading pandas dataframe, splitting dataframe into features and labels, saving machine learning model with hyperparameters and performance metrics, custom hyperparameter gridsearch function, Sklearn GridSearchCV hyperparameter gridsearch function, function to evaluate all machine learning models and a function to retrieve the best model, hyperprameters and metrics.
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

## License information
This project is licensed under the [MIT License](LICENSE).