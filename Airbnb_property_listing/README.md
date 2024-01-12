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

## Installation instructions
To run this project locally, follow these steps:
1. Clone the repository: `git clone https://github.com/ArmanNankali/modelling-airbnbs-property-listing-dataset-757.git`
2. Install dependencies: `pip install -r requirements.txt`

## Usage instructions
1. Navigate to the project directory: `cd modelling-airbnbs-property-listing-dataset-757`
2. Run the `tabular_data.py` script for data cleaning and `Data_cleaning_and_transformation.py` for numerical data pre-processing.

## File structure of the project

- `requirements.txt`: List of project dependencies.
- `tabular_data/`: Directory containing scripts for data cleaning and transformation.
    - `tabular_data/tabular_data.py`: Python scripts for data cleaning.
    - `listing.csv`: The original Airbnb property listing dataset in CSV format.
    - `airbnb_listings_data.pkl`: Cleaned and transformed data saved as a pickle file.
    - `numerical_data.pkl`: Numerical data for training machine learning models.
    
    - `tabular_data/Data_cleaning_and_transformation.py`: Python scripts for data transformation.
    -`YJH_transformed_no_price_night_numerical_data.pkl`: Yeo-Johnson transformed variables, excluding price_night label.
    -`YJH_transformed_no_bedrooms_numerical_data.pkl`: Yeo-Johnson transformed variables, excluding bedrooms label.

## License information
This project is licensed under the [MIT License](LICENSE).