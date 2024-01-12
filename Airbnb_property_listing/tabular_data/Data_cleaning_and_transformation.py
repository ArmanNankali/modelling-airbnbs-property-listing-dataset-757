import seaborn as sns
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import pickle
import pandas as pd
import csv as csv
import numpy as np
from scipy import stats
import pickle
import pandas as pd
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import PowerTransformer

with open("numerical_data.pkl", "rb") as file:
    numerical_data = pickle.load(file)

numerical_data.describe()

class Plotter():
    def __init__(self, df):
        """
        Instantiate the Plotter class.

        Parameters:
        - df (pd.DataFrame): The DataFrame containing numerical variables for plotting.
        """
        self.df = df
        self.df = df
     
    
    def correlation_matrix(self, matrix_name):
        """
        Plot a correlation matrix for numerical variables in the DataFrame.

        Parameters:
        - matrix_name (str): Name to be used for the plot.

        Returns:
        - None: Displays the correlation matrix plot.
        """
        self.matrix_name = matrix_name
        numerical_df = self.df.select_dtypes(include=['int64', 'float64'])
        corr = numerical_df.corr()
        mask = np.zeros_like(corr, dtype=np.bool_)
        #mask[np.triu_indices_from(mask)] = True
        plt.figure(figsize=(10, 8))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(corr, mask=mask, 
            square=True, linewidths=1, annot=True, fmt=".2f", cmap=cmap)
        plt.yticks(rotation=0)
        plt.title(f"Correlation Matrix of all Numerical Variables: {matrix_name}")
        plt.show()
    
    
    def skew_plot(self, col):
        """
        Plot a histogram and display the skewness of a specified column.

        Parameters:
        - col (str): The column for which to plot the histogram and display skewness.

        Returns:
        - None: Displays the histogram and skewness information.
        """
        self.col = col
        self.df[self.col].hist(bins=50)
        print(f"Skew of {self.col} column is {(self.df[self.col].skew()).round(2)}")
    
    
    def dak2(self, col):
        """
        Perform the D'Agostino's K^2 test for normality on a specified column.

        Parameters:
        - col (str): The column for which to perform the normality test.

        Returns:
        - None: Displays the results of the normality test.
        """
        self.col = col
        stat, p = stats.normaltest(self.df[col], nan_policy='omit')
        print(f"{self.col}")
        print("Statistics=%.3f, p=%.3f" % (stat, p))

nm1 = Plotter(numerical_data)
# From the correlation matrix it is apparent that bedrooms is strongly correlated with beds, guests and bathrooms.
# To mitigate the consequences of multicolinearity it could be warranted to remove one or all of those three columns.
# They will all be left in due to the low amount of data we have.
# Weight decay will be introduced to mitigate these effects.
nm1.correlation_matrix(numerical_data)

# Based on the D'Agostino's K^2 test, all of the columns are not normally distributed.
# This warrants the use of a batch normaliser for the neural net.
def all_dak2(object, df):
    dak2_success_list = []
    dak2_error_list = []
    for col in df:
        try:
            object.dak2(col)
            dak2_success_list.append(col)
        except:
            print(f"{col} unable to perfrom dak2")
            dak2_error_list.append(col)
    return dak2_success_list, dak2_error_list

all_dak2(nm1, numerical_data)

# Here the column names are converted to a list.
list1 = numerical_data.columns.tolist()

# All columns are tested for skew. Except for amenities_count all columns are significantly skewed.
# This will be remedied by finding the best transformation for each column.
for col in list1:
    nm1.skew_plot(col)

# Since the ranges of the columns vary a standard scaler will be introduced when training the final machine learning 
# models to make the mean 0 and the standard deviation 1.
columns = list1
df = numerical_data

for col in columns:
    min_value = df[col].min()
    max_value = df[col].max()
    range = max_value - min_value
    print(f'Range of {col}: {round(range,2)}, min:{min_value} to max:{max_value}')

class DtTransform():
    """
    A class for data transformations.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.

    Attributes:
    - df (pd.DataFrame): The DataFrame containing the data.
    - Log_list (list): A list to store columns that have undergone log transformation.
    - YJH_transformers (dict): A dictionary to store Yeo-Johnson transformers for each column.
    - BXC_transformers (dict): A dictionary to store Box-Cox transformers for each column.
    """
    def __init__(self, df):
        self.df = df
        self.Log_list = []
        self.YJH_transformers = {}
        self.BXC_transformers = {}

    def log_transform_check(self, col):
        """
        Apply log transformation to a column and check the skewness.

        Parameters:
        - col (str): The column to be log-transformed.

        Returns:
        - None: Modifies the DataFrame copy and plots the log-transformed data.
        """
        self.col = col
        log_population = self.df[self.col].map(lambda i: np.log(i) if i > 0 else 0)
        t=sns.histplot(log_population,label=f"{self.col} Log transformed Skewness: %.2f"%(log_population.skew()) )
        t.legend()
    
    def yeo_johnson_transform_check(self, col):
        """
        Apply Yeo-Johnson transformation to a column and check the skewness.

        Parameters:
        - col (str): The column to be Yeo-Johnson transformed.

        Returns:
        - None: Modifies the DataFrame copy and plots the Yeo-Johnson transformed data.
        """
        self.col = col
        pt = PowerTransformer(method="yeo-johnson")
        yeojohnson_population = self.df[self.col].values.reshape(-1, 1)
        yeojohnson_population = pt.fit_transform(yeojohnson_population)
        skewness = pd.Series(yeojohnson_population.squeeze()).skew()
        t = sns.histplot(yeojohnson_population, label=f"{col} Yeojohnson Population Skewness: %.2f" % skewness)
        t.legend()

    def box_cox_transform_check(self, col):
        """
        Apply Box-Cox transformation to a column and check the skewness.

        Parameters:
        - col (str): The column to be Box-Cox transformed.

        Returns:
        - None: Modifies the DataFrame copy and plots the Box-Cox transformed data.
        """
        self.col = col
        pt = PowerTransformer(method="box-cox")
        boxcox_population = self.df[self.col].values.reshape(-1, 1)
        boxcox_population = pt.fit_transform(boxcox_population)
        skewness = pd.Series(boxcox_population.squeeze()).skew()
        t = sns.histplot(boxcox_population, label=f"{col} Box-Cox Skewness: %.2f" % skewness)
        t.legend()
        
    def col_to_yeo_johnson_transform(self, col):
        """
        Apply Yeo-Johnson transformation to a column.

        Parameters:
        - col (str): The column to be Yeo-Johnson transformed.

        Returns:
        - None: Modifies the DataFrame and stores the transformer in YJH_transformers.
        """
        self.col = col
        pt = PowerTransformer(method="yeo-johnson")
        data = self.df[self.col].values.reshape(-1, 1)
        self.df[self.col] = pt.fit_transform(data)
        self.YJH_transformers[col] = pt
        t=sns.histplot(self.df[self.col],label=f"{self.col} Yeojohnson Population Skewness: %.2f"%(self.df[self.col].skew()) )
        t.legend()
# For the model predicting Price_night, all other columns will be transformed. 

list2 = [
        'guests',
 'beds',
 'bathrooms',
 'Cleanliness_rating',
 'Accuracy_rating',
 'Communication_rating',
 'Location_rating',
 'Check-in_rating',
 'Value_rating',
 'amenities_count',
 "bedrooms"]

# For the model predicting bedrooms, all other columns will be transformed.
list3 = [
        'guests',
 'beds',
 'bathrooms',
 'Cleanliness_rating',
 'Accuracy_rating',
 'Communication_rating',
 'Location_rating',
 'Check-in_rating',
 'Value_rating',
 'amenities_count',
 "Price_Night"]

# The class if instantiated for overall assessment of all columns after various transformations.
# Once a suitable transformation is chosen it will applied to both the data set for price_night prediction and that of bedrooms.
price_night_model_data = numerical_data.copy()
bedrooms_model_data = numerical_data.copy()
nm2 = DtTransform(numerical_data)
pn1 = DtTransform(price_night_model_data)
b1 = DtTransform(bedrooms_model_data)

# Of all the transformations, Yeo-Johnson produced the greatest reduction in skewness.
# This will be applied to all columns except the column we will try to predict.
for col in list1:
    nm2.yeo_johnson_transform_check(col)

# Transformations for price_night model
for col in list2:
    pn1.col_to_yeo_johnson_transform(col)

# Transformations for bedrooms model
for col in list3:
    b1.col_to_yeo_johnson_transform(col)

# The dataframes are saved in pickle format for later use.
with open(r"YJH_transformed_no_price_night_numerical_data.pkl", 'wb') as file:
    pickle.dump(price_night_model_data, file)

with open(r"YJH_transformed_no_bedrooms_numerical_data.pkl", 'wb') as file:
    pickle.dump(bedrooms_model_data, file)

