import pandas as pd
import csv
import chardet
import re
import ast
import pickle

def csv_to_dataframe(csv_path):
    '''
    A function to convert a csv file to a pandas dataframe.

    Parameters:
    - file_path (csv file path): The path to the csv file to be converted.
    '''
    with open(csv_path, "rb") as file:
        df = pd.read_csv(file)
    return df

class Tabular_data_cleaning:
    """
    A class for performing data cleaning operations on the tabular Airbnb property data.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame to be cleaned.
    """
    def __init__(self, df):
        self.df = df
        """
        Instantiate the class object.

        Parameters:
        - df (pandas.DataFrame): The DataFrame to be cleaned.
        """

    def remove_rows_with_missing_values(self, column):
        """
        Remove rows with missing values in a specified column.

        Parameters:
        - column (str): The column to check for missing values.

        Returns:
        - Tabular_data_cleaning: Returns the class object for method chaining.
        """
        self.df.dropna(subset=[column], inplace=True)
        return self
        
    def remove_null_rows_from_ratings(self, rating_string):
        """
        Remove rows containing null values from columns containing a specified rating string.

        Parameters:
        - rating_string (str): The rating string to identify columns.

        Returns:
        - Tabular_data_cleaning: Returns the class object for method chaining.
        """
        for column in self.df:
            try:
                if rating_string in str(column):
                    self.remove_rows_with_missing_values(column)
                    print(f"remove_null_rows_from_ratings success for {column}")
                else:
                    pass
                
            except Exception as e:
                print(f"remove_null_rows_from_ratings error: {e} for {column}")
        return self
    
    def row_column_value_shifter(self, row_index, columns, number_of_columns):
        """
        Shift values in specified columns for a given row.

        Parameters:
        - row_index (int): Index of the row to perform value shifting.
        - columns (list): List of columns to shift values.
        - number_of_columns (int): Number of columns to shift.

        Returns:
        - Tabular_data_cleaning: Returns the class object for method chaining.
        """
        n = 0
        while n < number_of_columns-1:
            try:
                self.df.at[row_index, columns[n]] = self.df.at[row_index, columns[n+1]]
                n += 1
                print("row_column_value_shifter success")
            except Exception as e:
                print(f"Error: {e}")
        print(f"{n} rows successfully shifted")
        self.df.drop("Unnamed: 19", axis=1, inplace=True)

        return self
    
    def set_default_feature_values(self, column_list, value):
        """
        Set default values for specified columns if they are null.

        Parameters:
        - column_list (list): List of columns to set default values.
        - value: Default value to set.

        Returns:
        - Tabular_data_cleaning: Returns the class object for method chaining.
        """
        def set_default_value(row):
            if pd.isnull(row):
                return value
            else:
                return row
        for column in column_list:
            try:
                self.df[column] = self.df[column].apply(set_default_value)
                print(f"set_default_value success for {column}")
            except Exception as e:
                print(f"set_default_value error: {e} for {column}")
        self.df.dropna(subset=["Description"], inplace=True)
        return self
    
    def unique_prefix_text(self, column):
        """
        Extract unique prefixes from a column containing lists.

        Parameters:
        - column (str): The column containing lists.

        Returns:
        - tuple: Returns a tuple containing sets of unique_text_list and error_causing_text.
        """
        unique_text_list = []
        error_causing_text = []
        for lists in self.df[column]:
            try:
                every_second_list = ast.literal_eval(lists)[::2]
                # Add the unique elements to unique_text_list
                unique_text_list += list(set(every_second_list))
            except Exception as e:
                print(f"Error: {e} with: '{lists}' ")
                if isinstance(lists, str):
                    error_causing_text += lists  
        return set(unique_text_list), (error_causing_text)

    def string_reformatter(self, column, string_list, replacement):
        """
        Reformats strings in a column based on a list of string replacements.

        Parameters:
        - column (str): The column containing strings to be reformatted.
        - string_list (list): List of strings to be replaced.
        - replacement: The replacement string.

        Returns:
        - None: Modifies the DataFrame in place.
        """
        successes = 0
        failures = 0
        
        def remove_from_description_string(description):
            nonlocal successes, failures
            for item in string_list:
                try:
                    if item in description:
                        description = description.replace(item, replacement)
                        successes += 1
                except Exception as e:
                    print(f"Error {e} with {description}")
                    failures += 1
                else:
                    pass
            description = description.replace(".", ". ")
            description = ' '.join(ast.literal_eval(description))

            return description
        
        self.df[column] = self.df[column].apply(remove_from_description_string)
        print(f"string_reformatter successes: {successes}, failures: {failures}")
    
    def convert_column_to_int64(self, column_list):
        """
        Convert specified columns to the int64 data type.

        Parameters:
        - column_list (list): List of columns to convert.

        Returns:
        - None: Modifies the DataFrame in place.
        """
        for column in column_list:
            self.df[column] = self.df[column].astype("int64")
            print(f"convert_column_to_int64 success for {column}")
        self.df.dropna(inplace=True)

# First the csv file for the Airbnb data is read from the csv file and then converted to a pandas dataframe
listings = csv_to_dataframe("listing.csv")
listings

def clean_tabular_data(object_name, dataframe, rating_column_string, row_shifter_column_list, default_value_column_list, unwanted_strings, int64_list):
    """
    Clean tabular data using a Tabular_data_cleaning object.

    Parameters:
    - object_name (str): Name of the object to be created.
    - dataframe (pandas.DataFrame): The input DataFrame to be cleaned.
    - rating_column_string (str): The rating string to identify columns for null row removal.
    - row_shifter_column_list (list): List of columns to shift values for a specific row.
    - default_value_column_list (list): List of columns to set default values.
    - unwanted_strings (list): List of strings to be removed from the 'Description' column.
    - int64_list (list): List of columns to be converted to the int64 data type.
    """
    object_name = Tabular_data_cleaning(dataframe)
    object_name.remove_null_rows_from_ratings(rating_column_string)
    object_name.unique_prefix_text("Description")
    object_name.row_column_value_shifter(586, row_shifter_column_list, 17)
    object_name.set_default_feature_values(default_value_column_list, 1)
    object_name.string_reformatter("Description", unwanted_strings, "")
    object_name.convert_column_to_int64(int64_list)


row_shifter_column_list = ["Description",
               "Amenities",
               "Location",
               "guests",
               "beds",
               "bathrooms",
               "Price_Night",
               "Cleanliness_rating",
               "Accuracy_rating",
               "Communication_rating",
               "Location_rating",
               "Check-in_rating",
               "Value_rating",
               "amenities_count",
               "url",
               "bedrooms",
               "Unnamed: 19"]

default_value_column_list = ["guests", "beds", "bathrooms", "bedrooms"]

unwanted_strings = ["The space",
                "Other things to note",
                "Licence number", 
                "Guest access", 
                "Dining and lounge areas", 
                "About this space", 
                "'',", r"\n", "', '", ", ,", ","]

int64_list = ["guests", "amenities_count", "bedrooms"]

clean_tabular_data("ls1", listings, "rating", row_shifter_column_list, default_value_column_list, unwanted_strings, int64_list)

# 1. Rows with null values for ratings columns are removed.
# 2. Upon using unique_prefix_text() it becomes apparent that other prefixes to each section of the description are repeated, similar to "About this place".
# However, it is also clear there is an erroneus value at row 586: " sleeps 6 with pool". 
# Upon further inspection, this rows values have been shifted out of place by 1 column to the right, creating the "Unnamed: 19" column.
# After shifting these values along by one, the redundant column can be deleted and the data cleaning can continue.
# 3. Row 586 values are shifted along into the right places and "Unnamed: 19" column is removed.
# 4. Null values in "guests", "beds", "bathrooms" and "bedrooms" column are replaced with a standard 1.
# 5. The "Description" strings have repeated prefixes to each section removed along with artifact whitespaces. Then they are converted to lists and joined where there were whitespaces (" ").
# 6. Finally columns that should be int64 are altered to that dtype and any rows with missing values are removed.

listings.info()
# 830 rows remain

# This dataframe will be saved as a pickle file for later use.
with open("airbnb_listings_data.pkl", "wb") as file:
    pickle.dump(listings, file)

# Non-numerical columns are dropped to leave behind numerical data only for training of models.
def drop_column_list(df, col_list):
    """
    Drops a list of columns from a DataFrame in place.

    Parameters:
    - df (pandas.DataFrame): The DataFrame from which to drop columns.
    - col_list (list): List of column names to be dropped.

    Returns:
    - pandas.DataFrame: The DataFrame after dropping the specified columns.
    """
    for col in col_list:
        df.drop(col, axis=1, inplace=True)
    return df

col_list = ["ID", "Category", "Title", "Description", "Amenities", "Location", "url"]
drop_column_list(listings, col_list)

import pickle

with open("numerical_data.pkl", "wb") as file:
    pickle.dump(listings, file)

