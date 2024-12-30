import warnings
from datetime import datetime
from typing import Optional

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from src.shared_functions import save_dataframe_to_csv

warnings.filterwarnings("ignore")


class DataPreprocessor:
    column_renaming = {
        "ORDER_ID": "Order_ID",
        "CLASS": "Class",
        "B_EMAIL": "Email_Bounce",
        "B_TELEFON": "Phone_Bounce",
        "B_BIRTHDATE": "Birth_Date",
        "FLAG_LRIDENTISCH": "Address_Match_Flag",
        "FLAG_NEWSLETTER": "Newsletter_Flag",
        "Z_METHODE": "Payment_Method",
        "Z_CARD_ART": "Card_Type",
        "Z_CARD_VALID": "Card_Validity",
        "Z_LAST_NAME": "Last_Name",
        "VALUE_ORDER": "Order_Value",
        "WEEKDAY_ORDER": "Order_Weekday",
        "TIME_ORDER": "Order_Time",
        "AMOUNT_ORDER": "Order_Amount",
        "ANUMMER_01": "Item_Number_01",
        "ANUMMER_02": "Item_Number_02",
        "ANUMMER_03": "Item_Number_03",
        "ANUMMER_04": "Item_Number_04",
        "ANUMMER_05": "Item_Number_05",
        "ANUMMER_06": "Item_Number_06",
        "ANUMMER_07": "Item_Number_07",
        "ANUMMER_08": "Item_Number_08",
        "ANUMMER_09": "Item_Number_09",
        "ANUMMER_10": "Item_Number_10",
        "CHK_LADR": "Billing_Address_Check",
        "CHK_RADR": "Shipping_Address_Check",
        "CHK_KTO": "Bank_Account_Check",
        "CHK_CARD": "Card_Check",
        "CHK_COOKIE": "Cookie_Check",
        "CHK_IP": "IP_Check",
        "FAIL_LPLZ": "Billing_Postal_Check",
        "FAIL_LORT": "Billing_City_Check",
        "FAIL_LPLZORTMATCH": "Billing_Postal_City_Match",
        "FAIL_RPLZ": "Shipping_Postal_Check",
        "FAIL_RORT": "Shipping_City_Check",
        "FAIL_RPLZORTMATCH": "Shipping_Postal_City_Match",
        "SESSION_TIME": "Session_Time",
        "NEUKUNDE": "New_Customer",
        "AMOUNT_ORDER_PRE": "Previous_Order_Amount",
        "VALUE_ORDER_PRE": "Previous_Order_Value",
        "DATE_LORDER": "Last_Order_Date",
        "MAHN_AKT": "Collection_Status",
        "MAHN_HOECHST": "Max_Collection_Level",
    }

    yes_no_columns = [
        "Class",
        "Email_Bounce",
        "Phone_Bounce",
        "Address_Match_Flag",
        "Newsletter_Flag",
        "Billing_Address_Check",
        "Shipping_Address_Check",
        "New_Customer",
        "Billing_Postal_Check",
        "Shipping_Postal_Check",
        "Billing_City_Check",
        "Card_Check",
        "Cookie_Check",
        "IP_Check",
        "Billing_Postal_City_Match",
        "Shipping_City_Check",
        "Shipping_Postal_City_Match",
        "Bank_Account_Check",
    ]

    def __init__(self, file_path: str, debug: bool = False):
        """
        Initializes the DataPreprocessor with the path to the data file.

        Parameters:
        - file_path (str): Path to the data file.
        """
        self.file_path = file_path
        self.data = pd.DataFrame()
        self.debug = debug

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Loads the dataset from a specified file path with custom NA values.

        Parameters:
            file_path (str): Path to the data file.

        Returns:
            pd.DataFrame: Loaded dataset with '?' values replaced by NaN.
        """
        self.data = pd.read_csv(
            file_path,
            delimiter="\t",
            header=0,  # Assumes the first row contains column names
            na_values=["NA", "NULL"],  # Custom NA values
            encoding="utf-8",  # Specify encoding if necessary
        )
        self.data.replace("?", np.nan, inplace=True)
        if self.debug:
            print("Data loaded successfully with shape:", self.data.shape)
        return self.data

    def rename_columns(self) -> pd.DataFrame:
        """
        Renames columns of the DataFrame according to a specified mapping.

        Parameters:
            data (pd.DataFrame): DataFrame with original column names.

        Returns:
            pd.DataFrame: DataFrame with renamed columns.
        """
        self.data.rename(columns=self.column_renaming, inplace=True)
        if self.debug:
            print("Columns renamed successfully.")

        return self.data

    def drop_columns(self, columns: list) -> pd.DataFrame:
        """
        Drops specified columns from the DataFrame.

        Parameters:
        - columns (list): List of column names to drop.

        Returns:
        - pd.DataFrame: DataFrame with specified columns removed.
        """
        self.data.drop(columns=columns, inplace=True, errors="ignore")
        if self.debug:
            print(f"Dropped columns: {columns}")
        return self.data

    def drop_na_threshold(self, threshold: float) -> pd.DataFrame:
        """
        Drops columns that have missing values exceeding a given threshold.

        Parameters:
        - threshold (float): Threshold for missing values as a fraction (e.g., 0.5 for 50%).

        Returns:
        - pd.DataFrame: DataFrame with columns dropped based on missing value threshold.
        """
        missing_threshold = int(threshold * len(self.data))
        self.data.dropna(thresh=missing_threshold, axis=1, inplace=True)
        if self.debug:
            print(f"Dropped columns with more than {threshold*100}% missing values.")
        return self.data

    def check_categorical_columns(self):
        """
        Check and print the number of unique categories in categorical columns of the DataFrame.

        This function identifies all categorical columns in the given DataFrame and counts the number of unique categories
        in each. It prints the name of each categorical column along with the number of categories it contains.
        If a column has more than one category, it specifies that; otherwise, it notes that there is only one category.

        Parameters:
        ----------
        df : pd.DataFrame
            The input DataFrame to analyze for categorical columns.

        Returns:
        -------
        None
            This function does not return a value; it prints output directly to the console.
        """
        categorical_columns = self.data.select_dtypes(
            include=["object", "category"]
        ).columns
        for col in categorical_columns:
            unique_values = self.data[col].unique()
            num_categories = len(unique_values)

            if num_categories >= 2:
                print(
                    f"{col}: {num_categories} categories (more than one category) - Values: {unique_values}"
                )
            else:
                print(
                    f"{col}: {num_categories} category (only one category) - Value: {unique_values[0]}"
                )

    def convert_yes_no_columns(self, columns: list) -> pd.DataFrame:
        """
        Convert specified 'yes'/'no' columns in the DataFrame to binary values (1 for 'yes', 0 for 'no').

        This function takes a DataFrame and a list of column names. It applies a transformation to each of the specified
        columns, replacing occurrences of 'yes' with 1 and 'no' with 0. If a value is neither 'yes' nor 'no', it remains unchanged.

        Parameters:
        ----------
        df : pd.DataFrame
            The input DataFrame containing the columns to convert.

        columns : list
            A list of column names in the DataFrame that should be converted from 'yes'/'no' to binary values.

        Returns:
        -------
        pd.DataFrame
            The modified DataFrame with specified columns converted to binary values.

        Raises:
        ------
        ValueError: If any of the specified columns are not present in the DataFrame.
        """
        # Check if the specified columns exist in the DataFrame
        missing_columns = [col for col in columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(
                f"The following columns are not in the DataFrame: {missing_columns}"
            )

        # Convert 'yes'/'no' values to binary using apply for each column
        for column in columns:
            self.data[column] = self.data[column].apply(
                lambda x: 1 if x == "yes" else (0 if x == "no" else x)
            )
        if self.debug:
            print(f"Converted columns to binary: {columns}")

        return self.data

    def transform_datetime_data(self) -> pd.DataFrame:
        """
        Transform the input DataFrame by calculating ordered hours and age from birth dates.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing the relevant columns.

        Returns:
        pd.DataFrame: The transformed DataFrame with new features and unnecessary columns dropped.
        """

        # Calculate 'Ordered_hours' from 'Order_Time'
        self.data["Ordered_hours"] = (
            pd.to_datetime(self.data["Order_Time"], format="%H:%M").dt.hour
            + pd.to_datetime(self.data["Order_Time"], format="%H:%M").dt.minute / 60
        )
        self.data.drop("Order_Time", axis=1, inplace=True)

        # Convert 'Birth_Date' to datetime
        self.data["Birth_Date"] = pd.to_datetime(self.data["Birth_Date"])

        # Calculate 'Age' based on 'Birth_Date'
        current_date = datetime.now()
        self.data["Age"] = current_date.year - self.data["Birth_Date"].dt.year

        # Drop the 'Birth_Date' column
        self.data.drop("Birth_Date", axis=1, inplace=True)

        if self.debug:
            print("Transformed datetime data: calculated Ordered_hours and Age.")
        return self.data

    def fill_missing_values(self):
        """
        Fills missing values in the DataFrame.
        Fills with mode for object columns and median for numeric columns.
        """
        for column in self.data.columns:
            if self.data[column].dtype == "object":
                self.data[column].fillna(self.data[column].mode()[0], inplace=True)
            else:
                self.data[column].fillna(self.data[column].median(), inplace=True)

    def one_hot_encode(self, categorical_columns: list) -> pd.DataFrame:
        """
        Performs one-hot encoding on specified categorical columns.

        Parameters:
            data (pd.DataFrame): DataFrame to apply one-hot encoding.

        Returns:
            pd.DataFrame: DataFrame with one-hot encoded columns.
        """

        self.data = pd.get_dummies(
            self.data, columns=categorical_columns, drop_first=True
        )
        return self.data

    def convert_to_boolean(self, boolean_columns: list) -> pd.DataFrame:
        """
        Converts specified columns to boolean (integer) type.

        Parameters:
            data (pd.DataFrame): DataFrame to convert columns.

        Returns:
            pd.DataFrame: DataFrame with specified columns converted to integer type.
        """

        for column in boolean_columns:
            self.data[column] = self.data[column].astype(int)  # Convert to int (0 or 1)
        return self.data

    def preprocess_data(self) -> pd.DataFrame:
        """
        Full preprocessing pipeline for loading, cleaning, and renaming columns.

        Parameters:
            file_path (str): Path to the data file.

        Returns:
            pd.DataFrame: Preprocessed DataFrame ready for analysis.
        """
        self.load_data(self.file_path)
        print("Data loaded successfully.")
        self.rename_columns()
        # print("\n Columns renamed successfully.")

        # Display data types and info
        # print(data.dtypes)
        # print("\n Initial Data Info:")
        # print(self.data.info())

        # Display initial missing values
        if self.debug:
            print("Initial Missing Values:")
            print(self.data.isnull().sum())

        # Check for missing values before dropping columns
        if self.debug:
            print(f"Total columns before dropping: {len(self.data.columns)}")

        self.drop_na_threshold(0.5)  # Drops columns with more than 50% missing values
        # Check for missing values after dropping columns
        # Display missing values after dropping columns
        if self.debug:
            print("Missing Values after dropping columns:")
            print(self.data.isnull().sum())

        self.drop_columns(["Last_Name", "Order_ID"])  # Drops the 'Last_Name' column

        # Fill missing values
        self.fill_missing_values()

        # Final check for missing values
        # Display final missing values
        if self.debug:
            print("Missing Values after filling:")
            print(self.data.isnull().sum())

        # data_dropped["Order_Weekday"].unique()

        # self.check_categorical_columns()

        self.convert_yes_no_columns(self.yes_no_columns)

        # self.check_categorical_columns()

        self.transform_datetime_data()
        categorical_columns = ["Payment_Method", "Order_Weekday"]
        self.one_hot_encode(categorical_columns)  # Call to one-hot encoding method
        boolean_columns = [
            "Payment_Method_credit_card",
            "Payment_Method_debit_card",
            "Payment_Method_debit_note",
            "Order_Weekday_Monday",
            "Order_Weekday_Saturday",
            "Order_Weekday_Sunday",
            "Order_Weekday_Thursday",
            "Order_Weekday_Tuesday",
            "Order_Weekday_Wednesday",
        ]
        self.convert_to_boolean(boolean_columns)  # Call to convert to boolean method

        save_dataframe_to_csv(
            self.data, "../data/processed/risk-train-preprocessed.csv"
        )

        return self.data
