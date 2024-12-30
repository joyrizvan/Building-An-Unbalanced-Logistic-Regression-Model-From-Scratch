import pandas as pd
import yaml

from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineering


class DataPipeline:
    def __init__(self, file_path: str, target_column: str):
        self.file_path = file_path
        self.target_column = target_column

    def run(self):

        testing_data_preprocessor = DataPreprocessor(self.file_path)
        data = testing_data_preprocessor.preprocess_data()

        feature_engineer = FeatureEngineering()
        df = feature_engineer.standardize_features(
            data, target_column=self.target_column
        )

        # These are the independent variables
        ind = df.columns.difference([self.target_column]).tolist()

        # These are the dependent variables
        dep = [self.target_column]

        return df, ind, dep
