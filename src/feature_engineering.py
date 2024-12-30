import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.shared_functions import save_dataframe_to_csv


class FeatureEngineering:
    def __init__(self):
        self.scaler = StandardScaler()

    def standardize_features(
        self, data: pd.DataFrame, target_column: str
    ) -> pd.DataFrame:
        """
        Standardizes the feature matrix using StandardScaler.

        Parameters:
        X (pd.DataFrame): The feature matrix to be standardized.

        Returns:
        pd.DataFrame: The standardized feature matrix.

        """

        data2 = data.drop(target_column, axis=1)
        X_continuous_scaled = self.scaler.fit_transform(data2)
        X_continuous_scaled = pd.DataFrame(
            X_continuous_scaled, columns=data2.columns, index=data2.index
        )
        X_continuous_scaled[target_column] = data[target_column].values

        save_dataframe_to_csv(
            X_continuous_scaled, "../data/transformed/risk-train-transformed.csv"
        )

        return X_continuous_scaled
