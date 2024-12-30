import os
from typing import Union

import pandas as pd


def save_dataframe_to_csv(
    dataframe: pd.DataFrame, path: Union[str, os.PathLike]
) -> None:
    """
    Save a DataFrame to a CSV file at the specified path.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame to save.
    path (Union[str, os.PathLike]): The path where the CSV will be saved.

    Returns:
    None
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save the DataFrame to a CSV file
    dataframe.to_csv(path, index=False)
