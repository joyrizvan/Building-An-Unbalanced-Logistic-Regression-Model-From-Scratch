import os
import pickle
import sys

import numpy as np
import pandas as pd
import yaml

from scripts.data_pipeline import DataPipeline
from src.CustomLogisticRegression import CustomLogisticRegression
from src.data_preprocessing import DataPreprocessor
from src.evaluate import Metrics
from src.feature_engineering import FeatureEngineering
from src.train_model import split_train_test

# Load configuration
config = None
try:
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
except FileNotFoundError:
    print("Error: config.yaml file not found.")
except yaml.YAMLError as e:
    print(f"Error reading config.yaml: {e}")

# Check if config was loaded successfully
if config is None:
    print("Failed to load configuration. Please check config.yaml file.")
    exit(1)


# Initialize DataPipeline with test data
data_pipeline = DataPipeline(
    file_path=config["data"]["test_data_path"], target_column="Class"
)

# Preprocess and retrieve dataframe, independent (X) and dependent (y) variables
df, ind, dep = data_pipeline.run()

# Load the trained model
with open(config["data"]["output_dir"], "rb") as f:
    clr = pickle.load(f)

# Getting the testing data to matrix form
x_test, y_test = clr.load_data(df, ind, dep)

# Get the y predicted values
y_pred = clr.predict(x_test)

# Evaluating the models
mt = Metrics()
y_pred, confusion_matrix, accuracy, precision, recall = mt.evaluate_model(
    y_pred, y_test, True
)
