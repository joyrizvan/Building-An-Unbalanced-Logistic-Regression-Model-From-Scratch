
## Customer Risk Prediction
## Authors: Rizvan Nahif
### Description

This project implements a custom logistic regression model from scratch, designed to classify binary data.
It demonstrates fundamental machine learning concepts by building a logistic regression model without relying on high-level machine learning libraries for the algorithm itself.

### Project Structure

ml_online_purchase_classification/  
│  
├── data/                              # Data storage (e.g., raw, interim, and processed data)  
│   ├── saved_model/                   # The trained LogisticRegressionModel  
│   ├── raw/                           # Original raw data (if available)  
│   ├── processed/                     # Cleaned and preprocessed data  
│   └── transformed/                   # Feature-selected/transformed data (optional)  
│  
├── src/                               # Core source code  
│   ├── __init__.py                    # Makes 'src' a package  
│   ├── data_preprocessing.py          # Data cleaning and preprocessing functions  
│   ├── feature_engineering.py         # Feature selection and transformation functions  
│   ├── CustomLogisticRegression.py    # Custom Logistic Regression model code  
│   ├── train_model.py                 # Code for training the model  
│   ├── evaluate.py                    # Model evaluation functions and metrics  
│   └── shared_functions.py            # Shared functions accross the project source code  
│  
├── notebooks/                         # Jupyter notebooks for experimentation and exploration  
│   ├── hyperparameter_tuning.ipynb    # Hyperparameter analysis  
│   └── results.ipynb                  # Project results displaying all steps taken  
│  
├── scripts/                           # Scripts for automation and running the pipeline   
│   └── data_pipeline.py               # Script to run data preprocessing and feature engineering  
│
├── config.yaml                        # Configuration file for data paths and parameters  
├── evaluate_test_data.py              # Evaluate trained model with test data   
├── main.py                            # Train the CustomLogistiRegression model  
├── requirements.txt                   # Python dependencies  
└── README.md                          # Project overview, how to run the code, and more  


### Execution Steps  
> <b>Train CustomLogisticRegression Model (Optional)</b>  

1. Run main.py file  
`python main.py`  

2. Trained model will be saved in path specified in config.yaml (_/data/saved_model_)  

> <b>Evaluate CustomLogisticRegression Model</b>  

1. Paste testing data file e.g. <b>risk-test.txt</b> in the directory _data/test_data_path_ specified in config.yaml  
2. run evaluate_test_data.py file  
`python evaluate_test_data.py`  

### Important Considerations  

* Review requirements.txt for installing dependencies

* Since our model requires for the data to go through preprocessing and feature engineering, 
the model should be tested againt a datasource that has all possible options for categorial features
this is so that onehot encoding gives the same result with the risk-test.txt as with 
the original datasource (risk-train.txt)


