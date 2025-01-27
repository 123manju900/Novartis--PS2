# main.py
"""
This script processes clinical trial data and trains a model using XGBoost.
Make sure to set up the environment using the provided requirements.txt file before running this script.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, f1_score
from sentence_transformers import SentenceTransformer
import xgboost as xgb
import matplotlib.pyplot as plt

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Load the dataset
def load_data(file_path):
    """Loads data from an Excel file."""
    return pd.read_excel(file_path)

def clean_data(df):
    """Performs basic data cleaning and preprocessing."""
    df['Study Title'] = df['Study Title'].fillna("").str.lower().str.replace(r"[^a-z0-9\s]", "", regex=True)
    df['Interventions'] = df['Interventions'].fillna("")
    return df

def extract_interventions(intervention_text):
    """Extracts interventions from text."""
    categories = {"DRUG": [], "DEVICE": [], "BIOLOGICAL": [], "PROCEDURE": []}
    if not pd.isna(intervention_text):
        items = str(intervention_text).split('|')
        for item in items:
            for key in categories.keys():
                if f'{key}:' in item:
                    categories[key].append(item.split(f'{key}:')[1].strip().lower())
    return categories

# Load and preprocess data
def preprocess_data(file_path):
    df = load_data(file_path)
    df = clean_data(df)
    df['interventions'] = df['Interventions'].apply(extract_interventions)

    # Extract primary intervention
    df['Primary Intervention'] = df['interventions'].apply(lambda x: next((k for k, v in x.items() if v), "OTHER"))
    return df

# Define the main function
def main():
    # File paths
    data_file = "clean-data.xlsx"  # Place your dataset here

    # Preprocess data
    df = preprocess_data(data_file)

    # Prepare data for training
    df['Study Results'] = df['Study Results'].map({'NO': 0, 'YES': 1})
    df = pd.get_dummies(df, columns=['Primary Intervention'], drop_first=True)

    # Extract features and target
    X = df.drop(columns=['Time taken for Enrollment'])
    y = df['Time taken for Enrollment']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    # Train XGBoost model
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        random_state=RANDOM_SEED
    )
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE: {rmse}")

    # Plot feature importance
    xgb.plot_importance(model, importance_type='gain')
    plt.show()

if __name__ == "__main__":
    main()
