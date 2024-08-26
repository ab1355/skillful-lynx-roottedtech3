import numpy as np
import pandas as pd
import os

def load_hr_data():
    """
    Load HR data from CSV files in the input directory.
    """
    hr_data = None
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            if filename.endswith('.csv'):
                file_path = os.path.join(dirname, filename)
                print(f"Loading data from: {file_path}")
                if hr_data is None:
                    hr_data = pd.read_csv(file_path)
                else:
                    hr_data = hr_data.append(pd.read_csv(file_path), ignore_index=True)
    
    return hr_data

def preprocess_hr_data(df):
    """
    Preprocess the HR data.
    """
    # Handle missing values
    df = df.dropna()
    
    # Convert categorical variables to numerical
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df[col] = pd.Categorical(df[col]).codes
    
    # Normalize numerical columns
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    df[numerical_columns] = (df[numerical_columns] - df[numerical_columns].mean()) / df[numerical_columns].std()
    
    return df

def get_processed_hr_data():
    """
    Load and preprocess HR data.
    """
    raw_data = load_hr_data()
    if raw_data is not None:
        processed_data = preprocess_hr_data(raw_data)
        return processed_data
    else:
        print("No data found in the input directory.")
        return None

if __name__ == "__main__":
    processed_data = get_processed_hr_data()
    if processed_data is not None:
        print(processed_data.head())
        print(f"Processed data shape: {processed_data.shape}")
    else:
        print("Failed to process HR data.")