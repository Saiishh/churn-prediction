import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the Telco Customer Churn dataset.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Preprocessed dataset
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    print(f"Original dataset shape: {df.shape}")
    
    # Handle missing values and data type issues
    df = clean_data(df)
    
    # Feature engineering
    df = engineer_features(df)
    
    print(f"Processed dataset shape: {df.shape}")
    return df

def clean_data(df):
    """
    Clean the dataset by handling missing values and data types.
    
    Args:
        df (pd.DataFrame): Raw dataset
        
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    df = df.copy()
    
    # Handle TotalCharges column (contains spaces instead of NaN)
    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
    
    # Fill missing TotalCharges with 0 (likely new customers)
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    # Convert SeniorCitizen to string for consistency
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    
    # Clean categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_columns:
        if col != 'customerID':  # Keep customer ID as is
            df[col] = df[col].str.strip()
    
    print(f"✓ Data cleaning completed. Missing values handled: {df.isnull().sum().sum()}")
    
    return df

def engineer_features(df):
    """
    Create additional features from existing ones.
    
    Args:
        df (pd.DataFrame): Cleaned dataset
        
    Returns:
        pd.DataFrame: Dataset with engineered features
    """
    df = df.copy()
    
    # Create tenure groups
    df['tenure_group'] = pd.cut(df['tenure'], 
                                bins=[0, 12, 24, 36, 48, 72], 
                                labels=['0-12', '13-24', '25-36', '37-48', '49-72'],
                                include_lowest=True)
    
    # Create monthly charges groups
    df['monthly_charges_group'] = pd.cut(df['MonthlyCharges'], 
                                         bins=[0, 35, 65, 100, df['MonthlyCharges'].max()], 
                                         labels=['Low', 'Medium', 'High', 'Very High'],
                                         include_lowest=True)
    
    # Average charges per month
    df['avg_charges_per_month'] = df['TotalCharges'] / (df['tenure'] + 1)  # Avoid division by zero
    
    # Total services count
    service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    df['total_services'] = 0
    for col in service_cols:
        if col in df.columns:
            df['total_services'] += (df[col] == 'Yes').astype(int)
    
    print("✓ Feature engineering completed")
    
    return df

def split_data(df, test_size=0.2, random_state=42):
    """
    Split the preprocessed data into training and testing sets.
    
    Args:
        df (pd.DataFrame): Preprocessed dataset
        test_size (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    X = df.drop('Churn', axis=1)
    y = df['Churn'].map({'Yes': 1, 'No': 0})  # Convert to binary
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"✓ Data split completed:")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Testing set: {X_test.shape[0]} samples")
    print(f"  Churn rate in training: {y_train.mean():.3f}")
    print(f"  Churn rate in testing: {y_test.mean():.3f}")
    
    return X_train, X_test, y_train, y_test
