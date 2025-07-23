import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Set paths
DATA_PATH = r"C:\Users\SAINATH NIKAM\Desktop\Celebal\final project\churn_prediction_project\data\telco_churn.csv"
MODEL_PATH = r"C:\Users\SAINATH NIKAM\Desktop\Celebal\final project\churn_prediction_project\models\best_model.pkl"

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    
    # Clean TotalCharges
    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
    df['TotalCharges'].fillna(0, inplace=True)

    # Convert SeniorCitizen to categorical
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})

    # Strip spaces from categorical values
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()

    return df

def feature_engineering(df):
    df['tenure_group'] = pd.cut(df['tenure'],
                                bins=[0, 12, 24, 36, 48, 72],
                                labels=['0-12', '13-24', '25-36', '37-48', '49-72'])

    df['monthly_charges_group'] = pd.cut(df['MonthlyCharges'],
                                         bins=[0, 35, 65, 100, df['MonthlyCharges'].max()],
                                         labels=['Low', 'Medium', 'High', 'Very High'])

    df['avg_charges_per_month'] = df['TotalCharges'] / (df['tenure'] + 1)

    service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

    df['total_services'] = df[service_cols].apply(lambda row: sum(row == 'Yes'), axis=1)

    return df

def prepare_data_for_training(df):
    df = df.drop(['customerID'], axis=1)

    X = df.drop('Churn', axis=1)
    y = df['Churn'].map({'Yes': 1, 'No': 0})

    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

    return X, y, preprocessor

def train_model(X, y, preprocessor):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Define model pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    print("\n✅ Model Evaluation Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return pipeline

def save_model(pipeline, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"\n🎉 Model saved successfully at: {model_path}")

if __name__ == "__main__":
    print("🔄 Loading and preprocessing data...")
    df = load_and_clean_data(DATA_PATH)
    df = feature_engineering(df)
    
    print("📦 Preparing data for training...")
    X, y, preprocessor = prepare_data_for_training(df)

    print("🧠 Training model...")
    model = train_model(X, y, preprocessor)

    print("💾 Saving model...")
    save_model(model, MODEL_PATH)
