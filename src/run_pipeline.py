import os
import sys
import pandas as pd
from pathlib import Path

# Set BASE_DIR as the root of the project (one level above /src)
BASE_DIR = Path(__file__).resolve().parent.parent

# Add 'src' directory to Python path
SRC_DIR = BASE_DIR / 'src'
sys.path.append(str(SRC_DIR))

# Import custom modules from src/
from my_preprocessing import load_and_preprocess_data, split_data
from model_training import train_model
from evaluation import evaluate_model
from interpretability import generate_shap_explanations

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = ['data', 'src', 'models', 'reports', 'dashboard']
    for directory in directories:
        dir_path = BASE_DIR / directory
        dir_path.mkdir(parents=True, exist_ok=True)
    print("✓ Directories created/verified")

def main():
    """Execute the complete ML pipeline."""
    print("🚀 Starting Customer Churn Prediction Pipeline")
    print("=" * 50)

    # Step 0: Setup
    create_directories()

    try:
        # Step 1: Data Preprocessing
        print("\n📊 Step 1: Data Preprocessing")
        data_path = BASE_DIR / "data" / "telco_churn.csv"

        if not data_path.exists():
            print(f"❌ Dataset not found at {data_path}")
            print("Please download the Telco Customer Churn dataset from:")
            print("https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
            print("And place it in the data/ directory as 'telco_churn.csv'")
            return

        df_processed = load_and_preprocess_data(str(data_path))
        X_train, X_test, y_train, y_test = split_data(df_processed)
        print("✓ Data preprocessing completed")

        # Step 2: Model Training
        print("\n🤖 Step 2: Model Training")
        model = train_model(X_train, y_train)
        print("✓ Model training completed")

        # Step 3: Model Evaluation
        print("\n📈 Step 3: Model Evaluation")
        evaluate_model(model, X_test, y_test)
        print("✓ Model evaluation completed")

        # Step 4: Model Interpretability
        print("\n🔍 Step 4: Model Interpretability")
        generate_shap_explanations(model, X_test)
        print("✓ SHAP explanations generated")

        print("\n🎉 Pipeline completed successfully!")
        print("=" * 50)
        print("Next steps:")
        print("1. Check the reports/ directory for SHAP summary plot")
        print("2. Run the Streamlit dashboard: streamlit run dashboard/app.py")

    except Exception as e:
        print(f"❌ Pipeline failed with error: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
