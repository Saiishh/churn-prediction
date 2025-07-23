# Customer Churn Prediction System

This project is a comprehensive machine learning solution to predict customer churn in a telecommunications company. It includes data preprocessing, model training, explainability (SHAP), and an interactive Streamlit dashboard.
---
## 📦 Project Structure

churn-prediction-main/
│
├── dashboard/ # Streamlit dashboard application
│ └── app.py
│
├── data/ # Dataset used for training
│ └── telco_churn.csv
│
├── models/ # Trained model files
│ └── best_model.pkl
│
├── notebooks/ # EDA and experiment notebooks/scripts
│ └── eda.py
│
├── reports/ # SHAP and evaluation plots
│ ├── evaluation_plots.png
│ ├── shap_summary.png
│ └── shap_waterfall.png
│
├── src/ # Source code for preprocessing and model training
│ ├── my_preprocessing.py
│ ├── feature_engineering.py
│ ├── model_training.py
│ ├── evaluation.py
│ └── interpretability.py
│
└── README.md

yaml
Copy
Edit

---

## 🚀 Features

- 📊 **Exploratory Data Analysis**
- 🧠 **Model Training & Selection**
- ⚙️ **Custom Feature Engineering**
- 📈 **Model Evaluation & Visualization**
- 📉 **SHAP-based Model Interpretability**
- 🖥️ **Streamlit Dashboard for Live Predictions**

---

## 🛠️ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/churn-prediction.git
cd churn-prediction-main
2. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
If requirements.txt is missing, manually install:

bash
Copy
Edit
pip install pandas scikit-learn shap streamlit matplotlib seaborn
3. Run the dashboard
bash
Copy
Edit
cd dashboard
streamlit run app.py
📊 Dataset
The dataset is a Telco Customer Churn dataset with the following columns:

customerID, gender, SeniorCitizen, Partner, tenure, InternetService, ...

Target column: Churn

📈 Model
The model is trained using Scikit-Learn with techniques like:

Label Encoding, Feature Scaling

Logistic Regression / Random Forest / XGBoost

Hyperparameter Tuning

SHAP-based Explainability

📷 Visualizations
SHAP summary and waterfall plots are available under reports/ to interpret model decisions.
