"""
Streamlit dashboard for Customer Churn Prediction.
Provides an interactive interface for making churn predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add parent directory to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Set page config
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    """Load the trained model."""
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error(f"Model file not found at {model_path}. Please run the training pipeline first.")
        return None

def create_input_features():
    """Create input widgets for all features."""
    st.sidebar.header("Customer Information")
    
    # Customer Demographics
    st.sidebar.subheader("Demographics")
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.sidebar.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.sidebar.selectbox("Has Dependents", ["No", "Yes"])
    
    # Account Information
    st.sidebar.subheader("Account Information")
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method = st.sidebar.selectbox(
        "Payment Method", 
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 18.0, 120.0, 70.0)
    total_charges = st.sidebar.number_input("Total Charges ($)", 0.0, 9000.0, float(monthly_charges * tenure))
    
    # Services
    st.sidebar.subheader("Services")
    phone_service = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.sidebar.selectbox(
        "Multiple Lines", 
        ["No phone service", "No", "Yes"] if phone_service == "No" else ["No", "Yes"]
    )
    
    internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    
    if internet_service == "No":
        online_security = online_backup = device_protection = tech_support = "No internet service"
        streaming_tv = streaming_movies = "No internet service"
    else:
        online_security = st.sidebar.selectbox("Online Security", ["No", "Yes"])
        online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes"])
        device_protection = st.sidebar.selectbox("Device Protection", ["No", "Yes"])
        tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes"])
        streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes"])
        streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes"])
    
    # Create feature dictionary
    features = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    return features

def add_engineered_features(features):
    """Add the same engineered features used in training."""
    # Tenure group
    tenure = features['tenure']
    if tenure <= 12:
        tenure_group = '0-12'
    elif tenure <= 24:
        tenure_group = '13-24'
    elif tenure <= 36:
        tenure_group = '25-36'
    elif tenure <= 48:
        tenure_group = '37-48'
    else:
        tenure_group = '49-72'
    
    # Monthly charges group
    monthly_charges = features['MonthlyCharges']
    if monthly_charges <= 35:
        monthly_charges_group = 'Low'
    elif monthly_charges <= 65:
        monthly_charges_group = 'Medium'
    elif monthly_charges <= 100:
        monthly_charges_group = 'High'
    else:
        monthly_charges_group = 'Very High'
    
    # Average charges per month
    avg_charges_per_month = features['TotalCharges'] / (features['tenure'] + 1)
    
    # Total services count
    services = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
               'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    total_services = sum(1 for service in services if features.get(service) == 'Yes')
    
    # Add engineered features
    features['tenure_group'] = tenure_group
    features['monthly_charges_group'] = monthly_charges_group
    features['avg_charges_per_month'] = avg_charges_per_month
    features['total_services'] = total_services
    
    return features

def create_gauge_chart(probability):
    """Create a gauge chart for churn probability."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Churn Probability (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkred" if probability > 0.7 else "orange" if probability > 0.3 else "darkgreen"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "lightcoral"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}))
    
    fig.update_layout(height=300)
    return fig

def create_feature_impact_chart(features):
    """Create a chart showing the impact of different features."""
    # Define risk factors and their weights (simplified example)
    risk_factors = {
        'Contract': 0.3 if features['Contract'] == 'Month-to-month' else 0.1 if features['Contract'] == 'One year' else 0.0,
        'InternetService': 0.2 if features['InternetService'] == 'Fiber optic' else 0.1 if features['InternetService'] == 'DSL' else 0.0,
        'PaymentMethod': 0.25 if features['PaymentMethod'] == 'Electronic check' else 0.05,
        'tenure': 0.3 if features['tenure'] < 12 else 0.15 if features['tenure'] < 24 else 0.0,
        'MonthlyCharges': 0.1 if features['MonthlyCharges'] > 80 else 0.0,
        'TechSupport': 0.15 if features['TechSupport'] == 'No' else 0.0,
        'OnlineSecurity': 0.1 if features['OnlineSecurity'] == 'No' else 0.0
    }
    
    # Create bar chart
    fig = px.bar(
        x=list(risk_factors.keys()),
        y=list(risk_factors.values()),
        title="Risk Factors Contributing to Churn",
        labels={'x': 'Features', 'y': 'Risk Score'},
        color=list(risk_factors.values()),
        color_continuous_scale='Reds'
    )
    fig.update_layout(height=400)
    return fig

def main():
    """Main Streamlit application."""
    # Header
    st.title("📊 Customer Churn Prediction Dashboard")
    st.markdown("---")
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    # Create two columns
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Prediction")
        
        # Get input features
        features = create_input_features()
        features = add_engineered_features(features)
        
        # Create DataFrame for prediction
        input_df = pd.DataFrame([features])
        
        # Make prediction
        try:
            prediction_proba = model.predict_proba(input_df)[0]
            churn_probability = prediction_proba[1]
            prediction = model.predict(input_df)[0]
            
            # Display results
            st.subheader("Prediction Results")
            
            # Prediction label
            if prediction == 1:
                st.error("🚨 **HIGH RISK** - Customer likely to churn")
            else:
                st.success("✅ **LOW RISK** - Customer likely to stay")
            
            # Probability
            st.metric(
                "Churn Probability", 
                f"{churn_probability:.1%}",
                delta=f"{churn_probability - 0.5:.1%}" if churn_probability > 0.5 else f"{0.5 - churn_probability:.1%}"
            )
            
            # Risk level
            if churn_probability > 0.7:
                risk_level = "🔴 HIGH"
                risk_color = "red"
            elif churn_probability > 0.3:
                risk_level = "🟡 MEDIUM"
                risk_color = "orange"
            else:
                risk_level = "🟢 LOW"
                risk_color = "green"
            
            st.markdown(f"**Risk Level:** <span style='color:{risk_color}'>{risk_level}</span>", 
                       unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.stop()
    
    with col2:
        st.header("Analysis")
        
        # Gauge chart
        gauge_fig = create_gauge_chart(churn_probability)
        st.plotly_chart(gauge_fig, use_container_width=True)
        
        # Feature impact chart
        impact_fig = create_feature_impact_chart(features)
        st.plotly_chart(impact_fig, use_container_width=True)
    
    # Customer insights
    st.markdown("---")
    st.header("Customer Insights & Recommendations")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Customer Profile Summary")
        st.write(f"• **Tenure:** {features['tenure']} months")
        st.write(f"• **Contract:** {features['Contract']}")
        st.write(f"• **Monthly Charges:** ${features['MonthlyCharges']:.2f}")
        st.write(f"• **Payment Method:** {features['PaymentMethod']}")
        st.write(f"• **Internet Service:** {features['InternetService']}")
        st.write(f"• **Total Services:** {features['total_services']}")
    
    with col4:
        st.subheader("Retention Recommendations")
        
        recommendations = []
        
        if features['Contract'] == 'Month-to-month':
            recommendations.append("• Offer long-term contract incentives")
        
        if features['PaymentMethod'] == 'Electronic check':
            recommendations.append("• Encourage automatic payment methods")
        
        if features['tenure'] < 12:
            recommendations.append("• Implement new customer retention program")
        
        if features['TechSupport'] == 'No':
            recommendations.append("• Offer complimentary tech support")
        
        if features['OnlineSecurity'] == 'No' and features['InternetService'] != 'No':
            recommendations.append("• Promote online security add-ons")
        
        if features['MonthlyCharges'] > 80:
            recommendations.append("• Consider loyalty discounts for high-value customers")
        
        if not recommendations:
            recommendations.append("• Customer shows good retention indicators")
            recommendations.append("• Continue providing excellent service")
        
        for rec in recommendations:
            st.write(rec)
    
    # Model information
    with st.expander("Model Information"):
        st.write("""
        **Model:** XGBoost Classifier with preprocessing pipeline
        
        **Features Used:**
        - Demographics (Gender, Age, Partner, Dependents)
        - Account Information (Tenure, Contract, Billing)
        - Services (Phone, Internet, Add-ons)
        - Engineered Features (Tenure groups, Service counts, etc.)
        
        **Performance Metrics:**
        - Training performed with 5-fold cross-validation
        - Hyperparameter tuning using GridSearchCV
        - Model saved with preprocessing pipeline
        
        **Note:** This prediction is based on historical patterns and should be used as a guide for retention strategies.
        """)

if __name__ == "__main__":
    main()