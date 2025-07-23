"""
Model interpretability module using SHAP for Customer Churn Prediction.
Generates SHAP explanations and visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import os
from sklearn.pipeline import Pipeline

def generate_shap_explanations(model, X_test, max_samples=1000):
    """
    Generate SHAP explanations for the trained model.
    
    Args:
        model: Trained model pipeline
        X_test (pd.DataFrame): Test features
        max_samples (int): Maximum number of samples to use for SHAP analysis
    """
    print("Generating SHAP explanations...")
    
    # Create reports directory
    os.makedirs("reports", exist_ok=True)
    
    # Limit samples for computational efficiency
    if len(X_test) > max_samples:
        print(f"Using {max_samples} samples for SHAP analysis (out of {len(X_test)})")
        X_test_shap = X_test.sample(n=max_samples, random_state=42)
    else:
        X_test_shap = X_test.copy()
    
    try:
        # Transform the data using the preprocessing pipeline
        X_test_transformed = model.named_steps['preprocessor'].transform(X_test_shap)
        
        # Get the classifier
        classifier = model.named_steps['classifier']
        
        # Create SHAP explainer for the classifier only
        print("Creating SHAP TreeExplainer...")
        explainer = shap.TreeExplainer(classifier)
        
        # Calculate SHAP values
        print("Calculating SHAP values...")
        shap_values = explainer.shap_values(X_test_transformed)
        
        # Create feature names for transformed data
        feature_names = create_feature_names(model, X_test_shap)
        
        # Generate and save SHAP plots
        create_shap_plots(shap_values, X_test_transformed, feature_names, explainer.expected_value)
        
        print("✓ SHAP explanations generated successfully")
        
        return shap_values, feature_names
        
    except Exception as e:
        print(f"Error generating SHAP explanations: {str(e)}")
        print("Trying alternative approach...")
        
        # Alternative approach: use the entire pipeline
        try:
            create_shap_pipeline_explanation(model, X_test_shap)
        except Exception as e2:
            print(f"Alternative approach also failed: {str(e2)}")

def create_feature_names(model, X_test):
    """
    Create feature names for the transformed data.
    
    Args:
        model: Trained model pipeline
        X_test (pd.DataFrame): Test features
        
    Returns:
        list: Feature names for transformed data
    """
    try:
        preprocessor = model.named_steps['preprocessor']
        feature_names = []
        
        # Get numerical features
        num_transformer = preprocessor.named_transformers_['num']
        num_features = preprocessor.transformers_[0][2]  # column names
        feature_names.extend(num_features)
        
        # Get categorical features
        cat_transformer = preprocessor.named_transformers_['cat']
        cat_features = preprocessor.transformers_[1][2]  # column names
        
        # Get categorical feature names after one-hot encoding
        if hasattr(cat_transformer, 'get_feature_names_out'):
            cat_feature_names = cat_transformer.get_feature_names_out(cat_features)
            feature_names.extend(cat_feature_names)
        else:
            # Fallback: create generic names
            n_cat_features = len(cat_transformer.transform(X_test[cat_features].iloc[:1]).flatten())
            cat_feature_names = [f'cat_feature_{i}' for i in range(n_cat_features)]
            feature_names.extend(cat_feature_names)
        
        return feature_names
        
    except Exception as e:
        print(f"Error creating feature names: {str(e)}")
        # Return generic names
        n_features = model.named_steps['preprocessor'].transform(X_test.iloc[:1]).shape[1]
        return [f'feature_{i}' for i in range(n_features)]

def create_shap_plots(shap_values, X_transformed, feature_names, expected_value):
    """
    Create and save SHAP visualization plots.
    
    Args:
        shap_values: SHAP values array
        X_transformed: Transformed feature data
        feature_names: List of feature names
        expected_value: Model's expected value
    """
    # Set matplotlib backend to avoid GUI issues
    plt.switch_backend('Agg')
    
    # Convert to DataFrame for better plotting
    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)
    
    # 1. SHAP Summary Plot (bar plot)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_transformed_df, plot_type="bar", show=False)
    plt.title('SHAP Feature Importance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('reports/shap_summary_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ SHAP summary bar plot saved to reports/shap_summary_bar.png")
    
    # 2. SHAP Summary Plot (beeswarm plot)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_transformed_df, show=False)
    plt.title('SHAP Summary Plot (Impact on Model Output)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('reports/shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ SHAP summary plot saved to reports/shap_summary.png")
    
    # 3. SHAP Waterfall Plot (for first prediction)
    try:
        plt.figure(figsize=(10, 8))
        shap_explanation = shap.Explanation(
            values=shap_values[0], 
            base_values=expected_value, 
            data=X_transformed[0],
            feature_names=feature_names
        )
        shap.waterfall_plot(shap_explanation, show=False)
        plt.title('SHAP Waterfall Plot (Single Prediction Example)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('reports/shap_waterfall.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ SHAP waterfall plot saved to reports/shap_waterfall.png")
    except Exception as e:
        print(f"Could not create waterfall plot: {str(e)}")

def create_shap_pipeline_explanation(model, X_test):
    """
    Alternative approach to create SHAP explanations using the entire pipeline.
    
    Args:
        model: Trained model pipeline
        X_test (pd.DataFrame): Test features
    """
    print("Using pipeline explainer approach...")
    
    # Create a wrapper function for the pipeline
    def model_predict(X):
        return model.predict_proba(pd.DataFrame(X, columns=X_test.columns))[:, 1]
    
    # Create SHAP explainer
    explainer = shap.Explainer(model_predict, X_test.iloc[:100])  # Use small background set
    
    # Calculate SHAP values for a subset
    shap_values = explainer(X_test.iloc[:200])  # Explain subset of test data
    
    # Create plots
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, show=False)
    plt.title('SHAP Summary Plot', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('reports/shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ SHAP summary plot saved using pipeline approach")

def analyze_feature_interactions(model, X_test, feature_names=None, max_samples=500):
    """
    Analyze feature interactions using SHAP.
    
    Args:
        model: Trained model pipeline
        X_test (pd.DataFrame): Test features
        feature_names (list): List of feature names
        max_samples (int): Maximum number of samples to analyze
    """
    print("Analyzing feature interactions...")
    
    try:
        # Use a subset for computational efficiency
        X_subset = X_test.sample(n=min(max_samples, len(X_test)), random_state=42)
        
        # Transform data
        X_transformed = model.named_steps['preprocessor'].transform(X_subset)
        classifier = model.named_steps['classifier']
        
        # Create explainer
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_transformed)
        
        if feature_names is None:
            feature_names = create_feature_names(model, X_subset)
        
        # Create interaction plot for top features
        X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)
        
        # Get top 5 most important features
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        top_features_idx = np.argsort(mean_abs_shap)[-5:]
        
        print("Top 5 most important features:")
        for i, idx in enumerate(reversed(top_features_idx)):
            print(f"{i+1}. {feature_names[idx]}: {mean_abs_shap[idx]:.4f}")
        
        # Create dependence plot for the most important feature
        if len(top_features_idx) > 0:
            most_important_feature = top_features_idx[-1]
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(
                most_important_feature, 
                shap_values, 
                X_transformed_df, 
                show=False
            )
            plt.title(f'SHAP Dependence Plot: {feature_names[most_important_feature]}', 
                     fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig('reports/shap_dependence.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ SHAP dependence plot saved to reports/shap_dependence.png")
            
    except Exception as e:
        print(f"Error in feature interaction analysis: {str(e)}")

def generate_shap_report(model, X_test):
    """
    Generate a comprehensive SHAP analysis report.
    
    Args:
        model: Trained model pipeline
        X_test (pd.DataFrame): Test features
    """
    print("\n" + "="*50)
    print("SHAP MODEL INTERPRETABILITY REPORT")
    print("="*50)
    
    try:
        # Generate main SHAP explanations
        shap_values, feature_names = generate_shap_explanations(model, X_test)
        
        if shap_values is not None:
            # Analyze feature interactions
            analyze_feature_interactions(model, X_test, feature_names)
            
            # Calculate global feature importance
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'SHAP_Importance': mean_abs_shap
            }).sort_values('SHAP_Importance', ascending=False)
            
            print("\nTop 15 Features by SHAP Importance:")
            print(feature_importance_df.head(15).to_string(index=False))
            
            # Save feature importance to CSV
            feature_importance_df.to_csv('reports/shap_feature_importance.csv', index=False)
            print("\n✓ SHAP feature importance saved to reports/shap_feature_importance.csv")
        
    except Exception as e:
        print(f"Error generating SHAP report: {str(e)}")
    
    print("\n" + "="*50)