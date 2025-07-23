"""
Model evaluation module for Customer Churn Prediction.
Provides comprehensive evaluation metrics and visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, 
    confusion_matrix, roc_curve, precision_recall_curve
)
import os

def evaluate_model(model, X_test, y_test, save_plots=True):
    """
    Evaluate the trained model on test data and generate comprehensive metrics.
    
    Args:
        model: Trained model pipeline
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        save_plots (bool): Whether to save evaluation plots
    """
    print("Evaluating model performance...")
    
    # Generate predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    if save_plots:
        create_evaluation_plots(y_test, y_pred, y_pred_proba)
    
    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

def create_evaluation_plots(y_test, y_pred, y_pred_proba):
    """
    Create and save evaluation plots.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        y_pred_proba: Prediction probabilities
    """
    # Create reports directory
    os.makedirs("reports", exist_ok=True)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Evaluation Results', fontsize=16, fontweight='bold')
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC Curve (AUC = {roc_auc:.4f})')
    axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend(loc="lower right")
    
    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    axes[1, 0].plot(recall, precision, color='blue', lw=2)
    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision-Recall Curve')
    
    # 4. Prediction Distribution
    axes[1, 1].hist(y_pred_proba[y_test == 0], bins=30, alpha=0.7, 
                    label='No Churn', color='blue')
    axes[1, 1].hist(y_pred_proba[y_test == 1], bins=30, alpha=0.7, 
                    label='Churn', color='red')
    axes[1, 1].set_xlabel('Prediction Probability')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Prediction Probability Distribution')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('reports/evaluation_plots.png', dpi=300, bbox_inches='tight')
    print("✓ Evaluation plots saved to reports/evaluation_plots.png")
    plt.close()

def generate_model_summary(model, X_test):
    """
    Generate a summary of the model including feature importance.
    
    Args:
        model: Trained model pipeline
        X_test (pd.DataFrame): Test features for feature names
    """
    print("\n" + "="*50)
    print("MODEL SUMMARY")
    print("="*50)
    
    # Get feature names after preprocessing
    try:
        # Get feature names from the preprocessor
        preprocessor = model.named_steps['preprocessor']
        feature_names = []
        
        # Get numerical feature names
        num_features = preprocessor.named_transformers_['num']
        if hasattr(preprocessor, 'transformers_'):
            num_feature_names = preprocessor.transformers_[0][2]  # numerical column names
            feature_names.extend(num_feature_names)
            
            # Get categorical feature names
            cat_transformer = preprocessor.named_transformers_['cat']
            if hasattr(cat_transformer, 'get_feature_names_out'):
                cat_feature_names = cat_transformer.get_feature_names_out()
                feature_names.extend(cat_feature_names)
        
        # Get feature importance from XGBoost
        xgb_model = model.named_steps['classifier']
        if hasattr(xgb_model, 'feature_importances_'):
            importances = xgb_model.feature_importances_
            
            # Create feature importance dataframe
            if len(feature_names) == len(importances):
                feature_importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                print("\nTop 10 Most Important Features:")
                print(feature_importance_df.head(10).to_string(index=False))
            else:
                print(f"Feature count mismatch: {len(feature_names)} names vs {len(importances)} importances")
                
    except Exception as e:
        print(f"Could not extract feature importance: {str(e)}")
    
    # Model parameters
    print(f"\nModel Parameters:")
    if hasattr(model.named_steps['classifier'], 'get_params'):
        params = model.named_steps['classifier'].get_params()
        for key, value in params.items():
            if not key.startswith('_'):
                print(f"  {key}: {value}")

def calculate_business_metrics(y_test, y_pred, y_pred_proba, threshold=0.5):
    """
    Calculate business-relevant metrics for churn prediction.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        y_pred_proba: Prediction probabilities
        threshold: Classification threshold
    """
    # Adjust predictions based on threshold
    y_pred_adjusted = (y_pred_proba >= threshold).astype(int)
    
    # Calculate confusion matrix components
    cm = confusion_matrix(y_test, y_pred_adjusted)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate business metrics
    total_customers = len(y_test)
    actual_churners = np.sum(y_test)
    predicted_churners = np.sum(y_pred_adjusted)
    
    # Precision and Recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print("\n" + "="*50)
    print("BUSINESS IMPACT ANALYSIS")
    print("="*50)
    print(f"Total customers in test set: {total_customers}")
    print(f"Actual churners: {actual_churners} ({actual_churners/total_customers*100:.1f}%)")
    print(f"Predicted churners: {predicted_churners} ({predicted_churners/total_customers*100:.1f}%)")
    print(f"Correctly identified churners: {tp} ({tp/actual_churners*100:.1f}% of actual churners)")
    print(f"False alarms: {fp} ({fp/total_customers*100:.1f}% of total customers)")
    print(f"Missed churners: {fn} ({fn/actual_churners*100:.1f}% of actual churners)")
    print(f"Precision: {precision:.3f} (of predicted churners, {precision*100:.1f}% actually churn)")
    print(f"Recall: {recall:.3f} (of actual churners, {recall*100:.1f}% are caught)")