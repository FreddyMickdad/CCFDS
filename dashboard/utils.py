import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, roc_curve, auc,
                           precision_recall_curve, average_precision_score)
import joblib
import json
from datetime import datetime
import os
import sys
from scipy import stats

# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import config

def load_model(model_path):
    """Load the model and return model, features, target"""
    if not os.path.isabs(model_path):
        model_path = os.path.join(parent_dir, model_path)
    
    print(f"Loading model from: {model_path}")
    model_data = joblib.load(model_path)
    
    if isinstance(model_data, tuple) and len(model_data) == 3:
        # Unpack the tuple and use the model's feature order
        clf, features, target = model_data
        print(f"Model features (in trained order): {features}")
        return clf, features, target
    else:
        raise ValueError("Model file does not contain expected data format")

def generate_dummy_data(schema_path, n=1, feature_order=None, distribution='uniform'):
    """
    Generate dummy data based on schema with different distribution options
    
    Parameters:
    -----------
    schema_path : str
        Path to the schema file
    n : int
        Number of samples to generate
    feature_order : list
        Order of features to generate
    distribution : str
        Type of distribution to use ('uniform', 'normal', 'realistic')
    """
    if not os.path.isabs(schema_path):
        schema_path = os.path.join(os.path.dirname(__file__), schema_path)
    
    print(f"Loading schema from: {schema_path}")
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    
    # Create data using the specified feature order
    data = pd.DataFrame(index=range(n))
    for feat in feature_order:  # Use the model's feature order
        props = schema['features'][feat]
        min_val = float(props['min'])
        max_val = float(props['max'])
        
        if distribution == 'normal':
            # Use normal distribution centered between min and max
            mean = (min_val + max_val) / 2
            std = (max_val - min_val) / 6  # 99.7% of data within range
            values = np.random.normal(mean, std, n)
            values = np.clip(values, min_val, max_val)
        elif distribution == 'realistic':
            # Use beta distribution for more realistic data
            if feat == 'amt':
                # Right-skewed for amount
                values = stats.beta(2, 5).rvs(n) * (max_val - min_val) + min_val
            elif feat == 'age':
                # Normal-like for age
                values = stats.beta(5, 5).rvs(n) * (max_val - min_val) + min_val
            else:
                values = np.random.uniform(min_val, max_val, n)
        else:  # uniform
            values = np.random.uniform(min_val, max_val, n)
        
        # Round integer features
        if isinstance(props['default'], int):
            values = np.round(values).astype(int)
        
        data[feat] = values
    
    print(f"Generated data columns: {data.columns.tolist()}")
    return data

def plot_confusion_matrix(cm, normalize=True):
    """Create an enhanced confusion matrix plot"""
    labels = ['Non-Fraud', 'Fraud']
    
    if normalize:
        cm_sum = cm.sum(axis=1)[:, np.newaxis]
        # Avoid division by zero
        cm_sum = np.where(cm_sum == 0, 1, cm_sum)
        cm_percent = cm.astype('float') / cm_sum * 100
    else:
        cm_percent = cm

    # Create text for annotations
    text = []
    for i, row in enumerate(cm):
        text_row = []
        for j, value in enumerate(row):
            if normalize:
                percentage = cm_percent[i, j]
                text_row.append(f'{percentage:.1f}%\n({int(value)})')
            else:
                text_row.append(f'{int(value)}')
        text.append(text_row)

    fig = go.Figure(data=go.Heatmap(
        z=cm_percent,
        x=labels,
        y=labels,
        text=text,
        texttemplate='%{text}',
        textfont={'size': 14},
        colorscale='RdYlBu_r',
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        width=600,
        height=500,
    )
    
    return fig

def plot_feature_importance(clf, features):
    """Plot feature importance"""
    importance = pd.DataFrame({
        'feature': features,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=True)
    
    fig = go.Figure(go.Bar(
        x=importance['importance'],
        y=importance['feature'],
        orientation='h'
    ))
    
    fig.update_layout(
        title='Feature Importance',
        xaxis_title='Importance Score',
        yaxis_title='Feature',
        width=800,
        height=400,
    )
    
    return fig

def plot_roc_curve(y_true, y_prob):
    """Plot ROC curve with additional metrics"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    
    # Add ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        name=f'ROC curve (AUC = {roc_auc:.3f})',
        mode='lines',
    ))
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name='Random classifier',
        mode='lines',
        line=dict(dash='dash'),
    ))
    
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=700,
        height=500,
    )
    
    return fig

def plot_precision_recall_curve(y_true, y_prob):
    """Plot Precision-Recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        name=f'PR curve (AP = {avg_precision:.3f})',
        mode='lines',
    ))
    
    fig.update_layout(
        title='Precision-Recall Curve',
        xaxis_title='Recall',
        yaxis_title='Precision',
        width=700,
        height=500,
    )
    
    return fig

def evaluate_batch(y_true, y_pred, y_prob):
    """Calculate comprehensive metrics for batch predictions"""
    return {
        'Accuracy': float(accuracy_score(y_true, y_pred)),
        'Precision': float(precision_score(y_true, y_pred)),
        'Recall': float(recall_score(y_true, y_pred)),
        'F1 Score': float(f1_score(y_true, y_pred)),
        'AUC-ROC': float(auc(roc_curve(y_true, y_prob)[0], roc_curve(y_true, y_prob)[1])),
        'Average Precision': float(average_precision_score(y_true, y_prob))
    }

def analyze_feature_distributions(data, schema):
    """Analyze feature distributions and detect anomalies"""
    analysis = {}
    for col in data.columns:
        analysis[col] = {
            'mean': float(data[col].mean()),
            'std': float(data[col].std()),
            'min': float(data[col].min()),
            'max': float(data[col].max()),
            'q1': float(data[col].quantile(0.25)),
            'median': float(data[col].median()),
            'q3': float(data[col].quantile(0.75)),
            'outliers': int(np.sum(np.abs(stats.zscore(data[col])) > 3)),
            'out_of_bounds': int(np.sum((data[col] < schema['features'][col]['min']) | 
                                      (data[col] > schema['features'][col]['max'])))
        }
    
    return analysis
