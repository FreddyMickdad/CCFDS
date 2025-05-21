import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import os
import sys
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils import (load_model, generate_dummy_data, plot_confusion_matrix,
                  plot_roc_curve, plot_precision_recall_curve, 
                  plot_feature_importance, analyze_feature_distributions,
                  evaluate_batch)
import config

# Page config
st.set_page_config(
    page_title="🔍 Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stAlert {
        background-color: #1e2227;
        border: 2px solid #00ff00;
        border-radius: 5px;
        padding: 10px;
    }
    .fraud-alert {
        color: #ff0000;
        font-weight: bold;
        font-size: 24px;
        animation: blink 1s linear infinite;
    }
    .safe-alert {
        color: #00ff00;
        font-weight: bold;
        font-size: 24px;
    }
    .metric-card {
        background-color: #1e2227;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .feature-importance {
        margin-top: 20px;
        padding: 10px;
        background-color: #262730;
        border-radius: 5px;
    }
    @keyframes blink {
        50% { opacity: 0.5; }
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🎛️ Control Panel")
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "Navigation",
        ["📊 Model Overview",
         "🧪 Single Transaction Test",
         "📈 Batch Testing",
         "🔬 Feature Analysis"]
    )

# Load model and schema
@st.cache_resource
def load_resources():
    try:
        model_path = os.path.abspath(os.path.join(parent_dir, config.MODEL_PATH))
        print(f"Loading model from: {model_path}")
        clf, features, target = load_model(model_path)
        
        schema_path = os.path.join(os.path.dirname(__file__), 'data_schema.json')
        print(f"Loading schema from: {schema_path}")
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        return clf, features, schema
    except Exception as e:
        st.error(f"Error loading resources: {str(e)}")
        raise

try:
    clf, features, schema = load_resources()
except Exception as e:
    st.error("Failed to load the model or schema. Please check the file paths and try again.")
    st.stop()

if page == "📊 Model Overview":
    st.title("Fraud Detection Model Overview")
    
    # Model Information
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Model Architecture")
        st.write("**Model Type:** Decision Tree Classifier")
        st.write("**Features Used (in order of importance):**")
        
        # Feature importance plot
        importance_fig = plot_feature_importance(clf, features)
        st.plotly_chart(importance_fig, use_container_width=True)
    
    with col2:
        st.subheader("Feature Ranges")
        for feat in features:
            with st.expander(f"📊 {feat}"):
                props = schema['features'][feat]
                st.write(f"**Min:** {props['min']}")
                st.write(f"**Max:** {props['max']}")
                st.write(f"**Default:** {props['default']}")
    
    st.markdown("---")
    
    # Model Performance Metrics (if available)
    st.subheader("Model Performance Overview")
    st.info("Generate test predictions in the Batch Testing section to view performance metrics")

elif page == "🧪 Single Transaction Test":
    st.title("Single Transaction Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.form("single_test"):
            st.subheader("Transaction Details")
            
            # Create input fields with tooltips and validation
            inputs = {}
            for feat in features:
                props = schema['features'][feat]
                col_left, col_right = st.columns([3, 1])
                
                with col_left:
                    inputs[feat] = st.number_input(
                        f"{feat}",
                        min_value=float(props['min']),
                        max_value=float(props['max']),
                        value=float(props['default']),
                        help=f"Range: {props['min']} to {props['max']}"
                    )
                
                with col_right:
                    if inputs[feat] < props['min'] or inputs[feat] > props['max']:
                        st.warning("⚠️ Out of range")
            
            submitted = st.form_submit_button("Analyze Transaction")
    
    if submitted:
        with st.spinner("Analyzing transaction..."):
            # Create DataFrame with features in model's order
            input_data = pd.DataFrame([inputs])[features]
            
            # Get prediction and probability
            prediction = clf.predict(input_data)[0]
            prob = clf.predict_proba(input_data)[0, 1]
            
            with col2:
                st.subheader("Analysis Results")
                
                # Prediction result with confidence level
                if prediction == 1:
                    st.markdown('<p class="fraud-alert">⚠️ FRAUD DETECTED</p>', unsafe_allow_html=True)
                    confidence = "High" if prob > 0.8 else "Medium" if prob > 0.6 else "Low"
                    st.error(f"Confidence Level: {confidence}")
                else:
                    st.markdown('<p class="safe-alert">✅ TRANSACTION SAFE</p>', unsafe_allow_html=True)
                    confidence = "High" if prob < 0.2 else "Medium" if prob < 0.4 else "Low"
                    st.success(f"Confidence Level: {confidence}")
                
                # Probability gauge
                st.metric("Fraud Probability", f"{prob:.1%}")
                
                # Feature analysis
                st.subheader("Feature Analysis")
                for feat, value in inputs.items():
                    props = schema['features'][feat]
                    percentile = (value - props['min']) / (props['max'] - props['min']) * 100
                    st.write(f"**{feat}:** {value:.2f}")
                    st.progress(min(100, max(0, percentile)) / 100)

elif page == "📈 Batch Testing":
    st.title("Batch Transaction Testing")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Generate Test Data")
        
        # Data generation options
        n_samples = st.number_input("Number of transactions", min_value=1, max_value=10000, value=100)
        distribution = st.selectbox(
            "Data Distribution",
            ["uniform", "normal", "realistic"],
            help="Choose how the test data should be distributed"
        )
        
        if st.button("Generate and Analyze"):
            with st.spinner("Generating and analyzing transactions..."):
                try:
                    # Generate data
                    X = generate_dummy_data('data_schema.json', n=n_samples, 
                                         feature_order=features, 
                                         distribution=distribution)
                    
                    # Make predictions
                    predictions = clf.predict(X)
                    probabilities = clf.predict_proba(X)[:, 1]
                    
                    # Create results DataFrame
                    results = X.copy()
                    results['fraud_probability'] = probabilities
                    results['prediction'] = predictions
                    
                    # For demonstration, we'll use predictions as ground truth
                    # In real scenarios, you would use actual labels
                    y_true = predictions  # Using predictions as ground truth for demo
                    
                    # Calculate metrics
                    cm = confusion_matrix(y_true, predictions)
                    metrics = evaluate_batch(y_true, predictions, probabilities)
                    
                    # Display results
                    st.write("### Sample Predictions")
                    st.dataframe(results)
                    
                    # Feature distribution analysis
                    analysis = analyze_feature_distributions(X, schema)
                    
                    with col2:
                        st.subheader("Analysis Results")
                        
                        # Display metrics
                        for metric, value in metrics.items():
                            st.metric(metric, f"{value:.3f}")
                        
                        # Confusion matrix
                        st.write("### Confusion Matrix")
                        cm_fig = plot_confusion_matrix(cm)
                        st.plotly_chart(cm_fig)
                        
                        # ROC curve
                        st.write("### ROC Curve")
                        roc_fig = plot_roc_curve(y_true, probabilities)
                        st.plotly_chart(roc_fig)
                        
                        # PR curve
                        st.write("### Precision-Recall Curve")
                        pr_fig = plot_precision_recall_curve(y_true, probabilities)
                        st.plotly_chart(pr_fig)
                    
                    # Download results
                    st.download_button(
                        "Download Results",
                        results.to_csv(index=False),
                        "fraud_detection_results.csv",
                        "text/csv",
                        key='download-csv'
                    )
                
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.exception(e)

elif page == "🔬 Feature Analysis":
    st.title("Feature Analysis Dashboard")
    
    # Generate sample data for analysis
    n_samples = st.number_input("Number of samples for analysis", 
                              min_value=100, max_value=10000, value=1000)
    
    if st.button("Analyze Features"):
        with st.spinner("Analyzing feature distributions..."):
            # Generate data with different distributions
            data_uniform = generate_dummy_data('data_schema.json', n=n_samples, 
                                            feature_order=features, 
                                            distribution='uniform')
            data_normal = generate_dummy_data('data_schema.json', n=n_samples, 
                                           feature_order=features, 
                                           distribution='normal')
            data_realistic = generate_dummy_data('data_schema.json', n=n_samples, 
                                              feature_order=features, 
                                              distribution='realistic')
            
            # Analyze each feature
            for feat in features:
                st.write(f"## {feat}")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Distribution comparison
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=data_uniform[feat], name='Uniform',
                                             opacity=0.6))
                    fig.add_trace(go.Histogram(x=data_normal[feat], name='Normal',
                                             opacity=0.6))
                    fig.add_trace(go.Histogram(x=data_realistic[feat], name='Realistic',
                                             opacity=0.6))
                    
                    fig.update_layout(
                        title=f"{feat} Distribution Comparison",
                        xaxis_title=feat,
                        yaxis_title="Count",
                        barmode='overlay'
                    )
                    st.plotly_chart(fig)
                
                with col2:
                    # Statistical summary
                    stats_df = pd.DataFrame({
                        'Uniform': data_uniform[feat].describe(),
                        'Normal': data_normal[feat].describe(),
                        'Realistic': data_realistic[feat].describe()
                    })
                    st.write("Statistical Summary")
                    st.dataframe(stats_df)
                    
                    # Schema bounds
                    props = schema['features'][feat]
                    st.write("Schema Bounds")
                    st.json(props)
