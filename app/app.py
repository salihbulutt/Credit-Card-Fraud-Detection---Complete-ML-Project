"""
Streamlit Web Application for Credit Card Fraud Detection
Provides interactive interface for fraud prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.inference import FraudDetector
from src.config import (
    FRAUD_PROBABILITY_THRESHOLD,
    HIGH_RISK_THRESHOLD,
    MEDIUM_RISK_THRESHOLD
)

# Page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .fraud-alert {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
    }
    .safe-alert {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detector' not in st.session_state:
    try:
        st.session_state.detector = FraudDetector()
        st.session_state.model_loaded = True
    except Exception as e:
        st.session_state.model_loaded = False
        st.session_state.error_message = str(e)

def create_gauge_chart(probability):
    """Create a gauge chart for fraud probability"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Fraud Probability", 'font': {'size': 24}},
        delta={'reference': 50, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#4caf50'},
                {'range': [50, 80], 'color': '#ff9800'},
                {'range': [80, 100], 'color': '#f44336'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': FRAUD_PROBABILITY_THRESHOLD * 100
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def create_feature_importance_chart(features):
    """Create feature importance bar chart"""
    if features and len(features) > 0:
        df = pd.DataFrame(features[:10])
        fig = px.bar(
            df,
            x='importance',
            y='feature',
            orientation='h',
            title="Top 10 Feature Importance",
            labels={'importance': 'Importance Score', 'feature': 'Feature'},
            color='importance',
            color_continuous_scale='Reds'
        )
        fig.update_layout(height=400, showlegend=False)
        return fig
    return None

def main():
    # Header
    st.markdown('<div class="main-header">💳 Credit Card Fraud Detection System</div>', unsafe_allow_html=True)
    
    # Check if model is loaded
    if not st.session_state.model_loaded:
        st.error(f"⚠️ Error loading model: {st.session_state.error_message}")
        st.info("Please ensure the model has been trained by running `python src/pipeline.py` first.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("🎯 Detection Settings")
        
        detection_mode = st.radio(
            "Select Mode",
            ["Single Transaction", "Batch Upload", "Example Transactions"]
        )
        
        st.markdown("---")
        st.header("ℹ️ Model Information")
        st.info(f"""
        **Model Type:** XGBoost
        **Features:** {len(st.session_state.detector.feature_names)}
        **Threshold:** {FRAUD_PROBABILITY_THRESHOLD:.1%}
        """)
        
        st.markdown("---")
        st.header("📊 Risk Levels")
        st.success(f"🟢 **LOW**: < {MEDIUM_RISK_THRESHOLD:.0%}")
        st.warning(f"🟡 **MEDIUM**: {MEDIUM_RISK_THRESHOLD:.0%} - {HIGH_RISK_THRESHOLD:.0%}")
        st.error(f"🔴 **HIGH**: > {HIGH_RISK_THRESHOLD:.0%}")
    
    # Main content
    if detection_mode == "Single Transaction":
        show_single_transaction_mode()
    elif detection_mode == "Batch Upload":
        show_batch_upload_mode()
    else:
        show_example_transactions_mode()

def show_single_transaction_mode():
    """Single transaction prediction interface"""
    st.header("🔍 Single Transaction Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Transaction Details")
        
        # Amount input
        amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0, step=10.0)
        
        # Time input
        time_seconds = st.number_input("Time (seconds since first transaction)", min_value=0, value=0)
        
        st.info("💡 PCA features (V1-V28) are pre-filled with example values. You can modify them below.")
        
        # PCA features in expandable section
        with st.expander("🔧 Advanced: PCA Features (V1-V28)", expanded=False):
            pca_cols = st.columns(4)
            pca_values = {}
            
            # Example PCA values (from a legitimate transaction)
            example_values = [-1.36, -0.07, 2.54, 1.38, -0.34, 0.46, 0.24, 0.10,
                            0.36, 0.09, -0.55, -0.62, -0.99, -0.31, 1.47, -0.47,
                            0.21, 0.03, 0.40, 0.25, -0.02, 0.28, -0.11, 0.07,
                            0.13, -0.19, 0.13, -0.02]
            
            for i in range(28):
                with pca_cols[i % 4]:
                    pca_values[f'V{i+1}'] = st.number_input(
                        f'V{i+1}',
                        value=float(example_values[i]),
                        format="%.6f",
                        key=f'v{i+1}'
                    )
        
        # Predict button
        if st.button("🔮 Predict Fraud", type="primary", use_container_width=True):
            # Prepare transaction
            transaction = {
                'Amount': amount,
                'Time': time_seconds,
                **pca_values
            }
            
            # Validate
            validation = st.session_state.detector.validate_transaction(transaction)
            
            if not validation['is_valid']:
                st.error("❌ Invalid transaction data:")
                for error in validation['errors']:
                    st.error(f"  • {error}")
                return
            
            if validation['warnings']:
                for warning in validation['warnings']:
                    st.warning(f"⚠️ {warning}")
            
            # Make prediction
            with st.spinner("Analyzing transaction..."):
                result = st.session_state.detector.predict_with_details(transaction)
            
            # Store result in session state
            st.session_state.last_result = result
    
    with col2:
        st.subheader("Quick Test")
        
        if st.button("🟢 Test Legitimate Transaction", use_container_width=True):
            legitimate_transaction = {
                'V1': -1.359807134, 'V2': -0.072781173, 'V3': 2.536346738,
                'V4': 1.378155224, 'V5': -0.338320770, 'V6': 0.462387778,
                'V7': 0.239598554, 'V8': 0.098697901, 'V9': 0.363786970,
                'V10': 0.090794172, 'V11': -0.551599533, 'V12': -0.617800856,
                'V13': -0.991389847, 'V14': -0.311169354, 'V15': 1.468176972,
                'V16': -0.470400525, 'V17': 0.207971242, 'V18': 0.025790720,
                'V19': 0.403992960, 'V20': 0.251412098, 'V21': -0.018306778,
                'V22': 0.277837576, 'V23': -0.110473910, 'V24': 0.066928075,
                'V25': 0.128539358, 'V26': -0.189114844, 'V27': 0.133558377,
                'V28': -0.021053053, 'Time': 0, 'Amount': 149.62
            }
            result = st.session_state.detector.predict_with_details(legitimate_transaction)
            st.session_state.last_result = result
        
        if st.button("🔴 Test Fraudulent Transaction", use_container_width=True):
            fraud_transaction = {
                'V1': 2.173, 'V2': 0.289, 'V3': -2.328, 'V4': 1.919,
                'V5': -0.863, 'V6': -2.301, 'V7': -3.241, 'V8': 0.247,
                'V9': -1.546, 'V10': -2.841, 'V11': 3.147, 'V12': -3.807,
                'V13': -0.429, 'V14': -6.858, 'V15': 0.213, 'V16': -1.892,
                'V17': -8.443, 'V18': 0.661, 'V19': 1.043, 'V20': 0.127,
                'V21': 0.217, 'V22': 0.437, 'V23': -0.107, 'V24': 0.294,
                'V25': 0.413, 'V26': 0.271, 'V27': 0.108, 'V28': 0.037,
                'Time': 44807, 'Amount': 1.00
            }
            result = st.session_state.detector.predict_with_details(fraud_transaction)
            st.session_state.last_result = result
    
    # Display results if available
    if 'last_result' in st.session_state:
        show_prediction_result(st.session_state.last_result)

def show_prediction_result(result):
    """Display prediction results"""
    st.markdown("---")
    st.header("📊 Analysis Results")
    
    # Main result card
    if result['is_fraud']:
        st.markdown(f"""
        <div class="fraud-alert">
            <h2>🚨 FRAUD DETECTED</h2>
            <h3>Fraud Probability: {result['fraud_probability']:.1%}</h3>
            <h3>Risk Level: {result['risk_level']}</h3>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="safe-alert">
            <h2>✅ LEGITIMATE TRANSACTION</h2>
            <h3>Fraud Probability: {result['fraud_probability']:.1%}</h3>
            <h3>Risk Level: {result['risk_level']}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Fraud Probability", f"{result['fraud_probability']:.2%}")
    with col2:
        st.metric("Risk Level", result['risk_level'])
    with col3:
        st.metric("Confidence", f"{result['confidence']:.2%}")
    with col4:
        st.metric("Threshold", f"{result['threshold_used']:.1%}")
    
    # Gauge chart
    st.plotly_chart(create_gauge_chart(result['fraud_probability']), use_container_width=True)
    
    # Recommendation
    st.subheader("💡 Recommended Action")
    st.info(result['recommendation'])
    
    # Feature importance
    if result['top_features']:
        st.subheader("🔍 Top Contributing Features")
        fig = create_feature_importance_chart([
            {'feature': f[0], 'importance': f[1]}
            for f in result['top_features']
        ])
        if fig:
            st.plotly_chart(fig, use_container_width=True)

def show_batch_upload_mode():
    """Batch upload and prediction interface"""
    st.header("📤 Batch Transaction Analysis")
    
    st.info("Upload a CSV file with multiple transactions for batch prediction.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} transactions")
            
            st.subheader("Data Preview")
            st.dataframe(df.head(10))
            
            if st.button("🔮 Analyze All Transactions", type="primary"):
                with st.spinner(f"Analyzing {len(df)} transactions..."):
                    results = st.session_state.detector.predict_batch(df, return_details=True)
                
                # Create results DataFrame
                results_df = pd.DataFrame(results)
                
                # Summary metrics
                st.subheader("📊 Summary Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    fraud_count = results_df['is_fraud'].sum()
                    st.metric("Fraudulent", fraud_count, f"{fraud_count/len(results_df):.1%}")
                
                with col2:
                    high_risk = (results_df['risk_level'] == 'HIGH').sum()
                    st.metric("High Risk", high_risk)
                
                with col3:
                    medium_risk = (results_df['risk_level'] == 'MEDIUM').sum()
                    st.metric("Medium Risk", medium_risk)
                
                with col4:
                    avg_prob = results_df['fraud_probability'].mean()
                    st.metric("Avg Probability", f"{avg_prob:.1%}")
                
                # Results table
                st.subheader("📋 Detailed Results")
                display_df = results_df[['transaction_id', 'is_fraud', 'fraud_probability', 'risk_level', 'confidence']]
                st.dataframe(display_df, use_container_width=True)
                
                # Download results
                csv = display_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name=f"fraud_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

def show_example_transactions_mode():
    """Show example transactions for testing"""
    st.header("📝 Example Transactions")
    
    st.info("Select from pre-loaded example transactions to see how the model performs.")
    
    examples = {
        "Legitimate - Small Purchase": {
            'V1': -1.36, 'V2': -0.07, 'V3': 2.54, 'V4': 1.38, 'V5': -0.34,
            'V6': 0.46, 'V7': 0.24, 'V8': 0.10, 'V9': 0.36, 'V10': 0.09,
            'V11': -0.55, 'V12': -0.62, 'V13': -0.99, 'V14': -0.31, 'V15': 1.47,
            'V16': -0.47, 'V17': 0.21, 'V18': 0.03, 'V19': 0.40, 'V20': 0.25,
            'V21': -0.02, 'V22': 0.28, 'V23': -0.11, 'V24': 0.07, 'V25': 0.13,
            'V26': -0.19, 'V27': 0.13, 'V28': -0.02, 'Time': 0, 'Amount': 149.62
        },
        "Fraudulent - Suspicious Pattern": {
            'V1': 2.17, 'V2': 0.29, 'V3': -2.33, 'V4': 1.92, 'V5': -0.86,
            'V6': -2.30, 'V7': -3.24, 'V8': 0.25, 'V9': -1.55, 'V10': -2.84,
            'V11': 3.15, 'V12': -3.81, 'V13': -0.43, 'V14': -6.86, 'V15': 0.21,
            'V16': -1.89, 'V17': -8.44, 'V18': 0.66, 'V19': 1.04, 'V20': 0.13,
            'V21': 0.22, 'V22': 0.44, 'V23': -0.11, 'V24': 0.29, 'V25': 0.41,
            'V26': 0.27, 'V27': 0.11, 'V28': 0.04, 'Time': 44807, 'Amount': 1.00
        },
        "Legitimate - Large Purchase": {
            'V1': -0.94, 'V2': 1.03, 'V3': 1.45, 'V4': 0.54, 'V5': -0.23,
            'V6': 0.78, 'V7': 0.11, 'V8': 0.05, 'V9': -0.87, 'V10': -0.45,
            'V11': 1.23, 'V12': -1.11, 'V13': 0.34, 'V14': -0.67, 'V15': 0.56,
            'V16': -0.89, 'V17': 0.23, 'V18': -0.12, 'V19': 0.45, 'V20': 0.19,
            'V21': 0.08, 'V22': 0.67, 'V23': -0.05, 'V24': -0.22, 'V25': 0.44,
            'V26': 0.11, 'V27': 0.03, 'V28': 0.01, 'Time': 12000, 'Amount': 2500.00
        }
    }
    
    selected_example = st.selectbox("Choose an example:", list(examples.keys()))
    
    if st.button("🔮 Analyze This Example", type="primary"):
        transaction = examples[selected_example]
        with st.spinner("Analyzing..."):
            result = st.session_state.detector.predict_with_details(transaction)
        
        show_prediction_result(result)
        
        # Show transaction details
        with st.expander("📋 View Transaction Details"):
            st.json(transaction)

if __name__ == "__main__":
    main()
