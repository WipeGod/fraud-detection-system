import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Import custom modules
from data_processor import FraudDataProcessor
from fraud_detector import FraudDetectionSystem
from alert_system import FraudAlertSystem

# Page config
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide"
)

# Initialize session state
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = FraudDataProcessor()
if 'fraud_detector' not in st.session_state:
    st.session_state.fraud_detector = FraudDetectionSystem()
if 'alert_system' not in st.session_state:
    st.session_state.alert_system = FraudAlertSystem()
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

def main():
    st.title("🛡️ Credit Card Fraud Detection System")
    st.markdown("**Advanced ML-powered fraud detection with real-time monitoring**")
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Navigation")
        page = st.selectbox(
            "Choose a page:",
            ["🏠 Dashboard", "📊 Data Analysis", "🤖 Model Training", "🔍 Real-time Detection"]
        )
    
    # Route to pages
    if page == "🏠 Dashboard":
        dashboard_page()
    elif page == "📊 Data Analysis":
        data_analysis_page()
    elif page == "🤖 Model Training":
        model_training_page()
    elif page == "🔍 Real-time Detection":
        real_time_detection_page()

def dashboard_page():
    """Main dashboard"""
    st.header("📊 System Overview")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first in the Data Analysis section")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    alert_summary = st.session_state.alert_system.get_alert_summary()
    
    with col1:
        st.metric("Total Alerts", alert_summary['total_alerts'])
    with col2:
        st.metric("Active Alerts", alert_summary['active_alerts'])
    with col3:
        st.metric("Critical Alerts", alert_summary['critical_alerts'])
    with col4:
        st.metric("Avg Fraud Prob", f"{alert_summary['avg_fraud_probability']:.1%}")

def data_analysis_page():
    """Data analysis and loading"""
    st.header("📊 Data Analysis")
    
    # Data loading
    st.subheader("📁 Load Data")
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if st.button("🚀 Load Data") or uploaded_file:
        with st.spinner("Loading data..."):
            df = st.session_state.data_processor.load_data(uploaded_file)
            st.session_state.df = df
            st.session_state.data_loaded = True
    
    if st.session_state.data_loaded:
        df = st.session_state.df
        
        # Data overview
        st.subheader("📋 Data Overview")
        analysis = st.session_state.data_processor.analyze_data(df)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", f"{analysis['total_transactions']:,}")
        with col2:
            st.metric("Fraud Cases", f"{analysis['fraud_count']:,}")
        with col3:
            st.metric("Normal Cases", f"{analysis['normal_count']:,}")
        with col4:
            st.metric("Fraud Rate", f"{analysis['fraud_percentage']:.2f}%")
        
        # Data visualization
        st.subheader("📈 Data Visualization")
        
        # Class distribution
        fig = px.pie(
            values=[analysis['normal_count'], analysis['fraud_count']], 
            names=['Normal', 'Fraud'],
            title="Transaction Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Amount distribution
        fig2 = px.histogram(
            df, 
            x='Amount', 
            color='Class',
            title="Transaction Amount Distribution",
            nbins=50
        )
        st.plotly_chart(fig2, use_container_width=True)

def model_training_page():
    """Model training interface"""
    st.header("🤖 Model Training")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first")
        return
    
    df = st.session_state.df
    
    # Training parameters
    st.subheader("⚙️ Training Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
        balance_method = st.selectbox(
            "Balancing Method", 
            ['smote', 'undersample', 'combined', 'none']
        )
    
    with col2:
        st.info(f"""
        **Dataset Info:**
        - Total samples: {len(df):,}
        - Features: {len(df.columns)-1}
        - Fraud rate: {(df['Class'].sum()/len(df)*100):.2f}%
        """)
    
    if st.button("🚀 Train Models", type="primary"):
        with st.spinner("Training models..."):
            # Preprocess data
            X_train, X_test, y_train, y_test = st.session_state.data_processor.preprocess_data(
                df, test_size, balance_method
            )
            
            # Train models
            st.session_state.fraud_detector.train_models(
                X_train, y_train, X_test, y_test, 
                st.session_state.data_processor.feature_names
            )
            
            st.session_state.models_trained = True
            st.success("✅ Models trained successfully!")
    
    # Model performance
    if st.session_state.models_trained:
        st.subheader("📊 Model Performance")
        
        performance_df = st.session_state.fraud_detector.get_model_comparison()
        if not performance_df.empty:
            st.dataframe(performance_df, use_container_width=True)
            
            # Performance visualization
            fig = px.bar(
                performance_df, 
                x='Model', 
                y=['precision', 'recall', 'f1_score'],
                title="Model Performance Comparison",
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)

def real_time_detection_page():
    """Real-time fraud detection"""
    st.header("🔍 Real-time Fraud Detection")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first")
        return
    
    st.subheader("💳 Enter Transaction Details")
    
    # Transaction input form
    with st.form("transaction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            amount = st.number_input("Amount ($)", min_value=0.0, value=100.0)
            time_val = st.number_input("Time (seconds)", min_value=0, value=3600)
            v1 = st.number_input("V1", value=0.0)
            v2 = st.number_input("V2", value=0.0)
            v3 = st.number_input("V3", value=0.0)
            v4 = st.number_input("V4", value=0.0)
            v5 = st.number_input("V5", value=0.0)
        
        with col2:
            v6 = st.number_input("V6", value=0.0)
            v7 = st.number_input("V7", value=0.0)
            v8 = st.number_input("V8", value=0.0)
            v9 = st.number_input("V9", value=0.0)
            v10 = st.number_input("V10", value=0.0)
            
            model_choice = st.selectbox(
                "Select Model", 
                ['xgboost', 'random_forest', 'isolation_forest', 'ensemble']
            )
        
        submitted = st.form_submit_button("🔍 Analyze Transaction", type="primary")
    
    if submitted:
        # Prepare transaction data
        transaction_data = {
            'Time': time_val,
            'Amount': amount,
            'V1': v1, 'V2': v2, 'V3': v3, 'V4': v4, 'V5': v5,
            'V6': v6, 'V7': v7, 'V8': v8, 'V9': v9, 'V10': v10
        }
        
        # Preprocess transaction
        processed_data = st.session_state.data_processor.preprocess_single_transaction(transaction_data)
        
        # Make prediction
        if model_choice == 'ensemble':
            result = st.session_state.fraud_detector.ensemble_predict(processed_data)
        else:
            result = st.session_state.fraud_detector.predict_fraud(processed_data, model_choice)
        
        # Generate alert
        alert = st.session_state.alert_system.generate_alert(transaction_data, result)
        
        # Display results
        st.subheader("🎯 Analysis Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Fraud Probability", f"{result['fraud_probability']:.1%}")
        with col2:
            st.metric("Risk Level", result['risk_level'])
        with col3:
            st.metric("Model Used", result['model_used'].title())
        
        # Risk assessment
        if result['fraud_probability'] > 0.8:
            st.error(f"🚨 HIGH RISK TRANSACTION - {result['fraud_probability']:.1%} fraud probability")
        elif result['fraud_probability'] > 0.5:
            st.warning(f"⚠️ MEDIUM RISK TRANSACTION - {result['fraud_probability']:.1%} fraud probability")
        else:
            st.success(f"✅ LOW RISK TRANSACTION - {result['fraud_probability']:.1%} fraud probability")
        
        # Alert details
        with st.expander("📋 Alert Details"):
            st.write("**Reasons:**")
            for reason in alert['reasons']:
                st.write(f"• {reason}")
            
            st.write("**Recommended Actions:**")
            for action in alert['recommended_actions']:
                st.write(f"• {action}")

if __name__ == "__main__":
    main()
