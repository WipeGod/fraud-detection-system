# 🛡️ Credit Card Fraud Detection System

An advanced machine learning system for detecting fraudulent credit card transactions in real-time using multiple algorithms and ensemble methods.

## 🚀 Live Demo
[**Try it now:**](https://fraud-detection-system-humamsxtenmwomqjs2m8qp.streamlit.app/)

## ✨ Features

### 🤖 Multiple ML Algorithms
- **XGBoost** - Gradient boosting for high accuracy
- **Isolation Forest** - Unsupervised anomaly detection
- **Local Outlier Factor** - Density-based outlier detection
- **Random Forest** - Ensemble decision trees
- **Logistic Regression** - Linear classification baseline
- **Ensemble Method** - Combines all models for best performance

### 📊 Advanced Data Processing
- **SMOTE** - Synthetic minority oversampling
- **Robust Scaling** - Handles outliers effectively
- **Imbalanced Dataset Handling** - Specialized techniques for fraud detection
- **Feature Engineering** - Automated feature importance analysis

### 🚨 Real-time Alert System
- **Risk Level Assessment** - LOW/MEDIUM/HIGH/CRITICAL
- **Automated Alerts** - Instant notifications for suspicious transactions
- **Business Rules** - Configurable thresholds and conditions
- **Action Recommendations** - Specific steps for each alert type

### 📈 Interactive Dashboard
- **Real-time Monitoring** - Live transaction analysis
- **Performance Metrics** - Model comparison and evaluation
- **Data Visualization** - Interactive charts and graphs
- **Alert Management** - Track and manage fraud alerts

## 🛠️ Technologies Used

- **Frontend:** Streamlit
- **Machine Learning:** Scikit-learn, XGBoost, Imbalanced-learn
- **Data Processing:** Pandas, NumPy
- **Visualizations:** Plotly, Seaborn, Matplotlib
- **Deployment:** Streamlit Cloud

## 📊 Key Algorithms

### **Anomaly Detection**
- **Isolation Forest:** Isolates anomalies using random forests
- **Local Outlier Factor:** Detects outliers based on local density

### **Supervised Learning**
- **XGBoost:** Gradient boosting with advanced regularization
- **Random Forest:** Ensemble of decision trees with voting
- **Logistic Regression:** Linear model with balanced class weights

### **Data Balancing**
- **SMOTE:** Generates synthetic minority samples
- **Random Undersampling:** Reduces majority class samples
- **Combined Approach:** Hybrid over/under-sampling strategy

## 🏃‍♂️ Quick Start

### Local Installation
```bash
git clone https://github.com/your-username/fraud-detection-system.git
cd fraud-detection-system
pip install -r requirements.txt
streamlit run app.py
