import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import streamlit as st
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class FraudDataProcessor:
    def __init__(self):
        self.scaler = RobustScaler()
        self.smote = SMOTE(random_state=42, k_neighbors=3)
        self.undersampler = RandomUnderSampler(random_state=42)
        self.feature_names = None
        self.is_fitted = False
    
    def create_synthetic_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """Create synthetic fraud detection dataset"""
        np.random.seed(42)
        
        # Generate normal transactions (99.8% of data)
        normal_transactions = int(n_samples * 0.998)
        fraud_transactions = n_samples - normal_transactions
        
        # Normal transaction features
        normal_data = {
            'Time': np.random.uniform(0, 172800, normal_transactions),  # 48 hours
            'Amount': np.random.lognormal(3, 1.5, normal_transactions),
            'V1': np.random.normal(0, 1, normal_transactions),
            'V2': np.random.normal(0, 1, normal_transactions),
            'V3': np.random.normal(0, 1, normal_transactions),
            'V4': np.random.normal(0, 1, normal_transactions),
            'V5': np.random.normal(0, 1, normal_transactions),
            'V6': np.random.normal(0, 1, normal_transactions),
            'V7': np.random.normal(0, 1, normal_transactions),
            'V8': np.random.normal(0, 1, normal_transactions),
            'V9': np.random.normal(0, 1, normal_transactions),
            'V10': np.random.normal(0, 1, normal_transactions),
            'Class': np.zeros(normal_transactions)
        }
        
        # Fraudulent transactions (different distributions)
        fraud_data = {
            'Time': np.random.uniform(0, 172800, fraud_transactions),
            'Amount': np.random.lognormal(4, 2, fraud_transactions),  # Higher amounts
            'V1': np.random.normal(2, 1.5, fraud_transactions),  # Shifted distributions
            'V2': np.random.normal(-1, 1.2, fraud_transactions),
            'V3': np.random.normal(1.5, 1.3, fraud_transactions),
            'V4': np.random.normal(-0.5, 1.1, fraud_transactions),
            'V5': np.random.normal(0.8, 1.4, fraud_transactions),
            'V6': np.random.normal(-1.2, 1.2, fraud_transactions),
            'V7': np.random.normal(0.6, 1.3, fraud_transactions),
            'V8': np.random.normal(-0.8, 1.1, fraud_transactions),
            'V9': np.random.normal(1.1, 1.2, fraud_transactions),
            'V10': np.random.normal(-0.9, 1.3, fraud_transactions),
            'Class': np.ones(fraud_transactions)
        }
        
        # Combine data
        all_data = {}
        for key in normal_data.keys():
            all_data[key] = np.concatenate([normal_data[key], fraud_data[key]])
        
        df = pd.DataFrame(all_data)
        return df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    def load_data(self, uploaded_file=None) -> pd.DataFrame:
        """Load fraud detection dataset"""
        try:
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Data loaded: {df.shape[0]} transactions")
            else:
                df = self.create_synthetic_data()
                st.info(f"📊 Generated synthetic data: {df.shape[0]} transactions")
            return df
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return self.create_synthetic_data()
    
    def analyze_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform exploratory data analysis"""
        fraud_count = df['Class'].sum()
        normal_count = len(df) - fraud_count
        
        analysis = {
            'total_transactions': len(df),
            'fraud_count': int(fraud_count),
            'normal_count': int(normal_count),
            'fraud_percentage': (fraud_count / len(df)) * 100,
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'feature_stats': df.describe(),
            'class_distribution': df['Class'].value_counts(),
            'amount_stats': {
                'normal_mean': df[df['Class'] == 0]['Amount'].mean(),
                'fraud_mean': df[df['Class'] == 1]['Amount'].mean(),
                'normal_median': df[df['Class'] == 0]['Amount'].median(),
                'fraud_median': df[df['Class'] == 1]['Amount'].median()
            }
        }
        
        return analysis
    
    def preprocess_data(self, df: pd.DataFrame, test_size: float = 0.2, 
                       balance_method: str = 'smote') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Preprocess data for machine learning"""
        
        # Separate features and target
        X = df.drop(['Class'], axis=1)
        y = df['Class']
        
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Handle class imbalance
        if balance_method == 'smote':
            X_train_balanced, y_train_balanced = self.smote.fit_resample(X_train_scaled, y_train)
        elif balance_method == 'undersample':
            X_train_balanced, y_train_balanced = self.undersampler.fit_resample(X_train_scaled, y_train)
        elif balance_method == 'combined':
            # First oversample, then undersample
            X_temp, y_temp = self.smote.fit_resample(X_train_scaled, y_train)
            X_train_balanced, y_train_balanced = self.undersampler.fit_resample(X_temp, y_temp)
        else:
            X_train_balanced, y_train_balanced = X_train_scaled, y_train
        
        self.is_fitted = True
        return X_train_balanced, X_test_scaled, y_train_balanced, y_test
    
    def preprocess_single_transaction(self, transaction_data: Dict[str, float]) -> np.ndarray:
        """Preprocess a single transaction for prediction"""
        if not self.is_fitted:
            raise ValueError("Processor must be fitted before preprocessing")
        
        # Convert to DataFrame
        df = pd.DataFrame([transaction_data])
        
        # Ensure all features are present
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0.0
        
        # Reorder columns
        df = df[self.feature_names]
        
        # Scale features
        scaled_data = self.scaler.transform(df)
        return scaled_data[0]
    
    def get_feature_importance_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate feature correlations with fraud"""
        correlations = df.corr()['Class'].abs().sort_values(ascending=False)
        
        importance_df = pd.DataFrame({
            'Feature': correlations.index[1:],  # Exclude 'Class'
            'Importance': correlations.values[1:]
        })
        
        return importance_df
