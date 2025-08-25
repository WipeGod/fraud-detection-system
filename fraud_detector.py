import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import streamlit as st
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionSystem:
    def __init__(self):
        self.models = {}
        self.model_performance = {}
        self.is_trained = False
        self.feature_names = None
        self.X_test_cache = None
        
    def initialize_models(self):
        """Initialize all fraud detection models"""
        self.models = {
            'isolation_forest': IsolationForest(
                contamination=0.002,
                random_state=42,
                n_estimators=100
            ),
            'local_outlier_factor': LocalOutlierFactor(
                contamination=0.002,
                novelty=True,
                n_neighbors=20
            ),
            'xgboost': xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=42,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced'
            ),
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            )
        }
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                    X_test: np.ndarray, y_test: np.ndarray, feature_names: List[str]):
        """Train all fraud detection models"""
        self.feature_names = feature_names
        self.X_test_cache = X_test
        self.initialize_models()
        
        st.info("🔄 Training fraud detection models...")
        progress_bar = st.progress(0)
        
        total_models = len(self.models)
        
        for i, (model_name, model) in enumerate(self.models.items()):
            try:
                st.write(f"Training {model_name.replace('_', ' ').title()}...")
                
                if model_name == 'isolation_forest':
                    # Unsupervised - train on normal transactions only
                    normal_data = X_train[y_train == 0]
                    model.fit(normal_data)
                    y_pred = model.predict(X_test)
                    y_pred_binary = np.where(y_pred == -1, 1, 0)
                    
                elif model_name == 'local_outlier_factor':
                    # Unsupervised - train on normal transactions only
                    normal_data = X_train[y_train == 0]
                    model.fit(normal_data)
                    y_pred = model.predict(X_test)
                    y_pred_binary = np.where(y_pred == -1, 1, 0)
                    
                else:
                    # Supervised models
                    model.fit(X_train, y_train)
                    y_pred_binary = model.predict(X_test)
                
                # Calculate performance metrics
                performance = self.calculate_performance_metrics(y_test, y_pred_binary, model_name)
                self.model_performance[model_name] = performance
                
                progress_bar.progress((i + 1) / total_models)
                
            except Exception as e:
                st.error(f"Error training {model_name}: {str(e)}")
                continue
        
        self.is_trained = True
        st.success("✅ All models trained successfully!")
        
    def calculate_performance_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, zero_division=0),
                'specificity': self.calculate_specificity(y_true, y_pred)
            }
            
            # Calculate AUC if possible
            try:
                if model_name in ['xgboost', 'random_forest', 'logistic_regression']:
                    model = self.models[model_name]
                    if hasattr(model, 'predict_proba'):
                        y_proba = model.predict_proba(self.X_test_cache)[:, 1]
                        metrics['auc'] = roc_auc_score(y_true, y_proba)
                    else:
                        metrics['auc'] = roc_auc_score(y_true, y_pred)
                else:
                    metrics['auc'] = roc_auc_score(y_true, y_pred)
            except:
                metrics['auc'] = 0.0
            
            return metrics
            
        except Exception as e:
            st.error(f"Error calculating metrics for {model_name}: {str(e)}")
            return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0, 'specificity': 0, 'auc': 0}
    
    def calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity (True Negative Rate)"""
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            return tn / (tn + fp) if (tn + fp) > 0 else 0.0
        except:
            return 0.0
    
    def predict_fraud(self, transaction_data: np.ndarray, model_name: str = 'xgboost') -> Dict[str, Any]:
        """Predict fraud for a single transaction"""
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        try:
            # Reshape if single transaction
            if transaction_data.ndim == 1:
                transaction_data = transaction_data.reshape(1, -1)
            
            if model_name in ['isolation_forest', 'local_outlier_factor']:
                prediction = model.predict(transaction_data)[0]
                fraud_probability = 0.8 if prediction == -1 else 0.2
                is_fraud = prediction == -1
            else:
                prediction = model.predict(transaction_data)[0]
                is_fraud = bool(prediction)
                
                # Get probability if available
                if hasattr(model, 'predict_proba'):
                    fraud_probability = model.predict_proba(transaction_data)[0][1]
                else:
                    fraud_probability = 0.8 if is_fraud else 0.2
            
            # Determine risk level
            if fraud_probability >= 0.8:
                risk_level = "HIGH"
                risk_color = "🔴"
            elif fraud_probability >= 0.5:
                risk_level = "MEDIUM"
                risk_color = "🟡"
            else:
                risk_level = "LOW"
                risk_color = "🟢"
            
            return {
                'is_fraud': is_fraud,
                'fraud_probability': fraud_probability,
                'risk_level': risk_level,
                'risk_color': risk_color,
                'model_used': model_name
            }
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            return {
                'is_fraud': False,
                'fraud_probability': 0.0,
                'risk_level': "UNKNOWN",
                'risk_color': "⚪",
                'model_used': model_name,
                'error': str(e)
            }
    
    def get_model_comparison(self) -> pd.DataFrame:
        """Get comparison of all model performances"""
        if not self.model_performance:
            return pd.DataFrame()
        
        comparison_data = []
        for model_name, metrics in self.model_performance.items():
            row = {'Model': model_name.replace('_', ' ').title()}
            row.update(metrics)
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Round numerical columns
        numerical_cols = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'auc']
        for col in numerical_cols:
            if col in df.columns:
                df[col] = df[col].round(4)
        
        return df
    
    def get_feature_importance(self, model_name: str = 'xgboost') -> pd.DataFrame:
        """Get feature importance from tree-based models"""
        if not self.is_trained or model_name not in self.models:
            return pd.DataFrame()
        
        model = self.models[model_name]
        
        try:
            if model_name == 'xgboost':
                importance = model.feature_importances_
            elif model_name == 'random_forest':
                importance = model.feature_importances_
            else:
                return pd.DataFrame()
            
            feature_importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            return feature_importance_df
            
        except Exception as e:
            st.error(f"Error getting feature importance: {str(e)}")
            return pd.DataFrame()
    
    def ensemble_predict(self, transaction_data: np.ndarray) -> Dict[str, Any]:
        """Make ensemble prediction using multiple models"""
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        predictions = {}
        fraud_probabilities = []
        
        # Get predictions from all models
        for model_name in self.models.keys():
            try:
                result = self.predict_fraud(transaction_data, model_name)
                predictions[model_name] = result
                fraud_probabilities.append(result['fraud_probability'])
            except:
                continue
        
        # Calculate ensemble results
        avg_fraud_probability = np.mean(fraud_probabilities)
        fraud_votes = sum(1 for pred in predictions.values() if pred['is_fraud'])
        total_votes = len(predictions)
        
        # Ensemble decision (majority vote + probability threshold)
        is_fraud_ensemble = (fraud_votes > total_votes / 2) or (avg_fraud_probability > 0.6)
        
        # Determine risk level
        if avg_fraud_probability >= 0.8:
            risk_level = "HIGH"
            risk_color = "🔴"
        elif avg_fraud_probability >= 0.5:
            risk_level = "MEDIUM"
            risk_color = "🟡"
        else:
            risk_level = "LOW"
            risk_color = "🟢"
        
        return {
            'is_fraud': is_fraud_ensemble,
            'fraud_probability': avg_fraud_probability,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'model_used': 'ensemble',
            'individual_predictions': predictions,
            'fraud_votes': fraud_votes,
            'total_votes': total_votes
        }
