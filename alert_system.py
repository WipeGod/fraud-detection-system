import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import time

class FraudAlertSystem:
    def __init__(self):
        self.alert_history = []
        self.alert_rules = {
            'high_amount_threshold': 10000,
            'multiple_transactions_window': 300,
            'multiple_transactions_count': 5,
            'velocity_threshold': 3,
            'suspicious_time_start': 23,
            'suspicious_time_end': 6,
        }
        self.active_alerts = []
    
    def generate_alert(self, transaction_data: Dict[str, Any], prediction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fraud alert"""
        alert = {
            'timestamp': datetime.now(),
            'transaction_id': f"TXN_{int(time.time())}",
            'alert_type': self.determine_alert_type(transaction_data, prediction_result),
            'severity': self.determine_severity(prediction_result),
            'fraud_probability': prediction_result.get('fraud_probability', 0),
            'risk_level': prediction_result.get('risk_level', 'UNKNOWN'),
            'transaction_amount': transaction_data.get('Amount', 0),
            'model_used': prediction_result.get('model_used', 'unknown'),
            'reasons': self.get_alert_reasons(transaction_data, prediction_result),
            'recommended_actions': self.get_recommended_actions(prediction_result),
            'status': 'ACTIVE'
        }
        
        self.alert_history.append(alert)
        
        if alert['severity'] in ['HIGH', 'CRITICAL']:
            self.active_alerts.append(alert)
        
        return alert
    
    def determine_alert_type(self, transaction_data: Dict[str, Any], prediction_result: Dict[str, Any]) -> str:
        """Determine alert type"""
        amount = transaction_data.get('Amount', 0)
        fraud_prob = prediction_result.get('fraud_probability', 0)
        
        if fraud_prob > 0.8:
            return "HIGH_FRAUD_PROBABILITY"
        elif amount > self.alert_rules['high_amount_threshold']:
            return "HIGH_AMOUNT_TRANSACTION"
        elif fraud_prob > 0.5:
            return "SUSPICIOUS_TRANSACTION"
        else:
            return "ANOMALY_DETECTED"
    
    def determine_severity(self, prediction_result: Dict[str, Any]) -> str:
        """Determine alert severity"""
        fraud_prob = prediction_result.get('fraud_probability', 0)
        
        if fraud_prob >= 0.9:
            return "CRITICAL"
        elif fraud_prob >= 0.7:
            return "HIGH"
        elif fraud_prob >= 0.5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def get_alert_reasons(self, transaction_data: Dict[str, Any], prediction_result: Dict[str, Any]) -> List[str]:
        """Get reasons for alert"""
        reasons = []
        amount = transaction_data.get('Amount', 0)
        fraud_prob = prediction_result.get('fraud_probability', 0)
        
        if fraud_prob > 0.8:
            reasons.append(f"High fraud probability: {fraud_prob:.2%}")
        
        if amount > self.alert_rules['high_amount_threshold']:
            reasons.append(f"High transaction amount: ${amount:,.2f}")
        
        if fraud_prob > 0.5:
            reasons.append("Suspicious transaction pattern detected")
        
        if 'Time' in transaction_data:
            hour = (transaction_data['Time'] // 3600) % 24
            if (hour >= self.alert_rules['suspicious_time_start'] or 
                hour <= self.alert_rules['suspicious_time_end']):
                reasons.append(f"Transaction at suspicious time: {int(hour):02d}:00")
        
        if not reasons:
            reasons.append("Anomalous transaction pattern")
        
        return reasons
    
    def get_recommended_actions(self, prediction_result: Dict[str, Any]) -> List[str]:
        """Get recommended actions"""
        actions = []
        severity = self.determine_severity(prediction_result)
        
        if severity == "CRITICAL":
            actions.extend([
                "🚨 IMMEDIATE ACTION REQUIRED",
                "Block card immediately",
                "Contact customer for verification",
                "Escalate to fraud investigation team",
                "Review recent transaction history"
            ])
        elif severity == "HIGH":
            actions.extend([
                "⚠️ Urgent review required",
                "Temporarily hold transaction",
                "Send SMS/email verification to customer",
                "Flag account for monitoring",
                "Review transaction details"
            ])
        elif severity == "MEDIUM":
            actions.extend([
                "📋 Review recommended",
                "Monitor account activity",
                "Consider additional verification",
                "Log for pattern analysis"
            ])
        else:
            actions.extend([
                "📝 Log for monitoring",
                "Continue normal processing",
                "Update risk profile"
            ])
        
        return actions
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary"""
        if not self.alert_history:
            return {
                'total_alerts': 0,
                'active_alerts': 0,
                'critical_alerts': 0,
                'high_alerts': 0,
                'medium_alerts': 0,
                'low_alerts': 0,
                'avg_fraud_probability': 0
            }
        
        df = pd.DataFrame(self.alert_history)
        
        return {
            'total_alerts': len(self.alert_history),
            'active_alerts': len(self.active_alerts),
            'critical_alerts': len(df[df['severity'] == 'CRITICAL']),
            'high_alerts': len(df[df['severity'] == 'HIGH']),
            'medium_alerts': len(df[df['severity'] == 'MEDIUM']),
            'low_alerts': len(df[df['severity'] == 'LOW']),
            'avg_fraud_probability': df['fraud_probability'].mean()
        }
    
    def get_recent_alerts(self, hours: int = 24) -> pd.DataFrame:
        """Get recent alerts"""
        if not self.alert_history:
            return pd.DataFrame()
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [alert for alert in self.alert_history 
                        if alert['timestamp'] >= cutoff_time]
        
        if not recent_alerts:
            return pd.DataFrame()
        
        df = pd.DataFrame(recent_alerts)
        return df.sort_values('timestamp', ascending=False)
    
    def dismiss_alert(self, transaction_id: str):
        """Dismiss an alert"""
        self.active_alerts = [alert for alert in self.active_alerts 
                             if alert['transaction_id'] != transaction_id]
        
        for alert in self.alert_history:
            if alert['transaction_id'] == transaction_id:
                alert['status'] = 'DISMISSED'
                break
