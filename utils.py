import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

class ChurnPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoders = {}
        self.feature_names = None
        
    def fit(self, df):
        """Fit preprocessor on training data"""
        # Fix TotalCharges
        df = df.copy()
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
        
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
            
        target = df['Churn']
        X = df.drop('Churn', axis=1)
        
        # Encode categoricals
        cat_cols = X.select_dtypes(include=['object']).columns.tolist()
        num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
        
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.encoders[col] = le
            
        # Store feature order
        self.feature_names = X.columns.tolist()
        
        # Scale
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, target.values
    
    def transform(self, df):
        """Transform new data"""
        df = df.copy()
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(0)  # or median from training
        
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
            
        X = df.drop('Churn', axis=1, errors='ignore')
        
        # Encode with fallback
        for col, le in self.encoders.items():
            if col in X.columns:
                try:
                    X[col] = le.transform(X[col].astype(str))
                except ValueError:
                    X[col] = 0  # fallback
                    
        # Add missing columns
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
                
        X = X[self.feature_names]
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
