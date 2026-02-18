"""
Data Processor Module
Handles data loading, preprocessing, and feature scaling for biomarker prediction
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataProcessor:
    """
    Class to handle all data preprocessing operations
    """
    
    def __init__(self):
        """Initialize the DataProcessor with default parameters"""
        self.scaler = StandardScaler()
        self.feature_names = None
        self.target_names = None
        
    def load_default_dataset(self):
        """
        Load the breast cancer dataset from sklearn
        
        Returns:
            tuple: (X, y, feature_names, target_names)
        """
        # Load breast cancer dataset
        data = load_breast_cancer()
        
        # Extract features and target
        X = data.data
        y = data.target
        
        # Store feature and target names
        self.feature_names = data.feature_names
        self.target_names = data.target_names
        
        print(f"✓ Loaded default dataset: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y, self.feature_names, self.target_names
    
    def load_custom_dataset(self, uploaded_file):
        """
        Load a custom dataset from uploaded CSV file
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            tuple: (X, y, feature_names, target_names)
        """
        try:
            # Read CSV file
            df = pd.read_csv(uploaded_file)
            
            # Assume last column is target
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            
            # Store feature names
            self.feature_names = df.columns[:-1].tolist()
            self.target_names = np.unique(y)
            
            print(f"✓ Loaded custom dataset: {X.shape[0]} samples, {X.shape[1]} features")
            
            return X, y, self.feature_names, self.target_names
            
        except Exception as e:
            raise ValueError(f"Error loading custom dataset: {str(e)}")
    
    def preprocess_data(self, X, y, test_size=0.3, random_state=42):
        """
        Preprocess the data: split and scale
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of test set (default: 0.3)
            random_state: Random seed for reproducibility (default: 42)
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y  # Maintain class distribution
        )
        
        print(f"✓ Train-test split: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
        
        # Feature scaling using StandardScaler
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"✓ Feature scaling completed using StandardScaler")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def get_feature_statistics(self, X):
        """
        Calculate basic statistics for features
        
        Args:
            X: Feature matrix
            
        Returns:
            DataFrame: Statistics for each feature
        """
        if self.feature_names is None:
            self.feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        
        stats_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Mean': np.mean(X, axis=0),
            'Std': np.std(X, axis=0),
            'Min': np.min(X, axis=0),
            'Max': np.max(X, axis=0)
        })
        
        return stats_df
