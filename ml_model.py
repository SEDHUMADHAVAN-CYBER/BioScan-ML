"""
Machine Learning Model Module
Implements Random Forest and SVM models for biomarker prediction
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)


class BiomarkerPredictor:
    """
    Class to train and evaluate ML models for biomarker prediction
    """
    
    def __init__(self):
        """Initialize the predictor with model configurations"""
        # Random Forest model
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Support Vector Machine model
        self.svm_model = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            random_state=42
        )
        
        # Store results
        self.results = {}
        self.feature_importance = None
        
    def train_models(self, X_train, y_train):
        """
        Train both Random Forest and SVM models
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        print("=" * 60)
        print("TRAINING MODELS")
        print("=" * 60)
        
        # Train Random Forest
        print("\n[1/2] Training Random Forest Classifier...")
        self.rf_model.fit(X_train, y_train)
        print("✓ Random Forest training completed")
        
        # Train SVM
        print("\n[2/2] Training Support Vector Machine...")
        self.svm_model.fit(X_train, y_train)
        print("✓ SVM training completed")
        
        # Extract feature importance from Random Forest
        self.feature_importance = self.rf_model.feature_importances_
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 60)
    
    def evaluate_model(self, model, model_name, X_test, y_test):
        """
        Evaluate a single model and return metrics
        
        Args:
            model: Trained model
            model_name: Name of the model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Dictionary containing all evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Store results
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'predictions': y_pred
        }
        
        return results
    
    def evaluate_all_models(self, X_test, y_test):
        """
        Evaluate both Random Forest and SVM models
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Results for both models
        """
        print("\n" + "=" * 60)
        print("EVALUATING MODELS")
        print("=" * 60)
        
        # Evaluate Random Forest
        print("\n[1/2] Evaluating Random Forest...")
        rf_results = self.evaluate_model(self.rf_model, "Random Forest", X_test, y_test)
        self.results['Random Forest'] = rf_results
        print(f"✓ Random Forest Accuracy: {rf_results['accuracy']:.4f}")
        
        # Evaluate SVM
        print("\n[2/2] Evaluating Support Vector Machine...")
        svm_results = self.evaluate_model(self.svm_model, "SVM", X_test, y_test)
        self.results['SVM'] = svm_results
        print(f"✓ SVM Accuracy: {svm_results['accuracy']:.4f}")
        
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETED")
        print("=" * 60)
        
        return self.results
    
    def get_feature_importance(self, feature_names, top_n=10):
        """
        Get top N important features from Random Forest
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features to return (default: 10)
            
        Returns:
            tuple: (top_features, top_importances)
        """
        if self.feature_importance is None:
            raise ValueError("Model not trained yet. Train the model first.")
        
        # Get indices of top N features
        top_indices = np.argsort(self.feature_importance)[-top_n:][::-1]
        
        # Get feature names and importance values
        top_features = [feature_names[i] for i in top_indices]
        top_importances = self.feature_importance[top_indices]
        
        return top_features, top_importances
    
    def get_comparison_dataframe(self):
        """
        Create a comparison dataframe for all models
        
        Returns:
            DataFrame: Comparison of model metrics
        """
        import pandas as pd
        
        comparison_data = []
        
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score']
            })
        
        return pd.DataFrame(comparison_data)
    
    def print_detailed_results(self):
        """
        Print detailed results for all models
        """
        print("\n" + "=" * 60)
        print("DETAILED RESULTS")
        print("=" * 60)
        
        for model_name, results in self.results.items():
            print(f"\n{model_name}:")
            print(f"  Accuracy:  {results['accuracy']:.4f}")
            print(f"  Precision: {results['precision']:.4f}")
            print(f"  Recall:    {results['recall']:.4f}")
            print(f"  F1-Score:  {results['f1_score']:.4f}")
