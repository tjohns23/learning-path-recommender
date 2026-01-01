"""
Ranking Pipeline: Predicts relevance scores for user-item pairs
"""

import pandas as pd
import numpy as np
from typing import Dict, Union
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge


class RankingPipeline:
    """
    Pipeline that scores items by relevance for each user.
    
    Flow: features + ranking_model -> relevance_scores
    """
    
    def __init__(self, model_type: str = 'random_forest', model_params: Dict = None):
        """
        Initialize the ranking pipeline.
        
        Args:
            model_type: 'random_forest' or 'ridge'
            model_params: Dictionary of model hyperparameters
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.is_trained = False
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              feature_columns: list = None, scale_features: bool = True):
        """
        Train the ranking model.
        
        Args:
            X_train: Training features
            y_train: Training relevance scores
            feature_columns: List of feature column names
            scale_features: Whether to scale features (needed for Ridge, not for RF)
        """
        self.feature_columns = feature_columns or list(X_train.columns)
        
        # Prepare data
        X_train_processed = X_train[self.feature_columns] if isinstance(X_train, pd.DataFrame) else X_train
        
        # Scale features if needed
        if scale_features and self.model_type == 'ridge':
            self.scaler = StandardScaler()
            X_train_processed = self.scaler.fit_transform(X_train_processed)
        elif scale_features and self.model_type == 'random_forest':
            # RF doesn't need scaling but we keep scaler for consistency
            self.scaler = None
        
        # Create and train model
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(**self.model_params)
        elif self.model_type == 'ridge':
            self.model = Ridge(**self.model_params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model.fit(X_train_processed, y_train)
        self.is_trained = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict relevance scores.
        
        Args:
            X: Features DataFrame or array
        
        Returns:
            Array of relevance scores
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        # Extract relevant columns
        X_processed = X[self.feature_columns] if isinstance(X, pd.DataFrame) else X
        
        # Scale if scaler was fit
        if self.scaler is not None:
            X_processed = self.scaler.transform(X_processed)
        
        return self.model.predict(X_processed)
    
    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get feature importance scores (only for Random Forest).
        
        Args:
            top_n: Number of top features to return
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.model_type != 'random_forest':
            raise ValueError("Feature importance only available for Random Forest model")
        
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        importances = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df.head(top_n)
