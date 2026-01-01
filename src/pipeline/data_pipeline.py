"""
Data Pipeline: Handles feature extraction from interactions
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from ..features.interaction_features import extract_interaction_features


class DataPipeline:
    """
    Pipeline that converts raw interactions into engineered features.
    
    Flow: interactions logs -> feature extraction -> feature DataFrame
    """
    
    def __init__(self, users: Dict[int, Dict], items: Dict[int, Dict]):
        """
        Initialize the data pipeline.
        
        Args:
            users: Dictionary mapping user_id to user data (mastery, learning_rate, etc.)
            items: Dictionary mapping item_id to item data (skills, difficulty, etc.)
        """
        self.users = users
        self.items = items
        self.features_df = None
        self.logs = None
        
    def process(self, logs: pd.DataFrame) -> pd.DataFrame:
        """
        Process interaction logs and extract features.
        
        Args:
            logs: DataFrame with columns [user_id, item_id, success, quiz_score, 
                  time_spent, difficulty_gap]
        
        Returns:
            DataFrame with engineered features for each interaction
        """
        self.logs = logs.copy()
        
        # Extract features from interactions
        self.features_df = extract_interaction_features(
            logs=self.logs,
            users=self.users,
            items=self.items
        )
        
        return self.features_df
    
    def get_features(self) -> pd.DataFrame:
        """Get the processed features DataFrame."""
        if self.features_df is None:
            raise RuntimeError("Pipeline not yet run. Call process() first.")
        return self.features_df
    
    def get_feature_columns(self) -> list:
        """Get list of feature column names (excluding user_id, item_id, success, etc.)."""
        if self.features_df is None:
            raise RuntimeError("Pipeline not yet run. Call process() first.")
        
        # Exclude outcome and metadata columns
        exclude_cols = {'user_id', 'item_id', 'success', 'quiz_score', 
                       'time_spent', 'difficulty_gap'}
        return [col for col in self.features_df.columns if col not in exclude_cols]
