"""
Recommender System: Generates top-K recommendations for users
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class RecommenderSystem:
    """
    System that generates personalized recommendations for each user.
    
    Flow: relevance_scores -> rank items -> return top-K recommendations
    """
    
    def __init__(self, top_k: int = 5, min_relevance: float = 0.0):
        """
        Initialize the recommender system.
        
        Args:
            top_k: Number of recommendations to return per user
            min_relevance: Minimum relevance threshold for recommendations
        """
        self.top_k = top_k
        self.min_relevance = min_relevance
        self.items = None
        self.users = None
        
    def set_context(self, users: Dict[int, Dict], items: Dict[int, Dict]):
        """
        Set user and item context for recommendations.
        
        Args:
            users: Dictionary mapping user_id to user data
            items: Dictionary mapping item_id to item data
        """
        self.users = users
        self.items = items
    
    def recommend(self, user_id: int, relevance_scores: Dict[int, float], 
                  exclude_items: set = None) -> List[Tuple[int, float]]:
        """
        Generate top-K recommendations for a specific user.
        
        Args:
            user_id: ID of the user
            relevance_scores: Dict mapping item_id to relevance score
            exclude_items: Set of item IDs to exclude from recommendations
        
        Returns:
            List of (item_id, relevance_score) tuples, sorted by relevance descending
        """
        exclude_items = exclude_items or set()
        
        # Filter items: must be above threshold and not excluded
        candidate_items = [
            (item_id, score) for item_id, score in relevance_scores.items()
            if score >= self.min_relevance and item_id not in exclude_items
        ]
        
        # Sort by relevance score descending
        candidate_items.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-K
        return candidate_items[:self.top_k]
    
    def recommend_batch(self, relevance_df: pd.DataFrame, 
                       exclude_seen: bool = True) -> pd.DataFrame:
        """
        Generate recommendations for all users at once.
        
        Args:
            relevance_df: DataFrame with columns [user_id, item_id, relevance_score]
            exclude_seen: If True, don't recommend items user has already interacted with
        
        Returns:
            DataFrame with columns [user_id, item_id, relevance_score, rank]
        """
        if relevance_df is None or len(relevance_df) == 0:
            return pd.DataFrame(columns=['user_id', 'item_id', 'relevance_score', 'rank'])
        
        # Check for required columns
        if 'user_id' not in relevance_df.columns or 'item_id' not in relevance_df.columns:
            return pd.DataFrame(columns=['user_id', 'item_id', 'relevance_score', 'rank'])
        
        recommendations = []
        
        for user_id in relevance_df['user_id'].unique():
            user_items = relevance_df[relevance_df['user_id'] == user_id]
            
            # Get set of items user has seen (if exclude_seen is True)
            exclude_items = set(user_items['item_id'].values) if exclude_seen else set()
            
            # Create relevance dict
            relevance_dict = dict(zip(user_items['item_id'], user_items['relevance_score']))
            
            # Get recommendations
            recs = self.recommend(user_id, relevance_dict, exclude_items=exclude_items)
            
            # Add to results with rank
            for rank, (item_id, score) in enumerate(recs, 1):
                recommendations.append({
                    'user_id': user_id,
                    'item_id': item_id,
                    'relevance_score': score,
                    'rank': rank
                })
        
        return pd.DataFrame(recommendations)
    
    def get_recommendations_for_user(self, user_id: int, 
                                    recommendations_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract recommendations for a specific user.
        
        Args:
            user_id: ID of the user
            recommendations_df: DataFrame from recommend_batch()
        
        Returns:
            Subset of recommendations for this user
        """
        if recommendations_df is None or len(recommendations_df) == 0:
            return pd.DataFrame()
        
        user_recs = recommendations_df[recommendations_df['user_id'] == user_id]
        return user_recs.sort_values('rank') if len(user_recs) > 0 else pd.DataFrame()
