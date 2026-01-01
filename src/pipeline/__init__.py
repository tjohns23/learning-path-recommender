"""
Pipeline: Complete end-to-end learning path recommendation pipeline
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional

from .data_pipeline import DataPipeline
from .ranking_pipeline import RankingPipeline
from .recommender import RecommenderSystem


class LearningPathPipeline:
    """
    Complete end-to-end pipeline orchestrating the recommendation workflow.
    
    Flow:
    1. users + items + interactions -> DataPipeline -> features
    2. features + ranking_model -> RankingPipeline -> relevance scores
    3. relevance scores -> RecommenderSystem -> top-K recommendations
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the learning path pipeline.
        
        Args:
            config: Configuration dictionary with pipeline parameters
        """
        self.config = config or {}
        
        # Pipeline stages
        self.data_pipeline = None
        self.ranking_pipeline = None
        self.recommender = None
        
        # State
        self.users = None
        self.items = None
        self.logs = None
        self.features = None
        self.relevance_scores = None
        self.recommendations = None
        
    def run(self, users: Dict[int, Dict], items: Dict[int, Dict], 
            logs: pd.DataFrame, ranking_model: str = 'random_forest',
            model_params: Dict = None, top_k: int = 5) -> Tuple[pd.DataFrame, Dict]:
        """
        Execute the complete pipeline.
        
        Args:
            users: Dictionary of user data
            items: Dictionary of item data
            logs: DataFrame of interaction logs
            ranking_model: Type of ranking model ('random_forest' or 'ridge')
            model_params: Model hyperparameters
            top_k: Number of recommendations per user
        
        Returns:
            Tuple of (recommendations_df, metadata_dict)
        """
        self.users = users
        self.items = items
        self.logs = logs.copy()
        
        print(f"[Pipeline] Starting with {len(users)} users, {len(items)} items, {len(logs)} interactions")
        
        # Stage 1: Data Pipeline - Extract features
        print("\n[Stage 1] Feature Extraction")
        self.data_pipeline = DataPipeline(users, items)
        self.features = self.data_pipeline.process(self.logs)
        print(f"    Extracted features: {self.features.shape[0]} samples × {self.features.shape[1]} columns")
        
        # Stage 2: Ranking Pipeline - Predict relevance scores
        print("\n[Stage 2] Ranking")
        self.ranking_pipeline = RankingPipeline(
            model_type=ranking_model,
            model_params=model_params
        )
        
        # For now, use all data (in production, would use train/test split)
        feature_cols = self.data_pipeline.get_feature_columns()
        X = self.features[feature_cols]
        
        # Use relevance if available, otherwise create a simple relevance score
        if 'relevance' in self.features.columns:
            y = self.features['relevance']
        else:
            # Create relevance from success and quiz score
            # relevance = 0.6 * success + 0.4 * (quiz_score / 100)
            success = self.features.get('success', 0.5)
            quiz_score = self.features.get('quiz_score', 50.0)
            y = 0.6 * success + 0.4 * (quiz_score / 100.0)
        
        self.ranking_pipeline.train(X, y, feature_columns=feature_cols)
        relevance_pred = self.ranking_pipeline.predict(X)
        print(f"    Trained {ranking_model} model")
        print(f"    Relevance score range: [{relevance_pred.min():.3f}, {relevance_pred.max():.3f}]")
        
        # Create relevance DataFrame
        relevance_df = pd.DataFrame({
            'user_id': self.features['user_id'],
            'item_id': self.features['item_id'],
            'relevance_score': relevance_pred
        })
        self.relevance_scores = relevance_df
        
        # Stage 3: Recommender System - Generate recommendations
        print("\n[Stage 3] Recommendation Generation")
        self.recommender = RecommenderSystem(top_k=top_k)
        self.recommender.set_context(users, items)
        # Note: exclude_seen=False because we want to recommend all high-relevance items,
        # not just those the user hasn't seen
        self.recommendations = self.recommender.recommend_batch(relevance_df, exclude_seen=False)
        print(f"    Generated recommendations: {len(self.recommendations)} total")
        
        # Compute metadata
        metadata = self._compute_metadata()
        
        print("\n[Pipeline] Complete!")
        return self.recommendations, metadata
    
    def _compute_metadata(self) -> Dict:
        """Compute metadata about the pipeline execution."""
        metadata = {
            'num_users': len(self.users),
            'num_items': len(self.items),
            'num_interactions': len(self.logs),
            'num_features': len(self.data_pipeline.get_feature_columns()),
            'num_recommendations': len(self.recommendations),
            'avg_recommendations_per_user': len(self.recommendations) / len(self.users) if len(self.users) > 0 else 0,
            'relevance_stats': {
                'min': float(self.relevance_scores['relevance_score'].min()),
                'max': float(self.relevance_scores['relevance_score'].max()),
                'mean': float(self.relevance_scores['relevance_score'].mean()),
                'std': float(self.relevance_scores['relevance_score'].std()),
            }
        }
        return metadata
    
    def get_user_recommendations(self, user_id: int) -> pd.DataFrame:
        """Get recommendations for a specific user."""
        if self.recommendations is None:
            raise RuntimeError("Pipeline not executed. Call run() first.")
        
        return self.recommender.get_recommendations_for_user(user_id, self.recommendations)
    
    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """Get top feature importances from the ranking model."""
        if self.ranking_pipeline is None:
            raise RuntimeError("Pipeline not executed. Call run() first.")
        
        return self.ranking_pipeline.get_feature_importance(top_n=top_n)
    
    def get_model_report(self) -> Dict:
        """Get summary report of the pipeline."""
        if self.recommendations is None:
            raise RuntimeError("Pipeline not executed. Call run() first.")
        
        return {
            'total_recommendations': len(self.recommendations),
            'users_with_recommendations': self.recommendations['user_id'].nunique(),
            'avg_relevance_score': float(self.recommendations['relevance_score'].mean()),
            'feature_importance': self.get_feature_importance().to_dict('records'),
        }
