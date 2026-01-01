"""
Model persistence utilities using joblib
"""

import joblib
import os
from pathlib import Path
from typing import Any, Dict


class ModelPersistence:
    """Handle saving and loading models with joblib."""
    
    def __init__(self, model_dir: str = None):
        """
        Initialize persistence utility.
        
        Args:
            model_dir: Directory to store models (default: PROJECT_ROOT/models)
        """
        if model_dir is None:
            from ..config import MODELS_DIR
            model_dir = MODELS_DIR
        
        self.model_dir = model_dir
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
    
    def save_model(self, model: Any, model_name: str) -> str:
        """
        Save a model using joblib.
        
        Args:
            model: Model object to save
            model_name: Name of the model (e.g., 'ranking_model', 'scaler')
        
        Returns:
            Path to saved model
        """
        filepath = os.path.join(self.model_dir, f"{model_name}.pkl")
        joblib.dump(model, filepath, compress=3)
        return filepath
    
    def load_model(self, model_name: str) -> Any:
        """
        Load a model using joblib.
        
        Args:
            model_name: Name of the model to load
        
        Returns:
            Loaded model object
        """
        filepath = os.path.join(self.model_dir, f"{model_name}.pkl")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model not found: {filepath}")
        return joblib.load(filepath)
    
    def model_exists(self, model_name: str) -> bool:
        """Check if a model exists."""
        filepath = os.path.join(self.model_dir, f"{model_name}.pkl")
        return os.path.exists(filepath)
    
    def list_models(self) -> list:
        """List all saved models."""
        if not os.path.exists(self.model_dir):
            return []
        return [f[:-4] for f in os.listdir(self.model_dir) if f.endswith('.pkl')]


def save_pipeline_models(pipeline, persistence: ModelPersistence = None):
    """
    Save all trained models from a pipeline.
    
    Args:
        pipeline: Trained LearningPathPipeline object
        persistence: ModelPersistence instance (creates new if None)
    
    Returns:
        Dictionary with paths to saved models
    """
    if persistence is None:
        persistence = ModelPersistence()
    
    saved = {}
    
    # Save ranking model
    if pipeline.ranking_pipeline and pipeline.ranking_pipeline.is_trained:
        saved['ranking_model'] = persistence.save_model(
            pipeline.ranking_pipeline.model,
            'ranking_model'
        )
        if pipeline.ranking_pipeline.scaler:
            saved['scaler'] = persistence.save_model(
                pipeline.ranking_pipeline.scaler,
                'scaler'
            )
    
    # Save feature columns
    if pipeline.data_pipeline:
        feature_cols = pipeline.data_pipeline.get_feature_columns()
        saved['feature_columns'] = persistence.save_model(
            feature_cols,
            'feature_columns'
        )
    
    return saved


def load_pipeline_models(persistence: ModelPersistence = None) -> Dict:
    """
    Load all trained models.
    
    Args:
        persistence: ModelPersistence instance (creates new if None)
    
    Returns:
        Dictionary with loaded models
    """
    if persistence is None:
        persistence = ModelPersistence()
    
    models = {}
    
    if persistence.model_exists('ranking_model'):
        models['ranking_model'] = persistence.load_model('ranking_model')
    
    if persistence.model_exists('scaler'):
        models['scaler'] = persistence.load_model('scaler')
    
    if persistence.model_exists('feature_columns'):
        models['feature_columns'] = persistence.load_model('feature_columns')
    
    return models
