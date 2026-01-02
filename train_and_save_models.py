"""
Train and save models for the API
This script runs the full pipeline and saves trained models using joblib
"""

import sys
import os

# Add src to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_PATH)

from src.simulator.simulate import run_simulation
from src.pipeline import LearningPathPipeline
from src.model.persistence import ModelPersistence, save_pipeline_models
from src.config import (
    NUM_USERS, NUM_ITEMS, STEPS_PER_USER, RANDOM_SEED,
    RANKING_MODEL_TYPE, RANKING_MODEL_PARAMS, TOP_K
)


def main():
    """Train pipeline and save models."""
    
    print("=" * 80)
    print("TRAINING AND SAVING MODELS FOR API")
    print("=" * 80)
    
    # Step 1: Generate synthetic data
    print("\n[Step 1] Generating synthetic data...")
    users, items, logs = run_simulation(
        num_users=NUM_USERS,
        num_items=NUM_ITEMS,
        steps_per_user=STEPS_PER_USER,
        seed=RANDOM_SEED
    )
    print(f"   Generated {len(users)} users, {len(items)} items, {len(logs)} interactions")
    
    # Step 2: Run pipeline
    print("\n[Step 2] Running recommendation pipeline...")
    pipeline = LearningPathPipeline()
    recommendations, metadata = pipeline.run(
        users=users,
        items=items,
        logs=logs,
        ranking_model=RANKING_MODEL_TYPE,
        model_params=RANKING_MODEL_PARAMS,
        top_k=TOP_K
    )
    
    # Step 3: Save models
    print("\n[Step 3] Saving models...")
    persistence = ModelPersistence()
    saved_paths = save_pipeline_models(pipeline, persistence)
    
    print("   Models saved:")
    for model_name, filepath in saved_paths.items():
        print(f"    - {model_name}: {filepath}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nModels are ready for the API!")
    print("Start the API with: python -m uvicorn src.api:app --reload")
    print("Access API at: http://localhost:8000")
    print("API docs at: http://localhost:8000/docs")
    print("=" * 80)


if __name__ == "__main__":
    main()
