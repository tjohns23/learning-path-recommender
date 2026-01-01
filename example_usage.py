"""
Example usage of the learning path recommender pipeline
"""

import sys
import os

# Add src to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_PATH)

from src.simulator.simulate import run_simulation
from src.pipeline import LearningPathPipeline
from src.config import (
    NUM_USERS, NUM_ITEMS, STEPS_PER_USER, RANDOM_SEED,
    RANKING_MODEL_TYPE, RANKING_MODEL_PARAMS, TOP_K
)


def main():
    """Run the complete recommendation pipeline."""
    
    print("=" * 80)
    print("LEARNING PATH RECOMMENDER PIPELINE")
    print("=" * 80)
    
    # Step 1: Generate synthetic data (users, items, interactions)
    print("\n[Step 1] Generating synthetic users, items, and interactions...")
    users, items, logs = run_simulation(
        num_users=NUM_USERS,
        num_items=NUM_ITEMS,
        steps_per_user=STEPS_PER_USER,
        seed=RANDOM_SEED
    )
    print(f"  ✓ Users: {len(users)}")
    print(f"  ✓ Items: {len(items)}")
    print(f"  ✓ Interactions: {len(logs)}")
    
    # Step 2: Run the pipeline
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
    
    # Step 3: Display results
    print("\n[Step 3] Pipeline Results")
    print("-" * 80)
    
    # Metadata
    print("\nMetadata:")
    for key, value in metadata.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v:.3f}" if isinstance(v, float) else f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    # Feature importance
    print("\nTop Features (Ranking Model):")
    feature_importance = pipeline.get_feature_importance(top_n=5)
    for idx, row in feature_importance.iterrows():
        print(f"  {idx + 1}. {row['feature']}: {row['importance']:.4f}")
    
    # Sample recommendations
    print("\nSample Recommendations for User 0:")
    try:
        user_0_recs = pipeline.get_user_recommendations(0)
    except (KeyError, AttributeError):
        user_0_recs = None
    
    if user_0_recs is not None and len(user_0_recs) > 0:
        for idx, row in user_0_recs.iterrows():
            item_id = int(row['item_id'])
            item = items[item_id]
            print(f"  Rank {int(row['rank'])}: Item {item_id} - Score: {row['relevance_score']:.3f}")
            print(f"    Skills: {sum(item['skills'])}, Difficulty: {item['difficulty']}")
    else:
        print("  No recommendations generated")
    
    # Full recommendations summary
    print("\nRecommendations Summary:")
    print(f"  Total recommendations: {len(recommendations)}")
    if len(recommendations) > 0:
        print(f"  Users with recommendations: {recommendations['user_id'].nunique()}")
        print(f"  Average relevance score: {recommendations['relevance_score'].mean():.3f}")
        print(f"  Relevance score range: [{recommendations['relevance_score'].min():.3f}, {recommendations['relevance_score'].max():.3f}]")
    else:
        print("  No recommendations generated (all items filtered by threshold)")
    
    print("\n" + "=" * 80)
    print("Pipeline execution complete!")
    print("=" * 80)
    
    return pipeline, recommendations


if __name__ == "__main__":
    pipeline, recommendations = main()
