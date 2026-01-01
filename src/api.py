"""
FastAPI application for learning path recommendations
"""

from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np

from .pipeline.persistence import ModelPersistence, load_pipeline_models
from .config import TOP_K, MIN_RELEVANCE_THRESHOLD


# Response models
class RecommendedItem(BaseModel):
    item_id: int
    relevance_score: float
    rank: int


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[RecommendedItem]
    total_recommendations: int
    average_relevance_score: float


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    available_models: List[str]


# Initialize FastAPI app
app = FastAPI(
    title="Learning Path Recommender API",
    description="Serves personalized learning path recommendations",
    version="1.0.0"
)

# Global state for loaded models
class AppState:
    ranking_model = None
    scaler = None
    feature_columns = None
    persistence = None
    models_loaded = False


@app.on_event("startup")
async def startup_event():
    """
    Load trained models on startup.
    This ensures models are loaded once and reused for all requests.
    """
    try:
        app.state.persistence = ModelPersistence()
        
        # Load models
        models = load_pipeline_models(app.state.persistence)
        
        if not models:
            app.state.models_loaded = False
            print("WARNING: No trained models found. API will not provide recommendations.")
            print("Run: python train_and_save_models.py")
            return
        
        app.state.ranking_model = models.get('ranking_model')
        app.state.scaler = models.get('scaler')
        app.state.feature_columns = models.get('feature_columns')
        
        if app.state.ranking_model is None:
            app.state.models_loaded = False
            print("WARNING: Ranking model not found.")
            return
        
        app.state.models_loaded = True
        print("✓ Models loaded successfully")
        print(f"  - Ranking model: {type(app.state.ranking_model).__name__}")
        if app.state.scaler:
            print(f"  - Scaler: StandardScaler")
        if app.state.feature_columns:
            print(f"  - Features: {len(app.state.feature_columns)} columns")
    
    except Exception as e:
        app.state.models_loaded = False
        print(f"ERROR loading models: {e}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Returns status of API and loaded models.
    """
    return HealthResponse(
        status="healthy",
        models_loaded=app.state.models_loaded,
        available_models=app.state.persistence.list_models() if app.state.persistence else []
    )


@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
async def get_recommendations(
    user_id: int = Path(..., description="ID of the user to get recommendations for"),
    top_k: Optional[int] = None
):
    """
    Get top-K recommendations for a user.
    
    Args:
        user_id: ID of the user
        top_k: Number of recommendations to return (default from config.TOP_K)
    
    Returns:
        RecommendationResponse with ranked items
    """
    # Check if models are loaded
    if not app.state.models_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. API is not ready to serve recommendations."
        )
    
    if top_k is None:
        top_k = TOP_K
    
    try:
        # For this minimal API, we'll return mock recommendations
        # In production, you would:
        # 1. Load user data and interaction history
        # 2. Compute features for all items
        # 3. Use the ranking model to score items
        # 4. Return top-K
        
        recommendations = _generate_recommendations(user_id, top_k)
        
        if not recommendations:
            raise HTTPException(
                status_code=404,
                detail=f"No recommendations generated for user {user_id}"
            )
        
        # Calculate statistics
        scores = [r['relevance_score'] for r in recommendations]
        avg_score = float(np.mean(scores))
        
        # Format response
        items = [
            RecommendedItem(
                item_id=int(r['item_id']),
                relevance_score=float(r['relevance_score']),
                rank=int(r['rank'])
            )
            for r in recommendations
        ]
        
        return RecommendationResponse(
            user_id=user_id,
            recommendations=items,
            total_recommendations=len(items),
            average_relevance_score=avg_score
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")


def _generate_recommendations(user_id: int, top_k: int) -> List[Dict]:
    """
    Generate recommendations for a user using the loaded model.
    
    This is a simplified version - in production you would:
    1. Query database for user's interaction history
    2. Compute features for all items
    3. Use ranking_model.predict() to score items
    4. Filter and rank
    
    For now, we generate mock recommendations for demo purposes.
    """
    # Generate mock recommendations (in production, use real model)
    # This simulates what would happen with actual user data
    recommendations = []
    
    # Generate K recommendations with decreasing scores
    for rank in range(1, top_k + 1):
        item_id = (user_id * 7 + rank) % 30  # Pseudo-random but deterministic
        # Score decreases with rank
        score = max(0.8 - (rank - 1) * 0.1, 0.5)
        
        recommendations.append({
            'item_id': item_id,
            'relevance_score': score,
            'rank': rank
        })
    
    # Sort by relevance score descending, then by rank
    recommendations.sort(key=lambda x: (-x['relevance_score'], x['rank']))
    
    return recommendations


@app.get("/")
async def root():
    """Welcome endpoint with API information."""
    return {
        "name": "Learning Path Recommender API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health - Check API status and loaded models",
            "recommend": "/recommend/{user_id} - Get recommendations for a user"
        },
        "models_loaded": app.state.models_loaded
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
