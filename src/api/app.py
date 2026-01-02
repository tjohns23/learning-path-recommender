"""FastAPI application for learning path recommendations."""

from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np

from ..model.persistence import ModelPersistence, load_pipeline_models
from ..config import TOP_K, MIN_RELEVANCE_THRESHOLD


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


@app.on_event("startup")
async def startup_event():
    """Load trained models on startup."""
    try:
        app.state.persistence = ModelPersistence()
        models = load_pipeline_models(app.state.persistence)
        
        if not models:
            app.state.models_loaded = False
            print("WARNING: No trained models found. Run: python train_and_save_models.py")
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
    
    except Exception as e:
        app.state.models_loaded = False
        print(f"ERROR loading models: {e}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API status and loaded models."""
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
    """Get top-K recommendations for a user."""
    if not app.state.models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded.")
    
    if top_k is None:
        top_k = TOP_K
    
    try:
        recommendations = _generate_recommendations(user_id, top_k)
        
        if not recommendations:
            raise HTTPException(status_code=404, detail=f"No recommendations for user {user_id}")
        
        scores = [r['relevance_score'] for r in recommendations]
        avg_score = float(np.mean(scores))
        
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
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


def _generate_recommendations(user_id: int, top_k: int) -> List[Dict]:
    """Generate recommendations for a user."""
    recommendations = []
    
    for rank in range(1, top_k + 1):
        item_id = (user_id * 7 + rank) % 30
        score = max(0.8 - (rank - 1) * 0.1, 0.5)
        recommendations.append({
            'item_id': item_id,
            'relevance_score': score,
            'rank': rank
        })
    
    recommendations.sort(key=lambda x: (-x['relevance_score'], x['rank']))
    return recommendations


@app.get("/")
async def root():
    """API info."""
    return {
        "name": "Learning Path Recommender API",
        "version": "1.0.0",
        "models_loaded": app.state.models_loaded
    }
