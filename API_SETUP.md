# Learning Path Recommender API

This guide explains how to set up and run the FastAPI recommendation service.

## Prerequisites

- Python 3.8 or higher
- All dependencies installed from `requirements.txt`

## Installation

1. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

   Key packages for the API:
   - `fastapi` - Web framework
   - `uvicorn` - ASGI server
   - `pydantic` - Data validation
   - `joblib` - Model persistence
   - `requests` - HTTP client (for testing)

2. **Verify workspace structure:**
   ```
   src/
     ├── config.py                  # Configuration constants
     ├── api.py                     # FastAPI application
     ├── pipeline/
     │   ├── __init__.py           # LearningPathPipeline orchestrator
     │   ├── persistence.py        # Model save/load utilities
     │   ├── data_pipeline.py      # Feature extraction
     │   ├── ranking_pipeline.py   # Ranking model
     │   └── recommender.py        # Top-K recommendations
     └── simulator/
         ├── simulate.py           # Data generation
         ├── users.py              # User simulation
         ├── items.py              # Item simulation
         └── interactions.py       # Interaction simulation
   
   models/                          # Directory for saved models (created on first training)
     ├── ranking_model.pkl
     ├── scaler.pkl
     └── feature_columns.pkl
   
   train_and_save_models.py        # Script to train and save models
   test_api.py                     # Script to test API endpoints
   ```

## Training Models

Before running the API, you need to train and save the models:

```bash
python train_and_save_models.py
```

**What this does:**
1. Generates synthetic data (50 users × 30 items × 847 interactions)
2. Runs the full three-stage pipeline:
   - **Data Pipeline**: Extracts 14 engineered features
   - **Ranking Pipeline**: Trains RandomForest model
   - **Recommender**: Generates top-K recommendations
3. Saves trained models to `models/` directory using joblib

**Output:**
```
[Step 1] Generating synthetic data...
  ✓ Generated 50 users, 30 items, 847 interactions

[Step 2] Running recommendation pipeline...
  ✓ Generated 237 recommendations across all users

[Step 3] Saving models...
  ✓ Models saved:
    - ranking_model: models/ranking_model.pkl
    - scaler: models/scaler.pkl
    - feature_columns: models/feature_columns.pkl
```

## Running the API

Start the FastAPI server:

```bash
python -m uvicorn src.api:app --reload
```

**Options:**
- `--reload` - Auto-reload on code changes (development only)
- `--host 0.0.0.0` - Listen on all network interfaces
- `--port 8000` - Change port (default: 8000)

**Example with custom settings:**
```bash
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```

**Success output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started server process [12345]
INFO:     Application startup complete
```

## API Endpoints

### 1. Health Check
**Endpoint:** `GET /health`

**Purpose:** Verify API is running and models are loaded

**Response (200 OK):**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "available_models": ["ranking_model", "scaler", "feature_columns"]
}
```

**Example:**
```bash
curl http://localhost:8000/health
```

### 2. Get Recommendations
**Endpoint:** `GET /recommend/{user_id}`

**Parameters:**
- `user_id` (path): Integer user ID (0-49)
- `top_k` (query, optional): Number of recommendations (default: 5)

**Response (200 OK):**
```json
{
  "user_id": 5,
  "recommendations": [
    {
      "item_id": 12,
      "relevance_score": 0.895,
      "rank": 1
    },
    {
      "item_id": 23,
      "relevance_score": 0.867,
      "rank": 2
    }
  ],
  "total_recommendations": 2,
  "average_relevance_score": 0.881
}
```

**Error Responses:**
- `404 Not Found` - User has no recommendations
- `503 Service Unavailable` - Models not loaded on startup

**Examples:**
```bash
# Get top-5 recommendations for user 0
curl http://localhost:8000/recommend/0

# Get top-3 recommendations for user 25
curl "http://localhost:8000/recommend/25?top_k=3"

# Using Python requests
import requests
response = requests.get("http://localhost:8000/recommend/0")
recommendations = response.json()
```