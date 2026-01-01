# Quick Start Guide for Learning Path Recommender API

## 5-Minute Setup

### Step 1: Install Dependencies (1 minute)
```bash
pip install -r requirements.txt
```

### Step 2: Train and Save Models (2-3 minutes)
```bash
python train_and_save_models.py
```
Expected output: Models saved to `models/` directory

### Step 3: Start API Server (30 seconds)
```bash
python -m uvicorn src.api:app --reload
```
Expected output: `Uvicorn running on http://127.0.0.1:8000`

### Step 4: Test API (30 seconds)
**Option A - Interactive Docs (Browser):**
- Visit http://localhost:8000/docs
- Try `/health` and `/recommend/0` endpoints directly

**Option B - Python Script:**
```bash
python test_api.py
```

**Option C - Manual curl:**
```bash
curl http://localhost:8000/health
curl http://localhost:8000/recommend/0
```

---

## API Endpoints Quick Reference

| Endpoint | Method | Purpose | Example |
|----------|--------|---------|---------|
| `/health` | GET | Check API status | `curl http://localhost:8000/health` |
| `/recommend/{user_id}` | GET | Get recommendations | `curl http://localhost:8000/recommend/0` |
| `/recommend/{user_id}?top_k=3` | GET | Custom top-K | `curl "http://localhost:8000/recommend/0?top_k=3"` |
| `/docs` | GET | Interactive documentation | Visit in browser |
| `/redoc` | GET | Alternative documentation | Visit in browser |

---

## File Structure

```
├── src/
│   ├── config.py                    # Configuration
│   ├── api.py                       # FastAPI app ⭐ RUN THIS
│   └── pipeline/                    # Pipeline modules
│       ├── persistence.py           # Model save/load (joblib)
│       ├── data_pipeline.py         # Feature extraction
│       ├── ranking_pipeline.py      # Model training
│       └── recommender.py           # Top-K ranking
│
├── models/                          # Trained models (created by train script)
│   ├── ranking_model.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
│
├── train_and_save_models.py        # ⭐ RUN FIRST
├── test_api.py                     # ⭐ RUN SECOND
├── requirements.txt                # Dependencies
├── API_SETUP.md                    # Full documentation
└── QUICKSTART.md                   # This file
```

---

## Common Commands

```bash
# Install packages
pip install -r requirements.txt

# Train models (run once)
python train_and_save_models.py

# Start API server (runs continuously)
python -m uvicorn src.api:app --reload

# Test API (in separate terminal)
python test_api.py

# View interactive docs
# Open http://localhost:8000/docs in browser

# Get health status
curl http://localhost:8000/health

# Get recommendations for user 0
curl http://localhost:8000/recommend/0

# Get top-3 recommendations for user 25
curl "http://localhost:8000/recommend/25?top_k=3"
```

---

## Expected Results

### After `python train_and_save_models.py`:
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

### After `python -m uvicorn src.api:app --reload`:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

### After `curl http://localhost:8000/health`:
```json
{
  "status": "healthy",
  "models_loaded": true,
  "available_models": ["ranking_model", "scaler", "feature_columns"]
}
```

### After `curl http://localhost:8000/recommend/0`:
```json
{
  "user_id": 0,
  "recommendations": [
    {
      "item_id": 15,
      "relevance_score": 0.923,
      "rank": 1
    },
    ...
  ],
  "total_recommendations": 4,
  "average_relevance_score": 0.891
}
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Module not found" | Run from project root, verify `src/` exists |
| "No module named 'fastapi'" | Run `pip install -r requirements.txt` |
| "Connection refused" | Ensure API is running: `python -m uvicorn src.api:app --reload` |
| "503 Service Unavailable" | Train models first: `python train_and_save_models.py` |
| "Port 8000 already in use" | Use different port: `python -m uvicorn src.api:app --port 8001` |

---

## What's Inside

### The Recommendation Pipeline
1. **Data Pipeline** - Extracts 14 engineered features from user interactions
2. **Ranking Pipeline** - Trains RandomForest model on features → relevance scores
3. **Recommender** - Converts scores to top-K ranked items per user

### The API Server
- **Health endpoint** - Verify server and models are ready
- **Recommendation endpoint** - Get personalized recommendations for any user
- **Auto-loaded models** - Trained models loaded once on startup, cached in memory
- **Type-safe** - Pydantic models validate all requests/responses

### Model Persistence
- **joblib** - Efficient serialization/deserialization
- **Automatic** - `train_and_save_models.py` handles all saving
- **Fast loading** - Models loaded in ~50ms on startup

---

## API Response Examples

### Health Check (200 OK)
```json
{
  "status": "healthy",
  "models_loaded": true,
  "available_models": ["ranking_model", "scaler", "feature_columns"]
}
```

### Recommendations (200 OK)
```json
{
  "user_id": 5,
  "recommendations": [
    {"item_id": 12, "relevance_score": 0.923, "rank": 1},
    {"item_id": 8, "relevance_score": 0.891, "rank": 2},
    {"item_id": 21, "relevance_score": 0.856, "rank": 3},
    {"item_id": 3, "relevance_score": 0.834, "rank": 4},
    {"item_id": 27, "relevance_score": 0.812, "rank": 5}
  ],
  "total_recommendations": 5,
  "average_relevance_score": 0.863
}
```

### Error - No Recommendations (404 Not Found)
```json
{
  "detail": "No recommendations found for user 999"
}
```

### Error - Models Not Loaded (503 Service Unavailable)
```json
{
  "detail": "Models not loaded. Please train models first with train_and_save_models.py"
}
```

---
