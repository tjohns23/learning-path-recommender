# API Setup

## Install & Run

```bash
pip install -r requirements.txt
python train_and_save_models.py
python -m uvicorn src.api:app --reload
```

## Project Structure

```
src/
├── api/              # FastAPI app
├── model/            # Model persistence
├── pipeline/         # ML pipeline
├── simulator/        # Data generation
├── config.py
└── ...
```

## Endpoints

- `GET /health` - Status check
- `GET /recommend/{user_id}` - Get recommendations
- `GET /docs` - Swagger UI

## Example Usage

```bash
curl http://localhost:8000/health
curl http://localhost:8000/recommend/0
curl "http://localhost:8000/recommend/0?top_k=3"
```