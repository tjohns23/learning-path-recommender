# Quick Start

## Setup (3 commands)

```bash
pip install -r requirements.txt
python train_and_save_models.py
python -m uvicorn src.api:app --reload
```

## Test
- Browser: http://localhost:8000/docs
- Curl: `curl http://localhost:8000/health`
- Python: `python test_api.py`

## API Endpoints
- `GET /health` - Health check
- `GET /recommend/{user_id}` - Get recommendations
- `GET /recommend/{user_id}?top_k=3` - Custom top-K
