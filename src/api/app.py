from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from pathlib import Path
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import traceback, os

MODEL_PATH = Path('model/model.joblib')
RELOAD_TOKEN = os.environ.get('RELOAD_TOKEN', 'changeme')

app = FastAPI(title='Churn Predictor (no-docker)')

# Allow requests from the local static server port we'll use (5500)
origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REQUEST_COUNT = Counter('churn_api_requests_total', 'Total requests', ['endpoint', 'status'])

class InputData(BaseModel):
    features: list[float]

model = None

def load_model():
    global model
    try:
        if MODEL_PATH.exists():
            import joblib
            model = joblib.load(MODEL_PATH)
            print(f'[load_model] Loaded model from {MODEL_PATH.resolve()}')
            return True, None
        else:
            model = None
            msg = f'Model file not found at {MODEL_PATH.resolve()}'
            print('[load_model]', msg)
            return False, msg
    except Exception as e:
        model = None
        tb = traceback.format_exc()
        print('[load_model] Exception while loading model:', tb)
        return False, tb

@app.on_event('startup')
def on_startup():
    ok, info = load_model()
    if not ok:
        print('[startup] Model not loaded. You can train and then POST /reload-model to load it.')

@app.get('/')
def root():
    return {'message':'Churn API is up. Use /docs for Swagger, /health for status, /metrics for Prometheus.'}

@app.get('/health')
def health():
    return {'status':'ok'}

@app.get('/metrics')
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post('/predict')
def predict(inp: InputData):
    if model is None:
        REQUEST_COUNT.labels(endpoint='/predict', status='500').inc()
        raise HTTPException(status_code=503, detail='Model not loaded. Train and then POST /reload-model.')
    try:
        x = np.array(inp.features).reshape(1, -1)
        proba = float(model.predict_proba(x)[0, 1])
        REQUEST_COUNT.labels(endpoint='/predict', status='200').inc()
        return {'churn_proba': proba}
    except Exception as e:
        REQUEST_COUNT.labels(endpoint='/predict', status='500').inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/reload-model')
def reload_model(x_reload_token: str | None = Header(None, alias='X-Reload-Token')):
    if x_reload_token != RELOAD_TOKEN:
        raise HTTPException(status_code=401, detail='Invalid reload token')
    ok, info = load_model()
    if ok:
        return {'status':'ok', 'message':'Model reloaded'}
    else:
        raise HTTPException(status_code=500, detail=f'Reload failed: {info}')
