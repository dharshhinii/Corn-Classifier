from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from app.config import load_selected_model
from app.inference import ImageClassifier
from app.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_config = load_selected_model()
    classifier = ImageClassifier(model_config)
    classifier.load()
    app.state.classifier = classifier

    print(f"Loaded {model_config.name} from {model_config.path}")
    yield
    app.state.classifier = None


app = FastAPI(
    title="Image Classification MVP",
    description="Three-class image classification inference API.",
    version="1.1.0",
    lifespan=lifespan,
)


@app.get("/")
def root(request: Request):
    config = request.app.state.classifier.config
    return {
        "service": "image-classification-api",
        "status": "ready",
        "model": config.name,
        "model_type": config.model_type,
        "classes": config.class_names,
        "health_endpoint": "/health",
        "model_endpoint": "/model",
        "prediction_endpoint": "/predict",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    classifier = getattr(request.app.state, "classifier", None)
    if classifier is None:
        return {
            "status": "not_ready",
            "model_loaded": False,
            "model_name": None,
            "model_type": None,
        }

    return {
        "status": "healthy",
        "model_loaded": True,
        "model_name": classifier.config.name,
        "model_type": classifier.config.model_type,
    }


@app.get("/model", response_model=ModelInfoResponse)
def model_details(request: Request):
    config = request.app.state.classifier.config
    model_path = Path(config.path)
    file_size_mb = (
        round(model_path.stat().st_size / (1024 * 1024), 3)
        if model_path.exists()
        else None
    )

    return {
        "name": config.name,
        "type": config.model_type,
        "path": str(model_path),
        "file_exists": model_path.exists(),
        "file_size_mb": file_size_mb,
        "class_names": config.class_names,
        "number_of_classes": len(config.class_names),
        "image_size": list(config.image_size),
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest, request: Request):
    classifier = request.app.state.classifier
    try:
        return await run_in_threadpool(classifier.predict, payload.image_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {exc}",
        ) from exc
