from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import list_available_models, load_selected_model
from app.inference import ImageClassifier
from app.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    SwitchModelRequest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_IMAGES_DIR = PROJECT_ROOT / "test_images"
FRONTEND_DIST_DIR = PROJECT_ROOT / "frontend" / "dist"


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
    description="Three-class image classification inference API with React Frontend.",
    version="1.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if TEST_IMAGES_DIR.exists():
    app.mount("/static/test_images", StaticFiles(directory=str(TEST_IMAGES_DIR)), name="test_images")

if FRONTEND_DIST_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST_DIR / "assets")), name="frontend_assets")


@app.get("/api-info")
def api_info(request: Request):
    config = request.app.state.classifier.config
    return {
        "service": "image-classification-api",
        "status": "ready",
        "model": config.name,
        "model_type": config.model_type,
        "classes": config.class_names,
        "health_endpoint": "/health",
        "model_endpoint": "/model",
        "models_endpoint": "/models",
        "prediction_endpoint": "/predict",
        "prediction_upload_endpoint": "/predict/upload",
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
    classifier = request.app.state.classifier
    config = classifier.config
    model_path = Path(config.path)
    file_size_mb = (
        round(model_path.stat().st_size / (1024 * 1024), 3)
        if model_path.exists()
        else None
    )

    available = list_available_models()
    return {
        "name": config.name,
        "type": config.model_type,
        "path": str(model_path),
        "file_exists": model_path.exists(),
        "file_size_mb": file_size_mb,
        "class_names": config.class_names,
        "number_of_classes": len(config.class_names),
        "image_size": list(config.image_size),
        "available_models": available,
    }


@app.get("/models")
def get_models():
    return list_available_models()


@app.post("/models/switch")
def switch_model(payload: SwitchModelRequest, request: Request):
    try:
        new_config = load_selected_model(payload.model_name)
        new_classifier = ImageClassifier(new_config)
        new_classifier.load()
        request.app.state.classifier = new_classifier
        return {
            "status": "success",
            "message": f"Switched model to '{new_config.name}'",
            "active_model": new_config.name,
            "model_type": new_config.model_type,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to switch model to '{payload.model_name}': {exc}",
        ) from exc


@app.get("/samples")
def get_sample_images(request: Request):
    base_url = str(request.base_url).rstrip("/")
    samples = []
    if TEST_IMAGES_DIR.exists():
        for file in sorted(TEST_IMAGES_DIR.glob("*")):
            if file.is_file() and file.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                samples.append({
                    "filename": file.name,
                    "url": f"{base_url}/static/test_images/{file.name}",
                    "server_path": str(file.resolve()),
                })
    return samples


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


@app.post("/predict/upload", response_model=PredictResponse)
async def predict_upload(request: Request, file: UploadFile = File(...)):
    classifier = request.app.state.classifier
    try:
        contents = await file.read()
        return await run_in_threadpool(
            classifier.predict,
            contents,
            file.filename,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed for uploaded image: {exc}",
        ) from exc


@app.get("/")
def serve_index():
    index_path = FRONTEND_DIST_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {
        "service": "image-classification-api",
        "status": "ready",
        "docs": "/docs",
    }
