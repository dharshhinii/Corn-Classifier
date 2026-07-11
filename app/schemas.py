from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    image_path: str = Field(..., min_length=1)


class TimingResponse(BaseModel):
    preprocessing: float
    inference: float
    total: float


class PredictResponse(BaseModel):
    model: str
    model_type: str
    image_path: str
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    timing_ms: TimingResponse


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None = None
    model_type: str | None = None


class ModelInfoResponse(BaseModel):
    name: str
    type: str
    path: str
    file_exists: bool
    file_size_mb: float | None
    class_names: list[str]
    number_of_classes: int
    image_size: list[int]
