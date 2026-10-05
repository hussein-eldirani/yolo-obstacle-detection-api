from contextlib import asynccontextmanager
import sys
import os
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.services.detector import YOLOObstacleDetector


# ----------------------------------------------------------------              
# Pydantic Schemas for API Documentation & Output Validation
# ----------------------------------------------------------------              
class DetectionResult(BaseModel):
    class_name: str = Field(..., description="Detected object class label")
    class_id: int = Field(..., description="Numeric class ID from model")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Detection confidence score"
    )
    bbox: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Bounding box coordinates [xmin, ymin, xmax, ymax]",
    )


class PredictResponse(BaseModel):
    count: int = Field(..., description="Total number of detected obstacles")
    predictions: list[DetectionResult] = Field(
        ..., description="List of detected obstacles"
    )


class HealthResponse(BaseModel):
    status: str
    model: str


# ----------------------------------------------------------------              
# Lifespan Management
# ----------------------------------------------------------------              
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles application startup and shutdown events.

    Loads model weights ONCE into app.state when the server boots up.
    """
    model_path = os.getenv("MODEL_PATH", "models/best.pt")
    # Store detector instance directly on app.state
    app.state.detector = YOLOObstacleDetector(model_path=model_path)
    yield
    # Clean up resources on shutdown if needed


# ----------------------------------------------------------------              
# Application Initialization (With Documentation)
# ----------------------------------------------------------------              
app = FastAPI(
    title="Obstacle Detection API",
    description=(
        "A high-performance, asynchronous FastAPI microservice serving YOLO26 "
        "for real-time obstacle detection with non-blocking threadpool offloading."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {
            "name": "System",
            "description": "Health check and system diagnostics endpoints.",
        },
        {
            "name": "Inference",
            "description": "Computer vision model prediction endpoints.",
        },
    ],
    contact={
        "name": "Obstacle Detection Team",
        "url": "https://github.com/YOUR_USERNAME/obstacle-detection-api",
    },
    license_info={
        "name": "MIT",
    },
)


# ----------------------------------------------------------------              
# Endpoints
# ----------------------------------------------------------------              
@app.get(
    "/",
    response_model=HealthResponse,
    tags=["System"],
    summary="Root Health Status",
)
def read_root():
    """Simple health check endpoint returning basic system health."""
    return {"status": "healthy", "model": "YOLO26 Nano"}

@app.get(
    "/health",
    tags=["System"],
    summary="System Health & Model Status Check",
    response_description="Returns runtime system health and loaded model availability.",
)
async def health_check(request: Request):
    """Detailed health diagnostic endpoint for load balancers and container orchestrators (Docker/Kubernetes)."""
    model_loaded = hasattr(request.app.state, "detector") and request.app.state.detector is not None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "python_version": sys.version.split(" ")[0],
        "api_version": app.version,
    }

@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Inference"],
    summary="Detect obstacles in an uploaded image",
    response_description="Structured JSON payload containing detected object counts, confidence scores, and bounding boxes.",
    status_code=200,
)
async def predict_obstacle(request: Request, file: UploadFile = File(...)):
    """Accepts an uploaded image file, processes it through YOLO,

    and returns bounding box information.
    """
    # 1. Validate MIME type before processing
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="Uploaded file must be an image."
        )

    try:
        # 2. Asynchronously read raw binary bytes from request stream
        image_bytes = await file.read()

        # 3. Retrieve model from app.state
        detector: YOLOObstacleDetector = request.app.state.detector

        # 4. Offload heavy CPU/GPU prediction task to thread pool
        # to avoid blocking FastAPI's async event loop
        results = await run_in_threadpool(detector.predict, image_bytes)

        return results

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Inference Engine Error: {str(e)}"
        )