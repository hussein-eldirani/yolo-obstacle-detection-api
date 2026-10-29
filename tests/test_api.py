import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


@pytest.fixture(scope="module")
def client():
    """
    Using 'with TestClient(app) as client:' triggers FastAPI's lifespan event,
    which loads the YOLO model weights into app.state.detector before tests run.
    """
    with TestClient(app) as client:
        yield client


# --- Health & Root Endpoint Tests ---
def test_read_root(client):
    """Verify that the root endpoint responds with 200 OK and expected status."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model" in data


def test_health_check(client):
    """Verify that /health diagnostics report model status and runtime details."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "model_loaded" in data
    assert data["model_loaded"] is True  # Verifies lifespan successfully loaded the model
    assert "python_version" in data
    assert "api_version" in data


# --- Input Validation Tests ---
def test_predict_invalid_file_type(client):
    """Verify that sending a non-image file returns HTTP 400 Bad Request."""
    fake_file = io.BytesIO(b"This is a text file, not an image.")
    response = client.post(
        "/predict",
        files={"file": ("test.txt", fake_file, "text/plain")},
    )
    assert response.status_code == 400
    assert "Uploaded file must be an image" in response.json()["detail"]


# --- Inference Pipeline Tests ---
def test_predict_valid_image(client):
    """Verify that uploading a valid JPEG image returns 200 OK with prediction schemas."""
    # Dynamically create a valid 640x640 blue test image in-memory via Pillow
    img = Image.new("RGB", (640, 640), color="blue")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_bytes = img_byte_arr.getvalue()

    response = client.post(
        "/predict",
        files={"file": ("test_obstacle.jpg", img_bytes, "image/jpeg")},
    )

    assert response.status_code == 200
    data = response.json()
    
    # Assert output conforms to PredictResponse schema
    assert "count" in data
    assert "predictions" in data
    assert isinstance(data["count"], int)
    assert isinstance(data["predictions"], list)