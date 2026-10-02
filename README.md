# Obstacle Detection API

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![YOLO26](https://img.shields.io/badge/YOLO-26-00FFFF?style=for-the-badge)

A FastAPI service serving a custom-trained YOLO model for real-time obstacle detection.

## Features

- **Fast Model Inference:** Runs bounding-box prediction on uploaded image binary streams using PyTorch/YOLO.
- **Async Threadpool Offloading:** Utilizes FastAPI's `run_in_threadpool` to offload CPU-heavy inference without blocking the async event loop.
- **Strict Validation:** Input MIME-type checking and Pydantic response models ensure structured output schemas.
- **State Management:** Loads YOLO weights once during application lifespan (`app.state`) rather than per-request.

## Architecture & Design Decisions

- **Async Offloading (`run_in_threadpool`):** Object detection involves CPU-heavy NumPy array operations and OpenCV decoding. Running these synchronously in standard FastAPI endpoints blocks the main async event loop. We route inference through `starlette.concurrency.run_in_threadpool` to execute PyTorch predictions in a separate thread pool, keeping the API responsive under concurrent traffic.
- **Lifespan State Management (`app.state`):** Model weight loading is expensive (~1–2 seconds). Weights are loaded once during application startup in the FastAPI `lifespan` handler and attached to `app.state`, ensuring zero-overhead per request.
- **Contract Enforcement:** Input files undergo strict MIME-type validation (`image/jpeg`, `image/png`) before reaching the model, returning early HTTP `400` errors for invalid payloads.

## Project Structure

```text
obstacle_detection_api/
├── app/
│   ├── services/
│   │   ├── __init__.py
│   │   └── detector.py    # YOLO inference logic & image decoding
│   ├── __init__.py
│   └── main.py            # FastAPI endpoints, validation & lifespan
├── models/
│   └── best.pt            # Trained YOLO model weights
├── .gitignore             # Version control exclusions
├── README.md              # Project documentation
└── requirements.txt       # Project dependencies