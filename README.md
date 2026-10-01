# Obstacle Detection API

A FastAPI service serving a custom-trained YOLO model for real-time obstacle detection.

## Features

- **Fast Model Inference:** Runs bounding-box prediction on uploaded image binary streams using PyTorch/YOLO.
- **Async Threadpool Offloading:** Utilizes FastAPI's `run_in_threadpool` to offload CPU-heavy inference without blocking the async event loop.
- **Strict Validation:** Input MIME-type checking and Pydantic response models ensure structured output schemas.
- **State Management:** Loads YOLO weights once during application lifespan (`app.state`) rather than per-request.

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