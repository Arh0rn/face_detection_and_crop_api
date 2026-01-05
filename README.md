# Photo Processing API

## Overview

This is a FastAPI-based backend service designed to process and optimize user photos. It provides an API endpoint that accepts an image, detects the face, and performs cropping and resizing operations to produce a standardized output format (JPEG).

## Features

-   **Deep Learning Face Detection**: uses OpenCV's YuNet model for high-accuracy face detection (robust against glasses, masks, and rotation).
-   **Smart Cropping**: Crops the image based on the detected face with configurable padding.
-   **Resizing**: Resizes the processed image to specific dimensions (default 700x800).
-   **Optimization**: Compresses and optimizes the output image (default 90% quality JPEG).

## How the API Works

The API exposes a single main endpoint:

### `POST /api/process`

-   **Input**: Accepts an image file (multipart/form-data).
-   **Processing**:
    1. Validates the input format.
    2. Detects the face in the image.
    3. Crops and resizes the image centered on the face.
    4. Optimizes the image size.
-   **Output**: Returns the processed image as `image/jpeg`.

## Getting Started

### Prerequisites

-   Python 3.8+
-   pip

### Installation

1. **Clone the repository** (if not already done):

    ```bash
    git clone <repository_url>
    cd face_detection_api
    ```

2. **Create a virtual environment** (recommended):

    ```bash
    # Windows
    python -m venv .venv
    .\.venv\Scripts\activate

    # Linux/macOS
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

To start the server locally with hot-reloading enabled (useful for development), run:

```bash
uvicorn main:app --reload
```

or if you are not using an active virtual environment but have it installed in `.venv`:

```bash
.\.venv\Scripts\python.exe -m uvicorn main:app --reload
```

The server will start at `http://localhost:8000` (by default).

## API Documentation

Once the server is running, you can access the interactive API documentation (Swagger UI) to test endpoints directly from your browser.

-   **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
-   **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
