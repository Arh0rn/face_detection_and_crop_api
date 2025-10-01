# config.py

# API settings
API_TITLE: str = "Photo Processing API"
API_DESCRIPTION: str = "API for processing and optimizing user photos."
API_VERSION: str = "1.0.0"


# Image processing settings
TARGET_WIDTH: int = 700
TARGET_HEIGHT: int = 800
PORTRAIT_PADDING_FACTOR: float = 2.2
JPG_QUALITY: int = 90
FACE_DETECTION_MIN_NEIGHBORS: int = 8
FACE_DETECTION_SCALE_FACTOR: float = 1.1
FACE_DETECTION_MIN_SIZE_W: int = 60
FACE_DETECTION_MIN_SIZE_H: int = 60

# Validation settings
MAX_OUTPUT_SIZE_KB: int = 100
MAX_OUTPUT_SIZE_BYTES: int = MAX_OUTPUT_SIZE_KB * 1024
SUPPORTED_FORMATS: tuple[str, ...] = ("JPEG", "JPG")
