# config.py

# API values
API_VERSION: str = "1.2.0"


# Image processing settings
TARGET_WIDTH: int = 700
TARGET_HEIGHT: int = 800
PORTRAIT_PADDING_FACTOR: float = 2.2
JPG_QUALITY: int = 90

# Validation settings
MAX_OUTPUT_SIZE_KB: int = 100
MAX_OUTPUT_SIZE_BYTES: int = MAX_OUTPUT_SIZE_KB * 1024
SUPPORTED_FORMATS: tuple = ("JPEG", "JPG", "PNG", "WEBP", "MPO")
