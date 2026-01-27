import io
import os
import cv2
import numpy as np
from PIL import Image
from PIL.Image import Image as ImageObject

from config import (
    JPG_QUALITY,
    MAX_OUTPUT_SIZE_BYTES,
    PORTRAIT_PADDING_FACTOR,
    SUPPORTED_FORMATS,
    TARGET_HEIGHT,
    TARGET_WIDTH,
)
from core.exceptions import (
    CompressionError,
    InvalidImageFormatError,
    NoFaceFoundError,
    PhotoProcessingError,
)

TARGET_ASPECT_RATIO: float = TARGET_WIDTH / TARGET_HEIGHT

# Path to the YuNet model
# We use a path relative to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "face_detection_yunet_2023mar.onnx")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Face detection model not found at {MODEL_PATH}")

# Initialize YuNet Face Detector
# We set input_size dynamically, but need an initial size
face_detector = cv2.FaceDetectorYN.create(
    model=MODEL_PATH,
    config="",
    input_size=(320, 320),
    score_threshold=0.5,
    nms_threshold=0.3,
    top_k=5000,
    backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
    target_id=cv2.dnn.DNN_TARGET_CPU,
)


def process_user_photo(image_bytes: bytes) -> bytes:
    """
    Full pipeline for processing a user's photo with smart face-aware cropping using OpenCV YuNet.
    """
    try:
        # --- 1. Open and validate ---
        image: ImageObject = Image.open(io.BytesIO(image_bytes))

        if image.format not in SUPPORTED_FORMATS:
            raise InvalidImageFormatError()

        # Handle EXIF rotation automatically
        from PIL import ImageOps

        image = ImageOps.exif_transpose(image)

        image = image.convert("RGB")
        original_width, original_height = image.size

        # --- 2. Face detection with YuNet ---
        open_cv_image = np.array(image)
        # Convert RGB to BGR for OpenCV
        open_cv_image_bgr = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

        # Update detector input size to match image size
        face_detector.setInputSize((original_width, original_height))

        # Run detection
        # retval is usually 1 if successful
        retval, faces = face_detector.detect(open_cv_image_bgr)

        if faces is None or len(faces) == 0:
            raise NoFaceFoundError()

        # --- 3. Find the main face ---
        # YuNet returns faces as a numpy array of shape [n_faces, 15]
        # Columns: x1, y1, w, h, x_right_eye, y_right_eye, ... conf
        # We select the face with the highest confidence (col 14) or largest area (w*h)
        # Let's use largest area as main metric for "main face"

        # Custom heuristic for "Main Face":
        # 1. Filter by confidence (>= 0.6) to reduce noise
        # 2. Keep top 3 largest faces by area
        # 3. Choose the highest one (smallest y) from the largest faces
        # This prevents selecting a "torso face" or background face over the true face.

        valid_faces = [f for f in faces if f[14] >= 0.6]
        if not valid_faces:
            valid_faces = faces

        # Sort by area (width * height) descending
        valid_faces = sorted(valid_faces, key=lambda f: f[2] * f[3], reverse=True)

        # Take top 3 largest
        top_candidates = valid_faces[:3]

        # Select the one with smallest y (top-most)
        main_face = min(top_candidates, key=lambda f: f[1])

        fx, fy, fw, fh = main_face[0:4]

        # --- 4. Compute face center ---
        face_center_x: float = fx + fw / 2
        face_center_y: float = fy + fh / 2

        # --- 5. Determine crop box size ---
        # We want the crop height to be PORTRAIT_PADDING_FACTOR times the face height
        crop_height: float = fh * PORTRAIT_PADDING_FACTOR
        crop_width: float = crop_height * TARGET_ASPECT_RATIO

        # --- 6. Compute crop box coordinates ---
        crop_x1: int = int(face_center_x - crop_width / 2)
        crop_y1: int = int(face_center_y - crop_height / 2)
        crop_x2: int = int(face_center_x + crop_width / 2)
        crop_y2: int = int(face_center_y + crop_height / 2)

        # --- 7. Adjust the box ---
        # Shift approach to preserve aspect ratio
        if crop_x1 < 0:
            crop_x2 -= crop_x1
            crop_x1 = 0
        if crop_y1 < 0:
            crop_y2 -= crop_y1
            crop_y1 = 0
        if crop_x2 > original_width:
            crop_x1 -= crop_x2 - original_width
            crop_x2 = original_width
        if crop_y2 > original_height:
            crop_y1 -= crop_y2 - original_height
            crop_y2 = original_height

        # Clamp
        crop_x1 = max(0, crop_x1)
        crop_y1 = max(0, crop_y1)
        crop_x2 = min(original_width, crop_x2)
        crop_y2 = min(original_height, crop_y2)

        # --- 8. Crop the image ---
        cropped_image: ImageObject = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

        # 9. Final resize to exact target size
        final_image: ImageObject = cropped_image.resize(
            (TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS
        )

        # --- 10. Save with optimization ---
        output_buffer = io.BytesIO()

        for quality in range(95, 10, -5):
            output_buffer.seek(0)
            output_buffer.truncate(0)

            final_image.save(
                output_buffer, format="JPEG", quality=quality, optimize=True
            )

            if output_buffer.tell() <= MAX_OUTPUT_SIZE_BYTES:
                return output_buffer.getvalue()

        raise CompressionError()

    except Exception as e:
        # Re-raise known errors, wrap unknown ones
        if isinstance(
            e,
            (
                PhotoProcessingError,
                NoFaceFoundError,
                InvalidImageFormatError,
                CompressionError,
            ),
        ):
            raise e
        raise PhotoProcessingError(f"Failed to process image: {e}")
