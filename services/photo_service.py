# services/photo_service.py

import io

import cv2
import numpy as np
from PIL import Image
from PIL.Image import Image as ImageObject

from config import (
    FACE_DETECTION_MIN_NEIGHBORS,
    FACE_DETECTION_MIN_SIZE_H,
    FACE_DETECTION_MIN_SIZE_W,
    FACE_DETECTION_SCALE_FACTOR,
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


face_cascade: cv2.CascadeClassifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore
)


def process_user_photo(image_bytes: bytes) -> bytes:
    """
    Full pipeline for processing a user's photo with smart face-aware cropping.
    """
    try:
        # --- 1. Open and validate ---
        image: ImageObject = Image.open(io.BytesIO(image_bytes))

        print(f"DEBUG: Pillow увидел формат: '{image.format}'")

        if image.format not in SUPPORTED_FORMATS:
            raise InvalidImageFormatError()

        image = image.convert("RGB")
        original_width: int
        original_height: int
        original_width, original_height = image.size

        # --- 2. Face detection ---
        open_cv_image = np.array(image)
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            FACE_DETECTION_SCALE_FACTOR,
            minNeighbors=FACE_DETECTION_MIN_NEIGHBORS,
            minSize=(FACE_DETECTION_MIN_SIZE_W, FACE_DETECTION_MIN_SIZE_H),
        )

        if len(faces) == 0:
            raise NoFaceFoundError()

        # --- NEW LOGIC: SMART CROPPING ---

        # 3. Find the largest face if there are multiple
        main_face = max(faces, key=lambda rect: rect[2] * rect[3])

        fx: int
        fy: int
        fw: int
        fh: int

        fx, fy, fw, fh = main_face

        # 4. Compute face center
        face_center_x: float = fx + fw / 2
        face_center_y: float = fy + fh / 2

        # 5. Determine crop box size
        # We want the crop height to be PORTRAIT_PADDING_FACTOR times the face height
        crop_height: float = fh * PORTRAIT_PADDING_FACTOR
        crop_width: float = crop_height * TARGET_ASPECT_RATIO

        # 6. Compute crop box coordinates so the face is centered
        crop_x1: int = int(face_center_x - crop_width / 2)
        crop_y1: int = int(face_center_y - crop_height / 2)
        crop_x2: int = int(face_center_x + crop_width / 2)
        crop_y2: int = int(face_center_y + crop_height / 2)

        # 7. Adjust the box so it doesn't go outside the image
        crop_x1 = max(0, crop_x1)
        crop_y1 = max(0, crop_y1)
        crop_x2 = min(original_width, crop_x2)
        crop_y2 = min(original_height, crop_y2)

        # 8. Crop the image
        cropped_image: ImageObject = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

        # 9. Final resize to exact 700x800
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

    except FileNotFoundError:
        raise RuntimeError("File 'haarcascade_frontalface_default.xml' not found")
    except Exception as e:
        raise PhotoProcessingError(f"Failed to process image: {e}")
