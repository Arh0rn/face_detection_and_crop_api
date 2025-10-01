# api/handlers.py

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response

from core.exceptions import PhotoProcessingError
from services.photo_service import process_user_photo

router = APIRouter()


@router.post(
    "/process",
    responses={
        200: {
            "content": {"image/jpeg": {}},
            "description": "Обработанное изображение в формате JPEG.",
        }
    },
)
async def process_photo_endpoint(file: UploadFile = File(...)):
    """
    Taking a user photo, processing it, and returning the optimized image.
    """
    try:
        image_bytes: bytes = await file.read()
        processed_image_bytes: bytes = process_user_photo(image_bytes)

        return Response(content=processed_image_bytes, media_type="image/jpeg")

    except PhotoProcessingError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
