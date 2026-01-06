# main.py

from fastapi import FastAPI

from api.handlers import router as photo_router
from config import API_VERSION

app = FastAPI(
    version=API_VERSION,
    title="Photo Processing API",
    description="Api that takes a face photo and resize, crop and center it.",
    contact={
        "name": "Amir Kurmanbekov",
        "email": "amir.kurmanbekov@gmail.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
)


app.include_router(photo_router, prefix="/api")
