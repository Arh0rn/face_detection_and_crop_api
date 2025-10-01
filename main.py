# main.py

from fastapi import FastAPI

from api.handlers import router as photo_router
from config import API_DESCRIPTION, API_TITLE, API_VERSION

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
)

# Подключаем наш роутер
app.include_router(photo_router, prefix="/api", tags=["Photo Processing"])


@app.get("/")
def read_root():
    return {"message": "Welcome to the Photo Processing API!"}
