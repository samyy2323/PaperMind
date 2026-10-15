import logging

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import answer_sheets, auth, classes, exams, knowledge_base, student_answers
from app.core.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="PaperMind API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware, allow_origins=settings.cors_origins_list, allow_credentials=True,
        allow_methods=["*"], allow_headers=["*"],
    )
    api = APIRouter(prefix="/api")
    for module in (auth, classes, exams, knowledge_base, answer_sheets, student_answers):
        api.include_router(module.router)
    app.include_router(api)

    @app.get("/health", tags=["meta"])
    def health():
        return {"status": "ok"}

    return app


app = create_app()
