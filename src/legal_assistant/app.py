"""Local web application. Full models are loaded lazily, never during import."""
import logging
import sqlite3
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .archive import count_answers, find_questions
from .service import Service
from .settings import mode

app = FastAPI(title="AI Legal Assistant")
service = Service()
ui = Path(__file__).parent / "ui"
app.mount("/ui", StaticFiles(directory=ui), name="ui")
logger = logging.getLogger(__name__)


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

    @field_validator("question")
    @classmethod
    def non_blank(cls, value):
        if not value.strip():
            raise ValueError("Question must not be blank")
        return value.strip()


@app.get("/")
def index():
    return FileResponse(ui / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "mode": mode()}


@app.get("/ready")
def ready():
    if mode() == "rag" and (service.engine is None or service.error):
        raise HTTPException(503, "Full RAG is not initialized. Check local artifacts, model weights and Ollama.")
    if mode() == "preview":
        try:
            if count_answers() != 2000:
                raise ValueError("Invalid archive")
        except (OSError, ValueError, sqlite3.Error):
            raise HTTPException(503, "Preview database unavailable.") from None
    return {"status": "ready", "mode": mode()}


@app.get("/examples")
def examples():
    fields = ("question", "question_en", "question_title_en", "topic_en")
    return {"archive_count": count_answers() if mode() == "preview" else 0, "mode": mode(), "questions": [x["question"] for x in service.examples],
            "examples": [{key: row[key] for key in fields} for row in service.examples]}


@app.get("/archive/questions")
def archive_questions(q: str = Query(default="", max_length=200), limit: int = Query(default=12, ge=1, le=20)):
    if mode() != "preview":
        raise HTTPException(404, "Question library is available in preview mode only.")
    return {"questions": find_questions(q, limit)}


@app.post("/ask")
def ask(body: Question):
    request_id = uuid.uuid4().hex
    try:
        return {**service.ask(body.question), "request_id": request_id}
    except BlockingIOError:
        raise HTTPException(429, "Another request is running. Please try again shortly.",
                            headers={"Retry-After": "5"}) from None
    except Exception as exc:  # noqa: BLE001 - sanitize provider/model errors at the API boundary
        logger.error("request_failed id=%s type=%s", request_id, type(exc).__name__)
        raise HTTPException(503, {"message": "The service could not complete the request. Check preview data or local artifacts, "
                                 "model availability and Ollama.", "request_id": request_id}) from None
