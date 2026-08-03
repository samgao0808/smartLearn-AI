import os
import sys
from pathlib import Path

# Allow `uvicorn smartlearn-backend.main:app --reload` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pypdf.errors import PdfReadError

from services import rag

app = FastAPI(title="SmartLearn Lite API")

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5173",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

BACKEND_DIR = Path(__file__).resolve().parent
ARTIFACT_ROOT = BACKEND_DIR / "artifacts" / "rag"
UPLOAD_ROOT = BACKEND_DIR / "uploads"

documents: dict[str, dict] = {}


class ChatRequest(BaseModel):
    chat_id: str = Field(default="day2-demo")
    message: str = Field(min_length=2, max_length=2000)


@app.get("/")
def root():
    return {"message": "SmartLearn Lite API is running"}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/upload")
async def upload(chat_id: str, file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Empty filename")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="File is empty")

    try:
        record = rag.prepare_rag_chat_record(
            chat_id=chat_id,
            filename=file.filename,
            pdf_bytes=pdf_bytes,
            upload_root=UPLOAD_ROOT,
            artifact_root=ARTIFACT_ROOT,
        )
    except (ValueError, PdfReadError) as e:
        raise HTTPException(status_code=400, detail=f"Could not read PDF: {e}")

    if not record["pages"]:
        raise HTTPException(
            status_code=422,
            detail="No extractable text found. OCR is not supported for scanned PDFs.",
        )

    # Store the richer Day 3 record only after it was built successfully,
    # so an error never leaves a half-written documents[chat_id] entry.
    documents[chat_id] = record
    return rag.build_upload_response(record)


@app.get("/documents/{chat_id}/file")
def document_file(chat_id: str):
    record = documents.get(chat_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"No document found for chat_id='{chat_id}'. "
                    "Upload a PDF via POST /upload first.",
        )

    file_path = record.get("saved_pdf_path") or record.get("file_path")
    if not file_path or not Path(file_path).exists():
        raise HTTPException(
            status_code=404,
            detail=f"No uploaded file found for chat_id='{chat_id}'.",
        )

    return FileResponse(
        str(file_path),
        media_type="application/pdf",
        filename=record.get("filename"),
        content_disposition_type="inline",
    )


@app.post("/chat")
def chat(request: ChatRequest):
    document = documents.get(request.chat_id)
    if document is None:
        raise HTTPException(
            status_code=404,
            detail=f"No document found for chat_id='{request.chat_id}'. "
                    "Upload a PDF via POST /upload first.",
        )

    try:
        result = rag.answer_chat_turn(document, request.message)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM service error: {e}")

    return {
        "answer": result["answer"],
        "citations": result["citations"],
        "sources": result["sources"],
    }
