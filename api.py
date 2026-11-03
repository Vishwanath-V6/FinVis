"""FastAPI service for document indexing and grounded questions."""

import asyncio
from pathlib import Path
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.inference import answer_question, load_qa_model
from src.preprocessing import chunk_pages, clean_pages, extract_text
from src.retrieval import RetrievalIndex

app = FastAPI(title="Financial RAG Research Assistant API")
index: RetrievalIndex | None = None
qa_models = None
index_lock = asyncio.Lock()
MAX_UPLOAD_SIZE = 50 * 1024 * 1024


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "indexed": index is not None}


@app.post("/index")
async def index_document(file: UploadFile = File(...)) -> dict[str, object]:
    global index
    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(status_code=415, detail="only PDF uploads are supported")
    content = bytearray()
    while len(content) <= MAX_UPLOAD_SIZE:
        chunk = await file.read(min(1024 * 1024, MAX_UPLOAD_SIZE + 1 - len(content)))
        if not chunk:
            break
        content.extend(chunk)
    if not content or len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="PDF must be non-empty and at most 50 MB")
    if bytes(content[:5]) != b"%PDF-":
        raise HTTPException(status_code=415, detail="uploaded content is not a valid PDF")
    safe_name = Path(file.filename or "document.pdf").name
    if not safe_name.lower().endswith(".pdf"):
        safe_name = f"{safe_name}.pdf"
    try:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "upload.pdf"
            path.write_bytes(content)
            pages = clean_pages(extract_text(path, document_id=safe_name))
            chunks = chunk_pages(pages)
            new_index = RetrievalIndex().build(chunks)
        async with index_lock:
            index = new_index
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"indexed_chunks": len(chunks), "document": safe_name}


@app.post("/ask")
def ask(request: QuestionRequest) -> dict[str, object]:
    global qa_models
    current_index = index
    if current_index is None:
        raise HTTPException(status_code=409, detail="index a PDF before asking a question")
    try:
        if qa_models is None:
            qa_models = load_qa_model()
        return answer_question(request.question, current_index, *qa_models, top_k=request.top_k)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="QA model is unavailable") from exc
