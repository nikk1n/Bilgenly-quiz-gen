import asyncio
import os
import tempfile

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import io
import traceback
import lecture_parser as lp
import question_generator as gen

app = FastAPI(title="MCQ Generator API")


# --- Request/Response schemas ---
class TextRequest(BaseModel):
    text: str
    num_questions: int = 20

class PdfRequest(BaseModel):
    file: UploadFile = File(...)
    num_questions: int = 20

class MCQResponse(BaseModel):
    total_questions: int
    results: dict  # chunk_1: [...], chunk_2: [...], etc.


# --- Shared processing logic ---
def process_chunks(chunks: list[str], num_questions: int) -> JSONResponse:
    if not chunks:
        raise HTTPException(status_code=422, detail="No usable content extracted.")
    results = {}
    total_questions = 0
    counter = 1
    for chunk in chunks:
        batch_results = gen.generate_mcqs(chunk, num_questions=1)
        results[f"chunk_{counter}"] = batch_results
        total_questions += len(batch_results)
        counter += 1
    return JSONResponse(content={
        "total_questions": total_questions,
        "results": results,
    })


# --- Endpoints ---

@app.post("/generate/text", response_model=MCQResponse)
async def generate_from_text(request: TextRequest):
    """Accept raw plain text and return MCQs."""
    loop = asyncio.get_event_loop()
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty.")
    try:
        chunk_clusters=lp.extract_context_from_text(request.text)
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=422, detail=f"Could not parse text: {e}")
    # Run blocking model inference in a thread so the API stays responsive
    return await loop.run_in_executor(
        None,
        lambda: process_chunks(chunk_clusters, request.num_questions)
    )


@app.post("/generate/pdf", response_model=MCQResponse)
async def generate_from_pdf(file: UploadFile = File(...), num_questions: int = 20):
    """Accept a PDF file and return MCQs."""
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    contents = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name
    # Extract text from PDF
    try:
        chunk_clusters=lp.extract_context_from_pdf(pdf_file_path=tmp_path)
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=422, detail=f"Could not parse PDF: {e}")
    finally:
        os.unlink(tmp_path)

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: process_chunks(chunk_clusters, num_questions)
    )


@app.get("/health")
async def health():
    return {"status": "ok"}