"""
Generation Router
=================
Endpoints:
  POST /v1/generate-key              — single question text → marking scheme
  POST /v1/generate-from-paper       — PDF file upload → full paper marking scheme
  POST /v1/generate-from-paper-path  — server-side PDF path → full paper marking scheme
"""

import asyncio
import os
from typing import Dict, Any

from fastapi import APIRouter, Form, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from app.models.request_models import QuestionGenerationRequest, PaperGenerationRequest
from app.models.domain_models import GeneratedQuestionResult, PaperGenerationResult
from app.services.generator_service import QuestionGeneratorService
from app.services.paper_extractor import QuestionPaperExtractor
from app.services.answer_key_extractor import AnswerKeyExtractor

router = APIRouter(prefix="/v1", tags=["Generation"])

# Singleton service instances
generator_service = QuestionGeneratorService()
paper_extractor = QuestionPaperExtractor()
answer_key_extractor = AnswerKeyExtractor()

# ─────────────────────────────────────────────────────────────────────────────
# Concurrency control — max 5 simultaneous Groq/Gemini calls to avoid rate limits
# ─────────────────────────────────────────────────────────────────────────────
_SEMAPHORE = asyncio.Semaphore(5)


async def _generate_single_safe(q_id: str, text: str, subject: str, marks: int, official_answer: str = None) -> tuple[str, Any]:
    """
    Generate a marking scheme for one question, rate-limited by the semaphore.
    Returns (q_id, result_or_error_dict).
    """
    async with _SEMAPHORE:
        try:
            result = await generator_service.generate_key(
                question_text=text,
                subject=subject,
                total_marks=marks,
                official_answer=official_answer
            )
            return q_id, result.model_dump()
        except Exception as e:
            return q_id, {"error": str(e), "question_id": q_id}


async def _run_paper_pipeline(
    pdf_bytes: bytes,
    subject: str,
    answer_key_bytes: bytes = None
) -> PaperGenerationResult:
    """
    Core pipeline used by both upload and path-based endpoints.
    1. Extract questions from PDF (Vision OCR + Gemini)
    2. Extract answers from Answer Key (if provided)
    3. Generate marking schemes concurrently (rate-limited)
    4. Package into PaperGenerationResult
    """
    # ── Step 1: Extract questions ────────────────────────────────────────────
    questions = await paper_extractor.extract_questions(pdf_bytes, subject=subject)

    if not questions:
        raise ValueError("No questions could be extracted from the PDF.")

    # ── Step 2: Extract official answers (Optional) ──────────────────────────
    official_mapping = {}
    if answer_key_bytes:
        try:
            print("[Router] Extracting official mapping from Answer Key...")
            raw_mapping = await answer_key_extractor.extract_mapping(answer_key_bytes)
            # Clean: uppercase and strip
            official_mapping = {str(k): str(v).strip().upper() for k, v in raw_mapping.items()}
            print(f"[Router] Successfully extracted {len(official_mapping)} official answer(s).")
        except Exception as e:
            print(f"[Router] Warning: Answer key extraction failed: {e}")

    print(f"[Router] Generating marking schemes for {len(questions)} question(s)...")

    # ── Step 3: Generate schemes concurrently ────────────────────────────────
    tasks = [
        _generate_single_safe(
            q["id"], 
            q["text"], 
            subject, 
            q["marks"], 
            official_answer=official_mapping.get(q["id"])
        )
        for q in questions
    ]
    outcomes = await asyncio.gather(*tasks, return_exceptions=False)

    # ── Step 4: Collect results ──────────────────────────────────────────────
    results: Dict[str, Any] = {}
    failed: list[str] = []

    for q_id, outcome in outcomes:
        results[q_id] = outcome
        if isinstance(outcome, dict) and "error" in outcome:
            failed.append(q_id)

    return PaperGenerationResult(
        subject=subject,
        total_questions=len(questions),
        successful=len(questions) - len(failed),
        failed=len(failed),
        results=results,
        failed_questions=failed
    )


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 1 — existing single-question endpoint (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/generate-key", response_model=GeneratedQuestionResult)
async def generate_key(request: QuestionGenerationRequest):
    """
    Takes a raw string input of a question, rephrases it for clarity,
    and generates a full JSON marking scheme from scratch.
    """
    try:
        result = await generator_service.generate_key(
            question_text=request.question_text,
            subject=request.subject,
            total_marks=request.total_marks
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 2 — PDF file upload
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/generate-from-paper", response_model=PaperGenerationResult)
async def generate_from_paper_upload(
    file: UploadFile = File(..., description="Question paper PDF file"),
    answer_key: UploadFile = File(None, description="Optional official answer key PDF"),
    subject: str = Form(default="Grade 12 CBSE Physics", description="Subject name")
):
    """
    Upload a question paper PDF and an optional answer key.

    The pipeline will:
    1. OCR every page using Google Cloud Vision
    2. Extract official answers from the answer key (if provided)
    3. Parse questions from the paper
    4. Generate a full marking scheme for each question
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    pdf_bytes = await file.read()
    
    answer_key_bytes = None
    if answer_key:
        if not answer_key.filename.lower().endswith(".pdf"):
             raise HTTPException(status_code=400, detail="Answer key must be a PDF.")
        answer_key_bytes = await answer_key.read()

    try:
        result = await _run_paper_pipeline(pdf_bytes, subject, answer_key_bytes)
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 3 — server-side file path (useful when PDF is already on disk)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/generate-from-paper-path", response_model=PaperGenerationResult)
async def generate_from_paper_path(request: PaperGenerationRequest):
    """
    Process a question paper PDF that is already stored on the server.

    Provide the absolute path to the PDF file and the subject name.
    Returns a `PaperGenerationResult` with all generated marking schemes.

    Example request body:
    ```json
    {
        "paper_path": "C:/question-analyzer/papers/physics_2024.pdf",
        "subject": "Grade 12 CBSE Physics"
    }
    ```
    """
    if not os.path.exists(request.paper_path):
        raise HTTPException(
            status_code=404,
            detail=f"PDF not found at path: {request.paper_path}"
        )

    if not request.paper_path.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF.")

    try:
        with open(request.paper_path, "rb") as f:
            pdf_bytes = f.read()

        answer_key_bytes = None
        if request.answer_key_path:
            if os.path.exists(request.answer_key_path):
                with open(request.answer_key_path, "rb") as f:
                    answer_key_bytes = f.read()

        result = await _run_paper_pipeline(pdf_bytes, request.subject, answer_key_bytes)
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")
