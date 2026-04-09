from fastapi import APIRouter, HTTPException
from app.models.request_models import QuestionGenerationRequest
from app.models.domain_models import GeneratedQuestionResult
from app.services.generator_service import QuestionGeneratorService

router = APIRouter(prefix="/v1", tags=["Generation"])

generator_service = QuestionGeneratorService()

@router.post("/generate-key", response_model=GeneratedQuestionResult)
async def generate_key(request: QuestionGenerationRequest):
    """
    Takes a raw string input of a question, rephrases it for clarity,
    and generates a full json marking scheme from scratch.
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
