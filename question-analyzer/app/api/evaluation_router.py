from fastapi import APIRouter, HTTPException, Depends
from app.models.request_models import EvaluationRequest
from app.models.domain_models import QuestionEvaluationResult
from app.services.evaluator_service import EvaluatorService
from typing import Dict, List

router = APIRouter(prefix="/v1", tags=["Evaluation"])

# Singleton instance for the demo
evaluator_service = EvaluatorService()

@router.post("/evaluate", response_model=Dict[str, QuestionEvaluationResult])
async def evaluate_answers(request: EvaluationRequest):
    """
    Evaluates a set of student answers against the marking scheme.
    Returns a mapping of question_id to evaluation results.
    """
    results = {}
    for q_id, answer in request.answers.items():
        try:
            results[q_id] = await evaluator_service.evaluate_answer(q_id, answer)
        except Exception as e:
            results[q_id] = {
                "question_id": q_id,
                "status": "error",
                "detailed_feedback": f"Internal error during evaluation: {str(e)}"
            }
    return results

@router.get("/questions", response_model=List[str])
async def get_available_questions():
    """Returns a list of all question IDs in the marking scheme."""
    return evaluator_service.get_marking_scheme_summary()

@router.get("/questions/{question_id}")
async def get_marking_item(question_id: str):
    """Returns the marking scheme details for a specific question."""
    if question_id not in evaluator_service.marking_scheme:
        raise HTTPException(status_code=404, detail="Question not found")
    return evaluator_service.marking_scheme[question_id]
