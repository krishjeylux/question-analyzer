from pydantic import BaseModel
from typing import Dict, Any, Optional

class StudentAnswerRequest(BaseModel):
    # Mapping of question_id to student's answer text/latex/etc
    # e.g., {"Q1": "B", "Q19": "Force is..."}
    answers: Dict[str, Any]
    
class EvaluationRequest(BaseModel):
    student_id: Optional[str] = "default_student"
    paper_id: str = "physics_paper_1"
    answers: Dict[str, Any]

class QuestionGenerationRequest(BaseModel):
    question_text: str
    subject: str = "Grade 12 CBSE Physics"
    total_marks: Optional[int] = None
