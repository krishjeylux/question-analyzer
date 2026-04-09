from enum import Enum
from typing import List, Dict, Union, Optional, Any
from pydantic import BaseModel, Field

class QuestionType(str, Enum):
    TEXT = "text"
    EQUATION = "equation"
    NUMERIC = "numeric"
    MIXED = "mixed"
    TABLE = "table"
    DIAGRAM = "diagram"

class MarkingSchemeItem(BaseModel):
    type: QuestionType
    question_note: str = ""
    allocated_marks: float
    correct_option: Optional[str] = None
    expected_answer: Any
    evaluation_criteria: List[str]
    criteria_status: str = "defined"
    reference_diagrams: Optional[List[str]] = None
    expected_table: Optional[Any] = None

class PaperMarkingScheme(BaseModel):
    questions: Dict[str, MarkingSchemeItem]

class EvaluationStatus(str, Enum):
    CORRECT = "correct"
    PARTIAL = "partial"
    INCORRECT = "incorrect"
    ERROR = "error"

class CriterionResult(BaseModel):
    criterion: str
    status: bool
    marks_awarded: float
    feedback: str

class QuestionEvaluationResult(BaseModel):
    question_id: str
    status: EvaluationStatus
    total_marks: float
    marks_obtained: float
    detailed_feedback: str
    criteria_results: List[CriterionResult]
    alternative_results: Optional[Dict[str, 'QuestionEvaluationResult']] = None

class GeneratedQuestionResult(BaseModel):
    original_question: str
    rephrased_question: str
    reasoning: str
    marking_scheme: MarkingSchemeItem

# For recursion support in alternative_results
QuestionEvaluationResult.model_rebuild()
