from enum import Enum
from typing import List, Dict, Union, Optional, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict

class QuestionType(str, Enum):
    TEXT = "text"
    EQUATION = "equation"
    NUMERIC = "numeric"
    MIXED = "mixed"
    TABLE = "table"
    DIAGRAM = "diagram"
    MCQ = "mcq"

# Map common LLM-returned variants to valid enum values
_TYPE_ALIASES = {
    "numerical": "numeric",
    "number": "numeric",
    "textual": "text",
    "string": "text",
    "math": "equation",
    "formula": "equation",
    "chart": "diagram",
    "image": "diagram",
    "tabular": "table",
    "mcq": "mcq",
    "multiple choice": "mcq",
    "multiple_choice": "mcq",
}

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

    @field_validator("type", mode="before")
    @classmethod
    def normalize_type_field(cls, v: Any) -> Any:
        if isinstance(v, str):
            v_lower = v.strip().lower()
            return _TYPE_ALIASES.get(v_lower, v_lower)
        return v

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


class ExtractedQuestion(BaseModel):
    """A single question extracted from a question paper PDF."""
    id: str
    text: str
    marks: int


class PaperGenerationResult(BaseModel):
    """Result of generating marking schemes for an entire question paper."""
    subject: str
    total_questions: int
    successful: int
    failed: int
    results: Dict[str, Any]   # question_id -> GeneratedQuestionResult or error dict
    failed_questions: List[str] = []
