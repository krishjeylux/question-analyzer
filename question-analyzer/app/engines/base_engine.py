from abc import ABC, abstractmethod
from app.models.domain_models import MarkingSchemeItem, QuestionEvaluationResult, CriterionResult
from typing import Any

class BaseEngine(ABC):
    @abstractmethod
    async def evaluate(self, question_id: str, student_answer: Any, marking_item: MarkingSchemeItem) -> QuestionEvaluationResult:
        pass
