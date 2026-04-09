import json
import os
from typing import Dict, Any, List
from app.core.config import settings
from app.models.domain_models import (
    PaperMarkingScheme, MarkingSchemeItem, QuestionEvaluationResult, EvaluationStatus, QuestionType
)
from app.engines.deterministic_engine import DeterministicEngine
from app.engines.llm_engine import LLMEngine

class EvaluatorService:
    def __init__(self):
        self.marking_scheme: Dict[str, MarkingSchemeItem] = {}
        self.deterministic_engine = DeterministicEngine()
        self.llm_engine = LLMEngine()
        self._load_marking_scheme()

    def _load_marking_scheme(self):
        if os.path.exists(settings.PHYSICS_PAPER_PATH):
            with open(settings.PHYSICS_PAPER_PATH, "r") as f:
                data = json.load(f)
                for q_id, q_data in data.items():
                    self.marking_scheme[q_id] = MarkingSchemeItem(**q_data)

    async def evaluate_answer(self, question_id: str, student_answer: Any) -> QuestionEvaluationResult:
        # Check for alternative version in marking scheme (e.g. Q21_A, Q21_B)
        # If user asks for Q21, and we have Q21_A and Q21_B, we should evaluate both.
        
        # Scenario 1: Exact match (e.g. Q1, Q19, Q21_A)
        if question_id in self.marking_scheme:
            return await self._run_evaluation(question_id, student_answer, self.marking_scheme[question_id])
        
        # Scenario 2: Question has alternatives (e.g. user sends "Q21")
        # Find all keys starting with {question_id}_
        alt_keys = [k for k in self.marking_scheme.keys() if k.startswith(f"{question_id}_")]
        if alt_keys:
            results = {}
            for k in alt_keys:
                results[k] = await self._run_evaluation(k, student_answer, self.marking_scheme[k])
            
            # Return a "Summary" result with alternatives nested
            return QuestionEvaluationResult(
                question_id=question_id,
                status=EvaluationStatus.PARTIAL, # Placeholder
                total_marks=results[alt_keys[0]].total_marks,
                marks_obtained=max(r.marks_obtained for r in results.values()),
                detailed_feedback=f"Evaluated against alternatives: {', '.join(alt_keys)}",
                criteria_results=[],
                alternative_results=results
            )

        return QuestionEvaluationResult(
            question_id=question_id,
            status=EvaluationStatus.ERROR,
            total_marks=0,
            marks_obtained=0,
            detailed_feedback=f"Question ID {question_id} not found in marking scheme.",
            criteria_results=[]
        )

    async def _run_evaluation(self, q_id: str, ans: Any, item: MarkingSchemeItem) -> QuestionEvaluationResult:
        # Routing logic
        # 1. Deterministic First for MCQs/Equations/Numerics
        if item.type in [QuestionType.EQUATION, QuestionType.NUMERIC] or item.correct_option:
            res = await self.deterministic_engine.evaluate(q_id, ans, item)
            if res.status == EvaluationStatus.CORRECT:
                return res
            # If deterministic failed or was incorrect, and type is NUMERIC/EQUATION, 
            # we might want to let LLM check for partial marks if it's mixed or complex.
            # But for straight MCQs, deterministic is final.
            if item.correct_option:
                return res

        # 2. LLM for Text, Mixed, or fallback for partial evaluation
        return await self.llm_engine.evaluate(q_id, ans, item)

    def get_marking_scheme_summary(self) -> List[str]:
        return list(self.marking_scheme.keys())
