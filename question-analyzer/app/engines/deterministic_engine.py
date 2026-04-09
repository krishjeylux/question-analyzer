import sympy
from sympy.parsing.latex import parse_latex
from app.engines.base_engine import BaseEngine
from app.models.domain_models import (
    MarkingSchemeItem, QuestionEvaluationResult, CriterionResult, EvaluationStatus, QuestionType
)
import re

class DeterministicEngine(BaseEngine):
    async def evaluate(self, question_id: str, student_answer: str, marking_item: MarkingSchemeItem) -> QuestionEvaluationResult:
        if marking_item.type == QuestionType.TEXT:
            # Handle MCQ string matching first if text is just an option
            if marking_item.correct_option:
                return self._evaluate_mcq(question_id, student_answer, marking_item)
        
        if marking_item.type in [QuestionType.EQUATION, QuestionType.NUMERIC]:
            return self._evaluate_math(question_id, student_answer, marking_item)
            
        # Fallback for now
        return QuestionEvaluationResult(
            question_id=question_id,
            status=EvaluationStatus.PARTIAL,
            total_marks=marking_item.allocated_marks,
            marks_obtained=0,
            detailed_feedback="Deterministic evaluation not fully supported for this type.",
            criteria_results=[]
        )

    def _evaluate_mcq(self, question_id: str, student_answer: str, marking_item: MarkingSchemeItem) -> QuestionEvaluationResult:
        student_ans_clean = str(student_answer).strip().upper()
        correct_ans = str(marking_item.correct_option).strip().upper()
        
        # Simple match for MCQ option
        if student_ans_clean == correct_ans or f"OPTION {correct_ans}" in student_ans_clean:
            return QuestionEvaluationResult(
                question_id=question_id,
                status=EvaluationStatus.CORRECT,
                total_marks=marking_item.allocated_marks,
                marks_obtained=marking_item.allocated_marks,
                detailed_feedback="Correct option identified.",
                criteria_results=[
                    CriterionResult(
                        criterion=f"Correct option {correct_ans}",
                        status=True,
                        marks_awarded=marking_item.allocated_marks,
                        feedback="Matches marking scheme."
                    )
                ]
            )
        
        return QuestionEvaluationResult(
            question_id=question_id,
            status=EvaluationStatus.INCORRECT,
            total_marks=marking_item.allocated_marks,
            marks_obtained=0,
            detailed_feedback=f"Incorrect. Expected {correct_ans}.",
            criteria_results=[
                CriterionResult(
                    criterion=f"Correct option {correct_ans}",
                    status=False,
                    marks_awarded=0,
                    feedback="Does not match."
                )
            ]
        )

    def _evaluate_math(self, question_id: str, student_answer: str, marking_item: MarkingSchemeItem) -> QuestionEvaluationResult:
        try:
            # Basic LaTeX cleaning
            def clean_latex(lat):
                return lat.replace("$", "").strip()

            expected_raw = marking_item.expected_answer
            if isinstance(expected_raw, dict):
                # Handle complex expected answers if needed, for now use first if string
                expected_raw = str(next(iter(expected_raw.values())))
            
            expr_student = parse_latex(clean_latex(student_answer))
            expr_expected = parse_latex(clean_latex(expected_raw))
            
            # Check for symbolic equality
            if (expr_student - expr_expected).simplify() == 0:
                return QuestionEvaluationResult(
                    question_id=question_id,
                    status=EvaluationStatus.CORRECT,
                    total_marks=marking_item.allocated_marks,
                    marks_obtained=marking_item.allocated_marks,
                    detailed_feedback="Mathematically equivalent answer.",
                    criteria_results=[
                        CriterionResult(
                            criterion=marking_item.evaluation_criteria[0],
                            status=True,
                            marks_awarded=marking_item.allocated_marks,
                            feedback="Symbolic match confirmed via SymPy."
                        )
                    ]
                )
        except Exception as e:
            # Fallback to string normalized match if parsing fails
            if clean_latex(student_answer).replace(" ", "") == clean_latex(str(marking_item.expected_answer)).replace(" ", ""):
                 return QuestionEvaluationResult(
                    question_id=question_id,
                    status=EvaluationStatus.CORRECT,
                    total_marks=marking_item.allocated_marks,
                    marks_obtained=marking_item.allocated_marks,
                    detailed_feedback="String-match match confirmed.",
                    criteria_results=[
                        CriterionResult(
                            criterion=marking_item.evaluation_criteria[0],
                            status=True,
                            marks_awarded=marking_item.allocated_marks,
                            feedback="Exact string match."
                        )
                    ]
                )

        return QuestionEvaluationResult(
            question_id=question_id,
            status=EvaluationStatus.INCORRECT,
            total_marks=marking_item.allocated_marks,
            marks_obtained=0,
            detailed_feedback="Answer does not match expected mathematical expression.",
            criteria_results=[]
        )
