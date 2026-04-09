from openai import AsyncOpenAI
from app.core.config import settings
from app.engines.base_engine import BaseEngine
from app.models.domain_models import (
    MarkingSchemeItem, QuestionEvaluationResult, CriterionResult, EvaluationStatus
)
from typing import Any, List
import json

class LLMEngine(BaseEngine):
    def __init__(self):
        # Using Groq API
        self.client = AsyncOpenAI(
            api_key=settings.GROK_API_KEY,
            base_url="https://api.groq.com/openai/v1"
        )

    async def evaluate(self, question_id: str, student_answer: str, marking_item: MarkingSchemeItem) -> QuestionEvaluationResult:
        prompt = self._build_prompt(question_id, student_answer, marking_item)
        
        try:
            response = await self.client.chat.completions.create(
                model="llama-3.3-70b-versatile", # Groq model
                messages=[
                    {"role": "system", "content": "You are an expert Physics teacher and examiner. Grade the student's answer based strictly on the provided marking scheme and evaluation criteria."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            result_data = json.loads(response.choices[0].message.content)
            
            return self._parse_llm_response(question_id, result_data, marking_item)
            
        except Exception as e:
            return QuestionEvaluationResult(
                question_id=question_id,
                status=EvaluationStatus.ERROR,
                total_marks=marking_item.allocated_marks,
                marks_obtained=0,
                detailed_feedback=f"LLM Evaluation failed: {str(e)}",
                criteria_results=[]
            )

    def _build_prompt(self, q_id: str, ans: str, item: MarkingSchemeItem) -> str:
        criteria_list = "\n".join([f"- {c}" for c in item.evaluation_criteria])
        
        prompt = f"""
Evaluate Question {q_id}.

[Expected Answer]
{json.dumps(item.expected_answer, indent=2)}

[Student Answer]
{ans}

[Evaluation Criteria]
{criteria_list}

[Instructions]
- For each criterion above, determine if it is met (true/false) and assign marks based on the parenthetical mark in the criterion (e.g., '(1)' means 1 mark).
- Provide concise feedback for each.
- Calculate total marks obtained.
- Output ONLY a JSON object in this format:
{{
  "status": "correct|partial|incorrect",
  "total_marks_obtained": float,
  "overall_feedback": "string",
  "criteria_results": [
    {{
      "criterion": "string",
      "status": boolean,
      "marks_awarded": float,
      "feedback": "string"
    }}
  ]
}}
"""
        return prompt

    def _parse_llm_response(self, q_id: str, data: dict, item: MarkingSchemeItem) -> QuestionEvaluationResult:
        criteria_results = [
            CriterionResult(**res) for res in data.get("criteria_results", [])
        ]
        
        # Ensure marks don't exceed allocated
        marks_obtained = min(data.get("total_marks_obtained", 0), item.allocated_marks)
        
        status_raw = data.get("status", "incorrect").lower()
        status = EvaluationStatus.INCORRECT
        if status_raw == "correct":
            status = EvaluationStatus.CORRECT
        elif status_raw == "partial":
            status = EvaluationStatus.PARTIAL

        return QuestionEvaluationResult(
            question_id=q_id,
            status=status,
            total_marks=item.allocated_marks,
            marks_obtained=marks_obtained,
            detailed_feedback=data.get("overall_feedback", ""),
            criteria_results=criteria_results
        )
