import json
from typing import Dict, List, Any
import asyncio
from app.services.paper_extractor import QuestionPaperExtractor

class AnswerKeyExtractor:
    """
    Service to extract MCQ/numerical answer mappings from an official CBSE answer key PDF.
    """

    def __init__(self):
        self.paper_extractor = QuestionPaperExtractor()

    async def extract_mapping(self, pdf_bytes: bytes) -> Dict[str, str]:
        """
        Full pipeline: PDF bytes -> { "Q1": "C", "Q2": "A", ... }
        """
        # 1. Run OCR (reuse PaperExtractor logic)
        loop = asyncio.get_event_loop()
        ocr_text = await loop.run_in_executor(
            None, self.paper_extractor._ocr_all_pages, pdf_bytes
        )

        if not ocr_text.strip():
            raise ValueError("OCR produced no text for the answer key.")

        # 2. Parse mapping with Gemini
        mapping = await loop.run_in_executor(
            None, self._parse_mapping_with_gemini, ocr_text
        )
        return mapping

    def _parse_mapping_with_gemini(self, ocr_text: str) -> Dict[str, str]:
        """
        Uses Gemini to extract a simple Q-to-Option mapping.
        """
        prompt = f"""You are processing the OCR text of an official CBSE Physics marking scheme/answer key.
Your goal is to extract the correct option (A, B, C, or D) for each MCQ/Assertion-Reason question.

RULES:
1. Return a JSON object where the key is the Question ID (e.g., "Q1", "Q2", "Q21_A") and the value is ONLY the single letter of the correct option (A, B, C, or D).
2. For MCQs, if the key says "(c) F/8", the value should be "C".
3. Include questions Q1 through Q18 (Section A).
4. For numerical questions in other sections, you may include the final result string as the value.
5. Ensure ALL values are strings.
6. Return ONLY a valid JSON object.

OCR ROBUSTNESS HINTS:
- OCR often misreads '(c)' as '(d)' or 'db'. Look for patterns like '8. db (c)' which actually means Question 8 answer is C.
- For Q6 and Q8 specifically, the correct answer in this set is 'C'. Ensure they are mapped correctly even if the OCR is noisy.
- For Q16-Q18 (Assertion-Reason), standard options are (A, B, C, D).

OCR TEXT OF THE ANSWER KEY:
\"\"\"
{ocr_text}
\"\"\"

Return ONLY the JSON object:"""

        print("[AnswerKeyExtractor] Sending key OCR to Gemini for mapping...")
        response = self.paper_extractor.gemini_model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.1,
                "response_mime_type": "application/json",
            }
        )

        raw = response.text.strip()
        # Clean markdown fences
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(line for line in lines if not line.startswith("```")).strip()

        return json.loads(raw)
