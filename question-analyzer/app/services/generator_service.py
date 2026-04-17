import json
import asyncio
from typing import Optional
import google.generativeai as genai
from app.core.config import settings
from app.models.domain_models import GeneratedQuestionResult, MarkingSchemeItem
from app.retrieval.qdrant_service import QdrantService

class QuestionGeneratorService:
    def __init__(self):
        # Configure Gemini
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        self.qdrant = QdrantService()
        
    async def generate_key(self, question_text: str, subject: str, total_marks: int = None, official_answer: Optional[str] = None) -> GeneratedQuestionResult:
        marks_str = f"The question is worth {total_marks} marks." if total_marks else "Assign appropriate marks based on the complexity."
        
        override_instruction = ""
        if official_answer:
            override_instruction = f"\nCRITICAL OVERRIDE: The official correct answer for this question is: '{official_answer}'. Your reasoning MUST lead directly to this answer. Do NOT contradict the official answer."

        # 1. Retrieve RAG Context (Increased top_k for better coverage)
        retrieved_docs = self.qdrant.search_similar(question_text, top_k=10)
        
        context_str = "No relevant context found in textbook corpus."
        if retrieved_docs:
            context_blocks = []
            for doc in retrieved_docs:
                block = f"[Source: {doc['source']}, Page: {doc['page']}]\n{doc['content']}"
                context_blocks.append(block)
            context_str = "\n\n".join(context_blocks)

        prompt = f"""
        You are an expert {subject} teacher and examiner. Your goal is to provide a 100% accurate, textbook-aligned answer key and marking scheme.
        
        [TASKS]
        1. Parse and understand the [RAW INPUT QUESTION].
        2. Rephrase the question for maximum clarity while keeping the EXACT academic meaning.
        3. Solve the question step-by-step.
        4. CRITICAL: For physical constants (like ε₀, μ₀, e, h, c) or specific formulas, refer ONLY to the [VERIFIED GROUND TRUTH CONTEXT] below. If the context is missing a specific constant, use standard CBSE values but document it in your reasoning.
        5. For MCQs, you MUST evaluate each option. If the [RAW INPUT QUESTION] has corrupted option text (due to OCR), reconstruct the intended options based on your physics calculation and pick the most appropriate letter (A, B, C, or D).
        6. CRITICAL: Maintain a scientific "Fact-Check" layer. If the RAG context is silent on a basic property (e.g., whether a material is diamagnetic), use verified scientific knowledge rather than guessing.
        
        {marks_str}
        {override_instruction}
        
        [VERIFIED GROUND TRUTH CONTEXT FROM TEXTBOOK]
        {context_str}
        
        [RAW INPUT QUESTION]
        {question_text}
        
        [INSTRUCTIONS FOR YOUR REASONING SCRATCHPAD]
        - First, identify the core physics concept.
        - List the knowns and unknowns.
        - Explicitly write down the formulas to be used.
        - Perform all calculations with high precision. Show unit conversions (e.g., cm to m).
        - DOUBLE-CHECK your math. If it's a numerical, solve it twice to verify.
        - For MCQs, explicitly state why the correct option is right and others are wrong.
        - MAPPING CHECK: Look at the raw OCR options. OCR often Misreads: '8' as 'ri', '4' as 'A', '2' as 'r', '1' as 'l' or 'i', '0' as 'o' or 'O', 'ε' as 'E'. If your calculated answer is 'F/8' and an option is 'ri', pick that option.
        - OPTICS GUARDRAIL: For a concave mirror, if the object is at the Center of Curvature (u=2f), the real image also forms at 2f. In this case, the distance BETWEEN the object and its image is ZERO. Do NOT mistake this for 4f (which is for lenses).
        
        [OUTPUT FORMAT]
        Output ONLY a JSON object. Ensure ALL LaTeX backslashes are double-escaped (e.g., \\\\frac, \\\\lambda).
        
        {{
            "rephrased_question": "Full rephrased question, keeping all sub-parts (a, b, i, ii) intact.",
            "reasoning": "Detailed step-by-step physics analysis for ALL sub-parts, knowns/unknowns, formula selection, and math verification.",
            "marking_scheme": {{
                "type": "text | equation | numeric | mixed | table | diagram | mcq",
                "question_note": "Mention specific constants used or assumptions made. If there are multiple parts, summarize the overall difficulty.",
                "allocated_marks": float,
                "correct_option": "A/B/C/D if MCQ, else empty",
                "expected_answer": {{
                    "Part (a) / (i)": "$...$",
                    "Part (b) / (ii)": "$...$",
                    "Part (c) [Choice 1] (if applicable)": "$...$",
                    "Part (c) [Choice 2] (if applicable)": "$...$",
                    "Final_Result": "$...$"
                }},
                "evaluation_criteria": [
                    "CRITICAL: This MUST be a flat list of strings. DO NOT nest dictionaries here.",
                    "Part (a): ... (Marks)",
                    "Part (b): ... (Marks)",
                    "Part (c) [Choice 1]: ... (Marks)",
                    "--- OR ---",
                    "Part (c) [Choice 2]: ... (Marks)"
                ],
                "criteria_status": "defined"
            }}
        }}
        [ASSERTION-REASON RULES]
        If the question is an "Assertion (A) - Reason (R)" type, you MUST use these standard options:
        (A) Both Assertion (A) and Reason (R) are true and Reason (R) is the correct explanation of Assertion (A).
        (B) Both Assertion (A) and Reason (R) are true but Reason (R) is NOT the correct explanation of Assertion (A). (Use this if R is a true statement but not the causal reason for A).
        (C) Assertion (A) is true but Reason (R) is false.
        (D) Assertion (A) is false and Reason (R) is false. (Use this if both statements are fundamentally scientifically incorrect).
        
        [REASONING COMPLIANCE CHECK]
        Before generating the final JSON, perform a MENTAL CHECK:
        - Does your `correct_option` match your math result?
        - Is your LaTeX perfectly escaped with DOUBLE backslashes (\\\\)?
        - For MCQs, is the selected option the best one based on the context?
        
        Return ONLY the JSON object.
        """

        try:
            # Gemini generation (synchronous SDK call, wrap in executor if needed for high load)
            # For this context, we'll run it directly as it's an async method
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: self.model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.1,
                        "response_mime_type": "application/json",
                    },
                    request_options={"timeout": 120}
                )
            )
            
            raw_text = response.text.strip()
            
            # Robust JSON parsing
            if "```json" in raw_text:
                raw_text = raw_text.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_text:
                raw_text = raw_text.split("```")[1].split("```")[0].strip()

            cleaned_text = self._clean_json_string(raw_text)
            try:
                result_data = json.loads(cleaned_text, strict=False)
            except json.JSONDecodeError as je:
                print(f"[Generator] JSON Decode Error: {je}")
                # Save to file for inspection
                with open("scratch/last_error_raw.txt", "w", encoding="utf-8") as f:
                    f.write(raw_text)
                with open("scratch/last_error_cleaned.txt", "w", encoding="utf-8") as f:
                    f.write(cleaned_text)
                raise je
            
            marking_item = MarkingSchemeItem(**result_data["marking_scheme"])
            
            return GeneratedQuestionResult(
                original_question=question_text,
                rephrased_question=result_data["rephrased_question"],
                reasoning=result_data["reasoning"],
                marking_scheme=marking_item
            )
            
        except Exception as e:
            raise Exception(f"Failed to generate key using Gemini: {str(e)}")

    def _clean_json_string(self, s: str) -> str:
        """
        Attempts to fix common LLM-generated JSON errors.
        """
        import re
        # 1. Double backslashes that are NOT preceded by a backslash AND NOT followed by a backslash or quote
        # This handles LaTeX like \lambda -> \\lambda
        s = re.sub(r'(?<!\\)\\(?![\\"])', r'\\\\', s)
        
        # 2. Fix literal newlines: if a string value contains a raw newline, escape it to \n
        # We only apply this to content between quotes. This is a heuristic.
        def replace_newline(match):
            return match.group(0).replace('\n', '\\n').replace('\r', '\\n')
        
        # Match "key": "value" where value has line breaks
        s = re.sub(r'":\s*"[^"]*?"', replace_newline, s, flags=re.DOTALL)
        
        # 3. Remove literal control characters (except \n, \r, \t)
        s = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', s)
        
        return s
