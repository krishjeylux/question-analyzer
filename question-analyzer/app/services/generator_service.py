import json
from openai import AsyncOpenAI
from app.core.config import settings
from app.models.domain_models import GeneratedQuestionResult, MarkingSchemeItem
from app.retrieval.qdrant_service import QdrantService

class QuestionGeneratorService:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.GROK_API_KEY,
            base_url="https://api.groq.com/openai/v1"
        )
        self.qdrant = QdrantService()
        
    async def generate_key(self, question_text: str, subject: str, total_marks: int = None) -> GeneratedQuestionResult:
        marks_str = f"The question is worth {total_marks} marks." if total_marks else "Assign appropriate marks based on the complexity."
        
        # 1. Retrieve RAG Context
        retrieved_docs = self.qdrant.search_similar(question_text, top_k=3)
        
        context_str = "No relevant context found in textbook corpus."
        if retrieved_docs:
            context_blocks = []
            print(f"\n--- [RAG RETRIEVAL SUCCESS] Found {len(retrieved_docs)} textbook paragraphs! ---")
            for doc in retrieved_docs:
                block = f"[Source: {doc['source']}, Page: {doc['page']}]\n{doc['content']}"
                context_blocks.append(block)
                print(f"\nRetrieved from {doc['source']} (Page {doc['page']}):\n{doc['content'][:150]}...")
            print("------------------------------------------------------\n")
            context_str = "\n\n".join(context_blocks)

        prompt = f"""
        You are an expert teacher setting a {subject} paper.
        
        I have a raw question input. I want you to:
        1. Understand the question text.
        2. Rephrase the question to be perfectly clear and academically sound for {subject}, while keeping the EXACT meaning.
        3. Solve the question and generate a comprehensive marking scheme for it.
        4. CRITICAL: Base your physics formulas and reasoning STRICTLY on the Verified Ground Truth Context provided below. Do not hallucinate physics concepts outside of typical CBSE framework.
        {marks_str}
        
        [VERIFIED GROUND TRUTH CONTEXT FROM TEXTBOOK]
        {context_str}
        
        [RAW INPUT QUESTION]
        {question_text}
        
        [INSTRUCTIONS]
        Output ONLY a JSON object that strictly follows this JSON schema. The "reasoning" key MUST come first so you can think step-by-step before finalizing the answer.
        
        CRITICAL FORMATTING RULES:
        1. Use LaTeX for ALL mathematical formulas, units, and calculations, wrapped in single dollar signs (e.g., $F = ma$, $ms^{-2}$).
        2. In the "expected_answer" dictionary, break down the solution into logical steps. Use descriptive keys like "Formula", "Calculation", "Final_Result", or specific names like "Force_formula", "Acceleration_calculation".
        3. The "evaluation_criteria" MUST be a list of strings following this EXACT format: "KeyName) LaTeX_Content (Marks)". "KeyName" must correspond to a key in your "expected_answer" dictionary.
        4. CRITICAL: Since your output must be valid JSON, ensure ALL backslashes in math formulas or LaTeX (like \lambda, \frac) are double-escaped (e.g., \\lambda, \\frac) to prevent JSON parse errors.
        5. You MUST double-check your arithmetic! Always convert decimals to fractions for calculations in your reasoning scratchpad.
        
        {{
            "rephrased_question": "String containing the clear, rephrased version of the question.",
            "reasoning": "Think step-by-step. Analyze the physics involved, explicitly write down formulas, and double-check numerical math.",
            "marking_scheme": {{
                "type": "text",
                "question_note": "Any notes or alternatives",
                "allocated_marks": float,
                "correct_option": "If it is an MCQ, string representing the correct option (e.g. 'A', 'B'), else empty string",
                "expected_answer": {{
                    "Step1_Name": "$LaTeX_Formula_or_Step$",
                    "Step2_Name": "$LaTeX_Calculation$"
                }},
                "evaluation_criteria": [
                    "Step1_Name) $LaTeX_Formula_or_Step$ (Marks)",
                    "Step2_Name) $LaTeX_Calculation$ (Marks)"
                ],
                "criteria_status": "defined"
            }}
        }}

        """

        try:
            response = await self.client.chat.completions.create(
                model="llama-3.3-70b-versatile", 
                messages=[
                    {"role": "system", "content": "You are a backend JSON output generator. You must analyze the physics carefully in the reasoning block before producing JSON. Output only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            result_data = json.loads(response.choices[0].message.content)
            
            marking_item = MarkingSchemeItem(**result_data["marking_scheme"])
            
            return GeneratedQuestionResult(
                original_question=question_text,
                rephrased_question=result_data["rephrased_question"],
                reasoning=result_data["reasoning"],
                marking_scheme=marking_item
            )

            
        except Exception as e:
            raise Exception(f"Failed to generate key using LLM: {str(e)}")
