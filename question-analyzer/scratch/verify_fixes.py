import asyncio
import json
from app.services.generator_service import QuestionGeneratorService

async def verify_q7():
    print("\n--- Verifying Q7 (Option Selection) ---")
    # Q7 from user's logs: A circular coil ... maximum emf induced is : (a) 0.12V (b) 0.15 V (c) 0.19 V (d) 0.22V
    # User said: "computed answer was correct but chosen option was wrong it chose D instead of C"
    # Actually, in the logs provided, Q6 was 0.22V and chose D. Q7 error was validation.
    # Let's try Q7 text if available, or just a generic MCQ where calculation leads to C.
    question = "A wire of resistance 10 Ohm is stretched to double its length. Its new resistance will be: (a) 10 Ohm (b) 20 Ohm (c) 40 Ohm (d) 5 Ohm"
    subject = "Grade 12 CBSE Physics"
    service = QuestionGeneratorService()
    result = await service.generate_key(question, subject, total_marks=1)
    print(f"Rephrased: {result.rephrased_question.encode('ascii', 'ignore').decode()}")
    # Use encode/decode to avoid character map errors in Windows terminal
    print(f"Reasoning snippet: {result.reasoning[:100].encode('ascii', 'ignore').decode()}...")
    print(f"Type: {result.marking_scheme.type}")
    print(f"Correct Option: {result.marking_scheme.correct_option}")
    assert result.marking_scheme.type == "mcq"
    assert "C" in result.marking_scheme.correct_option.upper()

async def verify_q31_json():
    print("\n--- Verifying Q31 (JSON Escaping) ---")
    # Simulate a response with bad LaTeX escaping
    service = QuestionGeneratorService()
    bad_json = '{"rephrased_question": "...", "reasoning": "...", "marking_scheme": {"type": "equation", "allocated_marks": 1, "expected_answer": {"formula": "\\lambda = h/p"}, "evaluation_criteria": []}}'
    # The \l in \lambda is an invalid escape sequence in JSON. It should be \\lambda.
    cleaned = service._clean_json_string(bad_json)
    print(f"Original: {bad_json}")
    print(f"Cleaned:  {cleaned}")
    # Try to load it
    data = json.loads(cleaned)
    print("Sucessfully parsed cleaned JSON!")
    assert "\\lambda" in data["marking_scheme"]["expected_answer"]["formula"]

async def main():
    try:
        await verify_q7()
        await verify_q31_json()
        print("\nALL FIXES VERIFIED!")
    except Exception as e:
        print(f"\nVERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
