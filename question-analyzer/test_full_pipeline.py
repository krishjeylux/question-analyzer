import asyncio
import os
import json
from app.services.paper_extractor import QuestionPaperExtractor
from app.api.generation_router import _generate_single_safe

async def test_pipeline():
    # 1. Path to the sample PDF
    pdf_path = r"C:\Users\krish\Downloads\cbse_12_physics_5511set1_2023_70.pdf"
    subject = "Grade 12 CBSE Physics"

    if not os.path.exists(pdf_path):
        print(f"[Test] ERROR: PDF not found at {pdf_path}")
        return

    print(f"[Test] Starting extraction for: {pdf_path}")
    extractor = QuestionPaperExtractor()

    # 2. Extract Questions
    try:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        print("[Test] Running extract_questions (this may take a minute)...")
        questions = await extractor.extract_questions(pdf_bytes, subject=subject)
        
        print(f"\n[Test] SUCCESS! Extracted {len(questions)} questions.")
        for q in questions[:3]:  # Show first 3
            print(f"  - {q['id']}: {q['text'][:100]}... ({q['marks']} marks)")
        
        if len(questions) > 3:
            print(f"  ... and {len(questions)-3} more.")

        # 3. Test a single marking scheme generation (optional, to verify LLM)
        if questions:
            print("\n[Test] Verifying marking scheme generation for the first question...")
            test_q = questions[0]
            q_id, result = await _generate_single_safe(
                test_q["id"], 
                test_q["text"], 
                subject, 
                test_q["marks"]
            )
            
            if "error" in result:
                print(f"[Test] Generation failed: {result['error']}")
            else:
                print(f"[Test] Generation SUCCESS for {q_id}!")
                print(f"  Rephrased: {result.get('rephrased_question', '')[:100]}...")

    except Exception as e:
        print(f"[Test] FATAL ERROR during pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_pipeline())
