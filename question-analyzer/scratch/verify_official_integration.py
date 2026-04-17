import asyncio
import json
import os
from app.api.generation_router import _run_paper_pipeline

async def test_full_pipeline_with_key():
    paper_path = r"C:\Users\krish\Downloads\cbse_12_physics_5511set1_2023_70.pdf"
    answer_key_path = r"C:\Users\krish\Downloads\cbse_12_physics_5511set1_2023_70_answer.pdf"
    
    if not os.path.exists(paper_path):
        print(f"Error: Paper not found at {paper_path}")
        return
    if not os.path.exists(answer_key_path):
        print(f"Error: Answer Key not found at {answer_key_path}")
        return

    print("--- Starting Full Pipeline Test with Official Key ---")
    
    with open(paper_path, "rb") as f:
        pdf_bytes = f.read()
    with open(answer_key_path, "rb") as f:
        key_bytes = f.read()

    try:
        result = await _run_paper_pipeline(
            pdf_bytes=pdf_bytes,
            subject="Grade 12 CBSE Physics",
            answer_key_bytes=key_bytes
        )
        
        # Save results to a file for inspect
        output_file = "scratch/full_paper_results_with_key.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, indent=2)
            
        print(f"SUCCESS: Processed {result.total_questions} questions.")
        print(f"Results saved to {output_file}")
        
        # Spot check Q1
        if "Q1" in result.results:
            q1 = result.results["Q1"]
            print(f"Q1 Final Result: {q1['marking_scheme']['correct_option']}")
            print(f"Q1 Reasoning snippet: {q1['reasoning'][:200]}...")
            
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    asyncio.run(test_full_pipeline_with_key())
