import asyncio
import json
from app.services.generator_service import QuestionGeneratorService

async def verify():
    print("Initializing final QuestionGeneratorService...")
    service = QuestionGeneratorService()
    
    sample_question = "A wire of resistance 8 R is bent in the form of a circle. What is the effective resistance between the ends of a diameter AB?"
    print(f"\n[Raw Input]: {sample_question}\n")
    print("Generating marking scheme... This may take a moment.")
    
    res = await service.generate_key(
        question_text=sample_question,
        subject="Grade 12 CBSE Physics",
        total_marks=2
    )
    
    print("\n--- Generation Success ---")
    print(f"Rephrased Question: {res.rephrased_question}")
    print("\nMarking Scheme Generated:")
    print(json.dumps(res.marking_scheme.model_dump(), indent=2))

if __name__ == "__main__":
    asyncio.run(verify())
