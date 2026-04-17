import asyncio
import json
from app.services.generator_service import QuestionGeneratorService

async def verify_numerical():
    # A sample numerical from a typical physics paper
    question = "A wire of resistance 8R is bent into a circle. What is the effective resistance between the ends of a diameter?"
    subject = "Grade 12 CBSE Physics"
    
    print(f"Generating answer for: {question}")
    service = QuestionGeneratorService()
    result = await service.generate_key(question, subject, total_marks=2)
    
    print("\n--- REASONING ---")
    print(result.reasoning)
    print("\n--- MARKING SCHEME ---")
    print(json.dumps(result.marking_scheme.dict(), indent=2))

if __name__ == "__main__":
    asyncio.run(verify_numerical())
