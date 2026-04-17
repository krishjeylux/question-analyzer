import asyncio
from app.services.generator_service import QuestionGeneratorService

async def test_q1_mapping():
    print("\n--- Testing Q1 (OCR Mapping) ---")
    question = """A point charge situated at a distance 'r' from a short electric dipole on its axis, experiences a force F. If the distance of the charge is '2r', the force on the charge will be: (a) 16 (b) 3 (c) ri (d) 5"""
    service = QuestionGeneratorService()
    result = await service.generate_key(question, "Grade 12 CBSE Physics", total_marks=1)
    print(f"Reasoning: {result.reasoning.encode('ascii', 'ignore').decode('ascii')}")
    print(f"Correct Option: {result.marking_scheme.correct_option}")

async def test_q9_optics():
    print("\n--- Testing Q9 (Optics Logic) ---")
    question = "For a concave mirror of focal length 'f', the minimum distance between the object and its real image is : (a) zero (b) f (c) 2f (d) 4f"
    service = QuestionGeneratorService()
    result = await service.generate_key(question, "Grade 12 CBSE Physics", total_marks=1)
    print(f"Reasoning: {result.reasoning.encode('ascii', 'ignore').decode('ascii')}")
    print(f"Correct Option: {result.marking_scheme.correct_option}")

async def test_q16_copper():
    print("\n--- Testing Q16 (Material Fact - Copper) ---")
    question = "Assertion (A): When a bar of copper is placed in an external magnetic field, the field lines get concentrated inside the bar. Reason (R): Copper is a paramagnetic substance."
    service = QuestionGeneratorService()
    result = await service.generate_key(question, "Grade 12 CBSE Physics", total_marks=1)
    print(f"Reasoning: {result.reasoning.encode('ascii', 'ignore').decode('ascii')}")
    print(f"Correct Option: {result.marking_scheme.correct_option}")

async def test_q18_ar():
    print("\n--- Testing Q18 (AR Option B Logic) ---")
    question = "Assertion (A): Work done by the electrostatic force in moving a charge from one point to another is independent of the path. Reason (R): Electrostatic force is a conservative force."
    # Both True, R is correct explanation of A. Should be A.
    service = QuestionGeneratorService()
    result = await service.generate_key(question, "Grade 12 CBSE Physics", total_marks=1)
    print(f"Reasoning: {result.reasoning.encode('ascii', 'ignore').decode('ascii')}")
    print(f"Correct Option: {result.marking_scheme.correct_option}")

async def main():
    await test_q1_mapping()
    await test_q9_optics()
    await test_q16_copper()
    await test_q18_ar()

if __name__ == "__main__":
    asyncio.run(main())
