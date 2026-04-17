import asyncio
import json
from app.services.paper_extractor import _pdf_to_images, _image_to_bytes, QuestionPaperExtractor
from app.services.generator_service import QuestionGeneratorService

async def main():
    e = QuestionPaperExtractor()
    with open(r'C:\Users\krish\Downloads\cbse_12_physics_5511set1_2023_70.pdf', 'rb') as f:
        pdf = f.read()
    images = _pdf_to_images(pdf)
    
    # Process only last few pages where Q33, Q34, Q35 are (pages 22, 23, 24.. depending on the 27 page pdf)
    # Actually let's just OCR pages 20-27
    last_images = images[19:] 
    
    text_parts = []
    print("Running OCR on last 8 pages...")
    for i, img in enumerate(last_images):
        try:
            page_text = e._ocr_page_vision(_image_to_bytes(img))
        except:
            page_text = e._ocr_page_tesseract(img)
        text_parts.append(page_text)
    
    ocr_text = '\n'.join(text_parts)
    print("OCR text length:", len(ocr_text))
    
    # Parse questions
    qs = await asyncio.get_event_loop().run_in_executor(None, e._parse_questions_with_gemini, ocr_text, 'Physics')
    
    g = QuestionGeneratorService()
    
    print("\n--- EXTRACTED QUESTIONS ---")
    for q in qs:
        qid = q.get('id', '')
        if "Q33" in qid or "Q34" in qid or "Q35" in qid:
            print(f"=== {qid} ===")
            print(q.get('text', '')[:300])
            try:
                print(f"Generating for {qid}...")
                res = await g.generate_key(q.get('text', ''), 'Physics', q.get('marks'))
                print("-> Success:", res.marking_scheme.type)
            except Exception as ex:
                print("-> Error:", ex)

if __name__ == "__main__":
    asyncio.run(main())
