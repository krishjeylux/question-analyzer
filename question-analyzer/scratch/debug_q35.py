import asyncio, json
from app.services.paper_extractor import _pdf_to_images, _image_to_bytes, QuestionPaperExtractor
from app.services.generator_service import QuestionGeneratorService

async def main():
    e = QuestionPaperExtractor()
    with open(r'C:\Users\krish\Downloads\cbse_12_physics_5511set1_2023_70.pdf', 'rb') as f:
        pdf = f.read()
    images = _pdf_to_images(pdf)
    
    last_images = images[24:] 
    
    text_parts = []
    print('OCR...')
    for img in last_images:
        try:
            pt = e._ocr_page_vision(_image_to_bytes(img))
        except:
            pt = e._ocr_page_tesseract(img)
        text_parts.append(pt)
    
    ocr_text = '\n'.join(text_parts)
    qs = await asyncio.get_event_loop().run_in_executor(None, e._parse_questions_with_gemini, ocr_text, 'Physics')
    
    g = QuestionGeneratorService()
    
    for q in qs:
        qid = q.get('id', '')
        if 'Q35' in qid:
            print(f'\n--- EXTRACTED RAW {qid} ---')
            print(q.get('text', ''))
            print('\n--- GENERATING... ---')
            try:
                res = await g.generate_key(q.get('text', ''), 'Physics', q.get('marks'))
                print('-> REPHRASED:')
                print(res.rephrased_question)
                print('\n-> MARKING SCHEME:')
                print(json.dumps(res.marking_scheme.model_dump(), indent=2))
                with open('scratch/q35_success.json', 'w') as out:
                    json.dump(res.model_dump(), out, indent=2)
            except Exception as ex:
                print('FAILED:', ex)

if __name__ == '__main__':
    asyncio.run(main())
