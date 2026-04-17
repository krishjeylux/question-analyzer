"""
QuestionPaperExtractor
======================
Extracts structured questions from a CBSE question paper PDF using:
  1. pdf2image (Poppler) — converts PDF pages to images
  2. Google Cloud Vision API — high-accuracy OCR per page
  3. Gemini Flash — semantic parsing into structured question list
"""

import os
import io
import json
import base64
import tempfile
import asyncio
from typing import List, Dict, Optional, Any
from pathlib import Path

from app.core.config import settings

# ─────────────────────────────────────────────
# Lazy imports (guarded so server still starts
# if a package is missing, giving a clear error)
# ─────────────────────────────────────────────

def _load_vision_client():
    """Return an authenticated Google Vision client."""
    try:
        from google.cloud import vision
        from google.oauth2 import service_account

        creds_path = settings.GOOGLE_APPLICATION_CREDENTIALS
        if creds_path and os.path.exists(creds_path):
            credentials = service_account.Credentials.from_service_account_file(
                creds_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            return vision.ImageAnnotatorClient(credentials=credentials)
        else:
            # Fall back to ADC (Application Default Credentials)
            return vision.ImageAnnotatorClient()
    except ImportError:
        raise RuntimeError(
            "google-cloud-vision is not installed. Run: pip install google-cloud-vision"
        )


def _load_gemini_client():
    """Return a configured Gemini generative model."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=settings.GEMINI_API_KEY)
        return genai.GenerativeModel("gemini-2.0-flash")
    except ImportError:
        raise RuntimeError(
            "google-generativeai is not installed. Run: pip install google-generativeai"
        )


def _load_tesseract():
    """Configure and return pytesseract."""
    try:
        import pytesseract
        if settings.TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD
        return pytesseract
    except ImportError:
        raise RuntimeError(
            "pytesseract is not installed. Run: pip install pytesseract"
        )


def _pdf_to_images(pdf_bytes: bytes, dpi: int = 200) -> List[Any]:
    """
    Convert a PDF (as raw bytes) into a list of PIL Images, one per page.
    Uses Poppler via pdf2image.
    """
    try:
        from pdf2image import convert_from_bytes
    except ImportError:
        raise RuntimeError(
            "pdf2image is not installed. Run: pip install pdf2image"
        )

    poppler_path = settings.POPPLER_PATH or None
    images = convert_from_bytes(
        pdf_bytes,
        dpi=dpi,
        poppler_path=poppler_path,
        fmt="jpeg"
    )
    return images


def _image_to_bytes(pil_image: Any, fmt: str = "JPEG") -> bytes:
    """Convert a PIL Image to raw bytes."""
    buf = io.BytesIO()
    pil_image.save(buf, format=fmt)
    return buf.getvalue()


# ─────────────────────────────────────────────
# Main service class
# ─────────────────────────────────────────────

class QuestionPaperExtractor:
    """
    Extracts structured questions from a question paper PDF.

    Usage:
        extractor = QuestionPaperExtractor()
        questions = await extractor.extract_questions(pdf_bytes, subject="Physics")
        # questions → [{"id": "Q1", "text": "...", "marks": 1}, ...]
    """

    def __init__(self):
        self._vision_client = None
        self._gemini_model = None
        self._tesseract = None

    @property
    def vision_client(self):
        try:
            if self._vision_client is None:
                self._vision_client = _load_vision_client()
            return self._vision_client
        except Exception as e:
            print(f"[Extractor] Warning: Could not initialize Vision client: {e}")
            return None

    @property
    def gemini_model(self):
        if self._gemini_model is None:
            self._gemini_model = _load_gemini_client()
        return self._gemini_model

    @property
    def tesseract(self):
        if self._tesseract is None:
            self._tesseract = _load_tesseract()
        return self._tesseract

    # ── Step 1: OCR ──────────────────────────────────────────────────────────

    def _ocr_page_vision(self, image_bytes: bytes) -> str:
        """
        Run Google Cloud Vision document_text_detection on a single page image.
        Returns the full text of that page.
        """
        client = self.vision_client
        if not client:
            raise RuntimeError("Vision client not available.")

        from google.cloud import vision

        image = vision.Image(content=image_bytes)
        response = client.document_text_detection(image=image)

        if response.error.message:
            raise RuntimeError(
                f"Vision API error: {response.error.message}"
            )

        annotation = response.full_text_annotation
        return annotation.text if annotation else ""

    def _ocr_page_tesseract(self, pil_image: Any) -> str:
        """
        Run local Tesseract OCR on a PIL image.
        """
        return self.tesseract.image_to_string(pil_image)

    def _ocr_all_pages(self, pdf_bytes: bytes) -> str:
        """
        Convert PDF to images, OCR each page, return combined text.
        Primary: Google Vision
        Fallback: Tesseract
        """
        print("[Extractor] Converting PDF pages to images...")
        images = _pdf_to_images(pdf_bytes)
        print(f"[Extractor] {len(images)} page(s) found. Running OCR...")

        full_text_parts = []
        for i, img in enumerate(images, start=1):
            img_bytes = _image_to_bytes(img)
            page_text = ""
            
            # --- Try Google Vision First ---
            try:
                page_text = self._ocr_page_vision(img_bytes)
                print(f"[Extractor]   Page {i}: Cloud Vision successful ({len(page_text)} chars).")
            except Exception as e:
                print(f"[Extractor]   Page {i}: Cloud Vision failed ({e}). Falling back to Tesseract...")
                
                # --- Try Tesseract Fallback ---
                try:
                    page_text = self._ocr_page_tesseract(img)
                    print(f"[Extractor]   Page {i}: Tesseract fallback successful ({len(page_text)} chars).")
                except Exception as te:
                    print(f"[Extractor]   Page {i}: Tesseract also failed ({te}). Using empty string.")
                    page_text = ""
            
            full_text_parts.append(f"--- PAGE {i} ---\n{page_text}")

        return "\n\n".join(full_text_parts)

    # ── Step 2: Gemini Parsing ────────────────────────────────────────────────

    def _parse_questions_with_gemini(self, ocr_text: str, subject: str) -> List[Dict]:
        """
        Use Gemini Flash to structure the raw OCR text into a list of questions.
        Returns: [{"id": "Q1", "text": "...", "marks": 1}, ...]
        """
        prompt = f"""You are processing the OCR text of a CBSE Grade 12 {subject} exam question paper.

Your mission is to extract EVERY SINGLE question from the text. Missing even one question is a failure.

CRITICAL RULES:
1. Identify EACH question by its number (e.g., Q1, Q2, Q3, etc.).
2. ROOT-LEVEL CHOICE: For major questions with a full internal choice (indicated by the word "OR" between two primary questions), create SEPARATE entries:
   - First choice: Q21_A, Q33_A, etc.
   - Second choice: Q21_B, Q33_B, etc.
3. SUBDIVISIONS & NESTING: For questions with subdivisions (e.g., (a), (b), (c) or (i), (ii)), you MUST keep ALL sub-parts within the SAME question block.
   - If a part has sub-sub-parts (e.g., (i) contains 1, 2), preserve this nesting in the text.
   - Example: Q33_A might have Part (i) [with sub-parts 1 and 2] and Part (ii) [with sub-parts 1 and 2]. Keep all these in Q33_A.
4. CASE STUDY INTERNAL OR: For Section E (Case Studies like Q34, Q35), a single sub-part (usually part 'c') might have an internal "OR". Do NOT split the whole question into _A and _B for this. Keep Q34 as one block and include both "OR" options for part (c) in the text.
5. Extract the marks allocated (e.g., [1], [3], (2)). If missing, estimate based on CBSE norms.
6. Preserve mathematical expressions, symbols, and LaTeX perfectly.
7. Skip instructions, headers, footers, and general instructions.
8. Return ONLY a valid JSON array. Double check the JSON syntax before finishing.

IMPORTANT: Look very closely at the start of each line for question numbers. Sometimes they are embedded in the text. Ensure you have a continuous sequence of questions.

OUTPUT FORMAT:
[
  {{"id": "Q1", "text": "...", "marks": 1}},
  ...
]

OCR TEXT OF THE QUESTION PAPER:
\"\"\"
{ocr_text}
\"\"\"

Return ONLY the JSON array:"""

        print("[Extractor] Sending OCR text to Gemini for question parsing...")
        response = self.gemini_model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 8192,
                "response_mime_type": "application/json",
            }
        )

        raw = response.text.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(
                line for line in lines
                if not line.startswith("```")
            ).strip()

        questions = json.loads(raw)
        print(f"[Extractor] Gemini extracted {len(questions)} question(s).")
        return questions

    # ── Step 3: Public API ────────────────────────────────────────────────────

    async def extract_questions(
        self,
        pdf_bytes: bytes,
        subject: str = "Physics"
    ) -> List[Dict]:
        """
        Full pipeline: PDF bytes → structured question list.

        Returns:
            List of dicts: [{"id": "Q1", "text": "...", "marks": 1}, ...]
        """
        # OCR runs synchronously (Vision API is sync); wrap in executor
        loop = asyncio.get_event_loop()
        ocr_text = await loop.run_in_executor(
            None, self._ocr_all_pages, pdf_bytes
        )

        if not ocr_text.strip():
            raise ValueError("OCR produced no text. The PDF may be corrupt or empty.")

        # Gemini parsing also sync; run in executor
        questions = await loop.run_in_executor(
            None, self._parse_questions_with_gemini, ocr_text, subject
        )

        # Validate structure and detect gaps
        validated = []
        q_numbers = []
        for q in questions:
            if not isinstance(q, dict):
                continue
            
            q_id = str(q.get("id", f"Q{len(validated)+1}"))
            validated.append({
                "id": q_id,
                "text": str(q.get("text", "")).strip(),
                "marks": int(q.get("marks", 1))
            })
            
            # Extract numeric part for gap detection (handles Q1, Q21_A etc.)
            import re
            match = re.search(r'Q(\d+)', q_id)
            if match:
                q_numbers.append(int(match.group(1)))

        # Gap detection
        if q_numbers:
            q_numbers = sorted(list(set(q_numbers)))
            expected = list(range(1, max(q_numbers) + 1))
            missing = [n for n in expected if n not in q_numbers]
            if missing:
                print(f"[Extractor] WARNING: Detected potentially missing questions: {missing}")

        return validated
