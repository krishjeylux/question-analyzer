from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import os

class Settings(BaseSettings):
    APP_NAME: str = "Question Analyzer"
    GROK_API_KEY: str
    GEMINI_API_KEY: str
    LOG_LEVEL: str = "INFO"
    
    # Google Cloud configuration
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = None
    
    # OCR Tool paths
    TESSERACT_CMD: Optional[str] = None
    POPPLER_PATH: Optional[str] = None
    
    # Qdrant configuration
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_URL: Optional[str] = None
    
    # Paper configuration
    PHYSICS_PAPER_PATH: str = os.path.join("app", "data", "physics_paper_1.json")
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
