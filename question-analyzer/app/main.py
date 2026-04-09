from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.evaluation_router import router as evaluation_router
from app.api.generation_router import router as generation_router
from app.core.config import settings
import uvicorn

app = FastAPI(
    title=settings.APP_NAME,
    description="Backend for analyzing and marking academic questions using deterministic and LLM-based engines.",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(evaluation_router)
app.include_router(generation_router)

@app.get("/")
async def root():
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "status": "online",
        "api_docs": "/docs"
    }

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
