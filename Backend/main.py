import os
import sys
from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure we can import from project root (for complete_system.py)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from complete_system import AITutorSystem  # type: ignore


class ChatRequest(BaseModel):
    query: str
    keywords: Optional[List[str]] = None


class ChatResponse(BaseModel):
    input: str
    answer: str
    intent: dict
    guardian: dict
    timestamp: str


app = FastAPI(title="EduAI Backend", version="1.0.0")

# CORS (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize system once
ai_system = AITutorSystem(mode="production")


@app.get("/")
def root():
    return {
        "message": "EduAI Backend running.",
        "docs": "/docs",
        "health": "/health",
        "chat_endpoint": {"method": "POST", "path": "/api/chat"},
    }

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    result = ai_system.handle_student_query(req.query)
    return ChatResponse(
        input=result["input"],
        answer=result["answer"],
        intent=result["intent"],
        guardian=result["guardian"],
        timestamp=result["timestamp"],
    )


