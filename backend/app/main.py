from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import AnalyzeRequest, AnalyzeResponse
from app.services.llm_service import JapaneseLearningLLMService

app = FastAPI(
    title="LLM Japanese Learning Assistant API",
    description="API for analyzing Japanese sentences with an LLM-powered tutor.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm_service = JapaneseLearningLLMService()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_sentence(payload: AnalyzeRequest) -> AnalyzeResponse:
    try:
        return await llm_service.analyze(payload)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
