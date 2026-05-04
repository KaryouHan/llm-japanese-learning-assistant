from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    KnowledgeIngestResponse,
    KnowledgeStatusResponse,
    PdfUploadResponse,
    RelatedExamplesRequest,
    RelatedExamplesResponse,
)
from app.services.knowledge_service import KnowledgeService
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
knowledge_service = KnowledgeService()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_sentence(payload: AnalyzeRequest) -> AnalyzeResponse:
    try:
        return await llm_service.analyze(payload)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/knowledge/status", response_model=KnowledgeStatusResponse)
def knowledge_status() -> KnowledgeStatusResponse:
    return knowledge_service.status()


@app.post("/api/knowledge/upload", response_model=PdfUploadResponse)
def upload_knowledge_pdf(file: UploadFile = File(...)) -> PdfUploadResponse:
    try:
        return knowledge_service.upload_pdf(file)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/knowledge/ingest", response_model=KnowledgeIngestResponse)
def ingest_knowledge_base() -> KnowledgeIngestResponse:
    try:
        return knowledge_service.ingest()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/knowledge/related", response_model=RelatedExamplesResponse)
def find_related_examples(payload: RelatedExamplesRequest) -> RelatedExamplesResponse:
    try:
        return knowledge_service.find_related(payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
