from typing import Literal

from pydantic import BaseModel, Field


JLPTLevel = Literal["N5", "N4", "N3", "N2", "N1"]
FocusArea = Literal["grammar", "vocabulary", "nuance", "exam", "general"]


class AnalyzeRequest(BaseModel):
    sentence: str = Field(..., min_length=1, max_length=500)
    jlpt_level: JLPTLevel = "N4"
    focus: FocusArea = "general"


class GrammarPoint(BaseModel):
    pattern: str
    explanation: str
    example: str


class VocabularyItem(BaseModel):
    word: str
    reading: str
    meaning: str
    note: str


class PracticeQuestion(BaseModel):
    question: str
    answer: str
    explanation: str


class AnalyzeResponse(BaseModel):
    summary: str
    natural_translation: str
    grammar_points: list[GrammarPoint]
    vocabulary: list[VocabularyItem]
    nuance: str
    examples: list[str]
    practice_questions: list[PracticeQuestion]
    model_used: str
    source: Literal["mock", "llm"]


class KnowledgeStatusResponse(BaseModel):
    raw_pdf_count: int
    uploaded_pdf_count: int
    indexed_document_count: int
    indexed_chunk_count: int
    indexed_sentence_count: int
    index_exists: bool


class PdfUploadResponse(BaseModel):
    filename: str
    saved_path: str


class KnowledgeIngestResponse(BaseModel):
    document_count: int
    chunk_count: int
    skipped_files: list[str]


class RelatedExamplesRequest(BaseModel):
    sentence: str = Field(..., min_length=1, max_length=500)
    jlpt_level: JLPTLevel = "N1"
    top_k: int = Field(default=5, ge=1, le=10)


class RelatedExample(BaseModel):
    source: str
    year: str | None = None
    month: str | None = None
    section: str
    question_id: str | None = None
    page: int
    related_pattern: str | None = None
    excerpt: str
    why_related: str
    score: float


class RelatedExamplesResponse(BaseModel):
    detected_patterns: list[str]
    related_examples: list[RelatedExample]
    study_note: str
