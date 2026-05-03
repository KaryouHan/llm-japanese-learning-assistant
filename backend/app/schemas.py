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

