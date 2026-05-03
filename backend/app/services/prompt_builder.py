from app.schemas import AnalyzeRequest


def build_analysis_prompt(payload: AnalyzeRequest) -> str:
    return f"""
You are a Japanese language tutor for an English-speaking learner.

Analyze the following Japanese sentence for a learner around JLPT {payload.jlpt_level}.
Focus area: {payload.focus}.

Sentence:
{payload.sentence}

Return valid JSON with this schema:
{{
  "summary": "short learner-friendly summary",
  "natural_translation": "natural English translation",
  "grammar_points": [
    {{
      "pattern": "grammar pattern",
      "explanation": "clear explanation",
      "example": "simple Japanese example with English meaning"
    }}
  ],
  "vocabulary": [
    {{
      "word": "Japanese word",
      "reading": "kana reading",
      "meaning": "English meaning",
      "note": "usage note"
    }}
  ],
  "nuance": "tone, politeness, or context notes",
  "examples": ["extra Japanese example sentence with English meaning"],
  "practice_questions": [
    {{
      "question": "short practice question",
      "answer": "correct answer",
      "explanation": "why the answer is correct"
    }}
  ]
}}

Keep explanations concise, practical, and appropriate for JLPT {payload.jlpt_level}.
""".strip()

