import json
import os
from typing import Any

import httpx
from dotenv import load_dotenv

from app.schemas import AnalyzeRequest, AnalyzeResponse
from app.services.prompt_builder import build_analysis_prompt

load_dotenv()


class JapaneseLearningLLMService:
    def __init__(self) -> None:
        self.provider = os.getenv("LLM_PROVIDER", "mock")
        self.api_key = os.getenv("LLM_API_KEY", "")
        self.base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        self.model = os.getenv("LLM_MODEL", "gpt-4.1-mini")
        self.thinking_type = os.getenv("LLM_THINKING_TYPE", "")
        self.reasoning_effort = os.getenv("LLM_REASONING_EFFORT", "")

    async def analyze(self, payload: AnalyzeRequest) -> AnalyzeResponse:
        if self.provider == "openai_compatible" and self.api_key:
            return self._analyze_with_openai_compatible_api(payload)

        return self._mock_response(payload)

    def _analyze_with_openai_compatible_api(
        self, payload: AnalyzeRequest
    ) -> AnalyzeResponse:
        prompt = build_analysis_prompt(payload)
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You return concise, valid JSON for Japanese learning.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }

        if self.thinking_type:
            body["thinking"] = {"type": self.thinking_type}

        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort

        with httpx.Client(timeout=60) as client:
            response = client.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
            response.raise_for_status()
            raw = response.json()

        content = raw["choices"][0]["message"]["content"]
        content = self._strip_json_fence(content)
        parsed: dict[str, Any] = json.loads(content)
        parsed["model_used"] = self.model
        parsed["source"] = "llm"
        return AnalyzeResponse.model_validate(parsed)

    def _strip_json_fence(self, content: str) -> str:
        text = content.strip()
        if text.startswith("```json"):
            text = text.removeprefix("```json").strip()
        elif text.startswith("```"):
            text = text.removeprefix("```").strip()

        if text.endswith("```"):
            text = text.removesuffix("```").strip()

        return text

    def _mock_response(self, payload: AnalyzeRequest) -> AnalyzeResponse:
        return AnalyzeResponse(
            summary=(
                "Demo response only. The backend is currently running without a model API key, "
                f"so this is not a real analysis of: {payload.sentence}"
            ),
            natural_translation=f"Mock translation for: {payload.sentence}",
            grammar_points=[
                {
                    "pattern": "Demo grammar point",
                    "explanation": (
                        "Connect an LLM API key to generate grammar points from the actual "
                        "sentence. This placeholder exists so the UI can be tested."
                    ),
                    "example": "日本語を勉強しています。= I am studying Japanese.",
                }
            ],
            vocabulary=[
                {
                    "word": payload.sentence[:20],
                    "reading": "-",
                    "meaning": "Input preview",
                    "note": "This is a placeholder because mock mode does not perform NLP.",
                },
            ],
            nuance=(
                "Mock mode does not infer tone or nuance. Set LLM_PROVIDER=openai_compatible "
                "and provide LLM_API_KEY in backend/.env for real analysis."
            ),
            examples=[
                "これはデモ用の例文です。= This is a demo example sentence.",
            ],
            practice_questions=[
                {
                    "question": "What should you do to get real analysis?",
                    "answer": "Add a model API key in backend/.env.",
                    "explanation": "The mock response is only for testing the frontend/backend flow.",
                }
            ],
            model_used="mock",
            source="mock",
        )
