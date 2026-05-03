# LLM Japanese Learning Assistant

An LLM-powered assistant for Japanese learners. It explains grammar, vocabulary, nuance, and example usage, with difficulty-aware output for JLPT-style learning.

This project is designed as an LLM Engineer portfolio project. It demonstrates prompt design, structured LLM output, API integration, frontend/backend separation, and a roadmap toward RAG with Japanese learning materials.

## Features

- Explain Japanese sentences in English
- Break down grammar points, vocabulary, and nuance
- Generate learner-friendly example sentences
- Create short practice questions
- Support JLPT level selection from N5 to N1
- Run with a mock LLM response by default
- Switch to a real model API through environment variables

## Tech Stack

- Backend: Python, FastAPI, Pydantic
- Frontend: Vue 3, TypeScript, Vite
- AI: LLM API integration, prompt engineering, structured JSON output
- Planned RAG: PDF ingestion, chunking, embeddings, vector search, source citations
- DevOps: Docker, docker compose

## Project Structure

```text
llm-japanese-learning-assistant/
  backend/
    app/
      main.py
      schemas.py
      services/
        llm_service.py
        prompt_builder.py
    requirements.txt
    .env.example
  frontend/
    src/
      App.vue
      main.ts
      styles.css
    package.json
    index.html
    vite.config.ts
    tsconfig.json
  docker-compose.yml
  README.md
```

## Quick Start

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

The backend will run at:

```text
http://localhost:8000
```

Health check:

```text
http://localhost:8000/health
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend will run at:

```text
http://localhost:5173
```

## API Example

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "sentence": "昨日、友達に日本語を教えてもらいました。",
    "jlpt_level": "N4",
    "focus": "grammar"
  }'
```

## Model API

By default, the backend returns a deterministic mock response so the project can run without an API key.

To connect a real model provider, configure:

```text
LLM_PROVIDER=openai_compatible
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4.1-mini
```

The code is intentionally provider-light: any OpenAI-compatible chat completions endpoint can be wired in.

### DeepSeek

DeepSeek provides an OpenAI-compatible API, so it can be used without changing the backend code.

```text
LLM_PROVIDER=openai_compatible
LLM_API_KEY=your_deepseek_api_key_here
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-v4-flash
```

For compatibility, `https://api.deepseek.com/v1` can also be used as the base URL. The `v1` path is API compatibility naming, not the model version.

For a stronger reasoning model, use:

```text
LLM_PROVIDER=openai_compatible
LLM_API_KEY=your_deepseek_api_key_here
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-v4-pro
LLM_THINKING_TYPE=enabled
LLM_REASONING_EFFORT=high
```

## RAG Roadmap

Future versions will add retrieval over Japanese learning resources:

- Upload JLPT grammar notes or exam preparation PDFs
- Extract and chunk Japanese text
- Build embeddings and vector indexes
- Retrieve grammar explanations and example sentences
- Return answers with source citations
- Evaluate retrieval quality and hallucination rate

## Portfolio Notes

This project is intended to demonstrate:

- Practical LLM application design
- Structured prompt engineering
- Full-stack AI product implementation
- API-first backend design
- Readable documentation and deployment readiness
- A clear path from prototype to RAG system
