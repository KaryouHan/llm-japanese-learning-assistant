import json
import math
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import UploadFile
from pypdf import PdfReader

from app.schemas import (
    KnowledgeIngestResponse,
    KnowledgeStatusResponse,
    PdfUploadResponse,
    RelatedExample,
    RelatedExamplesRequest,
    RelatedExamplesResponse,
)

load_dotenv()


class KnowledgeService:
    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parents[3]
        self.knowledge_root = self.project_root / "knowledge_base"
        self.raw_dir = self.knowledge_root / "raw"
        self.uploads_dir = self.knowledge_root / "uploads"
        self.index_dir = self.knowledge_root / "index"
        self.index_file = self.index_dir / "index.json"
        self.chroma_dir = self.index_dir / "chroma"
        self.chroma_collection_name = "n1_sentence_examples"
        self.llm_provider = os.getenv("LLM_PROVIDER", "mock")
        self.llm_api_key = os.getenv("LLM_API_KEY", "")
        self.llm_base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        self.llm_model = os.getenv("LLM_MODEL", "gpt-4.1-mini")
        self.vector_backend = os.getenv("RAG_VECTOR_BACKEND", "chroma").lower()
        self.embedding_model_name = os.getenv(
            "RAG_EMBEDDING_MODEL",
            "intfloat/multilingual-e5-small",
        )
        self.reranker_model_name = os.getenv(
            "RAG_RERANKER_MODEL",
            "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        )
        self.use_reranker = os.getenv("RAG_USE_RERANKER", "true").lower() == "true"
        self.retrieval_k = int(os.getenv("RAG_RETRIEVAL_K", "30"))
        self._embedding_model: Any | None = None
        self._reranker_model: Any | None = None

    def status(self) -> KnowledgeStatusResponse:
        index = self._load_index()
        return KnowledgeStatusResponse(
            raw_pdf_count=len(list(self.raw_dir.glob("*.pdf"))),
            uploaded_pdf_count=len(list(self.uploads_dir.glob("*.pdf"))),
            indexed_document_count=len(index.get("documents", [])),
            indexed_chunk_count=len(index.get("chunks", [])),
            indexed_sentence_count=len(index.get("sentences", [])),
            vector_backend=self.vector_backend,
            vector_index_exists=self._vector_index_exists(),
            embedding_model=self.embedding_model_name,
            reranker_model=self.reranker_model_name if self.use_reranker else None,
            index_exists=self.index_file.exists(),
        )

    def upload_pdf(self, upload: UploadFile) -> PdfUploadResponse:
        filename = self._safe_filename(upload.filename or "knowledge.pdf")
        if not filename.lower().endswith(".pdf"):
            filename = f"{filename}.pdf"

        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        target = self.uploads_dir / filename

        with target.open("wb") as output:
            shutil.copyfileobj(upload.file, output)

        return PdfUploadResponse(filename=filename, saved_path=str(target))

    def ingest(self) -> KnowledgeIngestResponse:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        pdf_by_name: dict[str, Path] = {}
        for path in [*self.raw_dir.glob("*.pdf"), *self.uploads_dir.glob("*.pdf")]:
            pdf_by_name[path.name] = path.resolve()

        pdf_paths = sorted(pdf_by_name.values())
        documents: list[dict[str, Any]] = []
        chunks: list[dict[str, Any]] = []
        sentences: list[dict[str, Any]] = []
        skipped_files: list[str] = []

        for pdf_path in pdf_paths:
            try:
                document_chunks = self._extract_pdf_chunks(pdf_path)
                document_sentences = self._extract_sentence_records(document_chunks)
            except Exception as exc:
                skipped_files.append(f"{pdf_path.name}: {exc}")
                continue

            documents.append(
                {
                    "source": pdf_path.name,
                    "path": str(pdf_path),
                    "chunk_count": len(document_chunks),
                    "sentence_count": len(document_sentences),
                }
            )
            chunks.extend(document_chunks)
            sentences.extend(document_sentences)

        payload = {
            "version": 2,
            "documents": documents,
            "chunks": chunks,
            "sentences": sentences,
        }
        self.index_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        vector_index_built = self._build_vector_index(sentences)

        return KnowledgeIngestResponse(
            document_count=len(documents),
            chunk_count=len(chunks),
            sentence_count=len(sentences),
            vector_backend=self.vector_backend,
            vector_index_built=vector_index_built,
            skipped_files=skipped_files,
        )

    def find_related(self, request: RelatedExamplesRequest) -> RelatedExamplesResponse:
        index = self._load_index()
        chunks = index.get("chunks", [])
        if not chunks:
            return RelatedExamplesResponse(
                detected_patterns=[],
                related_examples=[],
                study_note="No indexed PDFs found. Upload PDFs or place them under knowledge_base/raw, then run ingestion.",
            )

        records = index.get("sentences") or self._records_from_legacy_chunks(chunks)
        level_records = [
            record for record in records if record.get("level") == request.jlpt_level
        ]
        use_level_filter = bool(level_records)
        if use_level_filter:
            records = level_records

        patterns = self._extract_query_patterns(request.sentence)
        query_vector = self._vectorize(request.sentence + " " + " ".join(patterns))

        candidate_by_id: dict[str, tuple[float, dict[str, Any], str | None, str]] = {}
        for record in records:
            text = record.get("text", "")
            record_vector = Counter(record.get("vector", {}))
            vector_score = self._cosine(query_vector, record_vector)
            matched_pattern = self._best_pattern_match(patterns, text)
            if matched_pattern:
                score = 5.0 + vector_score
                candidate_by_id[record["id"]] = (score, record, matched_pattern, "pattern")
            elif vector_score >= 0.18:
                candidate_by_id[record["id"]] = (vector_score, record, None, "local-vector")

        for semantic_score, record in self._semantic_candidates(request.sentence):
            if use_level_filter and record.get("level") != request.jlpt_level:
                continue
            matched_pattern = self._best_pattern_match(patterns, record.get("text", ""))
            score = semantic_score + (5.0 if matched_pattern else 0.0)
            existing = candidate_by_id.get(record["id"])
            if not existing or score > existing[0]:
                method = "chroma-embedding" if self.vector_backend == "chroma" else "semantic"
                candidate_by_id[record["id"]] = (score, record, matched_pattern, method)

        candidates = list(candidate_by_id.values())
        if patterns:
            pattern_matches = [item for item in candidates if item[2]]
            if pattern_matches:
                candidates = pattern_matches
            else:
                candidates = []

        scored = self._rerank_candidates(request.sentence, candidates)
        examples = [
            self._to_related_example(score, record, matched_pattern, method)
            for score, record, matched_pattern, method in scored[: request.top_k]
        ]

        return RelatedExamplesResponse(
            detected_patterns=patterns,
            related_examples=examples,
            study_note=self._build_study_note(patterns, examples),
        )

    def _build_vector_index(self, records: list[dict[str, Any]]) -> bool:
        if self.vector_backend != "chroma" or not records:
            return False

        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            return False

        embedding_model = self._load_embedding_model()
        if not embedding_model:
            return False

        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(self.chroma_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        try:
            client.delete_collection(self.chroma_collection_name)
        except Exception:
            pass

        collection = client.get_or_create_collection(
            name=self.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        batch_size = 64
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            documents = [record["text"] for record in batch]
            embeddings = self._embed_texts(documents, mode="passage")
            if not embeddings:
                return False

            collection.add(
                ids=[record["id"] for record in batch],
                documents=documents,
                metadatas=[self._record_metadata(record) for record in batch],
                embeddings=embeddings,
            )

        return True

    def _semantic_candidates(self, sentence: str) -> list[tuple[float, dict[str, Any]]]:
        if self.vector_backend != "chroma" or not self._vector_index_exists():
            return []

        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            return []

        query_embedding = self._embed_texts([sentence], mode="query")
        if not query_embedding:
            return []

        try:
            client = chromadb.PersistentClient(
                path=str(self.chroma_dir),
                settings=Settings(anonymized_telemetry=False),
            )
            collection = client.get_collection(self.chroma_collection_name)
            result = collection.query(
                query_embeddings=query_embedding,
                n_results=max(self.retrieval_k, 10),
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return []

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        candidates: list[tuple[float, dict[str, Any]]] = []

        for document, metadata, distance in zip(documents, metadatas, distances, strict=False):
            score = max(0.0, 1.0 - float(distance))
            record = self._metadata_to_record(metadata, document)
            candidates.append((score, record))

        return candidates

    def _rerank_candidates(
        self,
        sentence: str,
        candidates: list[tuple[float, dict[str, Any], str | None, str]],
    ) -> list[tuple[float, dict[str, Any], str | None, str]]:
        if not candidates:
            return []

        candidates.sort(key=lambda item: item[0], reverse=True)
        shortlist = candidates[: max(self.retrieval_k, 10)]
        if not self.use_reranker:
            return candidates

        reranker = self._load_reranker_model()
        if not reranker:
            return candidates

        pairs = [(sentence, item[1].get("text", "")) for item in shortlist]
        try:
            rerank_scores = reranker.predict(pairs)
        except Exception:
            return candidates

        reranked: list[tuple[float, dict[str, Any], str | None, str]] = []
        for item, rerank_score in zip(shortlist, rerank_scores, strict=False):
            base_score, record, matched_pattern, method = item
            rerank_value = self._sigmoid(float(rerank_score))
            final_score = base_score + rerank_value * 2.0
            reranked.append((final_score, record, matched_pattern, f"{method}+reranker"))

        remaining = candidates[len(shortlist) :]
        reranked.extend(remaining)
        reranked.sort(key=lambda item: item[0], reverse=True)
        return reranked

    def _load_embedding_model(self) -> Any | None:
        if self._embedding_model is not None:
            return self._embedding_model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return None

        try:
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        except Exception:
            self._embedding_model = None

        return self._embedding_model

    def _load_reranker_model(self) -> Any | None:
        if self._reranker_model is not None:
            return self._reranker_model

        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            return None

        try:
            self._reranker_model = CrossEncoder(self.reranker_model_name)
        except Exception:
            self._reranker_model = None

        return self._reranker_model

    def _embed_texts(self, texts: list[str], mode: str) -> list[list[float]]:
        model = self._load_embedding_model()
        if not model:
            return []

        prepared_texts = [self._embedding_text(text, mode) for text in texts]
        try:
            embeddings = model.encode(
                prepared_texts,
                batch_size=32,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception:
            return []

        return embeddings.tolist()

    def _embedding_text(self, text: str, mode: str) -> str:
        if "e5" not in self.embedding_model_name.lower():
            return text

        prefix = "query" if mode == "query" else "passage"
        return f"{prefix}: {text}"

    def _record_metadata(self, record: dict[str, Any]) -> dict[str, str | int | float | bool]:
        return {
            "id": record["id"],
            "source": record.get("source", ""),
            "year": record.get("year") or "",
            "month": record.get("month") or "",
            "level": record.get("level") or "",
            "section": record.get("section") or "General",
            "question_id": record.get("question_id") or "",
            "page": int(record.get("page", 0)),
            "patterns": "|".join(record.get("patterns", [])),
        }

    def _metadata_to_record(self, metadata: dict[str, Any], document: str) -> dict[str, Any]:
        return {
            "id": metadata.get("id", ""),
            "source": metadata.get("source", ""),
            "year": metadata.get("year") or None,
            "month": metadata.get("month") or None,
            "level": metadata.get("level") or None,
            "section": metadata.get("section") or "General",
            "question_id": metadata.get("question_id") or None,
            "page": int(metadata.get("page", 0)),
            "text": document,
            "patterns": str(metadata.get("patterns", "")).split("|")
            if metadata.get("patterns")
            else [],
        }

    def _vector_index_exists(self) -> bool:
        return self.vector_backend == "chroma" and self.chroma_dir.exists()

    def _sigmoid(self, value: float) -> float:
        if value >= 0:
            z = math.exp(-value)
            return 1 / (1 + z)

        z = math.exp(value)
        return z / (1 + z)

    def _extract_sentence_records(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        seen: set[tuple[str, int, str]] = set()

        for chunk in chunks:
            for sentence_index, sentence in enumerate(self._split_sentences(chunk["text"]), start=1):
                if len(sentence) < 10:
                    continue

                key = (chunk["source"], int(chunk["page"]), sentence)
                if key in seen:
                    continue
                seen.add(key)

                record = {
                    "id": f"{chunk['id']}-s{sentence_index}",
                    "source": chunk["source"],
                    "year": chunk["year"],
                    "month": chunk["month"],
                    "level": chunk["level"],
                    "section": chunk["section"],
                    "question_id": chunk["question_id"],
                    "page": chunk["page"],
                    "text": sentence,
                    "patterns": self._detect_patterns(sentence),
                    "vector": dict(self._vectorize(sentence)),
                }
                records.append(record)

        return records

    def _records_from_legacy_chunks(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for chunk in chunks:
            for sentence_index, sentence in enumerate(self._split_sentences(chunk.get("text", "")), start=1):
                legacy_record = {
                    **chunk,
                    "id": f"{chunk.get('id', 'chunk')}-legacy-s{sentence_index}",
                    "text": sentence,
                    "patterns": self._detect_patterns(sentence),
                    "vector": dict(self._vectorize(sentence)),
                }
                records.append(legacy_record)
        return records

    def _extract_pdf_chunks(self, pdf_path: Path) -> list[dict[str, Any]]:
        reader = PdfReader(str(pdf_path))
        metadata = self._parse_pdf_metadata(pdf_path.name)
        chunks: list[dict[str, Any]] = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = self._clean_text(page.extract_text() or "")
            if len(text) < 40:
                continue

            section = self._detect_section(text)
            page_chunks = self._split_text(text)
            for chunk_index, chunk_text in enumerate(page_chunks, start=1):
                question_id = self._detect_question_id(chunk_text)
                chunk = {
                    "id": f"{pdf_path.stem}-p{page_number}-c{chunk_index}",
                    "source": pdf_path.name,
                    "year": metadata["year"],
                    "month": metadata["month"],
                    "level": metadata["level"],
                    "section": section,
                    "question_id": question_id,
                    "page": page_number,
                    "text": chunk_text,
                    "vector": dict(self._vectorize(chunk_text)),
                }
                chunks.append(chunk)

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []

        sentences = re.split(r"(?<=[。！？?])\s+", text)
        refined: list[str] = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) <= 220:
                refined.append(sentence)
                continue

            pieces = re.split(r"(?<=[）)])\s+|(?<=\))\s+|(?<=、)\s+", sentence)
            refined.extend(piece.strip() for piece in pieces if len(piece.strip()) >= 10)

        return refined

    def _split_text(self, text: str, max_chars: int = 900, overlap: int = 120) -> list[str]:
        if len(text) <= max_chars:
            return [text]

        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            chunks.append(text[start:end].strip())
            if end == len(text):
                break
            start = max(0, end - overlap)
        return chunks

    def _clean_text(self, text: str) -> str:
        replacements = {
            "\x00": "",
            "\u3000": " ",
            "⽇": "日",
            "⽉": "月",
            "⾔": "言",
            "⽂": "文",
            "⼒": "力",
            "⼀": "一",
            "⼊": "入",
            "⾒": "見",
            "⼈": "人",
            "⽣": "生",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        text = re.sub(r"微信公众号：[^）)]*", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _safe_filename(self, filename: str) -> str:
        name = Path(filename).name
        name = re.sub(r"[^A-Za-z0-9_.\-\u3040-\u30ff\u3400-\u9fff]", "_", name)
        return name or "knowledge.pdf"

    def _parse_pdf_metadata(self, filename: str) -> dict[str, str | None]:
        compact = re.sub(r"[^0-9A-Za-z]", "", filename)
        match = re.search(r"(20\d{2})(07|12)", compact)
        level_match = re.search(r"N[1-5]", filename.upper())
        return {
            "year": match.group(1) if match else None,
            "month": match.group(2) if match else None,
            "level": level_match.group(0) if level_match else None,
        }

    def _detect_section(self, text: str) -> str:
        if "文法" in text or "問題5" in text or "問題6" in text or "問題７" in text or "問題7" in text:
            return "Grammar"
        if "語彙" in text or "文字" in text:
            return "Vocabulary"
        if "読解" in text:
            return "Reading"
        if "聴解" in text or "聽解" in text:
            return "Listening"
        return "General"

    def _detect_question_id(self, text: str) -> str | None:
        match = re.search(r"(?:問題\s*)?([0-9０-９]{1,2})[、\s]", text)
        if not match:
            return None
        return f"Q{self._normalize_number(match.group(1))}"

    def _normalize_number(self, value: str) -> str:
        table = str.maketrans("０１２３４５６７８９", "0123456789")
        return value.translate(table)

    def _extract_query_patterns(self, sentence: str) -> list[str]:
        local_patterns = self._detect_patterns(sentence)
        llm_patterns = self._extract_patterns_with_llm(sentence)
        return list(dict.fromkeys([*local_patterns, *llm_patterns]))

    def _extract_patterns_with_llm(self, sentence: str) -> list[str]:
        if self.llm_provider != "openai_compatible" or not self.llm_api_key:
            return []

        url = f"{self.llm_base_url.rstrip('/')}/chat/completions"
        prompt = (
            "Extract Japanese grammar patterns from the learner sentence.\n"
            "Return strict JSON only: {\"patterns\":[\"～pattern\"]}.\n"
            "Rules:\n"
            "- Extract only reusable JLPT-style grammar expressions, not vocabulary.\n"
            "- Normalize with the leading ～, for example ～とは限らない.\n"
            "- If a pattern has a close exam variant, include both the exact pattern and variant.\n"
            "- Return at most 5 patterns.\n\n"
            f"Sentence: {sentence}"
        )
        body: dict[str, Any] = {
            "model": self.llm_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise Japanese grammar pattern extractor. Return JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        }

        try:
            with httpx.Client(timeout=20) as client:
                response = client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {self.llm_api_key}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                )
                response.raise_for_status()
                raw = response.json()
        except Exception:
            return []

        content = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
        try:
            parsed = json.loads(self._strip_json_fence(content))
        except json.JSONDecodeError:
            return []

        patterns = parsed.get("patterns", [])
        if not isinstance(patterns, list):
            return []

        normalized: list[str] = []
        for pattern in patterns:
            if not isinstance(pattern, str):
                continue
            pattern = pattern.strip()
            if not pattern:
                continue
            if not pattern.startswith("～"):
                pattern = f"～{pattern}"
            normalized.append(pattern)

        return normalized[:5]

    def _strip_json_fence(self, content: str) -> str:
        text = content.strip()
        if text.startswith("```json"):
            text = text.removeprefix("```json").strip()
        elif text.startswith("```"):
            text = text.removeprefix("```").strip()

        if text.endswith("```"):
            text = text.removesuffix("```").strip()

        return text

    def _detect_patterns(self, sentence: str) -> list[str]:
        patterns: list[str] = []
        if "ないとも限らない" in sentence:
            patterns.extend(["～ないとも限らない", "～とは限らない"])

        rules = [
            ("ないとも限らない", "～ないとも限らない"),
            ("とは限らない", "～とは限らない"),
            ("とはいえ", "～とはいえ"),
            ("といっても", "～といっても"),
            ("からといって", "～からといって"),
            ("に越したことはない", "～に越したことはない"),
            ("を余儀なくされ", "～を余儀なくされる"),
            ("余儀なくされた", "～を余儀なくされる"),
            ("ざるを得ない", "～ざるを得ない"),
            ("ずにはいられない", "～ずにはいられない"),
            ("にほかならない", "～にほかならない"),
            ("までもない", "～までもない"),
            ("わけではない", "～わけではない"),
            ("わけにはいかない", "～わけにはいかない"),
            ("ないわけにはいかない", "～ないわけにはいかない"),
            ("にしては", "～にしては"),
            ("としても", "～としても"),
            ("にしても", "～にしても"),
            ("に伴って", "～に伴って"),
            ("につれて", "～につれて"),
            ("に応じて", "～に応じて"),
            ("に基づいて", "～に基づいて"),
            ("をもとに", "～をもとに"),
            ("に即して", "～に即して"),
            ("に沿って", "～に沿って"),
            ("に反して", "～に反して"),
            ("にかかわらず", "～にかかわらず"),
            ("にもかかわらず", "～にもかかわらず"),
            ("ものの", "～ものの"),
            ("ものなら", "～ものなら"),
            ("ものだから", "～ものだから"),
            ("ものを", "～ものを"),
            ("ことなく", "～ことなく"),
            ("ことから", "～ことから"),
            ("ことだし", "～ことだし"),
            ("というもの", "～というもの"),
            ("というより", "～というより"),
            ("というか", "～というか"),
            ("とともに", "～とともに"),
            ("ともなると", "～ともなると"),
            ("となると", "～となると"),
            ("ないことには", "～ないことには"),
            ("かねない", "～かねない"),
            ("かねる", "～かねる"),
            ("に堪えない", "～に堪えない"),
            ("に足る", "～に足る"),
            ("に至る", "～に至る"),
            ("に至って", "～に至って"),
            ("に際して", "～に際して"),
            ("にあたって", "～にあたって"),
            ("をめぐって", "～をめぐって"),
            ("を問わず", "～を問わず"),
            ("を通じて", "～を通じて"),
            ("を通して", "～を通して"),
            ("を皮切りに", "～を皮切りに"),
            ("にとどまらず", "～にとどまらず"),
            ("のみならず", "～のみならず"),
            ("ばかりか", "～ばかりか"),
            ("だけあって", "～だけあって"),
            ("だけに", "～だけに"),
            ("に限って", "～に限って"),
            ("に限り", "～に限り"),
            ("に限らず", "～に限らず"),
            ("次第だ", "～次第だ"),
            ("次第で", "～次第で"),
            ("次第では", "～次第では"),
        ]
        patterns.extend(label for marker, label in rules if marker in sentence)
        return list(dict.fromkeys(patterns))

    def _best_pattern_match(self, patterns: list[str], text: str) -> str | None:
        normalized_text = text.replace(" ", "")
        for pattern in patterns:
            if any(keyword and keyword in normalized_text for keyword in self._pattern_keywords(pattern)):
                return pattern
        return None

    def _pattern_keyword(self, pattern: str) -> str:
        return pattern.replace("～", "").replace(" ", "")

    def _pattern_keywords(self, pattern: str) -> list[str]:
        keyword = self._pattern_keyword(pattern)
        aliases = {
            "を余儀なくされる": ["を余儀なくされ", "余儀なくされ"],
            "ざるを得ない": ["ざるを得ない", "ざるをえない"],
            "ずにはいられない": ["ずにはいられない", "ずにいられない"],
            "にほかならない": ["にほかならない", "に他ならない"],
            "に基づいて": ["に基づいて", "に基づき", "に基づく"],
            "に伴って": ["に伴って", "に伴い", "に伴う"],
            "につれて": ["につれて", "につれ"],
            "に応じて": ["に応じて", "に応じた", "に応じ"],
            "にかかわらず": ["にかかわらず", "に関わらず"],
            "にもかかわらず": ["にもかかわらず", "にも関わらず"],
            "に至る": ["に至る", "に至った", "に至り"],
            "に限らず": ["に限らず", "に限らない"],
        }
        return list(dict.fromkeys([keyword, *aliases.get(keyword, [])]))

    def _find_keyword_span(self, text: str, keyword: str) -> tuple[int, int] | None:
        if not keyword:
            return None

        compact_chars: list[str] = []
        original_indexes: list[int] = []
        for index, char in enumerate(text):
            if char.isspace():
                continue
            compact_chars.append(char)
            original_indexes.append(index)

        compact_text = "".join(compact_chars)
        compact_start = compact_text.find(keyword)
        if compact_start == -1:
            return None

        compact_end = compact_start + len(keyword) - 1
        return original_indexes[compact_start], original_indexes[compact_end] + 1

    def _extract_pattern_excerpt(self, text: str, pattern: str | None) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        if not pattern:
            return self._shorten_excerpt(text, 180)

        span = self._find_first_keyword_span(text, pattern)
        if not span:
            return self._shorten_excerpt(text, 180)

        start, end = span
        option_excerpt = self._extract_option_excerpt(text, start)
        if option_excerpt:
            return self._shorten_excerpt_around_span(option_excerpt, pattern, max_chars=120)

        left_boundaries = [text.rfind(mark, 0, start) for mark in "。！？?"]
        left = max(left_boundaries)
        left = 0 if left == -1 else left + 1

        right_candidates = [
            index for mark in "。！？?" if (index := text.find(mark, end)) != -1
        ]
        right = min(right_candidates) + 1 if right_candidates else len(text)

        excerpt = text[left:right].strip()
        return self._shorten_excerpt_around_span(excerpt, pattern, max_chars=180)

    def _extract_option_excerpt(self, text: str, keyword_start: int) -> str | None:
        marker_matches = list(re.finditer(r"(?<![0-9０-９])([1-4１-４])(?=\s|[ぁ-んァ-ン一-龥])", text))
        if len(marker_matches) < 2:
            return None

        current_marker = None
        next_marker = None
        for index, marker in enumerate(marker_matches):
            if marker.start() <= keyword_start:
                current_marker = marker
                next_marker = marker_matches[index + 1] if index + 1 < len(marker_matches) else None

        if not current_marker:
            return None

        option_start = current_marker.start()
        option_end = next_marker.start() if next_marker else len(text)
        if not (option_start <= keyword_start <= option_end):
            return None

        option = text[option_start:option_end].strip()
        if len(option) > 160:
            return None
        return option

    def _shorten_excerpt(self, text: str, max_chars: int) -> str:
        text = text.strip()
        if len(text) <= max_chars:
            return text
        return f"{text[:max_chars].rstrip()}..."

    def _shorten_excerpt_around_span(
        self, text: str, pattern: str | None, max_chars: int
    ) -> str:
        if len(text) <= max_chars or not pattern:
            return text

        span = self._find_first_keyword_span(text, pattern)
        if not span:
            return self._shorten_excerpt(text, max_chars)

        start, end = span
        left = max(0, start - 70)
        right = min(len(text), end + 90)
        excerpt = text[left:right].strip()
        if left > 0:
            excerpt = f"...{excerpt}"
        if right < len(text):
            excerpt = f"{excerpt}..."
        return excerpt

    def _find_first_keyword_span(self, text: str, pattern: str) -> tuple[int, int] | None:
        for keyword in self._pattern_keywords(pattern):
            span = self._find_keyword_span(text, keyword)
            if span:
                return span
        return None

    def _vectorize(self, text: str) -> Counter[str]:
        normalized = re.sub(r"\s+", "", text)
        tokens: list[str] = re.findall(r"[A-Za-z0-9]+", normalized)
        japanese = re.findall(r"[ぁ-んァ-ン一-龥ー]+", normalized)
        for sequence in japanese:
            for n in (2, 3):
                tokens.extend(sequence[i : i + n] for i in range(max(0, len(sequence) - n + 1)))
        return Counter(tokens)

    def _cosine(self, left: Counter[str], right: Counter[str]) -> float:
        if not left or not right:
            return 0.0

        dot = sum(value * right.get(token, 0) for token, value in left.items())
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        if not left_norm or not right_norm:
            return 0.0
        return dot / (left_norm * right_norm)

    def _to_related_example(
        self,
        score: float,
        chunk: dict[str, Any],
        matched_pattern: str | None,
        retrieval_method: str,
    ) -> RelatedExample:
        excerpt = self._extract_pattern_excerpt(chunk.get("text", ""), matched_pattern)

        if matched_pattern:
            why_related = (
                f"Contains the related grammar pattern {matched_pattern}; "
                f"retrieved via {retrieval_method}."
            )
        else:
            why_related = f"Retrieved via {retrieval_method} similarity search."

        return RelatedExample(
            source=chunk.get("source", ""),
            year=chunk.get("year"),
            month=chunk.get("month"),
            section=chunk.get("section", "General"),
            question_id=chunk.get("question_id"),
            page=int(chunk.get("page", 0)),
            related_pattern=matched_pattern,
            excerpt=excerpt,
            why_related=why_related,
            score=round(score, 4),
        )

    def _build_study_note(self, patterns: list[str], examples: list[RelatedExample]) -> str:
        if not examples:
            return "No related examples found. Try ingesting more JLPT PDFs or using a sentence with a clearer grammar pattern."
        if patterns:
            if "～ないとも限らない" in patterns and "～とは限らない" in patterns:
                return "Compare ～ないとも限らない with ～とは限らない. Both express that something cannot be ruled out, but ～ないとも限らない sounds more cautious and possibility-focused."
            return f"Review the detected pattern {', '.join(patterns)} and compare how it appears across similar JLPT questions."
        return "No explicit grammar pattern was detected, so results are based on local similarity. Try a sentence containing a clear JLPT grammar expression for better matches."

    def _load_index(self) -> dict[str, Any]:
        if not self.index_file.exists():
            return {"documents": [], "chunks": []}
        return json.loads(self.index_file.read_text(encoding="utf-8"))
