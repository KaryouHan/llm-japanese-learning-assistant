import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

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


class KnowledgeService:
    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parents[3]
        self.knowledge_root = self.project_root / "knowledge_base"
        self.raw_dir = self.knowledge_root / "raw"
        self.uploads_dir = self.knowledge_root / "uploads"
        self.index_dir = self.knowledge_root / "index"
        self.index_file = self.index_dir / "index.json"

    def status(self) -> KnowledgeStatusResponse:
        index = self._load_index()
        return KnowledgeStatusResponse(
            raw_pdf_count=len(list(self.raw_dir.glob("*.pdf"))),
            uploaded_pdf_count=len(list(self.uploads_dir.glob("*.pdf"))),
            indexed_document_count=len(index.get("documents", [])),
            indexed_chunk_count=len(index.get("chunks", [])),
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
        skipped_files: list[str] = []

        for pdf_path in pdf_paths:
            try:
                document_chunks = self._extract_pdf_chunks(pdf_path)
            except Exception as exc:
                skipped_files.append(f"{pdf_path.name}: {exc}")
                continue

            documents.append(
                {
                    "source": pdf_path.name,
                    "path": str(pdf_path),
                    "chunk_count": len(document_chunks),
                }
            )
            chunks.extend(document_chunks)

        payload = {
            "version": 1,
            "documents": documents,
            "chunks": chunks,
        }
        self.index_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return KnowledgeIngestResponse(
            document_count=len(documents),
            chunk_count=len(chunks),
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

        patterns = self._detect_patterns(request.sentence)
        query_vector = self._vectorize(request.sentence + " " + " ".join(patterns))

        scored: list[tuple[float, dict[str, Any], str | None]] = []
        for chunk in chunks:
            text = chunk.get("text", "")
            chunk_vector = Counter(chunk.get("vector", {}))
            vector_score = self._cosine(query_vector, chunk_vector)
            matched_pattern = self._best_pattern_match(patterns, text)
            pattern_score = 1.0 if matched_pattern else 0.0
            score = pattern_score * 3.0 + vector_score
            if score > 0:
                scored.append((score, chunk, matched_pattern))

        scored.sort(key=lambda item: item[0], reverse=True)
        examples = [
            self._to_related_example(score, chunk, matched_pattern)
            for score, chunk, matched_pattern in scored[: request.top_k]
        ]

        return RelatedExamplesResponse(
            detected_patterns=patterns,
            related_examples=examples,
            study_note=self._build_study_note(patterns, examples),
        )

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
        return {
            "year": match.group(1) if match else None,
            "month": match.group(2) if match else None,
            "level": "N1" if "N1" in filename.upper() else None,
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

    def _detect_patterns(self, sentence: str) -> list[str]:
        patterns: list[str] = []
        if "ないとも限らない" in sentence:
            patterns.extend(["～ないとも限らない", "～とは限らない"])

        rules = [
            ("ないとも限らない", "～ないとも限らない"),
            ("とは限らない", "～とは限らない"),
            ("に越したことはない", "～に越したことはない"),
            ("を余儀なくされ", "～を余儀なくされる"),
            ("ざるを得ない", "～ざるを得ない"),
            ("ずにはいられない", "～ずにはいられない"),
            ("にほかならない", "～にほかならない"),
            ("までもない", "～までもない"),
            ("わけではない", "～わけではない"),
            ("わけにはいかない", "～わけにはいかない"),
            ("にしては", "～にしては"),
            ("としても", "～としても"),
            ("に伴って", "～に伴って"),
            ("につれて", "～につれて"),
            ("に応じて", "～に応じて"),
            ("に基づいて", "～に基づいて"),
        ]
        patterns.extend(label for marker, label in rules if marker in sentence)
        return list(dict.fromkeys(patterns))

    def _best_pattern_match(self, patterns: list[str], text: str) -> str | None:
        normalized_text = text.replace(" ", "")
        for pattern in patterns:
            keyword = pattern.replace("～", "").replace(" ", "")
            if keyword and keyword in normalized_text:
                return pattern
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
        self, score: float, chunk: dict[str, Any], matched_pattern: str | None
    ) -> RelatedExample:
        excerpt = chunk.get("text", "")
        if len(excerpt) > 260:
            excerpt = f"{excerpt[:260].strip()}..."

        why_related = (
            f"Matches the detected grammar pattern {matched_pattern}."
            if matched_pattern
            else "Retrieved by local text-vector similarity with the input sentence."
        )

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
            return "No related examples found. Try ingesting more N1 PDFs or using a sentence with a clearer grammar pattern."
        if patterns:
            if "～ないとも限らない" in patterns and "～とは限らない" in patterns:
                return "Compare ～ないとも限らない with ～とは限らない. Both express that something cannot be ruled out, but ～ないとも限らない sounds more cautious and possibility-focused."
            return f"Review the detected pattern {', '.join(patterns)} and compare how it appears across similar N1 questions."
        return "No explicit grammar pattern was detected, so results are based on local similarity. Try a sentence containing a clear N1 grammar expression for better matches."

    def _load_index(self) -> dict[str, Any]:
        if not self.index_file.exists():
            return {"documents": [], "chunks": []}
        return json.loads(self.index_file.read_text(encoding="utf-8"))
