from __future__ import annotations

import re

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def question_text(item: dict[str, object]) -> str:
    return normalize_text(str(item.get("Question", "")))


def answer_text(item: dict[str, object]) -> str:
    return normalize_text(str(item.get("Answer", "")))


def document_text(item: dict[str, object], mode: str = "question") -> str:
    normalized_mode = mode.strip().lower()
    question = question_text(item)
    answer = answer_text(item)

    if normalized_mode == "question":
        return question
    if normalized_mode in {"question_answer", "question+answer"}:
        return f"Question: {question}\nAnswer: {answer}".strip()

    raise ValueError(f"Unsupported document_text_mode: {mode}")

