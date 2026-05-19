from __future__ import annotations

import re
from pathlib import Path

import jieba

_WHITESPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+")
_JIEBA_CACHE_READY = False


def _ensure_jieba_cache_dir() -> None:
    """將 jieba 快取固定到 repo 內可寫路徑，避免依賴 /tmp。"""
    global _JIEBA_CACHE_READY
    if _JIEBA_CACHE_READY:
        return

    cache_dir = Path(__file__).resolve().parents[1] / "outputs" / "cache" / "jieba"
    cache_dir.mkdir(parents=True, exist_ok=True)
    jieba.dt.tmp_dir = str(cache_dir)
    jieba.dt.cache_file = "jieba.cache"
    _JIEBA_CACHE_READY = True


def normalize_text(text: str) -> str:
    """將連續空白統一成單一空白。"""
    return _WHITESPACE_RE.sub(" ", text).strip()


def question_text(item: dict[str, object]) -> str:
    """從 query 或 document 物件中取出 Question 欄位並正規化。"""
    return normalize_text(str(item.get("Question", "")))


def answer_text(item: dict[str, object]) -> str:
    """從 document 物件中取出 Answer 欄位並正規化。"""
    return normalize_text(str(item.get("Answer", "")))


def document_text(item: dict[str, object]) -> str:
    """將 document 的 Question 與 Answer 串接成檢索文字。"""
    question = question_text(item)
    answer = answer_text(item)
    return normalize_text(f"{question} {answer}")


def bm25_tokenize(text: str) -> list[str]:
    """以 jieba 為主做中英文混合斷詞，供 BM25 使用。"""
    normalized = normalize_text(text)
    if not normalized:
        return []

    _ensure_jieba_cache_dir()

    tokens: list[str] = []
    for segment in jieba.lcut(normalized, cut_all=False):
        cleaned_segment = normalize_text(segment).lower()
        if not cleaned_segment:
            continue

        matches = _TOKEN_RE.findall(cleaned_segment)
        if matches:
            for match in matches:
                if not match.strip():
                    continue
                tokens.extend(_expand_token(match.lower()))
            continue

        if any(char.isalnum() or "\u4e00" <= char <= "\u9fff" for char in cleaned_segment):
            tokens.extend(_expand_token(cleaned_segment))

    return tokens


def _expand_token(token: str) -> list[str]:
    """將 token 展開成 BM25 使用的最小單位。"""
    if not token:
        return []
    if re.fullmatch(r"[A-Za-z0-9]+", token):
        return [token]
    if re.fullmatch(r"[\u4e00-\u9fff]+", token):
        return _expand_chinese_token(token)
    return [token]


def _expand_chinese_token(token: str) -> list[str]:
    if len(token) == 1:
        return [token]

    expanded = [token[index : index + 2] for index in range(len(token) - 1)]
    expanded.extend(char for char in token if char.strip())
    return expanded
