# 主要用途：集中處理文字正規化，供 dense retriever 與 reranker 共用。

from __future__ import annotations

import re

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """將連續空白統一成單一空白。"""
    return _WHITESPACE_RE.sub(" ", text).strip()


def question_text(item: dict[str, object]) -> str:
    """從 document 或 query 物件中取出 Question 欄位並正規化。"""
    return normalize_text(str(item.get("Question", "")))

