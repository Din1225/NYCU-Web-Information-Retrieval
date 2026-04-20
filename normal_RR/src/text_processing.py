# 主要用途：集中處理文字正規化與中文斷詞，供 BM25 和其他模組共用。

from __future__ import annotations

from pathlib import Path
import re

import jieba

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_JIEBA_CACHE_DIR = _PROJECT_ROOT / "outputs" / "cache" / "jieba"
_JIEBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
jieba.dt.tmp_dir = str(_JIEBA_CACHE_DIR)
jieba.dt.cache_file = "jieba.cache"

_WHITESPACE_RE = re.compile(r"\s+")


# 將文字中的連續空白統一成單一空白，避免換行或多空格影響檢索。
def normalize_text(text: str) -> str:
    """Normalize whitespace without changing Chinese punctuation or casing."""
    return _WHITESPACE_RE.sub(" ", text).strip()


# 使用 jieba 將中文文字切成 BM25 可使用的 token list。
def tokenize_zh(text: str) -> list[str]:
    """Tokenize Chinese text with jieba for BM25."""
    normalized = normalize_text(text)
    return [token.strip() for token in jieba.lcut(normalized) if token.strip()]


# 從 document 或 query 物件中取出 Question 欄位，並做基本文字正規化。
def question_text(item: dict[str, object]) -> str:
    """Return normalized question text from a document or query item."""
    return normalize_text(str(item.get("Question", "")))
