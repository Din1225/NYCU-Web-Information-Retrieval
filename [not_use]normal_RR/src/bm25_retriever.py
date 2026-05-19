# 主要用途：建立 BM25 檢索器，用 jieba 斷詞後從歷史問題中召回 top-k 候選。

from __future__ import annotations

from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from src.text_processing import question_text, tokenize_zh


# 封裝 rank-bm25 的索引建立與查詢流程。
class BM25Retriever:
    """BM25 retriever over historical question text."""

    # 初始化 BM25 檢索器，先將所有歷史問題斷詞，再建立 BM25 index。
    def __init__(self, documents: list[dict[str, Any]], show_progress: bool = True) -> None:
        self.documents = documents
        self.tokenized_documents = [
            tokenize_zh(question_text(document))
            for document in tqdm(documents, desc="Tokenizing documents for BM25", disable=not show_progress)
        ]
        self.index = BM25Okapi(self.tokenized_documents)

    # 對單一 query 執行 BM25 檢索，回傳分數最高的 top-k 歷史問題。
    def retrieve(self, query: str, top_k: int = 50) -> list[dict[str, Any]]:
        query_tokens = tokenize_zh(query)
        scores = np.asarray(self.index.get_scores(query_tokens), dtype=np.float32)
        if scores.size == 0:
            return []

        top_k = min(top_k, scores.size)
        top_indices = np.argpartition(-scores, top_k - 1)[:top_k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        results: list[dict[str, Any]] = []
        for doc_index in top_indices:
            results.append(
                {
                    "doc_index": int(doc_index),
                    "doc_id": self.documents[int(doc_index)]["ID"],
                    "bm25_score": float(scores[int(doc_index)]),
                    "source": "bm25",
                }
            )
        return results
