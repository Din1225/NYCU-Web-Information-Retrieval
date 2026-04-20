# 主要用途：使用 BAAI/bge-reranker-v2-m3 對合併後的候選問題重新排序。

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.text_processing import normalize_text, question_text


# 封裝 cross-encoder reranker，對 query 與候選歷史問題配對打分。
class CrossEncoderReranker:
    """Cross-encoder reranker using BAAI/bge-reranker-v2-m3 through FlagEmbedding."""

    # 初始化 reranker 的模型路徑、batch size、最大長度與分數正規化設定。
    def __init__(
        self,
        model_path: str | Path,
        batch_size: int = 16,
        max_length: int = 512,
        use_fp16: bool = False,
        normalize_scores: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.normalize_scores = normalize_scores
        self._model = None

    # 延遲載入 reranker 模型，只有第一次 rerank 時載入記憶體。
    @property
    def model(self):
        if self._model is None:
            from FlagEmbedding import FlagReranker

            self._model = FlagReranker(str(self.model_path), use_fp16=self.use_fp16)
        return self._model

    # 對同一個 query 的所有候選文件打 cross-encoder 分數，排序後回傳 top-k。
    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        documents: list[dict[str, Any]],
        top_k: int = 30,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        normalized_query = normalize_text(query)
        pairs = [
            [normalized_query, question_text(documents[int(candidate["doc_index"])])]
            for candidate in candidates
        ]
        scores = self._compute_scores(pairs)

        reranked: list[dict[str, Any]] = []
        for candidate, score in zip(candidates, scores, strict=True):
            item = dict(candidate)
            item["rerank_score"] = float(score)
            reranked.append(item)

        reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        return reranked[: min(top_k, len(reranked))]

    # 呼叫 FlagEmbedding 的 compute_score，並相容不同版本的參數支援情況。
    def _compute_scores(self, pairs: list[list[str]]) -> list[float]:
        try:
            scores = self.model.compute_score(
                pairs,
                batch_size=self.batch_size,
                max_length=self.max_length,
                normalize=self.normalize_scores,
            )
        except TypeError:
            scores = self.model.compute_score(pairs, normalize=self.normalize_scores)

        if isinstance(scores, (float, int, np.floating)):
            return [float(scores)]
        return [float(score) for score in scores]
