# 主要用途：使用 BAAI/bge-m3 建立 dense retriever，並支援 ReFIT 使用 query embedding 重新檢索。

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from src.text_processing import normalize_text, question_text


class DenseRetriever:
    """Dense retriever using BAAI/bge-m3 through FlagEmbedding."""

    def __init__(
        self,
        documents: list[dict[str, Any]],
        model_path: str | Path,
        cache_dir: str | Path,
        batch_size: int = 32,
        max_length: int = 512,
        use_fp16: bool = False,
        use_cache: bool = True,
    ) -> None:
        self.documents = documents
        self.model_path = Path(model_path)
        self.cache_dir = Path(cache_dir)
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.use_cache = use_cache
        self._model = None
        self.embeddings = self._load_or_build_embeddings()

    @property
    def model(self):
        """延遲載入 bge-m3 模型。"""
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel

            self._model = BGEM3FlagModel(str(self.model_path), use_fp16=self.use_fp16)
        return self._model

    def encode_query(self, query: str) -> np.ndarray:
        """將 query 文字編碼成 L2-normalized dense vector。"""
        return self._encode_texts([normalize_text(query)])[0]

    def retrieve(self, query: str, top_k: int = 50) -> list[dict[str, Any]]:
        """用 query 文字執行 dense retrieval。"""
        query_embedding = self.encode_query(query)
        return self.retrieve_by_embedding(query_embedding, top_k=top_k, score_key="dense_score")

    def retrieve_by_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int = 50,
        score_key: str = "dense_score",
        source: str = "dense",
    ) -> list[dict[str, Any]]:
        """用指定 query embedding 執行 dense retrieval。"""
        if self.embeddings.shape[0] == 0 or top_k <= 0:
            return []

        normalized_query = l2_normalize_vector(np.asarray(query_embedding, dtype=np.float32))
        scores = self.embeddings @ normalized_query
        top_k = min(top_k, scores.size)
        top_indices = np.argpartition(-scores, top_k - 1)[:top_k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        results: list[dict[str, Any]] = []
        for rank, doc_index in enumerate(top_indices, start=1):
            results.append(
                {
                    "doc_index": int(doc_index),
                    "doc_id": self.documents[int(doc_index)]["ID"],
                    "rank": rank,
                    score_key: float(scores[int(doc_index)]),
                    "source": source,
                }
            )
        return results

    def embeddings_for_indices(self, doc_indices: list[int]) -> np.ndarray:
        """取得候選文件的 dense embeddings。"""
        if not doc_indices:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        return self.embeddings[np.asarray(doc_indices, dtype=np.int64)]

    def scores_for_indices(self, query_embedding: np.ndarray, doc_indices: list[int]) -> np.ndarray:
        """計算指定 query embedding 對一批文件的內積分數。"""
        if not doc_indices:
            return np.empty((0,), dtype=np.float32)
        normalized_query = l2_normalize_vector(np.asarray(query_embedding, dtype=np.float32))
        candidate_embeddings = self.embeddings_for_indices(doc_indices)
        return candidate_embeddings @ normalized_query

    @property
    def embedding_dim(self) -> int:
        if self.embeddings.ndim != 2 or self.embeddings.shape[0] == 0:
            return 0
        return int(self.embeddings.shape[1])

    def _load_or_build_embeddings(self) -> np.ndarray:
        cache_paths = self._cache_paths()
        if self.use_cache and cache_paths["embeddings"].exists() and cache_paths["metadata"].exists():
            metadata = self._load_cache_metadata(cache_paths["metadata"])
            if metadata.get("cache_key") == cache_paths["cache_key"]:
                return np.load(cache_paths["embeddings"])

        texts = [question_text(document) for document in self.documents]
        embeddings = self._encode_in_batches(texts)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_paths["embeddings"], embeddings)
        cache_paths["metadata"].write_text(
            json.dumps(
                {
                    "cache_key": cache_paths["cache_key"],
                    "model_path": str(self.model_path),
                    "num_documents": len(self.documents),
                    "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return embeddings

    def _encode_in_batches(self, texts: list[str]) -> np.ndarray:
        batches: list[np.ndarray] = []
        for start in tqdm(range(0, len(texts), self.batch_size), desc="Encoding documents with bge-m3"):
            batch = texts[start : start + self.batch_size]
            batches.append(self._encode_texts(batch))
        if not batches:
            return np.empty((0, 0), dtype=np.float32)
        return np.vstack(batches).astype(np.float32)

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        encoded = self.model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        embeddings = np.asarray(encoded["dense_vecs"], dtype=np.float32)
        return l2_normalize_matrix(embeddings)

    def _cache_paths(self) -> dict[str, Any]:
        cache_key = self._cache_key()
        return {
            "cache_key": cache_key,
            "embeddings": self.cache_dir / f"bge_m3_{cache_key}.npy",
            "metadata": self.cache_dir / f"bge_m3_{cache_key}.json",
        }

    def _cache_key(self) -> str:
        digest = hashlib.sha256()
        digest.update(str(self.model_path.resolve()).encode("utf-8"))
        digest.update(str(self.max_length).encode("utf-8"))
        for document in self.documents:
            digest.update(str(document["ID"]).encode("utf-8"))
            digest.update(b"\0")
            digest.update(question_text(document).encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()[:16]

    @staticmethod
    def _load_cache_metadata(path: Path) -> dict[str, Any]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}


def l2_normalize_matrix(embeddings: np.ndarray) -> np.ndarray:
    """對 2D embedding matrix 做 L2 normalize。"""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return embeddings / norms


def l2_normalize_vector(embedding: np.ndarray) -> np.ndarray:
    """對單一 embedding vector 做 L2 normalize。"""
    norm = float(np.linalg.norm(embedding))
    if norm == 0.0:
        return embedding.astype(np.float32)
    return (embedding / norm).astype(np.float32)

