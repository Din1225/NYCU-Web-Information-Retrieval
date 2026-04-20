# 主要用途：使用 BAAI/bge-m3 建立 dense retriever，並快取歷史問題 embedding。

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from src.text_processing import normalize_text, question_text


# 封裝 bge-m3 向量檢索流程，包含文件 embedding 建立、快取與 cosine similarity 搜尋。
class DenseRetriever:
    """Dense retriever using BAAI/bge-m3 through FlagEmbedding."""

    # 初始化 dense retriever，設定模型路徑、快取路徑，並載入或建立文件 embedding。
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

    # 延遲載入 bge-m3 模型，只有第一次需要編碼時才真正載入模型到記憶體。
    @property
    def model(self):
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel

            self._model = BGEM3FlagModel(str(self.model_path), use_fp16=self.use_fp16)
        return self._model

    # 將 query 編碼成向量，與所有文件 embedding 做相似度計算並取 top-k。
    def retrieve(self, query: str, top_k: int = 50) -> list[dict[str, Any]]:
        if self.embeddings.shape[0] == 0:
            return []

        query_embedding = self._encode_texts([normalize_text(query)])
        scores = self.embeddings @ query_embedding[0]
        top_k = min(top_k, scores.size)
        top_indices = np.argpartition(-scores, top_k - 1)[:top_k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        results: list[dict[str, Any]] = []
        for doc_index in top_indices:
            results.append(
                {
                    "doc_index": int(doc_index),
                    "doc_id": self.documents[int(doc_index)]["ID"],
                    "dense_score": float(scores[int(doc_index)]),
                    "source": "dense",
                }
            )
        return results

    # 優先從快取讀取文件 embedding；若快取不存在或不相容，重新編碼並寫入快取。
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

    # 將大量文件文字分批送進模型編碼，避免一次佔用過多記憶體。
    def _encode_in_batches(self, texts: list[str]) -> np.ndarray:
        batches: list[np.ndarray] = []
        for start in tqdm(range(0, len(texts), self.batch_size), desc="Encoding documents with bge-m3"):
            batch = texts[start : start + self.batch_size]
            batches.append(self._encode_texts(batch))
        if not batches:
            return np.empty((0, 0), dtype=np.float32)
        return np.vstack(batches).astype(np.float32)

    # 將一批文字轉成 dense vector，並做 L2 normalize 以便用內積計算 cosine similarity。
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
        return _l2_normalize(embeddings)

    # 根據目前資料與模型設定產生 embedding 快取檔案路徑。
    def _cache_paths(self) -> dict[str, Any]:
        cache_key = self._cache_key()
        return {
            "cache_key": cache_key,
            "embeddings": self.cache_dir / f"bge_m3_{cache_key}.npy",
            "metadata": self.cache_dir / f"bge_m3_{cache_key}.json",
        }

    # 依據模型路徑、max_length、文件 ID 與文件問題內容產生快取識別碼。
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

    # 讀取 embedding 快取的 metadata；若檔案損壞或不存在可解析內容，回傳空 dict。
    @staticmethod
    def _load_cache_metadata(path: Path) -> dict[str, Any]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}


# 對 embedding 做 L2 normalize，避免向量長度影響相似度排序。
def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return embeddings / norms
