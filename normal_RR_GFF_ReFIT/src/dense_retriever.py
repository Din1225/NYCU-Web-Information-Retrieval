# 主要用途：使用 Qwen/Qwen3-Embedding-4B 建立 dense retriever，並支援 ReFIT 使用 query embedding 重新檢索。

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy.lib.format import open_memmap
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.qwen_utils import (
    DEFAULT_RETRIEVAL_INSTRUCTION,
    build_model_load_kwargs,
    canonical_model_reference,
    ensure_cuda_for_4bit,
    format_query_with_instruction,
    last_token_pool,
    resolve_pretrained_source,
)
from src.text_processing import normalize_text, question_text


def build_dense_cache_paths(
    documents: list[dict[str, Any]],
    model_name_or_path: str | Path,
    cache_dir: str | Path,
    max_length: int,
    use_4bit: bool,
    compute_dtype: str,
    query_instruction: str | None,
) -> dict[str, Any]:
    cache_key = build_dense_cache_key(
        documents=documents,
        model_name_or_path=model_name_or_path,
        max_length=max_length,
        use_4bit=use_4bit,
        compute_dtype=compute_dtype,
        query_instruction=query_instruction,
    )
    cache_dir_path = Path(cache_dir)
    return {
        "cache_key": cache_key,
        "embeddings": cache_dir_path / f"qwen3_embedding_{cache_key}.npy",
        "metadata": cache_dir_path / f"qwen3_embedding_{cache_key}.json",
        "partial_embeddings": cache_dir_path / f"qwen3_embedding_{cache_key}.partial.npy",
        "partial_progress": cache_dir_path / f"qwen3_embedding_{cache_key}.partial.json",
    }


def build_dense_cache_key(
    documents: list[dict[str, Any]],
    model_name_or_path: str | Path,
    max_length: int,
    use_4bit: bool,
    compute_dtype: str,
    query_instruction: str | None,
) -> str:
    digest = hashlib.sha256()
    digest.update(canonical_model_reference(model_name_or_path).encode("utf-8"))
    digest.update(str(max_length).encode("utf-8"))
    digest.update(str(use_4bit).encode("utf-8"))
    digest.update(compute_dtype.encode("utf-8"))
    digest.update(str(query_instruction).encode("utf-8"))
    for document in documents:
        digest.update(str(document["ID"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(normalize_text(question_text(document)).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def delete_dense_cache_files(cache_paths: dict[str, Any]) -> list[str]:
    deleted_paths: list[str] = []
    for key in ("embeddings", "metadata", "partial_embeddings", "partial_progress"):
        path = Path(cache_paths[key])
        try:
            path.unlink()
            deleted_paths.append(str(path))
        except FileNotFoundError:
            continue
    return deleted_paths


class DenseRetriever:
    """Dense retriever using Qwen/Qwen3-Embedding-4B through Transformers."""

    def __init__(
        self,
        documents: list[dict[str, Any]],
        model_name_or_path: str | Path,
        cache_dir: str | Path,
        model_cache_dir: str | Path | None = None,
        batch_size: int = 32,
        max_length: int = 512,
        use_4bit: bool = True,
        compute_dtype: str = "float16",
        use_cache: bool = True,
        query_instruction: str | None = DEFAULT_RETRIEVAL_INSTRUCTION,
    ) -> None:
        self.documents = documents
        self.model_name_or_path = canonical_model_reference(model_name_or_path)
        self.cache_dir = Path(cache_dir)
        self.model_cache_dir = Path(model_cache_dir) if model_cache_dir is not None else None
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_4bit = use_4bit
        self.compute_dtype = compute_dtype
        self.use_cache = use_cache
        self.query_instruction = query_instruction
        self._tokenizer = None
        self._model = None
        self._pretrained_source = None
        self.embeddings = self._load_or_build_embeddings()

    @property
    def pretrained_source(self) -> str:
        if self._pretrained_source is None:
            self._pretrained_source = resolve_pretrained_source(self.model_name_or_path, self.model_cache_dir)
        return self._pretrained_source

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.pretrained_source,
                padding_side="left",
                use_fast=False,
            )
            if self._tokenizer.pad_token is None and self._tokenizer.eos_token is not None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

    @property
    def model(self):
        """延遲載入 Qwen embedding 模型。"""
        if self._model is None:
            ensure_cuda_for_4bit(self.use_4bit)
            self._model = AutoModel.from_pretrained(
                self.pretrained_source,
                **build_model_load_kwargs(self.use_4bit, self.compute_dtype),
            )
            self._model.eval()
        return self._model

    def release_resources(self, clear_tokenizer: bool = True) -> None:
        """釋放 dense 模型與 tokenizer，降低單卡顯存峰值。"""
        self._model = None
        if clear_tokenizer:
            self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def encode_query(self, query: str) -> np.ndarray:
        """將 query 文字編碼成 L2-normalized dense vector。"""
        return self.encode_queries([query])[0]

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        """將多個 query 文字批次編碼成 L2-normalized dense vectors。"""
        if not queries:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        prepared_queries = [
            format_query_with_instruction(normalize_text(query), self.query_instruction)
            for query in queries
        ]
        return self._encode_texts(prepared_queries)

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
                print(f"Loaded cached dense embeddings from {cache_paths['embeddings']}", flush=True)
                return np.load(cache_paths["embeddings"])

        texts = [normalize_text(question_text(document)) for document in self.documents]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        embeddings = self._encode_in_batches(texts, cache_paths)
        np.save(cache_paths["embeddings"], embeddings)
        cache_paths["metadata"].write_text(
            json.dumps(
                {
                    "cache_key": cache_paths["cache_key"],
                    "model_name_or_path": self.model_name_or_path,
                    "num_documents": len(self.documents),
                    "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
                    "use_4bit": self.use_4bit,
                    "compute_dtype": self.compute_dtype,
                    "query_instruction": self.query_instruction,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        self._cleanup_partial_cache(cache_paths)
        return embeddings

    def _encode_in_batches(self, texts: list[str], cache_paths: dict[str, Any]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        next_index = 0
        memmap = None
        partial_progress = self._load_cache_metadata(cache_paths["partial_progress"])
        can_resume_partial_cache = self.use_cache and self._is_valid_partial_cache(
            partial_progress,
            cache_paths["cache_key"],
            len(texts),
        )
        if can_resume_partial_cache:
            next_index = int(partial_progress.get("next_index", 0))
            if cache_paths["partial_embeddings"].exists():
                try:
                    memmap = np.load(cache_paths["partial_embeddings"], mmap_mode="r+")
                    expected_dim = int(partial_progress.get("embedding_dim", -1))
                    if (
                        memmap.ndim != 2
                        or memmap.shape[0] != len(texts)
                        or (expected_dim > 0 and memmap.shape[1] != expected_dim)
                    ):
                        memmap = None
                        next_index = 0
                        self._cleanup_partial_cache(cache_paths)
                except (OSError, ValueError):
                    memmap = None
                    next_index = 0
                    self._cleanup_partial_cache(cache_paths)
            else:
                next_index = 0
                self._cleanup_partial_cache(cache_paths)
        elif cache_paths["partial_embeddings"].exists() or cache_paths["partial_progress"].exists():
            self._cleanup_partial_cache(cache_paths)

        if next_index > 0:
            print(
                f"Resuming dense embedding cache from document {next_index}/{len(texts)}.",
                flush=True,
            )

        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        initial_batches = min(next_index // self.batch_size, total_batches)
        for start in tqdm(
            range(next_index, len(texts), self.batch_size),
            desc="Encoding documents with Qwen3-Embedding-4B",
            initial=initial_batches,
            total=total_batches,
        ):
            batch = texts[start : start + self.batch_size]
            batch_embeddings = self._encode_texts(batch)
            end = start + len(batch)

            if memmap is None:
                memmap = self._create_partial_memmap(
                    cache_path=cache_paths["partial_embeddings"],
                    shape=(len(texts), int(batch_embeddings.shape[1])),
                )

            memmap[start:end] = batch_embeddings
            memmap.flush()
            if self.use_cache:
                self._write_partial_progress(
                    cache_paths=cache_paths,
                    next_index=end,
                    num_texts=len(texts),
                    embedding_dim=int(batch_embeddings.shape[1]),
                )

        if memmap is None:
            return np.empty((0, 0), dtype=np.float32)

        embeddings = np.asarray(memmap, dtype=np.float32)
        return embeddings.copy()

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        batch_dict = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch_dict = batch_dict.to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.detach().to(torch.float32).cpu().numpy()

    def _cache_paths(self) -> dict[str, Any]:
        return build_dense_cache_paths(
            documents=self.documents,
            model_name_or_path=self.model_name_or_path,
            cache_dir=self.cache_dir,
            max_length=self.max_length,
            use_4bit=self.use_4bit,
            compute_dtype=self.compute_dtype,
            query_instruction=self.query_instruction,
        )

    def _cache_key(self) -> str:
        return build_dense_cache_key(
            documents=self.documents,
            model_name_or_path=self.model_name_or_path,
            max_length=self.max_length,
            use_4bit=self.use_4bit,
            compute_dtype=self.compute_dtype,
            query_instruction=self.query_instruction,
        )

    @staticmethod
    def _load_cache_metadata(path: Path) -> dict[str, Any]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_partial_progress(
        self,
        cache_paths: dict[str, Any],
        next_index: int,
        num_texts: int,
        embedding_dim: int,
    ) -> None:
        cache_paths["partial_progress"].write_text(
            json.dumps(
                {
                    "cache_key": cache_paths["cache_key"],
                    "next_index": int(next_index),
                    "num_texts": int(num_texts),
                    "embedding_dim": int(embedding_dim),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _create_partial_memmap(cache_path: Path, shape: tuple[int, int]) -> np.memmap:
        try:
            return open_memmap(
                cache_path,
                mode="w+",
                dtype=np.float32,
                shape=shape,
            )
        except (OSError, ValueError):
            try:
                cache_path.unlink()
            except FileNotFoundError:
                pass
            return open_memmap(
                cache_path,
                mode="w+",
                dtype=np.float32,
                shape=shape,
            )

    @staticmethod
    def _is_valid_partial_cache(progress: dict[str, Any], cache_key: str, num_texts: int) -> bool:
        return (
            progress.get("cache_key") == cache_key
            and int(progress.get("num_texts", -1)) == int(num_texts)
            and int(progress.get("next_index", -1)) >= 0
        )

    @staticmethod
    def _cleanup_partial_cache(cache_paths: dict[str, Any]) -> None:
        for key in ("partial_embeddings", "partial_progress"):
            try:
                cache_paths[key].unlink()
            except FileNotFoundError:
                pass


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
