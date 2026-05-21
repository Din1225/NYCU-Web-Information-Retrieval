from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.model_utils import (
    DEFAULT_RETRIEVAL_INSTRUCTION,
    build_qwen_model_load_kwargs,
    canonical_model_reference,
    ensure_cuda_for_4bit,
    format_query_with_instruction,
    infer_dense_backend,
    last_token_pool,
)
from src.text_processing import document_text, normalize_text


def _l2_normalize_matrix(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.size == 0:
        return embeddings.astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (embeddings / norms).astype(np.float32)


def _l2_normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def _build_cache_key(
    documents: list[dict[str, Any]],
    model_name_or_path: str,
    backend: str,
    max_length: int,
    document_text_mode: str,
    query_instruction: str | None,
    use_4bit: bool,
    compute_dtype: str,
    bge_use_fp16: bool,
) -> str:
    digest = hashlib.sha256()
    digest.update(backend.encode("utf-8"))
    digest.update(canonical_model_reference(model_name_or_path).encode("utf-8"))
    digest.update(str(max_length).encode("utf-8"))
    digest.update(document_text_mode.encode("utf-8"))
    digest.update(str(query_instruction).encode("utf-8"))
    digest.update(str(use_4bit).encode("utf-8"))
    digest.update(compute_dtype.encode("utf-8"))
    digest.update(str(bge_use_fp16).encode("utf-8"))
    for document in documents:
        digest.update(str(document["ID"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(normalize_text(document_text(document, mode=document_text_mode)).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


class DenseRetriever:
    def __init__(
        self,
        documents: list[dict[str, Any]],
        model_name_or_path: str | Path,
        cache_dir: str | Path,
        backend: str = "auto",
        batch_size: int = 32,
        max_length: int = 512,
        use_4bit: bool = True,
        compute_dtype: str = "float16",
        bge_use_fp16: bool = False,
        use_cache: bool = True,
        document_text_mode: str = "question",
        query_instruction: str | None = DEFAULT_RETRIEVAL_INSTRUCTION,
    ) -> None:
        self.documents = documents
        self.model_name_or_path = canonical_model_reference(model_name_or_path)
        self.backend = infer_dense_backend(self.model_name_or_path, backend)
        self.cache_dir = Path(cache_dir)
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_4bit = use_4bit
        self.compute_dtype = compute_dtype
        self.bge_use_fp16 = bge_use_fp16
        self.use_cache = use_cache
        self.document_text_mode = document_text_mode
        self.query_instruction = query_instruction
        self._tokenizer = None
        self._model = None
        self._bge_tokenizer = None
        self._bge_impl = None
        self.embeddings = self._load_or_build_embeddings()

    @property
    def tokenizer(self):
        if self.backend != "qwen":
            raise AttributeError("Tokenizer is only used by the Qwen dense backend.")
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                padding_side="left",
                use_fast=False,
            )
            if self._tokenizer.pad_token is None and self._tokenizer.eos_token is not None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            if self.backend == "bge":
                try:
                    from FlagEmbedding import BGEM3FlagModel

                    self._model = BGEM3FlagModel(self.model_name_or_path, use_fp16=self.bge_use_fp16)
                    self._bge_impl = "flagembedding"
                except Exception as error:
                    try:
                        self._bge_tokenizer = AutoTokenizer.from_pretrained(
                            self.model_name_or_path,
                            use_fast=False,
                        )
                        self._model = AutoModel.from_pretrained(
                            self.model_name_or_path,
                            use_safetensors=True,
                        ).eval()
                        if torch.cuda.is_available():
                            self._model = self._model.to("cuda")
                            if self.bge_use_fp16:
                                self._model = self._model.half()
                        self._bge_impl = "transformers"
                        print(
                            "FlagEmbedding 載入失敗，改用 transformers+safetensors 作為 BGE dense fallback。"
                            f" 原始錯誤: {error}",
                            flush=True,
                        )
                    except Exception as transformers_error:
                        try:
                            from sentence_transformers import SentenceTransformer

                            self._model = SentenceTransformer(self.model_name_or_path)
                            if hasattr(self._model, "max_seq_length"):
                                self._model.max_seq_length = self.max_length
                            if torch.cuda.is_available():
                                self._model = self._model.to("cuda")
                            self._bge_impl = "sentence-transformers"
                            print(
                                "transformers+safetensors 載入失敗，改用 sentence-transformers 作為 BGE dense 次級 fallback。"
                                f" 原始錯誤: {transformers_error}",
                                flush=True,
                            )
                        except Exception as fallback_error:
                            raise RuntimeError(
                                "Unable to load BGE dense backend. "
                                "FlagEmbedding import failed, transformers+safetensors fallback failed, "
                                "and sentence-transformers fallback also failed."
                            ) from fallback_error
            else:
                ensure_cuda_for_4bit(self.use_4bit)
                self._model = AutoModel.from_pretrained(
                    self.model_name_or_path,
                    **build_qwen_model_load_kwargs(self.use_4bit, self.compute_dtype),
                )
                self._model.eval()
        return self._model

    @property
    def embedding_dim(self) -> int:
        if self.embeddings.ndim != 2 or self.embeddings.shape[0] == 0:
            return 0
        return int(self.embeddings.shape[1])

    def release_resources(self, clear_tokenizer: bool = True) -> None:
        self._model = None
        if clear_tokenizer:
            self._tokenizer = None
            self._bge_tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode_queries([query])[0]

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        if not queries:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        if self.backend == "bge":
            texts = [normalize_text(query) for query in queries]
        else:
            texts = [
                format_query_with_instruction(normalize_text(query), self.query_instruction)
                for query in queries
            ]
        return self._encode_texts(texts)

    def retrieve(self, query: str, top_k: int = 50, score_key: str = "dense_score") -> list[dict[str, Any]]:
        query_embedding = self.encode_query(query)
        return self.retrieve_by_embedding(query_embedding, top_k=top_k, score_key=score_key)

    def retrieve_by_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int = 50,
        score_key: str = "dense_score",
        source: str = "dense",
    ) -> list[dict[str, Any]]:
        if self.embeddings.shape[0] == 0 or top_k <= 0:
            return []

        normalized_query = _l2_normalize_vector(np.asarray(query_embedding, dtype=np.float32))
        scores = self.embeddings @ normalized_query
        top_k = min(top_k, int(scores.size))
        if top_k <= 0:
            return []

        top_indices = np.argpartition(-scores, top_k - 1)[:top_k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        results: list[dict[str, Any]] = []
        for rank, doc_index in enumerate(top_indices, start=1):
            results.append(
                {
                    "rank": rank,
                    "doc_index": int(doc_index),
                    "doc_id": self.documents[int(doc_index)]["ID"],
                    score_key: float(scores[int(doc_index)]),
                    "source": source,
                }
            )
        return results

    def _load_or_build_embeddings(self) -> np.ndarray:
        cache_paths = self._cache_paths()
        if self.use_cache and cache_paths["embeddings"].exists() and cache_paths["metadata"].exists():
            metadata = self._load_cache_metadata(cache_paths["metadata"])
            if metadata.get("cache_key") == cache_paths["cache_key"]:
                print(f"Loaded cached dense embeddings from {cache_paths['embeddings']}", flush=True)
                return np.load(cache_paths["embeddings"])

        texts = [normalize_text(document_text(document, mode=self.document_text_mode)) for document in self.documents]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        embeddings = self._encode_in_batches(texts)
        np.save(cache_paths["embeddings"], embeddings)
        cache_paths["metadata"].write_text(
            json.dumps(
                {
                    "cache_key": cache_paths["cache_key"],
                    "backend": self.backend,
                    "model_name_or_path": self.model_name_or_path,
                    "num_documents": len(self.documents),
                    "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
                    "document_text_mode": self.document_text_mode,
                    "query_instruction": self.query_instruction,
                    "use_4bit": self.use_4bit,
                    "compute_dtype": self.compute_dtype,
                    "bge_use_fp16": self.bge_use_fp16,
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
        desc = "Encoding documents with dense retriever"
        for start in tqdm(range(0, len(texts), self.batch_size), desc=desc):
            batch = texts[start : start + self.batch_size]
            batches.append(self._encode_texts(batch))
        if not batches:
            return np.empty((0, 0), dtype=np.float32)
        return np.vstack(batches).astype(np.float32)

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        if self.backend == "bge":
            model = self.model
            if self._bge_impl == "transformers":
                tokenizer = self._bge_tokenizer
                if tokenizer is None:
                    raise RuntimeError("BGE transformers fallback tokenizer is not initialized.")
                device = next(model.parameters()).device
                inputs = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {key: value.to(device) for key, value in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs, return_dict=True)
                    embeddings = outputs.last_hidden_state[:, 0]
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                return embeddings.detach().to(torch.float32).cpu().numpy()

            if self._bge_impl == "sentence-transformers":
                embeddings = model.encode(
                    texts,
                    batch_size=self.batch_size,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                return np.asarray(embeddings, dtype=np.float32)

            encoded = model.encode(
                texts,
                batch_size=self.batch_size,
                max_length=self.max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            embeddings = np.asarray(encoded["dense_vecs"], dtype=np.float32)
            return _l2_normalize_matrix(embeddings)

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
        cache_key = _build_cache_key(
            documents=self.documents,
            model_name_or_path=self.model_name_or_path,
            backend=self.backend,
            max_length=self.max_length,
            document_text_mode=self.document_text_mode,
            query_instruction=self.query_instruction if self.backend == "qwen" else None,
            use_4bit=self.use_4bit,
            compute_dtype=self.compute_dtype,
            bge_use_fp16=self.bge_use_fp16,
        )
        prefix = f"{self.backend}_dense"
        return {
            "cache_key": cache_key,
            "embeddings": self.cache_dir / f"{prefix}_{cache_key}.npy",
            "metadata": self.cache_dir / f"{prefix}_{cache_key}.json",
        }

    @staticmethod
    def _load_cache_metadata(path: Path) -> dict[str, Any]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
