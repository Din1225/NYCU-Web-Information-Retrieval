from __future__ import annotations

import hashlib
import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from src.text_processing import bm25_tokenize, document_text

_TOKENIZER_VERSION = "jieba_v1"


def build_bm25_cache_paths(
    documents: list[dict[str, Any]],
    cache_dir: str | Path,
    k1: float,
    b: float,
) -> dict[str, Any]:
    cache_key = build_bm25_cache_key(documents=documents, k1=k1, b=b)
    cache_dir_path = Path(cache_dir)
    return {
        "cache_key": cache_key,
        "index": cache_dir_path / f"bm25_index_{cache_key}.pkl",
        "metadata": cache_dir_path / f"bm25_index_{cache_key}.json",
    }


def build_bm25_cache_key(
    documents: list[dict[str, Any]],
    k1: float,
    b: float,
) -> str:
    digest = hashlib.sha256()
    digest.update(_TOKENIZER_VERSION.encode("utf-8"))
    digest.update(str(k1).encode("utf-8"))
    digest.update(str(b).encode("utf-8"))
    for document in documents:
        digest.update(str(document["ID"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(document_text(document).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def delete_bm25_cache_files(cache_paths: dict[str, Any]) -> list[str]:
    deleted_paths: list[str] = []
    for key in ("index", "metadata"):
        path = Path(cache_paths[key])
        try:
            path.unlink()
            deleted_paths.append(str(path))
        except FileNotFoundError:
            continue
    return deleted_paths


class BM25Retriever:
    """BM25 retriever over Question + Answer text."""

    def __init__(
        self,
        documents: list[dict[str, Any]],
        cache_dir: str | Path,
        k1: float = 1.5,
        b: float = 0.75,
        use_cache: bool = True,
    ) -> None:
        self.documents = documents
        self.cache_dir = Path(cache_dir)
        self.k1 = float(k1)
        self.b = float(b)
        self.use_cache = use_cache
        self.doc_lengths: np.ndarray = np.empty((0,), dtype=np.float32)
        self.avgdl: float = 0.0
        self.idf: dict[str, float] = {}
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._load_or_build_index()

    def retrieve(
        self,
        query: str,
        top_k: int = 100,
        score_key: str = "bm25_score",
        source: str = "bm25",
    ) -> list[dict[str, Any]]:
        if top_k <= 0 or not self.documents:
            return []

        scores = self.score(query)
        if scores.size == 0:
            return []

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

    def score(self, query: str) -> np.ndarray:
        query_tokens = bm25_tokenize(query)
        return self.score_tokens(query_tokens)

    def score_tokens(self, query_tokens: list[str]) -> np.ndarray:
        if not self.documents or not query_tokens:
            return np.zeros((len(self.documents),), dtype=np.float32)

        scores = np.zeros((len(self.documents),), dtype=np.float32)
        query_counter = Counter(query_tokens)
        avgdl = self.avgdl if self.avgdl > 0.0 else 1.0

        for token in query_counter:
            posting = self.postings.get(token)
            if posting is None:
                continue

            doc_indices, term_frequencies = posting
            idf = self.idf.get(token, 0.0)
            if idf == 0.0:
                continue

            doc_lengths = self.doc_lengths[doc_indices]
            numerator = term_frequencies * (self.k1 + 1.0)
            denominator = term_frequencies + self.k1 * (1.0 - self.b + self.b * doc_lengths / avgdl)
            scores[doc_indices] += idf * (numerator / denominator)

        return scores

    def _load_or_build_index(self) -> None:
        cache_paths = self._cache_paths()
        if self.use_cache and cache_paths["index"].exists() and cache_paths["metadata"].exists():
            metadata = self._load_json(cache_paths["metadata"])
            if metadata.get("cache_key") == cache_paths["cache_key"]:
                with cache_paths["index"].open("rb") as file:
                    payload = pickle.load(file)
                self.doc_lengths = np.asarray(payload["doc_lengths"], dtype=np.float32)
                self.avgdl = float(payload["avgdl"])
                self.idf = {str(key): float(value) for key, value in payload["idf"].items()}
                self.postings = {
                    str(token): (
                        np.asarray(doc_indices, dtype=np.int32),
                        np.asarray(term_frequencies, dtype=np.float32),
                    )
                    for token, (doc_indices, term_frequencies) in payload["postings"].items()
                }
                print(f"Loaded cached BM25 index from {cache_paths['index']}", flush=True)
                return

        payload = self._build_index_payload()
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with cache_paths["index"].open("wb") as file:
                pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
            cache_paths["metadata"].write_text(
                json.dumps(
                    {
                        "cache_key": cache_paths["cache_key"],
                        "num_documents": len(self.documents),
                        "k1": self.k1,
                        "b": self.b,
                        "tokenizer_version": _TOKENIZER_VERSION,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

    def _build_index_payload(self) -> dict[str, Any]:
        num_documents = len(self.documents)
        if num_documents == 0:
            self.doc_lengths = np.empty((0,), dtype=np.float32)
            self.avgdl = 0.0
            self.idf = {}
            self.postings = {}
            return {
                "doc_lengths": self.doc_lengths,
                "avgdl": self.avgdl,
                "idf": self.idf,
                "postings": self.postings,
            }

        doc_lengths = np.zeros((num_documents,), dtype=np.float32)
        document_frequencies: Counter[str] = Counter()
        posting_lists: dict[str, list[tuple[int, int]]] = defaultdict(list)

        for doc_index, document in enumerate(tqdm(self.documents, desc="Building BM25 index")):
            tokens = bm25_tokenize(document_text(document))
            doc_lengths[doc_index] = float(len(tokens))
            token_counter = Counter(tokens)
            document_frequencies.update(token_counter.keys())
            for token, frequency in token_counter.items():
                posting_lists[token].append((doc_index, int(frequency)))

        average_doc_length = float(doc_lengths.mean()) if num_documents > 0 else 0.0
        idf = {
            token: math.log(1.0 + (num_documents - df + 0.5) / (df + 0.5))
            for token, df in document_frequencies.items()
        }
        postings = {
            token: (
                np.asarray([doc_index for doc_index, _ in pairs], dtype=np.int32),
                np.asarray([frequency for _, frequency in pairs], dtype=np.float32),
            )
            for token, pairs in posting_lists.items()
        }

        self.doc_lengths = doc_lengths
        self.avgdl = average_doc_length
        self.idf = idf
        self.postings = postings
        return {
            "doc_lengths": doc_lengths,
            "avgdl": average_doc_length,
            "idf": idf,
            "postings": postings,
        }

    def _cache_paths(self) -> dict[str, Any]:
        return build_bm25_cache_paths(
            documents=self.documents,
            cache_dir=self.cache_dir,
            k1=self.k1,
            b=self.b,
        )

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
