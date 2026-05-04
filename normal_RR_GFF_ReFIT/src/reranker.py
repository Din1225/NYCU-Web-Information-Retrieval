# 主要用途：使用 Qwen/Qwen3-Reranker-4B 對 dense 候選文件計算 relevance feedback 分數。

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.qwen_utils import (
    DEFAULT_RETRIEVAL_INSTRUCTION,
    build_model_load_kwargs,
    canonical_model_reference,
    ensure_cuda_for_4bit,
    resolve_pretrained_source,
)
from src.text_processing import normalize_text, question_text


class CrossEncoderReranker:
    """Cross-encoder reranker using Qwen/Qwen3-Reranker-4B through Transformers."""

    _SYSTEM_PROMPT = (
        "請根據提供的 Instruct 和 Query，判斷 Document 是否符合檢索需求。"
        "回答只能是 yes 或 no。"
    )
    _SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(
        self,
        model_name_or_path: str | Path,
        model_cache_dir: str | Path | None = None,
        batch_size: int = 16,
        max_length: int = 512,
        use_4bit: bool = True,
        compute_dtype: str = "float16",
        normalize_scores: bool = True,
        instruction: str | None = DEFAULT_RETRIEVAL_INSTRUCTION,
    ) -> None:
        self.model_name_or_path = canonical_model_reference(model_name_or_path)
        self.model_cache_dir = Path(model_cache_dir) if model_cache_dir is not None else None
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_4bit = use_4bit
        self.compute_dtype = compute_dtype
        self.normalize_scores = normalize_scores
        self.instruction = instruction or DEFAULT_RETRIEVAL_INSTRUCTION
        self._tokenizer = None
        self._model = None
        self._prefix_tokens = None
        self._suffix_tokens = None
        self._token_false_id = None
        self._token_true_id = None
        self._pretrained_source = None

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
        """延遲載入 Qwen reranker 模型。"""
        if self._model is None:
            ensure_cuda_for_4bit(self.use_4bit)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.pretrained_source,
                **build_model_load_kwargs(self.use_4bit, self.compute_dtype),
            ).eval()
        return self._model

    def release_resources(self, clear_tokenizer: bool = True) -> None:
        """釋放 reranker 模型與 tokenizer，避免和其他模型同時佔用顯存。"""
        self._model = None
        if clear_tokenizer:
            self._tokenizer = None
            self._prefix_tokens = None
            self._suffix_tokens = None
            self._token_false_id = None
            self._token_true_id = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def prefix_tokens(self) -> list[int]:
        if self._prefix_tokens is None:
            prefix = (
                "<|im_start|>system\n"
                f"{self._SYSTEM_PROMPT}"
                "<|im_end|>\n<|im_start|>user\n"
            )
            self._prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        return self._prefix_tokens

    @property
    def suffix_tokens(self) -> list[int]:
        if self._suffix_tokens is None:
            self._suffix_tokens = self.tokenizer.encode(self._SUFFIX, add_special_tokens=False)
        return self._suffix_tokens

    @property
    def token_false_id(self) -> int:
        if self._token_false_id is None:
            self._token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        return self._token_false_id

    @property
    def token_true_id(self) -> int:
        if self._token_true_id is None:
            self._token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        return self._token_true_id

    def score(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        documents: list[dict[str, Any]],
    ) -> list[float]:
        """對 query-candidate pairs 計算 Qwen reranker 分數，不改變候選排序。"""
        if not candidates:
            return []

        normalized_query = normalize_text(query)
        pairs = [
            self._format_pair(normalized_query, question_text(documents[int(candidate["doc_index"])]))
            for candidate in candidates
        ]
        return self._compute_scores(pairs)

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        documents: list[dict[str, Any]],
        top_k: int = 30,
    ) -> list[dict[str, Any]]:
        """保留一般 rerank 能力，方便除錯比較。"""
        scores = self.score(query, candidates, documents)
        reranked: list[dict[str, Any]] = []
        for candidate, score in zip(candidates, scores, strict=True):
            item = dict(candidate)
            item["rerank_score"] = float(score)
            reranked.append(item)

        reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        return reranked[: min(top_k, len(reranked))]

    def _format_pair(self, query: str, document: str) -> str:
        return (
            f"<Instruct>: {self.instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {normalize_text(document)}"
        )

    def _process_inputs(self, pairs: list[str]):
        available_length = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        if available_length <= 0:
            raise ValueError(
                "reranker_max_length is too small for Qwen reranker prompt template. "
                f"Received {self.max_length}, but at least {len(self.prefix_tokens) + len(self.suffix_tokens) + 1} is required."
            )

        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=available_length,
        )
        for index, token_ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][index] = self.prefix_tokens + token_ids + self.suffix_tokens

        padded_inputs = self.tokenizer.pad(
            inputs,
            padding=True,
            return_tensors="pt",
            max_length=self.max_length,
        )
        return padded_inputs.to(self.model.device)

    @torch.no_grad()
    def _compute_scores(self, pairs: list[str]) -> list[float]:
        if not pairs:
            return []

        scores: list[float] = []
        for start in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[start : start + self.batch_size]
            inputs = self._process_inputs(batch_pairs)
            batch_logits = self.model(**inputs).logits[:, -1, :]
            true_vector = batch_logits[:, self.token_true_id]
            false_vector = batch_logits[:, self.token_false_id]

            if self.normalize_scores:
                batch_scores = torch.softmax(torch.stack([false_vector, true_vector], dim=1), dim=1)[:, 1]
            else:
                batch_scores = true_vector - false_vector

            scores.extend(batch_scores.detach().to(torch.float32).cpu().tolist())

        if isinstance(scores, (float, int, np.floating)):
            return [float(scores)]
        return [float(score) for score in scores]
