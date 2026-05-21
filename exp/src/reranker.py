from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from src.model_utils import (
    DEFAULT_RETRIEVAL_INSTRUCTION,
    build_qwen_model_load_kwargs,
    canonical_model_reference,
    ensure_cuda_for_4bit,
    infer_reranker_backend,
)
from src.text_processing import document_text, normalize_text


class CrossEncoderReranker:
    _SYSTEM_PROMPT = (
        "請根據提供的 Instruct 和 Query，判斷 Document 是否符合檢索需求。"
        "回答只能是 yes 或 no。"
    )
    _SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(
        self,
        model_name_or_path: str | Path,
        backend: str = "auto",
        batch_size: int = 16,
        max_length: int = 512,
        use_4bit: bool = True,
        compute_dtype: str = "float16",
        bge_use_fp16: bool = False,
        normalize_scores: bool = True,
        instruction: str | None = DEFAULT_RETRIEVAL_INSTRUCTION,
        document_text_mode: str = "question",
    ) -> None:
        self.model_name_or_path = canonical_model_reference(model_name_or_path)
        self.backend = infer_reranker_backend(self.model_name_or_path, backend)
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_4bit = use_4bit
        self.compute_dtype = compute_dtype
        self.bge_use_fp16 = bge_use_fp16
        self.normalize_scores = normalize_scores
        self.instruction = instruction or DEFAULT_RETRIEVAL_INSTRUCTION
        self.document_text_mode = document_text_mode
        self._tokenizer = None
        self._model = None
        self._bge_tokenizer = None
        self._bge_impl = None
        self._prefix_tokens = None
        self._suffix_tokens = None
        self._token_false_id = None
        self._token_true_id = None

    @property
    def tokenizer(self):
        if self.backend != "qwen":
            raise AttributeError("Tokenizer is only used by the Qwen reranker backend.")
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
                    from FlagEmbedding import FlagReranker

                    self._model = FlagReranker(self.model_name_or_path, use_fp16=self.bge_use_fp16)
                    self._bge_impl = "flagembedding"
                except Exception as error:
                    self._bge_tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name_or_path,
                        use_fast=False,
                    )
                    self._model = AutoModelForSequenceClassification.from_pretrained(
                        self.model_name_or_path,
                        use_safetensors=True,
                    ).eval()
                    if torch.cuda.is_available():
                        self._model = self._model.to("cuda")
                        if self.bge_use_fp16:
                            self._model = self._model.half()
                    self._bge_impl = "transformers"
                    print(
                        "FlagEmbedding 載入失敗，改用 transformers 作為 BGE reranker fallback。"
                        f" 原始錯誤: {error}",
                        flush=True,
                    )
            else:
                ensure_cuda_for_4bit(self.use_4bit)
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path,
                    **build_qwen_model_load_kwargs(self.use_4bit, self.compute_dtype),
                ).eval()
        return self._model

    def release_resources(self, clear_tokenizer: bool = True) -> None:
        self._model = None
        if clear_tokenizer:
            self._tokenizer = None
            self._bge_tokenizer = None
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
        if not candidates:
            return []

        normalized_query = normalize_text(query)
        if self.backend == "bge":
            pairs = [
                [
                    normalized_query,
                    document_text(documents[int(candidate["doc_index"])], mode=self.document_text_mode),
                ]
                for candidate in candidates
            ]
        else:
            pairs = [
                self._format_qwen_pair(
                    normalized_query,
                    document_text(documents[int(candidate["doc_index"])], mode=self.document_text_mode),
                )
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
        scores = self.score(query, candidates, documents)
        reranked: list[dict[str, Any]] = []
        for candidate, score in zip(candidates, scores, strict=True):
            item = dict(candidate)
            item["rerank_score"] = float(score)
            reranked.append(item)

        reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        return reranked[: min(top_k, len(reranked))]

    def _format_qwen_pair(self, query: str, document: str) -> str:
        return (
            f"<Instruct>: {self.instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {normalize_text(document)}"
        )

    def _process_qwen_inputs(self, pairs: list[str]):
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

    def _compute_scores(self, pairs: list[Any]) -> list[float]:
        if not pairs:
            return []
        if self.backend == "bge":
            _ = self.model
            if self._bge_impl == "transformers":
                return self._compute_bge_transformers_scores(pairs)

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

        scores: list[float] = []
        for start in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[start : start + self.batch_size]
            inputs = self._process_qwen_inputs(batch_pairs)
            batch_logits = self.model(**inputs).logits[:, -1, :]
            true_vector = batch_logits[:, self.token_true_id]
            false_vector = batch_logits[:, self.token_false_id]

            if self.normalize_scores:
                batch_scores = torch.softmax(torch.stack([false_vector, true_vector], dim=1), dim=1)[:, 1]
            else:
                batch_scores = true_vector - false_vector

            scores.extend(batch_scores.detach().to(torch.float32).cpu().tolist())
        return [float(score) for score in scores]

    @torch.no_grad()
    def _compute_bge_transformers_scores(self, pairs: list[list[str]]) -> list[float]:
        scores: list[float] = []
        model = self.model
        tokenizer = self._bge_tokenizer
        if tokenizer is None:
            raise RuntimeError("BGE transformers fallback tokenizer is not initialized.")

        device = next(model.parameters()).device
        for start in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[start : start + self.batch_size]
            inputs = tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            batch_scores = model(**inputs, return_dict=True).logits.view(-1).float()
            if self.normalize_scores:
                batch_scores = torch.sigmoid(batch_scores)
            scores.extend(batch_scores.detach().cpu().tolist())
        return [float(score) for score in scores]
