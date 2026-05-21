from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from transformers import BitsAndBytesConfig


DEFAULT_RETRIEVAL_INSTRUCTION = "給定一個問題，請檢索出語意最相關、問題表述最相近的問題。"


def canonical_model_reference(model_name_or_path: str | Path) -> str:
    candidate = Path(model_name_or_path)
    if candidate.exists():
        return str(candidate.resolve())
    return str(model_name_or_path)


def infer_dense_backend(model_name_or_path: str, backend: str = "auto") -> str:
    if backend != "auto":
        return backend

    model_ref = canonical_model_reference(model_name_or_path).lower()
    if "bge" in model_ref:
        return "bge"
    if "qwen" in model_ref:
        return "qwen"
    raise ValueError(
        "Unable to infer dense backend from model name. "
        "Please pass --dense_backend bge or --dense_backend qwen."
    )


def infer_reranker_backend(model_name_or_path: str, backend: str = "auto") -> str:
    if backend != "auto":
        return backend

    model_ref = canonical_model_reference(model_name_or_path).lower()
    if "bge" in model_ref and "reranker" in model_ref:
        return "bge"
    if "qwen" in model_ref:
        return "qwen"
    raise ValueError(
        "Unable to infer reranker backend from model name. "
        "Please pass --reranker_backend bge or --reranker_backend qwen."
    )


def torch_dtype_from_name(dtype_name: str) -> torch.dtype:
    normalized = dtype_name.strip().lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        supported = ", ".join(sorted(mapping))
        raise ValueError(f"Unsupported compute dtype: {dtype_name}. Expected one of: {supported}")
    return mapping[normalized]


def build_qwen_model_load_kwargs(use_4bit: bool, compute_dtype_name: str) -> dict[str, Any]:
    compute_dtype = torch_dtype_from_name(compute_dtype_name)
    load_kwargs: dict[str, Any] = {
        "torch_dtype": compute_dtype,
        "device_map": "auto",
    }
    if use_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    return load_kwargs


def ensure_cuda_for_4bit(use_4bit: bool) -> None:
    if use_4bit and not torch.cuda.is_available():
        raise RuntimeError("4-bit quantization requires CUDA. Disable 4-bit or run on a GPU-enabled environment.")


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = bool(torch.all(attention_mask[:, -1] == 1))
    if left_padding:
        return last_hidden_states[:, -1]

    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def format_query_with_instruction(query: str, instruction: str | None) -> str:
    if instruction is None or not instruction.strip():
        return query
    return f"Instruct: {instruction.strip()}\nQuery:{query}"

