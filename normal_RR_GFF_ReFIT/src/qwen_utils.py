from __future__ import annotations

from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
import torch
from torch import Tensor
from transformers import BitsAndBytesConfig


DEFAULT_RETRIEVAL_INSTRUCTION = "給定一個問題，請檢索出語意最相關、問題表述最相近的問題。"


def canonical_model_reference(model_name_or_path: str | Path) -> str:
    """將模型參考值轉成穩定字串；本地路徑會轉成 absolute path。"""
    candidate = Path(model_name_or_path)
    if candidate.exists():
        return str(candidate.resolve())
    return str(model_name_or_path)


def resolve_model_cache_dir(cache_dir: str | Path | None) -> str | None:
    """標準化模型快取資料夾，並確保目錄存在。"""
    if cache_dir is None:
        return None

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    return str(cache_path.resolve())


def resolve_pretrained_source(model_name_or_path: str | Path, cache_dir: str | Path | None) -> str:
    """回傳可直接交給 from_pretrained 的來源。

    - 若使用者已提供本地路徑，直接回傳該路徑。
    - 若提供 Hugging Face model id，則先下載或命中 repo 內快取，再回傳 snapshot 路徑。
    """
    model_reference = canonical_model_reference(model_name_or_path)
    local_candidate = Path(model_reference)
    if local_candidate.exists():
        return model_reference

    resolved_cache_dir = resolve_model_cache_dir(cache_dir)
    if resolved_cache_dir is None:
        return model_reference

    snapshot_path = snapshot_download(
        repo_id=model_reference,
        cache_dir=resolved_cache_dir,
    )
    return str(Path(snapshot_path).resolve())


def torch_dtype_from_name(dtype_name: str) -> torch.dtype:
    """把 CLI 傳入的 dtype 名稱轉成 torch dtype。"""
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


def build_model_load_kwargs(use_4bit: bool, compute_dtype_name: str) -> dict[str, Any]:
    """建立 Qwen 模型載入參數；預設使用 bitsandbytes 4-bit。"""
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
    """4-bit 量化預期跑在 CUDA 上；若環境不符則提早報錯。"""
    if use_4bit and not torch.cuda.is_available():
        raise RuntimeError("4-bit quantization requires CUDA. Disable 4-bit or run on a GPU-enabled environment.")


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """依 Qwen embedding 模型卡建議，取最後一個有效 token 當 embedding。"""
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
    """Qwen embedding 模型建議 query 端帶 instruction；文件端則維持原文。"""
    if instruction is None or not instruction.strip():
        return query
    return f"Instruct: {instruction.strip()}\nQuery:{query}"
