from __future__ import annotations

from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
import torch
from torch import Tensor
from transformers import BitsAndBytesConfig


DEFAULT_RETRIEVAL_INSTRUCTION = "給定一個問題，請檢索出語意最相關的歷史問答。"


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

    local_snapshot = _resolve_local_cached_model(model_reference, Path(resolved_cache_dir))
    if local_snapshot is not None:
        return str(local_snapshot)

    snapshot_path = snapshot_download(
        repo_id=model_reference,
        cache_dir=resolved_cache_dir,
    )
    return str(Path(snapshot_path).resolve())


def _resolve_local_cached_model(model_reference: str, cache_dir: Path) -> Path | None:
    """嘗試直接從本地 Hugging Face cache root 找到模型 snapshot。"""
    candidate_paths = [
        cache_dir / "models" / model_reference.replace("/", "--"),
        cache_dir / model_reference.replace("/", "--"),
        cache_dir / f"models--{model_reference.replace('/', '--')}",
        cache_dir / "hub" / f"models--{model_reference.replace('/', '--')}",
    ]
    for candidate in candidate_paths:
        if _is_pretrained_model_dir(candidate):
            return candidate.resolve()

    hub_roots = [
        cache_dir / f"models--{model_reference.replace('/', '--')}",
        cache_dir / "hub" / f"models--{model_reference.replace('/', '--')}",
    ]
    for hub_root in hub_roots:
        snapshot_path = _resolve_hf_snapshot_dir(hub_root)
        if snapshot_path is not None:
            return snapshot_path.resolve()

    return None


def _resolve_hf_snapshot_dir(hub_root: Path) -> Path | None:
    if not hub_root.exists():
        return None

    refs_main = hub_root / "refs" / "main"
    if refs_main.exists():
        snapshot_name = refs_main.read_text(encoding="utf-8").strip()
        if snapshot_name:
            snapshot_dir = hub_root / "snapshots" / snapshot_name
            if _is_pretrained_model_dir(snapshot_dir):
                return snapshot_dir

    snapshots_dir = hub_root / "snapshots"
    if not snapshots_dir.exists():
        return None

    snapshot_candidates = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
    for snapshot_dir in reversed(snapshot_candidates):
        if _is_pretrained_model_dir(snapshot_dir):
            return snapshot_dir
    return None


def _is_pretrained_model_dir(path: Path) -> bool:
    required_files = ("config.json", "tokenizer.json")
    return path.is_dir() and all((path / filename).exists() for filename in required_files)


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
