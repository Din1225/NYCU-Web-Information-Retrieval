from __future__ import annotations

"""
## Debug run

python normal_RR_ReFIT/scripts/run_refit_retrieval.py \
  --data data/IR_data.json \
  --query query/phase1_query.json \
  --output outputs/qwen_debug_refit_1000docs_results.json \
  --limit_docs 1000 \
  --limit_queries 1 \
  --feedback_top_k 100 \
  --final_top_k 30 \
  --refit_updates 100 \
  --dense_model Qwen/Qwen3-Embedding-4B \
  --reranker_model Qwen/Qwen3-Reranker-4B \
  --log_file outputs/logs/qwen_debug_refit_1000docs.log \
  --cuda_visible_devices 3 \
  --print_query_vectors \
  --query_vector_preview_dims 10

## Full run

python normal_RR_ReFIT/scripts/run_refit_retrieval.py \
  --data data/IR_data.json \
  --query query/phase1_query.json \
  --output outputs/qwen_refit_query1_results.json \
  --feedback_top_k 100 \
  --final_top_k 30 \
  --refit_updates 100 \
  --dense_model Qwen/Qwen3-Embedding-4B \
  --reranker_model Qwen/Qwen3-Reranker-4B \
  --log_file outputs/logs/qwen_refit_query1.log \
  --cuda_visible_devices 3 \
  --print_query_vectors \
  --query_vector_preview_dims 10

"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path


REFIT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = REFIT_ROOT.parent
sys.path.insert(0, str(REFIT_ROOT))


class TeeOutput:
    """同時把輸出寫到終端機與 log 檔案。"""

    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dense-only ReFIT retrieval.")
    parser.add_argument("--data", default="data/IR_data.json", help="Path to document JSON.")
    parser.add_argument("--query", default="query/phase1_query.json", help="Path to query JSON.")
    parser.add_argument("--output", default="outputs/refit_results.json", help="Output JSON path.")
    parser.add_argument(
        "--dense_model",
        default="Qwen/Qwen3-Embedding-4B",
        help="Dense model local path or Hugging Face model id.",
    )
    parser.add_argument(
        "--reranker_model",
        default="Qwen/Qwen3-Reranker-4B",
        help="Reranker local path or Hugging Face model id.",
    )
    parser.add_argument("--cache_dir", default="outputs/cache", help="Directory for reusable caches.")
    parser.add_argument("--feedback_top_k", type=int, default=100)
    parser.add_argument("--final_top_k", type=int, default=30)
    parser.add_argument("--refit_updates", type=int, default=100)
    parser.add_argument("--refit_learning_rate", type=float, default=0.005)
    parser.add_argument("--refit_temperature", type=float, default=2.0)
    parser.add_argument("--refit_no_minmax", action="store_true", help="Disable min-max score normalization.")
    parser.add_argument("--dense_batch_size", type=int, default=32)
    parser.add_argument("--reranker_batch_size", type=int, default=16)
    parser.add_argument("--dense_max_length", type=int, default=512)
    parser.add_argument("--reranker_max_length", type=int, default=512)
    parser.add_argument("--disable_4bit", action="store_true", help="Disable bitsandbytes 4-bit loading.")
    parser.add_argument(
        "--compute_dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Compute dtype used during model inference.",
    )
    parser.add_argument(
        "--use_fp16",
        action="store_true",
        help="Legacy flag; kept for backward compatibility. It is equivalent to --compute_dtype float16.",
    )
    parser.add_argument("--no_cache", action="store_true", help="Disable dense embedding cache loading.")
    parser.add_argument("--limit_docs", type=int, default=None, help="Debug only: limit document count.")
    parser.add_argument("--limit_queries", type=int, default=None, help="Debug only: limit query count.")
    parser.add_argument(
        "--retrieval_instruction",
        default="給定一個問題，請檢索出語意最相關、問題表述最相近的問題。",
        help="Instruction injected into Qwen query-side embedding and reranker prompts.",
    )
    parser.add_argument(
        "--print_query_vectors",
        action="store_true",
        help="Print and save query embedding values before and after ReFIT.",
    )
    parser.add_argument(
        "--query_vector_preview_dims",
        type=int,
        default=10,
        help="Number of query embedding dimensions to print/save when --print_query_vectors is enabled.",
    )
    parser.add_argument("--log_file", default=None, help="Path to log file.")
    parser.add_argument(
        "--cuda_visible_devices",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES value, for example 0 or 1.",
    )
    return parser.parse_args()


def resolve_existing_path(path: str) -> Path:
    """相對路徑先找 ReFIT 專案，再找 repo 根目錄。"""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate

    refit_candidate = REFIT_ROOT / candidate
    if refit_candidate.exists():
        return refit_candidate

    repo_candidate = REPO_ROOT / candidate
    if repo_candidate.exists():
        return repo_candidate

    return refit_candidate


def resolve_model_reference(model_name_or_path: str) -> str:
    """模型參數支援本地路徑與 Hugging Face model id。"""
    candidate = Path(model_name_or_path)
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)

    if not candidate.is_absolute():
        refit_candidate = REFIT_ROOT / candidate
        if refit_candidate.exists():
            return str(refit_candidate)

        repo_candidate = REPO_ROOT / candidate
        if repo_candidate.exists():
            return str(repo_candidate)

    return model_name_or_path


def resolve_output_path(path: str) -> Path:
    """輸出類路徑預設寫在 normal_RR_ReFIT 底下。"""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REFIT_ROOT / candidate


def default_log_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REFIT_ROOT / "outputs" / "logs" / f"refit_{timestamp}.log"


def enable_log_file(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeOutput(original_stdout, log_file)
    sys.stderr = TeeOutput(original_stderr, log_file)
    return log_file, original_stdout, original_stderr


def disable_log_file(log_file, original_stdout, original_stderr) -> None:
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file.close()


def print_run_config(config, log_path: Path) -> None:
    print("=== Dense-only ReFIT Run Config ===")
    print(f"data_path: {config.data_path}")
    print(f"query_path: {config.query_path}")
    print(f"output_path: {config.output_path}")
    print(f"dense_model_name_or_path: {config.dense_model_name_or_path}")
    print(f"reranker_model_name_or_path: {config.reranker_model_name_or_path}")
    print(f"cache_dir: {config.cache_dir}")
    print(f"log_file: {log_path}")
    print(f"feedback_top_k: {config.feedback_top_k}")
    print(f"final_top_k: {config.final_top_k}")
    print(f"refit_updates: {config.refit_updates}")
    print(f"refit_learning_rate: {config.refit_learning_rate}")
    print(f"refit_temperature: {config.refit_temperature}")
    print(f"refit_use_minmax: {config.refit_use_minmax}")
    print(f"dense_batch_size: {config.dense_batch_size}")
    print(f"reranker_batch_size: {config.reranker_batch_size}")
    print(f"dense_max_length: {config.dense_max_length}")
    print(f"reranker_max_length: {config.reranker_max_length}")
    print(f"use_4bit: {config.use_4bit}")
    print(f"compute_dtype: {config.compute_dtype}")
    print(f"use_cache: {config.use_cache}")
    print(f"limit_docs: {config.limit_docs}")
    print(f"limit_queries: {config.limit_queries}")
    print(f"retrieval_instruction: {config.retrieval_instruction}")
    print(f"print_query_vectors: {config.print_query_vectors}")
    print(f"query_vector_preview_dims: {config.query_vector_preview_dims}")
    print("====================================")


def main() -> None:
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    if args.use_fp16:
        args.compute_dtype = "float16"

    from src.pipeline import DenseOnlyReFITConfig, run_refit_retrieval

    log_path = resolve_output_path(args.log_file) if args.log_file else default_log_path()
    config = DenseOnlyReFITConfig(
        data_path=resolve_existing_path(args.data),
        query_path=resolve_existing_path(args.query),
        output_path=resolve_output_path(args.output),
        dense_model_name_or_path=resolve_model_reference(args.dense_model),
        reranker_model_name_or_path=resolve_model_reference(args.reranker_model),
        cache_dir=resolve_output_path(args.cache_dir),
        feedback_top_k=args.feedback_top_k,
        final_top_k=args.final_top_k,
        refit_updates=args.refit_updates,
        refit_learning_rate=args.refit_learning_rate,
        refit_temperature=args.refit_temperature,
        refit_use_minmax=not args.refit_no_minmax,
        dense_batch_size=args.dense_batch_size,
        reranker_batch_size=args.reranker_batch_size,
        dense_max_length=args.dense_max_length,
        reranker_max_length=args.reranker_max_length,
        use_4bit=not args.disable_4bit,
        compute_dtype=args.compute_dtype,
        use_cache=not args.no_cache,
        limit_docs=args.limit_docs,
        limit_queries=args.limit_queries,
        retrieval_instruction=args.retrieval_instruction,
        print_query_vectors=args.print_query_vectors,
        query_vector_preview_dims=args.query_vector_preview_dims,
    )

    log_file, original_stdout, original_stderr = enable_log_file(log_path)
    try:
        started_at = time.perf_counter()
        print(f"Log file: {log_path}")
        print_run_config(config, log_path)
        print("開始執行 dense-only ReFIT：第一次 dense retrieval -> reranker feedback -> query 更新 -> 第二次 dense retrieval。")
        run_refit_retrieval(config)
        elapsed = time.perf_counter() - started_at
        print(f"Dense-only ReFIT 流程完成，總耗時 {elapsed:.2f} 秒。")
    finally:
        disable_log_file(log_file, original_stdout, original_stderr)


if __name__ == "__main__":
    main()
