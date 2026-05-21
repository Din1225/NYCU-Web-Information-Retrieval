from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path


EXP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXP_ROOT.parent
sys.path.insert(0, str(EXP_ROOT))


class TeeOutput:
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
    parser = argparse.ArgumentParser(description="Run dense retrieval + rerank baseline experiment.")
    parser.add_argument("--data", default="data/IR_data.json", help="Path to document JSON.")
    parser.add_argument("--query", default="query/phase1_query.json", help="Path to query JSON.")
    parser.add_argument("--output", default="outputs/dense_rerank_results.json", help="Output JSON path.")
    parser.add_argument(
        "--dense_model",
        default="BAAI/bge-m3",
        help="Dense model local path or Hugging Face model id.",
    )
    parser.add_argument(
        "--reranker_model",
        default="BAAI/bge-reranker-v2-m3",
        help="Reranker model local path or Hugging Face model id.",
    )
    parser.add_argument("--cache_dir", default="outputs/cache", help="Directory for reusable caches.")
    parser.add_argument("--dense_backend", default="auto", choices=["auto", "bge", "qwen"])
    parser.add_argument("--reranker_backend", default="auto", choices=["auto", "bge", "qwen"])
    parser.add_argument("--dense_top_k", type=int, default=100)
    parser.add_argument("--rerank_top_k", type=int, default=30)
    parser.add_argument("--dense_batch_size", type=int, default=32)
    parser.add_argument("--reranker_batch_size", type=int, default=16)
    parser.add_argument("--dense_max_length", type=int, default=512)
    parser.add_argument("--reranker_max_length", type=int, default=512)
    parser.add_argument("--disable_4bit", action="store_true", help="Disable Qwen 4-bit loading.")
    parser.add_argument(
        "--compute_dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Compute dtype for Qwen models.",
    )
    parser.add_argument(
        "--bge_use_fp16",
        action="store_true",
        help="Use fp16 for BGE dense/reranker backends.",
    )
    parser.add_argument("--no_cache", action="store_true", help="Disable dense embedding cache loading.")
    parser.add_argument("--limit_docs", type=int, default=None, help="Debug only: limit document count.")
    parser.add_argument("--limit_queries", type=int, default=None, help="Debug only: limit query count.")
    parser.add_argument(
        "--document_text_mode",
        default="question",
        choices=["question", "question_answer"],
        help="Document text used by dense retrieval and reranker.",
    )
    parser.add_argument(
        "--retrieval_instruction",
        default="給定一個問題，請檢索出語意最相關、問題表述最相近的問題。",
        help="Instruction used by Qwen embedding and Qwen reranker prompts.",
    )
    parser.add_argument(
        "--annotation",
        default=None,
        help="Optional annotation JSON path. If provided, the script also runs masked evaluation.",
    )
    parser.add_argument(
        "--evaluation_output",
        default=None,
        help="Optional evaluation JSON path. Defaults to a sibling file next to --output.",
    )
    parser.add_argument(
        "--binary_relevance_threshold",
        type=int,
        default=2,
        choices=[1, 2],
        help="Label threshold used by MAP/MRR. Relevant means label >= threshold.",
    )
    parser.add_argument("--log_file", default=None, help="Path to log file.")
    parser.add_argument("--cuda_visible_devices", default=None, help="Optional CUDA_VISIBLE_DEVICES value.")
    return parser.parse_args()


def resolve_existing_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate

    exp_candidate = EXP_ROOT / candidate
    if exp_candidate.exists():
        return exp_candidate

    repo_candidate = REPO_ROOT / candidate
    if repo_candidate.exists():
        return repo_candidate

    return exp_candidate


def resolve_output_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return EXP_ROOT / candidate


def default_log_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return EXP_ROOT / "outputs" / "logs" / f"dense_rerank_{timestamp}.log"


def default_evaluation_path(result_path: Path) -> Path:
    return result_path.with_name(f"{result_path.stem}.evaluation.json")


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


def print_run_config(config, log_path: Path, annotation_path: Path | None, evaluation_path: Path | None) -> None:
    print("=== Dense Retrieve + Rerank Run Config ===")
    print(f"data_path: {config.data_path}")
    print(f"query_path: {config.query_path}")
    print(f"output_path: {config.output_path}")
    print(f"dense_model_name_or_path: {config.dense_model_name_or_path}")
    print(f"reranker_model_name_or_path: {config.reranker_model_name_or_path}")
    print(f"dense_backend: {config.dense_backend}")
    print(f"reranker_backend: {config.reranker_backend}")
    print(f"cache_dir: {config.cache_dir}")
    print(f"log_file: {log_path}")
    print(f"annotation_path: {annotation_path}")
    print(f"evaluation_output_path: {evaluation_path}")
    print(f"dense_top_k: {config.dense_top_k}")
    print(f"rerank_top_k: {config.rerank_top_k}")
    print(f"dense_batch_size: {config.dense_batch_size}")
    print(f"reranker_batch_size: {config.reranker_batch_size}")
    print(f"dense_max_length: {config.dense_max_length}")
    print(f"reranker_max_length: {config.reranker_max_length}")
    print(f"use_4bit: {config.use_4bit}")
    print(f"compute_dtype: {config.compute_dtype}")
    print(f"bge_use_fp16: {config.bge_use_fp16}")
    print(f"use_cache: {config.use_cache}")
    print(f"limit_docs: {config.limit_docs}")
    print(f"limit_queries: {config.limit_queries}")
    print(f"document_text_mode: {config.document_text_mode}")
    print(f"retrieval_instruction: {config.retrieval_instruction}")
    print("==========================================")


def main() -> None:
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    from src.evaluation import evaluate_multiple_result_fields, save_evaluation
    from src.pipeline import DenseRerankConfig, run_dense_rerank_experiment

    output_path = resolve_output_path(args.output)
    annotation_path = resolve_existing_path(args.annotation) if args.annotation else None
    evaluation_path = (
        resolve_output_path(args.evaluation_output)
        if args.evaluation_output
        else (default_evaluation_path(output_path) if annotation_path else None)
    )
    log_path = resolve_output_path(args.log_file) if args.log_file else default_log_path()

    config = DenseRerankConfig(
        data_path=resolve_existing_path(args.data),
        query_path=resolve_existing_path(args.query),
        output_path=output_path,
        dense_model_name_or_path=str(args.dense_model),
        reranker_model_name_or_path=str(args.reranker_model),
        cache_dir=resolve_output_path(args.cache_dir),
        dense_backend=args.dense_backend,
        reranker_backend=args.reranker_backend,
        dense_top_k=args.dense_top_k,
        rerank_top_k=args.rerank_top_k,
        dense_batch_size=args.dense_batch_size,
        reranker_batch_size=args.reranker_batch_size,
        dense_max_length=args.dense_max_length,
        reranker_max_length=args.reranker_max_length,
        use_4bit=not args.disable_4bit,
        compute_dtype=args.compute_dtype,
        bge_use_fp16=args.bge_use_fp16,
        use_cache=not args.no_cache,
        limit_docs=args.limit_docs,
        limit_queries=args.limit_queries,
        document_text_mode=args.document_text_mode,
        retrieval_instruction=args.retrieval_instruction,
    )

    log_file, original_stdout, original_stderr = enable_log_file(log_path)
    try:
        started_at = time.perf_counter()
        print(f"Log file: {log_path}", flush=True)
        print_run_config(config, log_path, annotation_path, evaluation_path)
        run_dense_rerank_experiment(config)

        if annotation_path is not None and evaluation_path is not None:
            evaluation = evaluate_multiple_result_fields(
                annotation_path=annotation_path,
                result_path=output_path,
                result_fields=["dense_results", "rerank_results"],
                binary_relevance_threshold=args.binary_relevance_threshold,
                top_ks=(10, 30),
            )
            save_evaluation(evaluation, evaluation_path)
            print(f"Saved evaluation to {evaluation_path}", flush=True)
            print("dense_results aggregate:", evaluation["dense_results"]["aggregate"], flush=True)
            print("rerank_results aggregate:", evaluation["rerank_results"]["aggregate"], flush=True)

        elapsed = time.perf_counter() - started_at
        print(f"流程完成，總耗時 {elapsed:.2f} 秒。", flush=True)
    finally:
        disable_log_file(log_file, original_stdout, original_stderr)


if __name__ == "__main__":
    main()
