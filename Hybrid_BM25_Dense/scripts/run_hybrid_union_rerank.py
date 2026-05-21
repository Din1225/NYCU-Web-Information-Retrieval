from __future__ import annotations

"""
## Debug run

python hybrid_BM25_Dense_Rerank/scripts/run_hybrid_union_rerank.py \
  --data data/IR_data.json \
  --query query/phase1_query.json \
  --subquery_path query/phase1_subqueries.json \
  --output outputs/hybrid_union_debug_results.json \
  --limit_docs 1000 \
  --limit_queries 1 \
  --RRF \
  --bm25_top_k 100 \
  --dense_top_k 100 \
  --final_top_k 30 \
  --dense_model Qwen/Qwen3-Embedding-4B \
  --reranker_model Qwen/Qwen3-Reranker-4B \
  --log_file outputs/logs/hybrid_union_debug.log \
  --cuda_visible_devices 0
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path


METHOD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = METHOD_ROOT.parent
SHARED_MODEL_CACHE_ROOT = Path("/workplace/Share/LLM_model")
sys.path.insert(0, str(METHOD_ROOT))


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
    parser = argparse.ArgumentParser(
        description="Run hybrid BM25 + dense fusion retrieval with reranking.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", default="data/IR_data.json", help="文件資料池 corpus 的 JSON 路徑。")
    parser.add_argument("--query", default="query/phase2_query.json", help="query JSON 檔案路徑。")
    parser.add_argument(
        "--subquery_path",
        default=None,
        help="query 對應的 sub-query JSON 路徑；指定後會啟用 risk penalty 流程。",
    )
    parser.add_argument("--output", default="outputs/hybrid_union_results.json", help="最終輸出結果 JSON 路徑。")
    parser.add_argument(
        "--dense_model",
        default="Qwen/Qwen3-Embedding-4B",
        help="Dense retriever 模型，用來建立文件/query embeddings。",
    )
    parser.add_argument(
        "--reranker_model",
        default="Qwen/Qwen3-Reranker-4B",
        help="Cross-encoder reranker 模型，用來對融合後的 candidates 重新排序。",
    )
    parser.add_argument(
        "--cache_dir",
        default="outputs/cache",
        help="可重複使用的快取根目錄，例如 dense embeddings cache 與 BM25 index cache。",
    )
    parser.add_argument(
        "--model_cache_dir",
        default=None,
        help="模型 snapshot 下載快取目錄；未指定時使用 <cache_dir>/models。",
    )
    parser.add_argument(
        "--bm25_top_k",
        type=int,
        default=100,
        help="BM25 第一次關鍵字檢索保留幾篇候選文件。",
    )
    parser.add_argument(
        "--dense_top_k",
        type=int,
        default=100,
        help="Dense 第一次語意檢索保留幾篇候選文件。",
    )
    parser.add_argument(
        "--final_top_k",
        type=int,
        default=30,
        help="融合候選集合經過 rerank 後最後輸出幾筆結果。",
    )
    parser.add_argument(
        "--risk_top_k",
        type=int,
        default=50,
        help="先用 reranker 保留前幾篇文件，再交給 risk detector 做 penalty。",
    )
    parser.add_argument(
        "--RRF",
        "--rrf",
        dest="use_rrf",
        action="store_true",
        help="啟用 Reciprocal Rank Fusion，改用 RRF 取代預設的 union seed 排序。",
    )
    parser.add_argument(
        "--rrf_k",
        type=int,
        default=60,
        help="RRF 的常數 k；只有在啟用 --RRF 時會使用。",
    )
    parser.add_argument(
        "--bm25_k1",
        type=float,
        default=1.5,
        help="BM25 的 k1 參數。",
    )
    parser.add_argument(
        "--bm25_b",
        type=float,
        default=0.75,
        help="BM25 的 b 參數。",
    )
    parser.add_argument(
        "--dense_batch_size",
        type=int,
        default=32,
        help="Dense retriever 編碼文件與 query 文字時使用的 batch size。",
    )
    parser.add_argument(
        "--reranker_batch_size",
        type=int,
        default=16,
        help="Cross-encoder reranker 計算 query-document pair 分數時使用的 batch size。",
    )
    parser.add_argument(
        "--dense_max_length",
        type=int,
        default=512,
        help="Dense retrieval 編碼文字時使用的最大 token 長度。",
    )
    parser.add_argument(
        "--reranker_max_length",
        type=int,
        default=512,
        help="Reranker 打包 query-document pair 時使用的最大 token 長度。",
    )
    parser.add_argument(
        "--disable_4bit",
        action="store_true",
        help="關閉 bitsandbytes 4-bit 載入，改用一般精度路徑載入模型。",
    )
    parser.add_argument(
        "--compute_dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="模型載入後推論時計算使用的 dtype。",
    )
    parser.add_argument(
        "--use_fp16",
        action="store_true",
        help="相容舊指令的快捷參數；等同強制把 --compute_dtype 設為 float16。",
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="不讀取也不重用既有的 dense/BM25 cache。",
    )
    parser.add_argument(
        "--limit_docs",
        type=int,
        default=None,
        help="Debug 用：只載入前 N 篇文件。",
    )
    parser.add_argument(
        "--limit_queries",
        type=int,
        default=None,
        help="Debug 用：只載入前 N 筆 queries。",
    )
    parser.add_argument(
        "--retrieval_instruction",
        default="給定一個問題，請檢索出語意最相關的歷史問答。",
        help="注入到 Qwen query-side embedding 與 reranker prompt 的 instruction。",
    )
    parser.add_argument(
        "--risk_model",
        default="Qwen/Qwen3.5-4B",
        help="Risk detector 使用的 Qwen causal LM。",
    )
    parser.add_argument(
        "--risk_model_cache_dir",
        default=str(SHARED_MODEL_CACHE_ROOT),
        help="Risk detector 模型快取根目錄；預設使用共享快取 /workplace/Share/LLM_model。",
    )
    parser.add_argument(
        "--risk_lambda",
        type=float,
        default=0.1,
        help="最終分數中的 risk penalty 權重 lambda。",
    )
    parser.add_argument(
        "--risk_batch_size",
        type=int,
        default=4,
        help="Risk detector 批次推論時使用的 batch size。",
    )
    parser.add_argument(
        "--risk_max_length",
        type=int,
        default=1024,
        help="Risk detector prompt 的最大 token 長度。",
    )
    parser.add_argument(
        "--risk_max_new_tokens",
        type=int,
        default=256,
        help="Risk detector 生成 JSON 時允許的最大新 token 數。",
    )
    parser.add_argument(
        "--log_file",
        default=None,
        help="可選的 log 檔案路徑；會同步保存 stdout 與 stderr。",
    )
    parser.add_argument(
        "--cuda_visible_devices",
        default=None,
        help="模型載入前要設定的 CUDA_VISIBLE_DEVICES，例如 '0' 或 '0,1'。",
    )
    return parser.parse_args()


def resolve_existing_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate

    method_candidate = METHOD_ROOT / candidate
    if method_candidate.exists():
        return method_candidate

    repo_candidate = REPO_ROOT / candidate
    if repo_candidate.exists():
        return repo_candidate

    return method_candidate


def resolve_model_reference(model_name_or_path: str) -> str:
    candidate = Path(model_name_or_path)
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)

    if not candidate.is_absolute():
        method_candidate = METHOD_ROOT / candidate
        if method_candidate.exists():
            return str(method_candidate)

        repo_candidate = REPO_ROOT / candidate
        if repo_candidate.exists():
            return str(repo_candidate)

    return model_name_or_path


def resolve_output_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return METHOD_ROOT / path


def default_log_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return METHOD_ROOT / "outputs" / "logs" / f"hybrid_union_{timestamp}.log"


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
    print("=== Hybrid BM25 + Dense Fusion Run Config ===")
    print(f"data_path: {config.data_path}")
    print(f"query_path: {config.query_path}")
    print(f"subquery_path: {config.subquery_path}")
    print(f"output_path: {config.output_path}")
    print(f"dense_model_name_or_path: {config.dense_model_name_or_path}")
    print(f"reranker_model_name_or_path: {config.reranker_model_name_or_path}")
    print(f"risk_model_name_or_path: {config.risk_model_name_or_path}")
    print(f"cache_dir: {config.cache_dir}")
    print(f"model_cache_dir: {config.model_cache_dir}")
    print(f"risk_model_cache_dir: {config.risk_model_cache_dir}")
    print(f"log_file: {log_path}")
    print(f"bm25_top_k: {config.bm25_top_k}")
    print(f"dense_top_k: {config.dense_top_k}")
    print(f"risk_top_k: {config.risk_top_k}")
    print(f"final_top_k: {config.final_top_k}")
    print(f"fusion_method: {config.fusion_method}")
    print(f"rrf_k: {config.rrf_k}")
    print(f"risk_lambda: {config.risk_lambda}")
    print(f"bm25_k1: {config.bm25_k1}")
    print(f"bm25_b: {config.bm25_b}")
    print(f"use_4bit: {config.use_4bit}")
    print(f"compute_dtype: {config.compute_dtype}")
    print(f"use_cache: {config.use_cache}")
    print("================================================")


def cleanup_debug_caches(config) -> None:
    if config.limit_docs is None:
        return

    from src.bm25_retriever import build_bm25_cache_paths, delete_bm25_cache_files
    from src.data_io import load_documents
    from src.dense_retriever import build_dense_cache_paths, delete_dense_cache_files

    documents = load_documents(config.data_path, limit=config.limit_docs)
    dense_cache_paths = build_dense_cache_paths(
        documents=documents,
        model_name_or_path=config.dense_model_name_or_path,
        cache_dir=config.cache_dir / "dense",
        max_length=config.dense_max_length,
        use_4bit=config.use_4bit,
        compute_dtype=config.compute_dtype,
        query_instruction=config.retrieval_instruction,
    )
    bm25_cache_paths = build_bm25_cache_paths(
        documents=documents,
        cache_dir=config.cache_dir / "bm25",
        k1=config.bm25_k1,
        b=config.bm25_b,
    )

    deleted_paths = []
    deleted_paths.extend(delete_dense_cache_files(dense_cache_paths))
    deleted_paths.extend(delete_bm25_cache_files(bm25_cache_paths))
    if deleted_paths:
        print("Debug run 已清除本次 cache：")
        for path in deleted_paths:
            print(f"  - {path}")
    else:
        print("Debug run 未找到需清除的 cache 檔案。")


def main() -> None:
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    if args.use_fp16:
        args.compute_dtype = "float16"

    from src.pipeline_hybrid_union_rerank import (
        HybridUnionRerankConfig,
        run_hybrid_union_rerank_retrieval,
    )

    log_path = resolve_output_path(args.log_file) if args.log_file else default_log_path()
    config = HybridUnionRerankConfig(
        data_path=resolve_existing_path(args.data),
        query_path=resolve_existing_path(args.query),
        subquery_path=resolve_existing_path(args.subquery_path) if args.subquery_path else None,
        output_path=resolve_output_path(args.output),
        dense_model_name_or_path=resolve_model_reference(args.dense_model),
        reranker_model_name_or_path=resolve_model_reference(args.reranker_model),
        cache_dir=resolve_output_path(args.cache_dir),
        model_cache_dir=(
            resolve_output_path(args.model_cache_dir)
            if args.model_cache_dir
            else resolve_output_path(args.cache_dir) / "models"
        ),
        bm25_top_k=args.bm25_top_k,
        dense_top_k=args.dense_top_k,
        final_top_k=args.final_top_k,
        risk_top_k=args.risk_top_k,
        dense_batch_size=args.dense_batch_size,
        reranker_batch_size=args.reranker_batch_size,
        risk_batch_size=args.risk_batch_size,
        dense_max_length=args.dense_max_length,
        reranker_max_length=args.reranker_max_length,
        risk_max_length=args.risk_max_length,
        risk_max_new_tokens=args.risk_max_new_tokens,
        bm25_k1=args.bm25_k1,
        bm25_b=args.bm25_b,
        use_4bit=not args.disable_4bit,
        compute_dtype=args.compute_dtype,
        use_cache=not args.no_cache,
        limit_docs=args.limit_docs,
        limit_queries=args.limit_queries,
        retrieval_instruction=args.retrieval_instruction,
        fusion_method="rrf" if args.use_rrf else "union",
        rrf_k=args.rrf_k,
        risk_model_name_or_path=resolve_model_reference(args.risk_model) if args.subquery_path else None,
        risk_model_cache_dir=resolve_existing_path(args.risk_model_cache_dir) if args.risk_model_cache_dir else None,
        risk_lambda=args.risk_lambda,
    )

    log_file, original_stdout, original_stderr = enable_log_file(log_path)
    try:
        started_at = time.perf_counter()
        print(f"Log file: {log_path}")
        print_run_config(config, log_path)
        print(
            "開始執行 Hybrid retrieval："
            f"BM25 top-k + Dense top-k -> {config.fusion_method} -> rerank top-{config.risk_top_k}"
            + (" -> risk penalty -> final top-k。" if config.subquery_path else " -> final top-k。")
        )
        run_hybrid_union_rerank_retrieval(config)
        elapsed = time.perf_counter() - started_at
        print(f"Hybrid retrieval 流程完成，總耗時 {elapsed:.2f} 秒。")
    finally:
        try:
            cleanup_debug_caches(config)
        except Exception as exc:
            print(f"清理 debug cache 時發生錯誤：{exc}")
        disable_log_file(log_file, original_stdout, original_stderr)


if __name__ == "__main__":
    main()
