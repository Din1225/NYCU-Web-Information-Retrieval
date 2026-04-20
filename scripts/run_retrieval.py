from __future__ import annotations

"""
測試的話
python scripts/run_retrieval.py \
  --data data/IR_data.json \
  --query query/phase1_query.json \
  --output outputs/debug_100docs_results.json \
  --limit_docs 100 \
  --limit_queries 1 \
  --bm25_top_k 50 \
  --dense_top_k 50 \
  --rerank_top_k 30 \
  --use_fp16 \
  --log_file outputs/logs/debug_100docs.log

正式執行的話
python scripts/run_retrieval.py \
  --data data/IR_data.json \
  --query query/phase1_query.json \
  --output outputs/ver1_results.json \
  --bm25_top_k 50 \
  --dense_top_k 50 \
  --rerank_top_k 30 \
  --use_fp16 \
  --log_file outputs/logs/ver11.log
"""


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 指定使用 GPU

# 主要用途：提供命令列入口，讓使用者用參數控制完整檢索與 rerank 流程。

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import RetrievalConfig, run_retrieval  # noqa: E402


class TeeOutput:
    """同時把輸出寫到終端機與 log 檔案。"""

    # 初始化多個輸出 stream，例如原本的 stdout/stderr 和 log 檔案。
    def __init__(self, *streams) -> None:
        self.streams = streams

    # 將文字同步寫到所有 stream，tqdm 的進度條也會透過這裡被記錄。
    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    # 手動 flush 所有 stream，避免程式中斷時 log 還留在 buffer。
    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    # 讓 tqdm 可以判斷終端機是否支援互動式進度條。
    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


# 建立並解析命令列參數，所有參數最後會轉成 RetrievalConfig。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BM25 + dense retrieval + cross-encoder reranking.")
    parser.add_argument("--data", default="data/IR_data.json", help="Path to document JSON.") # --data：歷史問答資料池 JSON，檢索器會從這個檔案的 Question 欄位找候選，最後輸出 Answer。
    parser.add_argument("--query", default="query/phase1_query.json", help="Path to query JSON.") # --query：待檢索的 query JSON，每筆 query 需要有 ID 和 Question 欄位。
    parser.add_argument("--output", default="outputs/phase1_results.json", help="Output JSON path.") # --output：完整流程完成後的輸出 JSON 路徑，會包含每個 query 的 top-k 結果。
    parser.add_argument(
        "--dense_model",
        default="model_cache/BAAI/bge-m3",
        help="Local path to BAAI/bge-m3.",
    ) # --dense_model：bge-m3 dense retriever 的本地模型資料夾路徑。
    parser.add_argument(
        "--reranker_model",
        default="model_cache/BAAI/bge-reranker-v2-m3",
        help="Local path to BAAI/bge-reranker-v2-m3.",
    ) # --reranker_model：bge-reranker-v2-m3 cross-encoder reranker 的本地模型資料夾路徑。
    parser.add_argument("--cache_dir", default="outputs/cache", help="Directory for reusable caches.") # --cache_dir：快取資料夾，目前主要用來存 dense 文件 embedding，避免每次重算。
    parser.add_argument("--bm25_top_k", type=int, default=50) # --bm25_top_k：BM25 第一階段召回的歷史問題數量。
    parser.add_argument("--dense_top_k", type=int, default=50) # --dense_top_k：dense retriever 第一階段召回的歷史問題數量。
    parser.add_argument("--rerank_top_k", type=int, default=30) # --rerank_top_k：BM25 和 dense 合併去重後，reranker 最後保留的結果數量。
    parser.add_argument("--dense_batch_size", type=int, default=32) # --dense_batch_size：bge-m3 做文件或 query embedding 時，每批送入模型的文字數量。
    parser.add_argument("--reranker_batch_size", type=int, default=16) # --reranker_batch_size：cross-encoder reranker 每批計算 query-candidate pair 的數量。
    parser.add_argument("--dense_max_length", type=int, default=512) # --dense_max_length：dense retriever tokenizer 的最大輸入長度，超過會被模型截斷。
    parser.add_argument("--reranker_max_length", type=int, default=512) # --reranker_max_length：reranker tokenizer 的最大輸入長度，超過會被模型截斷。
    parser.add_argument("--use_fp16", action="store_true", help="Enable fp16 model inference.") # --use_fp16：使用 fp16 推論，可降低 GPU 記憶體用量並加速；CPU 或不支援時不要開。
    parser.add_argument("--no_cache", action="store_true", help="Disable dense embedding cache loading.") # --no_cache：關閉 dense embedding 快取讀取，強制重新計算並覆寫新的快取。
    parser.add_argument("--limit_docs", type=int, default=None, help="Debug only: limit document count.") # --limit_docs：除錯用，只讀取前 N 筆歷史資料，方便快速測試流程。
    parser.add_argument("--limit_queries", type=int, default=None, help="Debug only: limit query count.") # --limit_queries：除錯用，只讀取前 N 筆 query，方便快速測試流程。
    parser.add_argument("--log_file", default=None, help="Path to log file.") # --log_file：執行時的 log 檔案路徑；不指定時會自動寫到 outputs/logs。
    return parser.parse_args()


# 將相對路徑轉成以專案根目錄為基準的絕對路徑；若已是絕對路徑則直接使用。
def resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


# 產生預設 log 檔路徑，使用時間戳避免覆蓋前一次執行紀錄。
def default_log_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "outputs" / "logs" / f"retrieval_{timestamp}.log"


# 將 stdout 和 stderr 同時導到終端機與 log 檔，讓一般輸出、警告、錯誤和 tqdm 都能被記錄。
def enable_log_file(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeOutput(original_stdout, log_file)
    sys.stderr = TeeOutput(original_stderr, log_file)
    return log_file, original_stdout, original_stderr


# 還原 stdout 和 stderr，並關閉 log 檔。
def disable_log_file(log_file, original_stdout, original_stderr) -> None:
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file.close()


# 印出本次執行設定，方便從終端機或 log 回看使用了哪些參數。
def print_run_config(config: RetrievalConfig, log_path: Path) -> None:
    print("=== Retrieval Run Config ===")
    print(f"data_path: {config.data_path}")
    print(f"query_path: {config.query_path}")
    print(f"output_path: {config.output_path}")
    print(f"dense_model_path: {config.dense_model_path}")
    print(f"reranker_model_path: {config.reranker_model_path}")
    print(f"cache_dir: {config.cache_dir}")
    print(f"log_file: {log_path}")
    print(f"bm25_top_k: {config.bm25_top_k}")
    print(f"dense_top_k: {config.dense_top_k}")
    print(f"rerank_top_k: {config.rerank_top_k}")
    print(f"dense_batch_size: {config.dense_batch_size}")
    print(f"reranker_batch_size: {config.reranker_batch_size}")
    print(f"dense_max_length: {config.dense_max_length}")
    print(f"reranker_max_length: {config.reranker_max_length}")
    print(f"use_fp16: {config.use_fp16}")
    print(f"use_cache: {config.use_cache}")
    print(f"limit_docs: {config.limit_docs}")
    print(f"limit_queries: {config.limit_queries}")
    print("============================")


# 程式進入點：解析 CLI 參數、建立 RetrievalConfig，並啟動完整檢索流程。
def main() -> None:
    args = parse_args()
    log_path = resolve_path(args.log_file) if args.log_file else default_log_path()
    config = RetrievalConfig(
        data_path=resolve_path(args.data),
        query_path=resolve_path(args.query),
        output_path=resolve_path(args.output),
        dense_model_path=resolve_path(args.dense_model),
        reranker_model_path=resolve_path(args.reranker_model),
        cache_dir=resolve_path(args.cache_dir),
        bm25_top_k=args.bm25_top_k,
        dense_top_k=args.dense_top_k,
        rerank_top_k=args.rerank_top_k,
        dense_batch_size=args.dense_batch_size,
        reranker_batch_size=args.reranker_batch_size,
        dense_max_length=args.dense_max_length,
        reranker_max_length=args.reranker_max_length,
        use_fp16=args.use_fp16,
        use_cache=not args.no_cache,
        limit_docs=args.limit_docs,
        limit_queries=args.limit_queries,
    )

    log_file, original_stdout, original_stderr = enable_log_file(log_path)
    try:
        started_at = time.perf_counter()
        print(f"Log file: {log_path}")
        print_run_config(config, log_path)
        print("開始執行檢索流程，後續會顯示 BM25 斷詞、dense embedding 和 query retrieval 進度。")
        run_retrieval(config)
        elapsed = time.perf_counter() - started_at
        print(f"檢索流程完成，總耗時 {elapsed:.2f} 秒。")
    finally:
        disable_log_file(log_file, original_stdout, original_stderr)


if __name__ == "__main__":
    main()

