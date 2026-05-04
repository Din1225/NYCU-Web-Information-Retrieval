from __future__ import annotations

"""
## Debug run

python normal_RR_GFF_ReFIT/scripts/run_gff_refit_retrieval.py \
  --data data/IR_data.json \
  --query query/phase2_query.json \
  --output outputs/qwen_debug_gff_refit_results.json \
  --limit_docs 1000 \
  --limit_queries 1 \
  --feedback_top_k 100 \
  --final_top_k 30 \
  --dense_model Qwen/Qwen3-Embedding-4B \
  --reranker_model Qwen/Qwen3-Reranker-4B \
  --generator_model Qwen/Qwen3.5-4B \
  --log_file outputs/logs/qwen_debug_gff_refit.log \
  --cuda_visible_devices 0 \
  --print_query_vectors
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
    parser = argparse.ArgumentParser(
        description="Run GFF + ReFIT retrieval.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", default="data/IR_data.json", help="文件資料池 corpus 的 JSON 路徑。")
    parser.add_argument("--query", default="query/phase2_query.json", help="query JSON 檔案路徑。")
    parser.add_argument("--output", default="outputs/gff_refit_results.json", help="最終輸出結果 JSON 路徑。")
    parser.add_argument(
        "--dense_model",
        default="Qwen/Qwen3-Embedding-4B",
        help="Dense retriever 模型，用來建立文件/query embeddings，並執行兩次 dense retrieval。",
    )
    parser.add_argument(
        "--reranker_model",
        default="Qwen/Qwen3-Reranker-4B",
        help="Cross-encoder reranker 模型，用於第一階段 GFF feedback scoring。",
    )
    parser.add_argument(
        "--generator_model",
        default="Qwen/Qwen3.5-4B",
        help="Q2D2K 使用的 instruction-following generator，用來生成 passages 與 keywords。",
    )
    parser.add_argument(
        "--cache_dir",
        default="outputs/cache",
        help="可重複使用的快取根目錄，例如 dense document embeddings cache。",
    )
    parser.add_argument(
        "--model_cache_dir",
        default=None,
        help="模型 snapshot 下載快取目錄；未指定時使用 <cache_dir>/models。",
    )
    parser.add_argument(
        "--feedback_top_k",
        type=int,
        default=100,
        help="第一次 dense retrieval 要保留多少篇候選文件，供後續 GFF reranking 與 ReFIT feedback 使用。",
    )
    parser.add_argument(
        "--final_top_k",
        type=int,
        default=30,
        help="ReFIT 更新 query embedding 後，第二次 dense retrieval 最後輸出幾筆結果。",
    )
    parser.add_argument(
        "--refit_updates",
        type=int,
        default=100,
        help="ReFIT 更新 query embedding 時要做幾步 gradient descent。",
    )
    parser.add_argument(
        "--refit_learning_rate",
        type=float,
        default=0.005,
        help="ReFIT 只更新 query embedding 時使用的 learning rate。",
    )
    parser.add_argument(
        "--refit_temperature",
        type=float,
        default=2.0,
        help="ReFIT 中將 teacher feedback scores 轉成 soft target distribution 時使用的 temperature。",
    )
    parser.add_argument(
        "--refit_no_minmax",
        action="store_true",
        help="關閉 min-max normalization；也就是在把 reranker / retriever scores 轉成 distribution 前不先做 min-max。",
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
        "--generator_batch_size",
        type=int,
        default=1,
        help="Q2D2K 生成 passage 與 keyword 時使用的 batch size。",
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
        help="不讀取也不重用既有的 dense document embeddings cache。",
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
        default="給定一個問題，請檢索出語意最相關、問題表述最相近的問題。",
        help="注入到 Qwen query-side embedding 與 reranker prompt 的 instruction。",
    )
    parser.add_argument(
        "--print_query_vectors",
        action="store_true",
        help="輸出 ReFIT 更新前後的 query embedding 摘要，並一併寫入 output JSON。",
    )
    parser.add_argument(
        "--query_vector_preview_dims",
        type=int,
        default=10,
        help="query embedding debug preview 要顯示前幾個維度。",
    )
    parser.add_argument(
        "--gff_rounds",
        type=int,
        default=3,
        help="GFF query expansion 的 self-consistency 輪數。",
    )
    parser.add_argument(
        "--gff_passages_per_round",
        type=int,
        default=2,
        help="每一輪 self-consistency 中，Q2D2K 對同一個 query 要生成幾段 passages。",
    )
    parser.add_argument(
        "--gff_keywords_per_passage",
        type=int,
        default=15,
        help="每段生成 passage 最多抽取幾個 keywords。",
    )
    parser.add_argument(
        "--gff_top_keywords",
        type=int,
        default=3,
        help="經過 frequency aggregation 與 query-keyword similarity filtering 後，最後保留幾個 keywords。",
    )
    parser.add_argument(
        "--gff_passage_max_new_tokens",
        type=int,
        default=256,
        help="Q2D2K 每段 synthetic passage 最多生成幾個新 token。",
    )
    parser.add_argument(
        "--gff_passage_temperature",
        type=float,
        default=0.8,
        help="Q2D2K 生成 synthetic passages 時使用的 sampling temperature。",
    )
    parser.add_argument(
        "--gff_passage_top_p",
        type=float,
        default=0.95,
        help="Q2D2K 生成 synthetic passages 時使用的 top-p nucleus sampling 門檻。",
    )
    parser.add_argument(
        "--gff_keyword_max_new_tokens",
        type=int,
        default=64,
        help="從單一 passage 生成 keyword text 時，最多生成幾個新 token。",
    )
    parser.add_argument(
        "--gff_keyword_temperature",
        type=float,
        default=0.7,
        help="從每段 passage 生成 keyword text 時使用的 sampling temperature。",
    )
    parser.add_argument(
        "--gff_keyword_top_p",
        type=float,
        default=0.9,
        help="keyword generation 使用的 top-p nucleus sampling 門檻。",
    )
    parser.add_argument(
        "--gff_rrf_k",
        type=int,
        default=60,
        help="original query 與 expanded queries 做 reciprocal-rank fusion 時使用的 k 值。",
    )
    parser.add_argument(
        "--gff_original_query_weight",
        type=float,
        default=0.3,
        help="在融合 expanded-query feedback 前，original-query ranking 的固定權重。",
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

    refit_candidate = REFIT_ROOT / candidate
    if refit_candidate.exists():
        return refit_candidate

    repo_candidate = REPO_ROOT / candidate
    if repo_candidate.exists():
        return repo_candidate

    return refit_candidate


def resolve_model_reference(model_name_or_path: str) -> str:
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
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REFIT_ROOT / candidate


def default_log_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REFIT_ROOT / "outputs" / "logs" / f"gff_refit_{timestamp}.log"


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
    print("=== GFF + ReFIT Run Config ===")
    print(f"data_path: {config.data_path}")
    print(f"query_path: {config.query_path}")
    print(f"output_path: {config.output_path}")
    print(f"dense_model_name_or_path: {config.dense_model_name_or_path}")
    print(f"reranker_model_name_or_path: {config.reranker_model_name_or_path}")
    print(f"generator_model_name_or_path: {config.generator_model_name_or_path}")
    print(f"cache_dir: {config.cache_dir}")
    print(f"model_cache_dir: {config.model_cache_dir}")
    print(f"log_file: {log_path}")
    print(f"feedback_top_k: {config.feedback_top_k}")
    print(f"final_top_k: {config.final_top_k}")
    print(f"refit_updates: {config.refit_updates}")
    print(f"refit_learning_rate: {config.refit_learning_rate}")
    print(f"refit_temperature: {config.refit_temperature}")
    print(f"refit_use_minmax: {config.refit_use_minmax}")
    print(f"use_4bit: {config.use_4bit}")
    print(f"compute_dtype: {config.compute_dtype}")
    print(f"use_cache: {config.use_cache}")
    print(f"gff_rounds: {config.gff_rounds}")
    print(f"gff_passages_per_round: {config.gff_passages_per_round}")
    print(f"gff_keywords_per_passage: {config.gff_keywords_per_passage}")
    print(f"gff_top_keywords: {config.gff_top_keywords}")
    print(f"gff_rrf_k: {config.gff_rrf_k}")
    print(f"gff_original_query_weight: {config.gff_original_query_weight}")
    print("================================")


def cleanup_debug_dense_cache(config) -> None:
    if config.limit_docs is None and config.limit_queries is None:
        return

    from src.data_io import load_documents
    from src.dense_retriever import build_dense_cache_paths, delete_dense_cache_files

    documents = load_documents(config.data_path, limit=config.limit_docs)
    cache_paths = build_dense_cache_paths(
        documents=documents,
        model_name_or_path=config.dense_model_name_or_path,
        cache_dir=config.cache_dir / "dense",
        max_length=config.dense_max_length,
        use_4bit=config.use_4bit,
        compute_dtype=config.compute_dtype,
        query_instruction=config.retrieval_instruction,
    )
    deleted_paths = delete_dense_cache_files(cache_paths)
    if deleted_paths:
        print("Debug run 已清除本次 dense cache：")
        for path in deleted_paths:
            print(f"  - {path}")
    else:
        print("Debug run 未找到需清除的 dense cache 檔案。")


def main() -> None:
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    if args.use_fp16:
        args.compute_dtype = "float16"

    from src.pipeline_gff_refit import GFFReFITConfig, run_gff_refit_retrieval

    log_path = resolve_output_path(args.log_file) if args.log_file else default_log_path()
    config = GFFReFITConfig(
        data_path=resolve_existing_path(args.data),
        query_path=resolve_existing_path(args.query),
        output_path=resolve_output_path(args.output),
        dense_model_name_or_path=resolve_model_reference(args.dense_model),
        reranker_model_name_or_path=resolve_model_reference(args.reranker_model),
        generator_model_name_or_path=resolve_model_reference(args.generator_model),
        cache_dir=resolve_output_path(args.cache_dir),
        model_cache_dir=(
            resolve_output_path(args.model_cache_dir)
            if args.model_cache_dir
            else resolve_output_path(args.cache_dir) / "models"
        ),
        feedback_top_k=args.feedback_top_k,
        final_top_k=args.final_top_k,
        refit_updates=args.refit_updates,
        refit_learning_rate=args.refit_learning_rate,
        refit_temperature=args.refit_temperature,
        refit_use_minmax=not args.refit_no_minmax,
        dense_batch_size=args.dense_batch_size,
        reranker_batch_size=args.reranker_batch_size,
        generator_batch_size=args.generator_batch_size,
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
        gff_rounds=args.gff_rounds,
        gff_passages_per_round=args.gff_passages_per_round,
        gff_keywords_per_passage=args.gff_keywords_per_passage,
        gff_top_keywords=args.gff_top_keywords,
        gff_passage_max_new_tokens=args.gff_passage_max_new_tokens,
        gff_passage_temperature=args.gff_passage_temperature,
        gff_passage_top_p=args.gff_passage_top_p,
        gff_keyword_max_new_tokens=args.gff_keyword_max_new_tokens,
        gff_keyword_temperature=args.gff_keyword_temperature,
        gff_keyword_top_p=args.gff_keyword_top_p,
        gff_rrf_k=args.gff_rrf_k,
        gff_original_query_weight=args.gff_original_query_weight,
    )

    log_file, original_stdout, original_stderr = enable_log_file(log_path)
    try:
        started_at = time.perf_counter()
        print(f"Log file: {log_path}")
        print_run_config(config, log_path)
        print("開始執行 GFF + ReFIT：Q2D2K -> self-consistency -> reciprocal-rank fusion -> ReFIT。")
        run_gff_refit_retrieval(config)
        elapsed = time.perf_counter() - started_at
        print(f"GFF + ReFIT 流程完成，總耗時 {elapsed:.2f} 秒。")
    finally:
        try:
            cleanup_debug_dense_cache(config)
        except Exception as exc:
            print(f"清理 debug dense cache 時發生錯誤：{exc}")
        disable_log_file(log_file, original_stdout, original_stderr)


if __name__ == "__main__":
    main()
