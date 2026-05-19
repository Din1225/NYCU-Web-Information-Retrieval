# 主要用途：串接資料讀取、BM25 召回、dense 召回、候選合併、rerank 與結果輸出。

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.bm25_retriever import BM25Retriever
from src.data_io import load_documents, load_queries, save_json
from src.dense_retriever import DenseRetriever
from src.reranker import CrossEncoderReranker
from src.text_processing import question_text


# 定義整個檢索流程會用到的路徑、top-k、batch size、快取與測試限制等設定。
@dataclass(frozen=True)
class RetrievalConfig:
    data_path: Path
    query_path: Path
    output_path: Path
    dense_model_path: Path
    reranker_model_path: Path
    cache_dir: Path
    bm25_top_k: int = 50
    dense_top_k: int = 50
    rerank_top_k: int = 30
    dense_batch_size: int = 32
    reranker_batch_size: int = 16
    dense_max_length: int = 512
    reranker_max_length: int = 512
    use_fp16: bool = False
    use_cache: bool = True
    limit_docs: int | None = None
    limit_queries: int | None = None


# 執行完整檢索流程，針對每個 query 產生 rerank 後的 top-k 回答結果。
def run_retrieval(config: RetrievalConfig) -> list[dict[str, Any]]:
    documents = load_documents(config.data_path, limit=config.limit_docs)
    queries = load_queries(config.query_path, limit=config.limit_queries)

    print(f"Loaded {len(documents)} documents from {config.data_path}")
    print(f"Loaded {len(queries)} queries from {config.query_path}")
    print("資料讀取已完成。", flush=True)

    bm25 = BM25Retriever(documents)
    print("BM25 索引建立已完成。", flush=True)

    dense = DenseRetriever(
        documents=documents,
        model_path=config.dense_model_path,
        cache_dir=config.cache_dir / "dense",
        batch_size=config.dense_batch_size,
        max_length=config.dense_max_length,
        use_fp16=config.use_fp16,
        use_cache=config.use_cache,
    )
    print("Dense retriever 初始化已完成。", flush=True)

    reranker = CrossEncoderReranker(
        model_path=config.reranker_model_path,
        batch_size=config.reranker_batch_size,
        max_length=config.reranker_max_length,
        use_fp16=config.use_fp16,
    )
    print("Reranker 初始化已完成。", flush=True)

    all_results: list[dict[str, Any]] = []
    for query in tqdm(queries, desc="Retrieving queries"):
        query_text = question_text(query)
        query_id = query["ID"]

        bm25_results = bm25.retrieve(query_text, top_k=config.bm25_top_k)
        print(f"{query_id} BM25 召回已完成。", flush=True)

        dense_results = dense.retrieve(query_text, top_k=config.dense_top_k)
        print(f"{query_id} Dense 召回已完成。", flush=True)

        candidates = merge_candidates(bm25_results, dense_results)
        print(f"{query_id} 候選合併已完成，共 {len(candidates)} 筆候選。", flush=True)

        reranked = reranker.rerank(
            query=query_text,
            candidates=candidates,
            documents=documents,
            top_k=config.rerank_top_k,
        )
        print(f"{query_id} Cross-encoder rerank 已完成。", flush=True)

        results = build_output_results(reranked, documents)
        print(f"{query_id} 結果整理已完成，共 {len(results)} 筆結果。", flush=True)

        all_results.append(
            {
                "query_id": query_id,
                "query": query_text,
                "num_candidates": len(candidates),
                "results": results,
            }
        )

    save_json(all_results, config.output_path)
    print(f"Saved results to {config.output_path}")
    print("結果輸出已完成。", flush=True)
    return all_results


# 將 BM25 和 dense retriever 的候選結果依 doc_index 去重合併，並保留來源與分數。
def merge_candidates(
    bm25_results: list[dict[str, Any]],
    dense_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge BM25 and dense candidates by document index."""
    merged: dict[int, dict[str, Any]] = {}

    for rank, item in enumerate(bm25_results, start=1):
        doc_index = int(item["doc_index"])
        merged[doc_index] = {
            "doc_index": doc_index,
            "doc_id": item["doc_id"],
            "bm25_rank": rank,
            "bm25_score": item.get("bm25_score"),
            "dense_rank": None,
            "dense_score": None,
            "sources": ["bm25"],
        }

    for rank, item in enumerate(dense_results, start=1):
        doc_index = int(item["doc_index"])
        if doc_index not in merged:
            merged[doc_index] = {
                "doc_index": doc_index,
                "doc_id": item["doc_id"],
                "bm25_rank": None,
                "bm25_score": None,
                "dense_rank": rank,
                "dense_score": item.get("dense_score"),
                "sources": ["dense"],
            }
        else:
            merged[doc_index]["dense_rank"] = rank
            merged[doc_index]["dense_score"] = item.get("dense_score")
            merged[doc_index]["sources"].append("dense")

    return list(merged.values())


# 將 rerank 結果補上歷史問題、答案與各階段分數，整理成最終輸出的 JSON 格式。
def build_output_results(
    reranked: list[dict[str, Any]],
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for rank, item in enumerate(reranked, start=1):
        document = documents[int(item["doc_index"])]
        output.append(
            {
                "rank": rank,
                "doc_index": int(item["doc_index"]),
                "doc_id": document["ID"],
                "question": document["Question"],
                "answer": document["Answer"],
                "rerank_score": item.get("rerank_score"),
                "bm25_rank": item.get("bm25_rank"),
                "bm25_score": item.get("bm25_score"),
                "dense_rank": item.get("dense_rank"),
                "dense_score": item.get("dense_score"),
                "sources": item.get("sources", []),
            }
        )
    return output
