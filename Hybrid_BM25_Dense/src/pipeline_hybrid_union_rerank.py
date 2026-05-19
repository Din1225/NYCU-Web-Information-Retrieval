from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.bm25_retriever import BM25Retriever
from src.data_io import load_documents, load_queries, save_json
from src.dense_retriever import DenseRetriever
from src.qwen_utils import DEFAULT_RETRIEVAL_INSTRUCTION
from src.reranker import CrossEncoderReranker
from src.text_processing import question_text


@dataclass(frozen=True)
class HybridUnionRerankConfig:
    data_path: Path
    query_path: Path
    output_path: Path
    dense_model_name_or_path: str
    reranker_model_name_or_path: str
    cache_dir: Path
    model_cache_dir: Path | None = None
    bm25_top_k: int = 100
    dense_top_k: int = 100
    final_top_k: int = 30
    dense_batch_size: int = 32
    reranker_batch_size: int = 16
    dense_max_length: int = 512
    reranker_max_length: int = 512
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    use_4bit: bool = True
    compute_dtype: str = "float16"
    use_cache: bool = True
    limit_docs: int | None = None
    limit_queries: int | None = None
    retrieval_instruction: str | None = DEFAULT_RETRIEVAL_INSTRUCTION


@dataclass
class QueryWorkItem:
    query_id: str
    query_text: str
    bm25_results: list[dict[str, Any]] | None = None
    dense_results: list[dict[str, Any]] | None = None
    union_candidates: list[dict[str, Any]] | None = None
    final_results: list[dict[str, Any]] | None = None


def run_hybrid_union_rerank_retrieval(config: HybridUnionRerankConfig) -> list[dict[str, Any]]:
    """執行 BM25 + Dense union candidates，再以 reranker 取 top-30。"""
    documents = load_documents(config.data_path, limit=config.limit_docs)
    queries = load_queries(config.query_path, limit=config.limit_queries)

    print(f"Loaded {len(documents)} documents from {config.data_path}", flush=True)
    print(f"Loaded {len(queries)} queries from {config.query_path}", flush=True)
    _log_step_result(
        step_no=1,
        title="讀取文件與查詢",
        payload={
            "num_documents": len(documents),
            "num_queries": len(queries),
            "document_sample_ids": [item["ID"] for item in documents[:3]],
            "query_sample_ids": [item["ID"] for item in queries[:3]],
        },
    )

    model_cache_dir = config.model_cache_dir or (config.cache_dir / "models")

    bm25 = BM25Retriever(
        documents=documents,
        cache_dir=config.cache_dir / "bm25",
        k1=config.bm25_k1,
        b=config.bm25_b,
        use_cache=config.use_cache,
    )
    _log_step_result(
        step_no=2,
        title="建立或讀取 BM25 index",
        payload={
            "cache_dir": str(config.cache_dir / "bm25"),
            "bm25_k1": config.bm25_k1,
            "bm25_b": config.bm25_b,
        },
    )

    dense = DenseRetriever(
        documents=documents,
        model_name_or_path=config.dense_model_name_or_path,
        cache_dir=config.cache_dir / "dense",
        model_cache_dir=model_cache_dir / "dense",
        batch_size=config.dense_batch_size,
        max_length=config.dense_max_length,
        use_4bit=config.use_4bit,
        compute_dtype=config.compute_dtype,
        use_cache=config.use_cache,
        query_instruction=config.retrieval_instruction,
    )
    _log_step_result(
        step_no=3,
        title="建立或讀取 Dense embeddings",
        payload={
            "embedding_dim": dense.embedding_dim,
            "num_document_embeddings": int(dense.embeddings.shape[0]),
            "cache_dir": str(config.cache_dir / "dense"),
            "model_cache_dir": str(model_cache_dir / "dense"),
        },
    )

    query_items = [
        QueryWorkItem(
            query_id=query["ID"],
            query_text=question_text(query),
        )
        for query in queries
    ]

    for item in tqdm(query_items, desc="Hybrid candidate retrieval"):
        item.bm25_results = bm25.retrieve(
            item.query_text,
            top_k=config.bm25_top_k,
            score_key="bm25_score",
            source="bm25",
        )
        query_embedding = dense.encode_query(item.query_text)
        item.dense_results = dense.retrieve_by_embedding(
            query_embedding,
            top_k=config.dense_top_k,
            score_key="dense_score",
            source="dense",
        )
        item.union_candidates = union_candidates(item.bm25_results, item.dense_results)
        _log_step_result(
            step_no=4,
            title=f"{item.query_id} Hybrid 候選集合",
            payload={
                "query_id": item.query_id,
                "query": item.query_text,
                "bm25_top_k": len(item.bm25_results),
                "dense_top_k": len(item.dense_results),
                "union_size": len(item.union_candidates),
                "bm25_preview": _candidate_preview(item.bm25_results, score_key="bm25_score"),
                "dense_preview": _candidate_preview(item.dense_results, score_key="dense_score"),
                "union_preview": _candidate_preview(item.union_candidates, score_key="union_seed_rank"),
            },
        )

    dense.release_resources(clear_tokenizer=True)
    print("Dense retriever 模型已釋放。", flush=True)

    reranker = CrossEncoderReranker(
        model_name_or_path=config.reranker_model_name_or_path,
        model_cache_dir=model_cache_dir / "reranker",
        batch_size=config.reranker_batch_size,
        max_length=config.reranker_max_length,
        use_4bit=config.use_4bit,
        compute_dtype=config.compute_dtype,
        instruction=config.retrieval_instruction,
    )
    _log_step_result(
        step_no=5,
        title="載入 Reranker",
        payload={
            "reranker_model_name_or_path": config.reranker_model_name_or_path,
            "model_cache_dir": str(model_cache_dir / "reranker"),
            "reranker_batch_size": config.reranker_batch_size,
        },
    )

    all_results: list[dict[str, Any]] = []
    for item in tqdm(query_items, desc="Reranking union candidates"):
        reranked_results = reranker.rerank(
            query=item.query_text,
            candidates=item.union_candidates or [],
            documents=documents,
            top_k=config.final_top_k,
        )
        item.final_results = build_hybrid_output_results(reranked_results, documents)
        _log_step_result(
            step_no=6,
            title=f"{item.query_id} Rerank top-{config.final_top_k}",
            payload={
                "query_id": item.query_id,
                "final_top_k": len(item.final_results),
                "result_preview": [
                    {
                        "rank": result["rank"],
                        "doc_id": result["doc_id"],
                        "rerank_score": result["rerank_score"],
                        "candidate_sources": result["candidate_sources"],
                    }
                    for result in item.final_results[:3]
                ],
            },
        )

        all_results.append(
            {
                "query_id": item.query_id,
                "query": item.query_text,
                "bm25_top_k": config.bm25_top_k,
                "dense_top_k": config.dense_top_k,
                "union_size": len(item.union_candidates or []),
                "final_top_k": len(item.final_results),
                "dense_model_name_or_path": config.dense_model_name_or_path,
                "reranker_model_name_or_path": config.reranker_model_name_or_path,
                "retrieval_instruction": config.retrieval_instruction,
                "document_text_mode": "Question + Answer",
                "bm25": {
                    "k1": config.bm25_k1,
                    "b": config.bm25_b,
                },
                "results": item.final_results,
            }
        )

    reranker.release_resources(clear_tokenizer=True)
    print("Reranker 模型已釋放。", flush=True)

    save_json(all_results, config.output_path)
    print(f"Saved results to {config.output_path}", flush=True)
    return all_results


def union_candidates(
    bm25_results: list[dict[str, Any]],
    dense_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """將 BM25 與 Dense 的 top-k 結果做 union，並保留來源資訊。"""
    merged: dict[int, dict[str, Any]] = {}

    for result in bm25_results:
        _merge_candidate(
            merged=merged,
            result=result,
            source_name="bm25",
            score_key="bm25_score",
            rank_key="bm25_rank",
        )

    for result in dense_results:
        _merge_candidate(
            merged=merged,
            result=result,
            source_name="dense",
            score_key="dense_score",
            rank_key="dense_rank",
        )

    union_results = list(merged.values())
    union_results.sort(
        key=lambda item: (
            item["union_seed_rank"],
            item.get("dense_rank", 10**9),
            item.get("bm25_rank", 10**9),
            item["doc_index"],
        )
    )
    for rank, item in enumerate(union_results, start=1):
        item["rank"] = rank
        item["union_rank"] = rank
        item["source"] = "union"
    return union_results


def build_hybrid_output_results(
    reranked_results: list[dict[str, Any]],
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for rank, item in enumerate(reranked_results, start=1):
        doc_index = int(item["doc_index"])
        document = documents[doc_index]
        output.append(
            {
                "rank": rank,
                "doc_index": doc_index,
                "doc_id": document["ID"],
                "question": document["Question"],
                "answer": document["Answer"],
                "candidate_sources": item.get("candidate_sources", []),
                "union_rank": item.get("union_rank"),
                "union_seed_rank": item.get("union_seed_rank"),
                "bm25_score": item.get("bm25_score"),
                "bm25_rank": item.get("bm25_rank"),
                "dense_score": item.get("dense_score"),
                "dense_rank": item.get("dense_rank"),
                "rerank_score": item.get("rerank_score"),
                "rerank_rank": item.get("rank"),
            }
        )
    return output


def _merge_candidate(
    merged: dict[int, dict[str, Any]],
    result: dict[str, Any],
    source_name: str,
    score_key: str,
    rank_key: str,
) -> None:
    candidate = dict(result)
    doc_index = int(candidate["doc_index"])
    entry = merged.setdefault(
        doc_index,
        {
            "doc_index": doc_index,
            "doc_id": candidate["doc_id"],
            "candidate_sources": [],
        },
    )
    if source_name not in entry["candidate_sources"]:
        entry["candidate_sources"].append(source_name)
    entry[score_key] = candidate.get(score_key)
    entry[rank_key] = int(candidate["rank"])

    bm25_rank = entry.get("bm25_rank", 10**9)
    dense_rank = entry.get("dense_rank", 10**9)
    entry["union_seed_rank"] = min(bm25_rank, dense_rank)


def _candidate_preview(
    candidates: list[dict[str, Any]] | None,
    score_key: str,
    limit: int = 3,
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    preview: list[dict[str, Any]] = []
    for item in candidates[:limit]:
        preview.append(
            {
                "rank": item.get("rank"),
                "doc_id": item.get("doc_id"),
                "doc_index": item.get("doc_index"),
                score_key: item.get(score_key),
                "candidate_sources": item.get("candidate_sources"),
            }
        )
    return preview


def _log_step_result(step_no: int, title: str, payload: dict[str, Any]) -> None:
    print(f"\n=== Step {step_no}: {title} ===", flush=True)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
