from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.bm25_retriever import BM25Retriever
from src.data_io import load_documents, load_queries, load_subqueries_map, save_json
from src.dense_retriever import DenseRetriever
from src.qwen_utils import DEFAULT_RETRIEVAL_INSTRUCTION
from src.risk_detector import QwenRiskDetector
from src.reranker import CrossEncoderReranker
from src.text_processing import question_text

FUSION_METHOD_UNION = "union"
FUSION_METHOD_RRF = "rrf"


@dataclass(frozen=True)
class HybridUnionRerankConfig:
    data_path: Path
    query_path: Path
    output_path: Path
    dense_model_name_or_path: str
    reranker_model_name_or_path: str
    cache_dir: Path
    subquery_path: Path | None = None
    model_cache_dir: Path | None = None
    bm25_top_k: int = 100
    dense_top_k: int = 100
    final_top_k: int = 30
    risk_top_k: int = 50
    dense_batch_size: int = 32
    reranker_batch_size: int = 16
    risk_batch_size: int = 4
    dense_max_length: int = 512
    reranker_max_length: int = 512
    risk_max_length: int = 1024
    risk_max_new_tokens: int = 256
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    use_4bit: bool = True
    compute_dtype: str = "float16"
    use_cache: bool = True
    limit_docs: int | None = None
    limit_queries: int | None = None
    retrieval_instruction: str | None = DEFAULT_RETRIEVAL_INSTRUCTION
    fusion_method: str = FUSION_METHOD_UNION
    rrf_k: int = 60
    risk_model_name_or_path: str | None = None
    risk_model_cache_dir: Path | None = None
    risk_lambda: float = 0.1


@dataclass
class QueryWorkItem:
    query_id: str
    query_text: str
    sub_queries: list[str] | None = None
    bm25_results: list[dict[str, Any]] | None = None
    dense_results: list[dict[str, Any]] | None = None
    fused_candidates: list[dict[str, Any]] | None = None
    reranked_candidates: list[dict[str, Any]] | None = None
    final_results: list[dict[str, Any]] | None = None


def run_hybrid_union_rerank_retrieval(config: HybridUnionRerankConfig) -> list[dict[str, Any]]:
    """執行 BM25 + Dense 融合候選，reranker top-k，再做風險懲罰與最終排序。"""
    documents = load_documents(config.data_path, limit=config.limit_docs)
    queries = load_queries(config.query_path, limit=config.limit_queries)
    subqueries_map = load_subqueries_map(config.subquery_path) if config.subquery_path is not None else {}
    risk_penalty_enabled = _risk_penalty_enabled(config)

    print(f"Loaded {len(documents)} documents from {config.data_path}", flush=True)
    print(f"Loaded {len(queries)} queries from {config.query_path}", flush=True)
    _log_step_result(
        step_no=1,
        title="讀取文件、查詢與子查詢",
        payload={
            "num_documents": len(documents),
            "num_queries": len(queries),
            "subquery_path": str(config.subquery_path) if config.subquery_path is not None else None,
            "num_subquery_groups": len(subqueries_map),
            "risk_penalty_enabled": risk_penalty_enabled,
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
            sub_queries=subqueries_map.get(query["ID"], []),
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
        item.fused_candidates = fuse_candidates(
            bm25_results=item.bm25_results,
            dense_results=item.dense_results,
            fusion_method=config.fusion_method,
            rrf_k=config.rrf_k,
        )
        fusion_preview_score_key = "rrf_score" if config.fusion_method == FUSION_METHOD_RRF else "union_seed_rank"
        _log_step_result(
            step_no=4,
            title=f"{item.query_id} Hybrid 候選集合",
            payload={
                "query_id": item.query_id,
                "query": item.query_text,
                "sub_query_count": len(item.sub_queries or []),
                "fusion_method": config.fusion_method,
                "rrf_k": config.rrf_k if config.fusion_method == FUSION_METHOD_RRF else None,
                "bm25_top_k": len(item.bm25_results),
                "dense_top_k": len(item.dense_results),
                "fusion_size": len(item.fused_candidates),
                "union_size": len(item.fused_candidates),
                "bm25_preview": _candidate_preview(item.bm25_results, score_key="bm25_score"),
                "dense_preview": _candidate_preview(item.dense_results, score_key="dense_score"),
                "fusion_preview": _candidate_preview(item.fused_candidates, score_key=fusion_preview_score_key),
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
    reranker_top_k = config.risk_top_k if risk_penalty_enabled else min(config.final_top_k, config.risk_top_k)
    _log_step_result(
        step_no=5,
        title="載入 Reranker",
        payload={
            "reranker_model_name_or_path": config.reranker_model_name_or_path,
            "model_cache_dir": str(model_cache_dir / "reranker"),
            "reranker_batch_size": config.reranker_batch_size,
            "reranker_top_k": reranker_top_k,
        },
    )

    for item in tqdm(query_items, desc="Reranking fused candidates"):
        item.reranked_candidates = reranker.rerank(
            query=item.query_text,
            candidates=item.fused_candidates or [],
            documents=documents,
            top_k=reranker_top_k,
        )
        _log_step_result(
            step_no=6,
            title=f"{item.query_id} Rerank top-{reranker_top_k}",
            payload={
                "query_id": item.query_id,
                "sub_query_count": len(item.sub_queries or []),
                "rerank_top_k": len(item.reranked_candidates),
                "result_preview": [
                    {
                        "rank": result["rank"],
                        "doc_id": result["doc_id"],
                        "rerank_score": result["rerank_score"],
                        "candidate_sources": result["candidate_sources"],
                    }
                    for result in (item.reranked_candidates or [])[:3]
                ],
            },
        )

    reranker.release_resources(clear_tokenizer=True)
    print("Reranker 模型已釋放。", flush=True)

    if risk_penalty_enabled:
        risk_detector = QwenRiskDetector(
            model_name_or_path=config.risk_model_name_or_path or "Qwen/Qwen3.5-4B",
            model_cache_dir=config.risk_model_cache_dir,
            batch_size=config.risk_batch_size,
            max_length=config.risk_max_length,
            max_new_tokens=config.risk_max_new_tokens,
            use_4bit=config.use_4bit,
            compute_dtype=config.compute_dtype,
        )
        _log_step_result(
            step_no=7,
            title="載入 Risk Detector",
            payload={
                "risk_model_name_or_path": config.risk_model_name_or_path,
                "risk_model_cache_dir": str(config.risk_model_cache_dir) if config.risk_model_cache_dir is not None else None,
                "risk_batch_size": config.risk_batch_size,
                "risk_lambda": config.risk_lambda,
            },
        )

        for item in tqdm(query_items, desc="Applying risk penalty"):
            penalized_candidates = apply_risk_penalty(
                query=item.query_text,
                sub_queries=item.sub_queries or [],
                reranked_candidates=item.reranked_candidates or [],
                documents=documents,
                risk_detector=risk_detector,
                risk_lambda=config.risk_lambda,
            )
            item.final_results = build_hybrid_output_results(
                penalized_candidates[: config.final_top_k],
                documents,
            )
            _log_step_result(
                step_no=8,
                title=f"{item.query_id} Risk-adjusted top-{config.final_top_k}",
                payload={
                    "query_id": item.query_id,
                    "sub_query_count": len(item.sub_queries or []),
                    "final_top_k": len(item.final_results),
                    "result_details": [
                        {
                            "rank": result["rank"],
                            "doc_id": result["doc_id"],
                            "risk_parse_ok": result["risk_parse_ok"],
                            "rerank_score_norm": result["rerank_score_norm"],
                            "risk": result["risk"],
                            "final_score": result["final_score"],
                        }
                        for result in item.final_results
                    ],
                },
            )

        risk_detector.release_resources(clear_tokenizer=True)
        print("Risk detector 模型已釋放。", flush=True)
    else:
        for item in query_items:
            finalized_candidates = finalize_without_risk(
                reranked_candidates=item.reranked_candidates or [],
                risk_lambda=config.risk_lambda,
            )
            item.final_results = build_hybrid_output_results(
                finalized_candidates[: config.final_top_k],
                documents,
            )

    all_results: list[dict[str, Any]] = []
    for item in query_items:
        all_results.append(
            {
                "query_id": item.query_id,
                "query": item.query_text,
                "sub_queries": item.sub_queries or [],
                "sub_query_count": len(item.sub_queries or []),
                "fusion_method": config.fusion_method,
                "rrf_k": config.rrf_k if config.fusion_method == FUSION_METHOD_RRF else None,
                "bm25_top_k": config.bm25_top_k,
                "dense_top_k": config.dense_top_k,
                "fusion_size": len(item.fused_candidates or []),
                "union_size": len(item.fused_candidates or []),
                "rerank_top_k": len(item.reranked_candidates or []),
                "final_top_k": len(item.final_results or []),
                "dense_model_name_or_path": config.dense_model_name_or_path,
                "reranker_model_name_or_path": config.reranker_model_name_or_path,
                "risk_penalty_enabled": risk_penalty_enabled,
                "risk_model_name_or_path": config.risk_model_name_or_path if risk_penalty_enabled else None,
                "risk_lambda": config.risk_lambda if risk_penalty_enabled else None,
                "risk_top_k": config.risk_top_k if risk_penalty_enabled else None,
                "retrieval_instruction": config.retrieval_instruction,
                "document_text_mode": "Question + Answer",
                "bm25": {
                    "k1": config.bm25_k1,
                    "b": config.bm25_b,
                },
                "results": item.final_results or [],
            }
        )

    save_json(all_results, config.output_path)
    print(f"Saved results to {config.output_path}", flush=True)
    return all_results


def union_candidates(
    bm25_results: list[dict[str, Any]],
    dense_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """將 BM25 與 Dense 的 top-k 結果做 union，並保留來源資訊。"""
    return fuse_candidates(
        bm25_results=bm25_results,
        dense_results=dense_results,
        fusion_method=FUSION_METHOD_UNION,
    )


def fuse_candidates(
    bm25_results: list[dict[str, Any]],
    dense_results: list[dict[str, Any]],
    fusion_method: str = FUSION_METHOD_UNION,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """融合 BM25 與 Dense 的 top-k 結果，保留來源資訊與融合排名。"""
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

    if fusion_method == FUSION_METHOD_UNION:
        return _finalize_union_candidates(merged)
    if fusion_method == FUSION_METHOD_RRF:
        return _finalize_rrf_candidates(merged, rrf_k=rrf_k)

    raise ValueError(f"Unsupported fusion_method: {fusion_method}")


def _finalize_union_candidates(merged: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    union_results = list(merged.values())
    union_results.sort(
        key=lambda item: (
            item["fusion_seed_rank"],
            item.get("dense_rank", 10**9),
            item.get("bm25_rank", 10**9),
            item["doc_index"],
        )
    )
    for rank, item in enumerate(union_results, start=1):
        item["rank"] = rank
        item["fusion_method"] = FUSION_METHOD_UNION
        item["fusion_rank"] = rank
        item["fusion_score"] = None
        item["union_rank"] = rank
        item["source"] = "union"
    return union_results


def _finalize_rrf_candidates(
    merged: dict[int, dict[str, Any]],
    rrf_k: int,
) -> list[dict[str, Any]]:
    if rrf_k < 0:
        raise ValueError(f"rrf_k must be >= 0, got {rrf_k}")

    fused_results = list(merged.values())
    for item in fused_results:
        item["rrf_score"] = _compute_rrf_score(item, rrf_k=rrf_k)
        item["fusion_score"] = item["rrf_score"]

    fused_results.sort(
        key=lambda item: (
            -item["rrf_score"],
            item["fusion_seed_rank"],
            item.get("dense_rank", 10**9),
            item.get("bm25_rank", 10**9),
            item["doc_index"],
        )
    )
    for rank, item in enumerate(fused_results, start=1):
        item["rank"] = rank
        item["fusion_method"] = FUSION_METHOD_RRF
        item["fusion_rank"] = rank
        item["union_rank"] = rank
        item["source"] = FUSION_METHOD_RRF
    return fused_results


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
                "fusion_method": item.get("fusion_method"),
                "fusion_rank": item.get("fusion_rank"),
                "fusion_score": item.get("fusion_score"),
                "candidate_sources": item.get("candidate_sources", []),
                "union_rank": item.get("union_rank"),
                "union_seed_rank": item.get("union_seed_rank"),
                "bm25_score": item.get("bm25_score"),
                "bm25_rank": item.get("bm25_rank"),
                "dense_score": item.get("dense_score"),
                "dense_rank": item.get("dense_rank"),
                "rrf_score": item.get("rrf_score"),
                "rerank_score_norm": item.get("rerank_score_norm"),
                "rerank_score": item.get("rerank_score"),
                "rerank_rank": item.get("rank"),
                "risk_lambda": item.get("risk_lambda"),
                "risk": item.get("risk"),
                "risk_parse_ok": item.get("risk_parse_ok"),
                "topic_mismatch_risk": item.get("topic_mismatch_risk"),
                "missing_subquery_risk": item.get("missing_subquery_risk"),
                "keyword_only_risk": item.get("keyword_only_risk"),
                "shallow_answer_risk": item.get("shallow_answer_risk"),
                "final_score": item.get("final_score"),
                "final_rank": item.get("final_rank", rank),
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
    entry["fusion_seed_rank"] = min(bm25_rank, dense_rank)
    entry["union_seed_rank"] = entry["fusion_seed_rank"]


def _compute_rrf_score(entry: dict[str, Any], rrf_k: int) -> float:
    score = 0.0
    for rank_key in ("bm25_rank", "dense_rank"):
        rank = entry.get(rank_key)
        if rank is not None:
            score += 1.0 / (rrf_k + int(rank))
    return float(score)


def apply_risk_penalty(
    query: str,
    sub_queries: list[str],
    reranked_candidates: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    risk_detector: QwenRiskDetector,
    risk_lambda: float,
) -> list[dict[str, Any]]:
    if not reranked_candidates:
        return []

    if not sub_queries:
        return finalize_without_risk(reranked_candidates=reranked_candidates, risk_lambda=risk_lambda)

    risk_outputs = risk_detector.detect(
        query=query,
        sub_queries=sub_queries,
        candidates=reranked_candidates,
        documents=documents,
    )
    penalized_candidates: list[dict[str, Any]] = []
    for candidate, risk_output in zip(reranked_candidates, risk_outputs, strict=True):
        rerank_score_norm = float(candidate.get("rerank_score", 0.0))
        item = dict(candidate)
        item["rerank_score_norm"] = rerank_score_norm
        item["risk_lambda"] = risk_lambda
        item.update(risk_output)
        item["final_score"] = rerank_score_norm - risk_lambda * float(item["risk"])
        penalized_candidates.append(item)
    return _sort_final_candidates(penalized_candidates)


def finalize_without_risk(
    reranked_candidates: list[dict[str, Any]],
    risk_lambda: float,
) -> list[dict[str, Any]]:
    finalized_candidates: list[dict[str, Any]] = []
    for candidate in reranked_candidates:
        rerank_score_norm = float(candidate.get("rerank_score", 0.0))
        item = dict(candidate)
        item["rerank_score_norm"] = rerank_score_norm
        item["risk_lambda"] = risk_lambda
        item["risk"] = 0.0
        item["risk_parse_ok"] = None
        item["topic_mismatch_risk"] = 0.0
        item["missing_subquery_risk"] = 0.0
        item["keyword_only_risk"] = 0.0
        item["shallow_answer_risk"] = 0.0
        item["final_score"] = rerank_score_norm
        finalized_candidates.append(item)
    return _sort_final_candidates(finalized_candidates)


def _sort_final_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates.sort(
        key=lambda item: (
            -float(item.get("final_score", 0.0)),
            -float(item.get("rerank_score_norm", 0.0)),
            int(item.get("rank", 10**9)),
            int(item.get("doc_index", 10**9)),
        )
    )
    for final_rank, item in enumerate(candidates, start=1):
        item["final_rank"] = final_rank
    return candidates


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


def _risk_penalty_enabled(config: HybridUnionRerankConfig) -> bool:
    return config.subquery_path is not None and config.risk_model_name_or_path is not None
