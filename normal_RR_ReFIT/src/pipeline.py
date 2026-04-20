# 主要用途：串接 dense retrieval、reranker feedback、ReFIT query 更新與第二次 dense retrieval。

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from src.data_io import load_documents, load_queries, save_json
from src.dense_retriever import DenseRetriever
from src.refit import ReFITConfig, optimize_query_embedding
from src.reranker import CrossEncoderReranker
from src.text_processing import question_text


@dataclass(frozen=True)
class DenseOnlyReFITConfig:
    data_path: Path
    query_path: Path
    output_path: Path
    dense_model_path: Path
    reranker_model_path: Path
    cache_dir: Path
    feedback_top_k: int = 100
    final_top_k: int = 30
    refit_updates: int = 100
    refit_learning_rate: float = 0.005
    refit_temperature: float = 2.0
    refit_use_minmax: bool = True
    dense_batch_size: int = 32
    reranker_batch_size: int = 16
    dense_max_length: int = 512
    reranker_max_length: int = 512
    use_fp16: bool = False
    use_cache: bool = True
    limit_docs: int | None = None
    limit_queries: int | None = None
    print_query_vectors: bool = False
    query_vector_preview_dims: int = 10


def run_refit_retrieval(config: DenseOnlyReFITConfig) -> list[dict[str, Any]]:
    """執行 dense-only ReFIT 檢索流程。"""
    documents = load_documents(config.data_path, limit=config.limit_docs)
    queries = load_queries(config.query_path, limit=config.limit_queries)

    print(f"Loaded {len(documents)} documents from {config.data_path}")
    print(f"Loaded {len(queries)} queries from {config.query_path}")
    print("資料讀取已完成。", flush=True)

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

    refit_config = ReFITConfig(
        updates=config.refit_updates,
        learning_rate=config.refit_learning_rate,
        temperature=config.refit_temperature,
        use_minmax=config.refit_use_minmax,
    )

    all_results: list[dict[str, Any]] = []
    for query in tqdm(queries, desc="Dense-only ReFIT queries"):
        query_text = question_text(query)
        query_id = query["ID"]

        original_query_embedding = dense.encode_query(query_text)
        first_results = dense.retrieve_by_embedding(
            original_query_embedding,
            top_k=config.feedback_top_k,
            score_key="first_dense_score",
            source="dense_first",
        )
        print(f"{query_id} 第一次 dense retrieval 已完成，共 {len(first_results)} 筆候選。", flush=True)

        reranker_scores = reranker.score(query_text, first_results, documents)
        feedback_candidates = attach_feedback_scores(first_results, reranker_scores)
        first_reranked_candidates = sort_feedback_by_reranker(feedback_candidates)
        print(f"{query_id} Reranker feedback 分數計算已完成。", flush=True)

        feedback_doc_indices = [int(item["doc_index"]) for item in feedback_candidates]
        feedback_embeddings = dense.embeddings_for_indices(feedback_doc_indices)
        updated_query_embedding = optimize_query_embedding(
            query_embedding=original_query_embedding,
            candidate_embeddings=feedback_embeddings,
            reranker_scores=reranker_scores,
            config=refit_config,
        )
        query_embedding_debug = build_query_embedding_debug(
            original_query_embedding,
            updated_query_embedding,
            preview_dims=config.query_vector_preview_dims,
        )
        if config.print_query_vectors:
            print_query_embedding_debug(query_id, query_embedding_debug)
        print(f"{query_id} ReFIT query 向量更新已完成。", flush=True)

        second_results = dense.retrieve_by_embedding(
            updated_query_embedding,
            top_k=config.final_top_k,
            score_key="refit_dense_score",
            source="refit_dense_second",
        )
        print(f"{query_id} 第二次 dense retrieval 已完成，共 {len(second_results)} 筆結果。", flush=True)

        refit_results = build_refit_output_results(second_results, first_reranked_candidates, documents)
        query_output = {
            "query_id": query_id,
            "query": query_text,
            "feedback_top_k": len(feedback_candidates),
            "final_top_k": len(refit_results),
            "refit_updates": config.refit_updates,
            "refit_learning_rate": config.refit_learning_rate,
            "refit_temperature": config.refit_temperature,
            "refit_use_minmax": config.refit_use_minmax,
            "query_vector_shift_l2": query_embedding_debug["shift_l2"],
        }
        if config.print_query_vectors:
            query_output["query_embedding_debug"] = query_embedding_debug
        query_output.update(
            {
                "refit_results": refit_results,
            }
        )
        all_results.append(query_output)
        print(f"{query_id} 結果整理已完成。", flush=True)

    save_json(all_results, config.output_path)
    print(f"Saved results to {config.output_path}")
    print("結果輸出已完成。", flush=True)
    return all_results


def attach_feedback_scores(
    first_results: list[dict[str, Any]],
    reranker_scores: list[float],
) -> list[dict[str, Any]]:
    """將 reranker feedback 分數掛回第一次 dense retrieval 候選。"""
    feedback_candidates: list[dict[str, Any]] = []
    for item, score in zip(first_results, reranker_scores, strict=True):
        candidate = dict(item)
        candidate["first_dense_rank"] = int(item["rank"])
        candidate["rerank_feedback_score"] = float(score)
        feedback_candidates.append(candidate)
    return feedback_candidates


def build_query_embedding_debug(
    original_query_embedding: np.ndarray,
    updated_query_embedding: np.ndarray,
    preview_dims: int,
) -> dict[str, Any]:
    """建立 query embedding 更新前後的 debug 摘要。"""
    original = np.asarray(original_query_embedding, dtype=np.float32)
    updated = np.asarray(updated_query_embedding, dtype=np.float32)
    preview_size = max(0, min(int(preview_dims), int(original.shape[0]), int(updated.shape[0])))
    delta = updated - original
    return {
        "embedding_dim": int(original.shape[0]),
        "preview_dims": preview_size,
        "original_norm_l2": float(np.linalg.norm(original)),
        "updated_norm_l2": float(np.linalg.norm(updated)),
        "shift_l2": float(np.linalg.norm(delta)),
        "original_query_vector_preview": original[:preview_size].astype(float).tolist(),
        "updated_query_vector_preview": updated[:preview_size].astype(float).tolist(),
        "delta_vector_preview": delta[:preview_size].astype(float).tolist(),
    }


def print_query_embedding_debug(query_id: str, debug: dict[str, Any]) -> None:
    """將 query embedding 更新前後摘要印到 log。"""
    print(f"{query_id} Query embedding debug:", flush=True)
    print(f"  embedding_dim: {debug['embedding_dim']}", flush=True)
    print(f"  preview_dims: {debug['preview_dims']}", flush=True)
    print(f"  original_norm_l2: {debug['original_norm_l2']:.8f}", flush=True)
    print(f"  updated_norm_l2: {debug['updated_norm_l2']:.8f}", flush=True)
    print(f"  shift_l2: {debug['shift_l2']:.8f}", flush=True)
    print(f"  original_query_vector_preview: {debug['original_query_vector_preview']}", flush=True)
    print(f"  updated_query_vector_preview: {debug['updated_query_vector_preview']}", flush=True)
    print(f"  delta_vector_preview: {debug['delta_vector_preview']}", flush=True)


def sort_feedback_by_reranker(feedback_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """依 reranker feedback 分數排序第一次 dense 候選。"""
    reranked = [dict(item) for item in feedback_candidates]
    reranked.sort(key=lambda item: item["rerank_feedback_score"], reverse=True)
    for rank, item in enumerate(reranked, start=1):
        item["first_rerank_rank"] = rank
    return reranked


def build_refit_output_results(
    second_results: list[dict[str, Any]],
    feedback_candidates: list[dict[str, Any]],
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """整理第二次 dense retrieval 的最終輸出；第二次結果不再 rerank。"""
    feedback_by_doc_index = {
        int(item["doc_index"]): item
        for item in feedback_candidates
    }

    output: list[dict[str, Any]] = []
    for rank, item in enumerate(second_results, start=1):
        doc_index = int(item["doc_index"])
        document = documents[doc_index]
        feedback = feedback_by_doc_index.get(doc_index, {})
        output.append(
            {
                "rank": rank,
                "doc_index": doc_index,
                "doc_id": document["ID"],
                "question": document["Question"],
                "answer": document["Answer"],
                "first_dense_score": feedback.get("first_dense_score"),
                "first_dense_rank": feedback.get("first_dense_rank"),
                "rerank_feedback_score": feedback.get("rerank_feedback_score"),
                "first_rerank_rank": feedback.get("first_rerank_rank"),
                "refit_dense_score": item.get("refit_dense_score"),
                "refit_rank": rank,
            }
        )
    return output
