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
        print(f"{query_id} Reranker feedback 分數計算已完成。", flush=True)

        feedback_doc_indices = [int(item["doc_index"]) for item in feedback_candidates]
        feedback_embeddings = dense.embeddings_for_indices(feedback_doc_indices)
        updated_query_embedding = optimize_query_embedding(
            query_embedding=original_query_embedding,
            candidate_embeddings=feedback_embeddings,
            reranker_scores=reranker_scores,
            config=refit_config,
        )
        print(f"{query_id} ReFIT query 向量更新已完成。", flush=True)

        second_results = dense.retrieve_by_embedding(
            updated_query_embedding,
            top_k=config.final_top_k,
            score_key="refit_dense_score",
            source="refit_dense_second",
        )
        print(f"{query_id} 第二次 dense retrieval 已完成，共 {len(second_results)} 筆結果。", flush=True)

        results = build_output_results(second_results, feedback_candidates, documents)
        all_results.append(
            {
                "query_id": query_id,
                "query": query_text,
                "feedback_top_k": len(feedback_candidates),
                "final_top_k": len(results),
                "refit_updates": config.refit_updates,
                "refit_learning_rate": config.refit_learning_rate,
                "refit_temperature": config.refit_temperature,
                "refit_use_minmax": config.refit_use_minmax,
                "query_vector_shift_l2": float(np.linalg.norm(updated_query_embedding - original_query_embedding)),
                "results": results,
            }
        )
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


def build_output_results(
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
                "refit_dense_score": item.get("refit_dense_score"),
                "first_dense_rank": feedback.get("first_dense_rank"),
                "first_dense_score": feedback.get("first_dense_score"),
                "rerank_feedback_score": feedback.get("rerank_feedback_score"),
                "sources": ["refit_dense_second"],
            }
        )
    return output

