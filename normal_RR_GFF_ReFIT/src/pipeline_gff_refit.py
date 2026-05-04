from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from src.data_io import load_documents, load_queries, save_json
from src.dense_retriever import DenseRetriever
from src.generator import InstructionGenerationModel
from src.gff_fusion import GFFFusionConfig, build_ranked_list, fuse_ranked_lists, fusion_result_to_dict
from src.qwen_utils import DEFAULT_RETRIEVAL_INSTRUCTION
from src.query_expansion import (
    GFFKeywordConfig,
    build_expanded_queries,
    generate_keywords_with_q2d2k,
    q2d2k_result_to_dict,
    select_keywords_by_frequency,
    unique_candidate_keywords,
)
from src.refit import ReFITConfig, optimize_query_embedding
from src.reranker import CrossEncoderReranker
from src.text_processing import question_text


@dataclass(frozen=True)
class GFFReFITConfig:
    data_path: Path
    query_path: Path
    output_path: Path
    dense_model_name_or_path: str
    reranker_model_name_or_path: str
    generator_model_name_or_path: str
    cache_dir: Path
    model_cache_dir: Path | None = None
    feedback_top_k: int = 100
    final_top_k: int = 30
    refit_updates: int = 100
    refit_learning_rate: float = 0.005
    refit_temperature: float = 2.0
    refit_use_minmax: bool = True
    dense_batch_size: int = 32
    reranker_batch_size: int = 16
    generator_batch_size: int = 1
    dense_max_length: int = 512
    reranker_max_length: int = 512
    use_4bit: bool = True
    compute_dtype: str = "float16"
    use_cache: bool = True
    limit_docs: int | None = None
    limit_queries: int | None = None
    print_query_vectors: bool = False
    query_vector_preview_dims: int = 10
    retrieval_instruction: str | None = DEFAULT_RETRIEVAL_INSTRUCTION
    gff_rounds: int = 3
    gff_passages_per_round: int = 2
    gff_keywords_per_passage: int = 15
    gff_top_keywords: int = 3
    gff_passage_max_new_tokens: int = 256
    gff_passage_temperature: float = 0.8
    gff_passage_top_p: float = 0.95
    gff_keyword_max_new_tokens: int = 64
    gff_keyword_temperature: float = 0.7
    gff_keyword_top_p: float = 0.9
    gff_rrf_k: int = 60
    gff_original_query_weight: float = 0.3


@dataclass
class QueryWorkItem:
    query_id: str
    query_text: str
    original_query_embedding: Any | None = None
    first_results: list[dict[str, Any]] | None = None
    feedback_doc_indices: list[int] | None = None
    expansion_result: Any | None = None
    expanded_queries: list[dict[str, str]] | None = None
    first_stage: dict[str, Any] | None = None
    query_embedding_debug: dict[str, Any] | None = None
    second_stage_pool: list[dict[str, Any]] | None = None
    refit_results: list[dict[str, Any]] | None = None


def run_gff_refit_retrieval(config: GFFReFITConfig) -> list[dict[str, Any]]:
    """執行 GFF(Q2D2K + self-consistency + reciprocal-rank fusion) + ReFIT。

    GFF 只用在第一次 feedback 階段；最終結果直接採用第二次 dense retrieval 排名。
    """
    documents = load_documents(config.data_path, limit=config.limit_docs)
    queries = load_queries(config.query_path, limit=config.limit_queries)

    print(f"Loaded {len(documents)} documents from {config.data_path}")
    print(f"Loaded {len(queries)} queries from {config.query_path}")
    print("資料讀取已完成。", flush=True)
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
    print("Dense retriever 初始化已完成。", flush=True)
    _log_step_result(
        step_no=2,
        title="建立或讀取 Dense embeddings",
        payload={
            "embedding_dim": dense.embedding_dim,
            "num_document_embeddings": int(dense.embeddings.shape[0]),
            "cache_dir": str(config.cache_dir / "dense"),
            "model_cache_dir": str(model_cache_dir / "dense"),
        },
    )

    keyword_config = GFFKeywordConfig(
        rounds=config.gff_rounds,
        passages_per_round=config.gff_passages_per_round,
        keywords_per_passage=config.gff_keywords_per_passage,
        top_keywords=config.gff_top_keywords,
        passage_generation=_build_generation_config(
            max_new_tokens=config.gff_passage_max_new_tokens,
            temperature=config.gff_passage_temperature,
            top_p=config.gff_passage_top_p,
            do_sample=True,
        ),
        keyword_generation=_build_generation_config(
            max_new_tokens=config.gff_keyword_max_new_tokens,
            temperature=config.gff_keyword_temperature,
            top_p=config.gff_keyword_top_p,
            do_sample=False,
        ),
    )
    fusion_config = GFFFusionConfig(
        rrf_k=config.gff_rrf_k,
        original_query_weight=config.gff_original_query_weight,
    )
    refit_config = ReFITConfig(
        updates=config.refit_updates,
        learning_rate=config.refit_learning_rate,
        temperature=config.refit_temperature,
        use_minmax=config.refit_use_minmax,
    )

    query_items = [
        QueryWorkItem(
            query_id=query["ID"],
            query_text=question_text(query),
        )
        for query in queries
    ]

    for item in tqdm(query_items, desc="Dense first-stage queries"):
        item.original_query_embedding = dense.encode_query(item.query_text)
        _log_step_result(
            step_no=3,
            title=f"{item.query_id} Query 向量化",
            payload={
                "query_id": item.query_id,
                "query": item.query_text,
                "embedding_dim": int(item.original_query_embedding.shape[0]),
                "query_vector_preview": _vector_preview(item.original_query_embedding),
            },
        )
        item.first_results = dense.retrieve_by_embedding(
            item.original_query_embedding,
            top_k=config.feedback_top_k,
            score_key="first_dense_score",
            source="dense_first",
        )
        print(f"{item.query_id} 第一次 dense retrieval 已完成，共 {len(item.first_results)} 筆候選。", flush=True)
        _log_step_result(
            step_no=4,
            title=f"{item.query_id} 第一次 Dense Retrieval",
            payload={
                "top_k": len(item.first_results),
                "top_candidates": _candidate_preview(item.first_results, score_key="first_dense_score"),
            },
        )
        item.feedback_doc_indices = [int(candidate["doc_index"]) for candidate in item.first_results]
        _log_step_result(
            step_no=5,
            title=f"{item.query_id} Dense 階段完成候選保留",
            payload={
                "retained_candidate_embeddings_shape": [len(item.feedback_doc_indices), dense.embedding_dim],
                "retained_candidate_doc_indices_preview": item.feedback_doc_indices[:5],
                "dense_model_release_deferred": True,
            },
        )
    dense.release_resources(clear_tokenizer=True)
    print("Dense retriever 模型已釋放。", flush=True)

    generator = InstructionGenerationModel(
        model_name_or_path=config.generator_model_name_or_path,
        model_cache_dir=model_cache_dir / "generator",
        batch_size=config.generator_batch_size,
        use_4bit=config.use_4bit,
        compute_dtype=config.compute_dtype,
    )
    print("Generator 初始化已完成。", flush=True)
    for item in tqdm(query_items, desc="Q2D2K keyword generation"):
        item.expansion_result = generate_keywords_with_q2d2k(
            query=item.query_text,
            generator=generator,
            config=keyword_config,
        )
        print(
            f"{item.query_id} Q2D2K + jieba keyword parsing 已完成，共產生 {len(item.expansion_result.candidate_keywords)} 個候選 keywords。",
            flush=True,
        )
        _log_step_result(
            step_no=6,
            title=f"{item.query_id} Q2D2K 與 jieba keyword parsing",
            payload={
                "selected_keywords": list(item.expansion_result.selected_keywords),
                "keyword_frequencies": item.expansion_result.keyword_frequencies,
                "candidate_keywords_preview": list(item.expansion_result.candidate_keywords[:10]),
                "generated_passages_preview": [
                    {
                        "round_index": passage_item.round_index,
                        "passage_index": passage_item.passage_index,
                        "passage_preview": passage_item.passage[:200],
                        "keywords": list(passage_item.keywords),
                    }
                    for passage_item in item.expansion_result.generated_passages[:2]
                ],
            },
        )
    generator.release_resources(clear_tokenizer=True)
    print("Generator 模型已釋放。", flush=True)
    dense.release_resources(clear_tokenizer=True)
    print("Dense retriever 模型已釋放（Q2D2K 完成）。", flush=True)

    for item in tqdm(query_items, desc="Keyword frequency selection"):
        item.expansion_result = select_keywords_by_frequency(
            result=item.expansion_result,
            top_k=config.gff_top_keywords,
        )
        item.expanded_queries = build_expanded_queries(item.query_text, item.expansion_result.selected_keywords)
        print(
            f"{item.query_id} Keyword frequency selection 已完成，選出 {len(item.expansion_result.selected_keywords)} 個 keywords。",
            flush=True,
        )
        _log_step_result(
            step_no=7,
            title=f"{item.query_id} Keyword Frequency Selection",
            payload={
                "selected_keywords": list(item.expansion_result.selected_keywords),
                "keyword_frequencies": item.expansion_result.keyword_frequencies,
                "candidate_keyword_count": len(item.expansion_result.candidate_keywords),
            },
        )

    reranker = CrossEncoderReranker(
        model_name_or_path=config.reranker_model_name_or_path,
        model_cache_dir=model_cache_dir / "reranker",
        batch_size=config.reranker_batch_size,
        max_length=config.reranker_max_length,
        use_4bit=config.use_4bit,
        compute_dtype=config.compute_dtype,
        instruction=config.retrieval_instruction,
    )
    print("Reranker 初始化已完成。", flush=True)
    for item in tqdm(query_items, desc="GFF reranking queries"):
        item.first_stage = _rerank_with_gff(
            query_text=item.query_text,
            candidates=item.first_results,
            documents=documents,
            reranker=reranker,
            expanded_queries=item.expanded_queries,
            fusion_config=fusion_config,
        )
        print(f"{item.query_id} 第一次 GFF fusion rerank 已完成。", flush=True)
        _log_step_result(
            step_no=8,
            title=f"{item.query_id} Original/Expanded Query Rerank + Fusion",
            payload={
                "expanded_queries": item.expanded_queries,
                "original_top_doc_indices": list(item.first_stage["original_ranked_list"].ranked_doc_indices[:5]),
                "fusion_top_doc_indices": list(item.first_stage["fusion_result"].ranked_doc_indices[:5]),
                "fusion_top_candidates": _candidate_preview(
                    item.first_stage["feedback_candidates"],
                    score_key="gff_fused_score",
                    rank_key="gff_fused_rank",
                ),
                "fusion_contributions": [
                    {
                        "label": contribution.label,
                        "query_text": contribution.query_text,
                        "list_weight": contribution.list_weight,
                        "anchor_doc_rank": contribution.anchor_doc_rank,
                    }
                    for contribution in item.first_stage["fusion_result"].contributions
                ],
            },
        )
        _log_step_result(
            step_no=9,
            title=f"{item.query_id} Reranker 階段完成",
            payload={
                "num_feedback_candidates": len(item.first_stage["feedback_candidates"]),
                "reranker_model_release_deferred": True,
            },
        )
    reranker.release_resources(clear_tokenizer=True)
    print("Reranker 模型已釋放。", flush=True)

    all_results: list[dict[str, Any]] = []
    for item in tqdm(query_items, desc="ReFIT final retrieval queries"):
        feedback_embeddings = dense.embeddings_for_indices(item.feedback_doc_indices)
        updated_query_embedding = optimize_query_embedding(
            query_embedding=item.original_query_embedding,
            candidate_embeddings=feedback_embeddings,
            reranker_scores=list(item.first_stage["fusion_result"].fused_scores),
            config=refit_config,
        )
        item.query_embedding_debug = build_query_embedding_debug(
            item.original_query_embedding,
            updated_query_embedding,
            preview_dims=config.query_vector_preview_dims,
        )
        if config.print_query_vectors:
            print_query_embedding_debug(item.query_id, item.query_embedding_debug)
        print(f"{item.query_id} ReFIT query 向量更新已完成。", flush=True)
        _log_step_result(
            step_no=10,
            title=f"{item.query_id} ReFIT Query Update",
            payload={
                "refit_updates": config.refit_updates,
                "query_vector_shift_l2": item.query_embedding_debug["shift_l2"],
                "original_query_vector_preview": item.query_embedding_debug["original_query_vector_preview"],
                "updated_query_vector_preview": item.query_embedding_debug["updated_query_vector_preview"],
                "teacher_score_preview": list(item.first_stage["fusion_result"].fused_scores[:10]),
            },
        )

        item.second_stage_pool = dense.retrieve_by_embedding(
            updated_query_embedding,
            top_k=config.final_top_k,
            score_key="refit_dense_score",
            source="refit_dense_second",
        )
        print(f"{item.query_id} 第二次 dense retrieval 已完成，共 {len(item.second_stage_pool)} 筆候選。", flush=True)
        _log_step_result(
            step_no=11,
            title=f"{item.query_id} 第二次 Dense Retrieval",
            payload={
                "top_k": len(item.second_stage_pool),
                "top_candidates": _candidate_preview(item.second_stage_pool, score_key="refit_dense_score"),
            },
        )

        item.refit_results = build_gff_refit_output_results(
            second_results=item.second_stage_pool,
            first_stage_details=item.first_stage,
            documents=documents,
        )
        query_output = {
            "query_id": item.query_id,
            "query": item.query_text,
            "feedback_top_k": len(item.first_results),
            "final_top_k": len(item.refit_results),
            "refit_updates": config.refit_updates,
            "refit_learning_rate": config.refit_learning_rate,
            "refit_temperature": config.refit_temperature,
            "refit_use_minmax": config.refit_use_minmax,
            "query_vector_shift_l2": item.query_embedding_debug["shift_l2"],
            "dense_model_name_or_path": config.dense_model_name_or_path,
            "reranker_model_name_or_path": config.reranker_model_name_or_path,
            "generator_model_name_or_path": config.generator_model_name_or_path,
            "use_4bit": config.use_4bit,
            "compute_dtype": config.compute_dtype,
            "retrieval_instruction": config.retrieval_instruction,
            "gff": {
                "rounds": config.gff_rounds,
                "passages_per_round": config.gff_passages_per_round,
                "keywords_per_passage": config.gff_keywords_per_passage,
                "top_keywords": config.gff_top_keywords,
                "rrf_k": config.gff_rrf_k,
                "original_query_weight": config.gff_original_query_weight,
                "query_expansion": q2d2k_result_to_dict(item.expansion_result),
                "first_stage_fusion": fusion_result_to_dict(item.first_stage["fusion_result"]),
            },
            "refit_results": item.refit_results,
        }
        if config.print_query_vectors:
            query_output["query_embedding_debug"] = item.query_embedding_debug
        all_results.append(query_output)
        print(f"{item.query_id} 結果整理已完成。", flush=True)
        _log_step_result(
            step_no=12,
            title=f"{item.query_id} 輸出單筆結果",
            payload={
                "query_id": item.query_id,
                "num_results": len(item.refit_results),
                "result_preview": [
                    {
                        "rank": result["rank"],
                        "doc_index": result["doc_index"],
                        "doc_id": result["doc_id"],
                        "refit_dense_score": result["refit_dense_score"],
                    }
                    for result in item.refit_results[:3]
                ],
            },
        )

    save_json(all_results, config.output_path)
    print(f"Saved results to {config.output_path}")
    print("結果輸出已完成。", flush=True)
    return all_results


def build_gff_refit_output_results(
    second_results: list[dict[str, Any]],
    first_stage_details: dict[str, Any],
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """整理最終 GFF + ReFIT 輸出。

    最終結果直接沿用第二次 dense retrieval 排名，不再做第二次 GFF rerank。
    """
    first_stage_lookup = {
        int(item["doc_index"]): item
        for item in first_stage_details["feedback_candidates"]
    }

    output: list[dict[str, Any]] = []
    for rank, item in enumerate(second_results, start=1):
        doc_index = int(item["doc_index"])
        document = documents[doc_index]
        first_stage_feedback = first_stage_lookup.get(doc_index, {})
        output.append(
            {
                "rank": rank,
                "doc_index": doc_index,
                "doc_id": document["ID"],
                "question": document["Question"],
                "answer": document["Answer"],
                "first_dense_score": first_stage_feedback.get("first_dense_score"),
                "first_dense_rank": first_stage_feedback.get("first_dense_rank"),
                "first_original_rerank_score": first_stage_feedback.get("original_rerank_score"),
                "first_original_rerank_rank": first_stage_feedback.get("original_rerank_rank"),
                "first_gff_fused_score": first_stage_feedback.get("gff_fused_score"),
                "first_gff_fused_rank": first_stage_feedback.get("gff_fused_rank"),
                "refit_dense_score": item.get("refit_dense_score"),
                "refit_dense_rank": item.get("rank"),
            }
        )
    return output


def _rerank_with_gff(
    query_text: str,
    candidates: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    reranker: CrossEncoderReranker,
    expanded_queries: list[dict[str, str]],
    fusion_config: GFFFusionConfig,
) -> dict[str, Any]:
    original_scores = reranker.score(query_text, candidates, documents)
    original_ranked_list = build_ranked_list(
        label="original_query",
        query_text=query_text,
        candidates=candidates,
        scores=original_scores,
    )

    expansion_ranked_lists = []
    expansion_debug: list[dict[str, Any]] = []
    for index, expansion in enumerate(expanded_queries, start=1):
        expanded_query = expansion["expanded_query"]
        scores = reranker.score(expanded_query, candidates, documents)
        ranked_list = build_ranked_list(
            label=f"keyword_{index}",
            query_text=expanded_query,
            candidates=candidates,
            scores=scores,
        )
        expansion_ranked_lists.append(ranked_list)
        expansion_debug.append(
            {
                "keyword": expansion["keyword"],
                "expanded_query": expanded_query,
                "scores": [float(score) for score in scores],
            }
        )

    fusion_result = fuse_ranked_lists(
        candidates=candidates,
        original_ranked_list=original_ranked_list,
        expansion_ranked_lists=expansion_ranked_lists,
        config=fusion_config,
    )
    feedback_candidates = _attach_gff_feedback_scores(
        candidates=candidates,
        original_scores=original_scores,
        original_ranked_list=original_ranked_list,
        fusion_result=fusion_result,
    )
    return {
        "original_scores": [float(score) for score in original_scores],
        "original_ranked_list": original_ranked_list,
        "expansion_ranked_lists": expansion_ranked_lists,
        "expansion_debug": expansion_debug,
        "fusion_result": fusion_result,
        "feedback_candidates": feedback_candidates,
    }


def _attach_gff_feedback_scores(
    candidates: list[dict[str, Any]],
    original_scores: list[float],
    original_ranked_list,
    fusion_result,
) -> list[dict[str, Any]]:
    fused_rank_by_doc_index = {
        int(doc_index): rank
        for rank, doc_index in enumerate(fusion_result.ranked_doc_indices, start=1)
    }
    feedback_candidates: list[dict[str, Any]] = []
    for candidate, original_score, fused_score in zip(
        candidates,
        original_scores,
        fusion_result.fused_scores,
        strict=True,
    ):
        doc_index = int(candidate["doc_index"])
        item = dict(candidate)
        item["first_dense_rank"] = int(candidate["rank"])
        item["original_rerank_score"] = float(original_score)
        item["original_rerank_rank"] = int(original_ranked_list.rank_by_doc_index.get(doc_index, len(candidates) + 1))
        item["gff_fused_score"] = float(fused_score)
        item["gff_fused_rank"] = int(fused_rank_by_doc_index.get(doc_index, len(candidates) + 1))
        feedback_candidates.append(item)
    return feedback_candidates


def _build_generation_config(max_new_tokens: int, temperature: float, top_p: float, do_sample: bool):
    from src.generator import GenerationConfig

    return GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
    )


def _log_step_result(step_no: int, title: str, payload: dict[str, Any]) -> None:
    print(f"[Step {step_no}] {title}", flush=True)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def _candidate_preview(
    candidates: list[dict[str, Any]],
    score_key: str,
    rank_key: str = "rank",
    limit: int = 5,
) -> list[dict[str, Any]]:
    sorted_candidates = sorted(
        candidates,
        key=lambda item: (item.get(rank_key) is None, item.get(rank_key, 10**9)),
    )
    preview: list[dict[str, Any]] = []
    for item in sorted_candidates[:limit]:
        preview.append(
            {
                "doc_index": int(item["doc_index"]),
                "doc_id": item.get("doc_id"),
                rank_key: item.get(rank_key),
                score_key: item.get(score_key),
            }
        )
    return preview


def _vector_preview(vector, limit: int = 10) -> list[float]:
    return [float(value) for value in vector[:limit]]


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
