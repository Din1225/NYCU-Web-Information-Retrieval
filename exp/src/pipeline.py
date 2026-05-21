from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.data_io import load_documents, load_queries, save_json
from src.dense_retriever import DenseRetriever
from src.model_utils import DEFAULT_RETRIEVAL_INSTRUCTION
from src.reranker import CrossEncoderReranker
from src.text_processing import question_text


@dataclass(frozen=True)
class DenseRerankConfig:
    data_path: Path
    query_path: Path
    output_path: Path
    dense_model_name_or_path: str
    reranker_model_name_or_path: str
    cache_dir: Path
    dense_backend: str = "auto"
    reranker_backend: str = "auto"
    dense_top_k: int = 100
    rerank_top_k: int = 30
    dense_batch_size: int = 32
    reranker_batch_size: int = 16
    dense_max_length: int = 512
    reranker_max_length: int = 512
    use_4bit: bool = True
    compute_dtype: str = "float16"
    bge_use_fp16: bool = False
    use_cache: bool = True
    limit_docs: int | None = None
    limit_queries: int | None = None
    document_text_mode: str = "question"
    retrieval_instruction: str | None = DEFAULT_RETRIEVAL_INSTRUCTION


def run_dense_rerank_experiment(config: DenseRerankConfig) -> list[dict[str, Any]]:
    documents = load_documents(config.data_path, limit=config.limit_docs)
    queries = load_queries(config.query_path, limit=config.limit_queries)

    print(f"Loaded {len(documents)} documents from {config.data_path}", flush=True)
    print(f"Loaded {len(queries)} queries from {config.query_path}", flush=True)

    dense = DenseRetriever(
        documents=documents,
        model_name_or_path=config.dense_model_name_or_path,
        cache_dir=config.cache_dir / "dense",
        backend=config.dense_backend,
        batch_size=config.dense_batch_size,
        max_length=config.dense_max_length,
        use_4bit=config.use_4bit,
        compute_dtype=config.compute_dtype,
        bge_use_fp16=config.bge_use_fp16,
        use_cache=config.use_cache,
        document_text_mode=config.document_text_mode,
        query_instruction=config.retrieval_instruction,
    )
    print("Dense retriever 初始化已完成。", flush=True)

    query_items: list[dict[str, Any]] = []
    for query in tqdm(queries, desc="Dense retrieval"):
        query_text = question_text(query)
        dense_results = dense.retrieve(query_text, top_k=config.dense_top_k, score_key="dense_score")
        query_items.append(
            {
                "query_id": query["ID"],
                "query": query_text,
                "dense_results": build_dense_output_results(dense_results, documents),
            }
        )
        print(f"{query['ID']} dense retrieval 已完成，共 {len(dense_results)} 筆結果。", flush=True)

    dense.release_resources(clear_tokenizer=True)
    print("Dense retriever 模型已釋放。", flush=True)

    reranker = CrossEncoderReranker(
        model_name_or_path=config.reranker_model_name_or_path,
        backend=config.reranker_backend,
        batch_size=config.reranker_batch_size,
        max_length=config.reranker_max_length,
        use_4bit=config.use_4bit,
        compute_dtype=config.compute_dtype,
        bge_use_fp16=config.bge_use_fp16,
        instruction=config.retrieval_instruction,
        document_text_mode=config.document_text_mode,
    )
    print("Reranker 初始化已完成。", flush=True)

    all_results: list[dict[str, Any]] = []
    for item in tqdm(query_items, desc="Dense rerank"):
        dense_candidates = [
            {
                "doc_index": int(result["doc_index"]),
                "doc_id": result["doc_id"],
                "rank": int(result["rank"]),
                "dense_score": float(result["dense_score"]),
                "source": "dense",
            }
            for result in item["dense_results"]
        ]
        reranked = reranker.rerank(
            query=item["query"],
            candidates=dense_candidates,
            documents=documents,
            top_k=config.rerank_top_k,
        )
        rerank_results = build_rerank_output_results(reranked, documents)
        print(f"{item['query_id']} rerank 已完成，共 {len(rerank_results)} 筆結果。", flush=True)

        all_results.append(
            {
                "query_id": item["query_id"],
                "query": item["query"],
                "dense_top_k": len(item["dense_results"]),
                "rerank_top_k": len(rerank_results),
                "dense_backend": dense.backend,
                "reranker_backend": reranker.backend,
                "dense_model_name_or_path": config.dense_model_name_or_path,
                "reranker_model_name_or_path": config.reranker_model_name_or_path,
                "document_text_mode": config.document_text_mode,
                "retrieval_instruction": config.retrieval_instruction,
                "dense_results": item["dense_results"],
                "rerank_results": rerank_results,
            }
        )

    reranker.release_resources(clear_tokenizer=True)
    print("Reranker 模型已釋放。", flush=True)

    save_json(all_results, config.output_path)
    print(f"Saved results to {config.output_path}", flush=True)
    return all_results


def build_dense_output_results(
    dense_results: list[dict[str, Any]],
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in dense_results:
        document = documents[int(item["doc_index"])]
        output.append(
            {
                "rank": int(item["rank"]),
                "doc_index": int(item["doc_index"]),
                "doc_id": document["ID"],
                "question": document["Question"],
                "answer": document["Answer"],
                "dense_score": float(item["dense_score"]),
            }
        )
    return output


def build_rerank_output_results(
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
                "dense_rank": int(item["rank"]),
                "dense_score": float(item["dense_score"]),
                "rerank_score": float(item["rerank_score"]),
            }
        )
    return output

