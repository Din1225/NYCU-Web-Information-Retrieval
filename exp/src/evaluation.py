from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import math

from src.data_io import load_json, save_json


@dataclass(frozen=True)
class MaskedEvaluationConfig:
    annotation_path: Path
    result_path: Path
    result_field: str = "rerank_results"
    binary_relevance_threshold: int = 2
    top_ks: tuple[int, ...] = (10, 30)


def evaluate_result_file(config: MaskedEvaluationConfig) -> dict[str, Any]:
    raw_annotations = load_json(config.annotation_path)
    raw_results = load_json(config.result_path)

    annotation_index = build_annotation_index(raw_annotations)
    if not isinstance(raw_results, list):
        raise ValueError(f"Result file must contain a JSON list: {config.result_path}")

    per_query: list[dict[str, Any]] = []
    for query_result in raw_results:
        query_id = str(query_result["query_id"])
        if query_id not in annotation_index:
            raise ValueError(f"Query {query_id} not found in annotations.")
        if config.result_field not in query_result:
            raise ValueError(f"Result field {config.result_field} not found for query {query_id}.")

        labels_by_doc = annotation_index[query_id]["labels_by_doc"]
        raw_ranked_docs = query_result[config.result_field]
        masked_docs = compress_labeled_results(raw_ranked_docs, labels_by_doc)
        binary_labels = [
            1 if int(item["relevance"]) >= config.binary_relevance_threshold else 0
            for item in masked_docs
        ]
        graded_labels = [int(item["relevance"]) for item in masked_docs]
        total_relevant = sum(
            1
            for relevance in labels_by_doc.values()
            if int(relevance) >= config.binary_relevance_threshold
        )

        query_metrics: dict[str, Any] = {
            "query_id": query_id,
            "annotated_doc_count": len(labels_by_doc),
            "annotated_relevant_doc_count": total_relevant,
            "retrieved_doc_count": len(raw_ranked_docs),
            "scored_doc_count": len(masked_docs),
            "first_scored_doc_original_rank": masked_docs[0]["original_rank"] if masked_docs else None,
            "MRR": reciprocal_rank(binary_labels),
        }
        for top_k in config.top_ks:
            query_metrics[f"MAP@{top_k}"] = average_precision_at_k(binary_labels, total_relevant, top_k)
            query_metrics[f"NDCG@{top_k}"] = ndcg_at_k(graded_labels, top_k)
            query_metrics[f"scored_hits@{top_k}"] = int(sum(binary_labels[:top_k]))
            query_metrics[f"scored_docs@{top_k}"] = int(min(len(masked_docs), top_k))
        per_query.append(query_metrics)

    aggregate = aggregate_query_metrics(per_query, config.top_ks)
    return {
        "annotation_path": str(config.annotation_path),
        "result_path": str(config.result_path),
        "result_field": config.result_field,
        "binary_relevance_threshold": config.binary_relevance_threshold,
        "filter_rule": "remove_unlabeled_docs_then_apply_metrics",
        "query_count": len(per_query),
        "per_query": per_query,
        "aggregate": aggregate,
    }


def evaluate_multiple_result_fields(
    annotation_path: Path,
    result_path: Path,
    result_fields: list[str],
    binary_relevance_threshold: int = 2,
    top_ks: tuple[int, ...] = (10, 30),
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for result_field in result_fields:
        summaries[result_field] = evaluate_result_file(
            MaskedEvaluationConfig(
                annotation_path=annotation_path,
                result_path=result_path,
                result_field=result_field,
                binary_relevance_threshold=binary_relevance_threshold,
                top_ks=top_ks,
            )
        )
    return summaries


def save_evaluation(data: Any, path: str | Path) -> None:
    save_json(data, path)


def build_annotation_index(raw_annotations: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw_annotations, list):
        raise ValueError("Annotation file must contain a JSON list.")

    annotation_index: dict[str, dict[str, Any]] = {}
    for item in raw_annotations:
        query_id = str(item["id"])
        answers = item["answers"]
        labels_by_doc = {
            str(answer["id"]): int(answer["final_answer"])
            for answer in answers
        }
        annotation_index[query_id] = {
            "question": item.get("question"),
            "labels_by_doc": labels_by_doc,
        }
    return annotation_index


def compress_labeled_results(
    ranked_docs: list[dict[str, Any]],
    labels_by_doc: dict[str, int],
) -> list[dict[str, Any]]:
    compressed: list[dict[str, Any]] = []
    for original_rank, item in enumerate(ranked_docs, start=1):
        doc_id = str(item["doc_id"])
        if doc_id not in labels_by_doc:
            continue
        compressed.append(
            {
                "doc_id": doc_id,
                "relevance": int(labels_by_doc[doc_id]),
                "original_rank": original_rank,
                "scored_rank": len(compressed) + 1,
            }
        )
    return compressed


def average_precision_at_k(binary_labels: list[int], total_relevant: int, top_k: int) -> float:
    if total_relevant <= 0 or top_k <= 0:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for rank, label in enumerate(binary_labels[:top_k], start=1):
        if label <= 0:
            continue
        hits += 1
        precision_sum += hits / rank

    denominator = min(total_relevant, top_k)
    if denominator <= 0:
        return 0.0
    return precision_sum / denominator


def reciprocal_rank(binary_labels: list[int]) -> float:
    for rank, label in enumerate(binary_labels, start=1):
        if label > 0:
            return 1.0 / rank
    return 0.0


def dcg_at_k(graded_labels: list[int], top_k: int) -> float:
    score = 0.0
    for rank, relevance in enumerate(graded_labels[:top_k], start=1):
        gain = (2 ** int(relevance)) - 1
        score += gain / math.log2(rank + 1)
    return score


def ndcg_at_k(graded_labels: list[int], top_k: int) -> float:
    if top_k <= 0:
        return 0.0
    ideal_labels = sorted(graded_labels, reverse=True)
    ideal_dcg = dcg_at_k(ideal_labels, top_k)
    if ideal_dcg == 0.0:
        return 0.0
    return dcg_at_k(graded_labels, top_k) / ideal_dcg


def aggregate_query_metrics(per_query: list[dict[str, Any]], top_ks: tuple[int, ...]) -> dict[str, float]:
    if not per_query:
        aggregate = {"MRR": 0.0}
        for top_k in top_ks:
            aggregate[f"MAP@{top_k}"] = 0.0
            aggregate[f"NDCG@{top_k}"] = 0.0
        return aggregate

    metric_names = ["MRR"]
    for top_k in top_ks:
        metric_names.append(f"MAP@{top_k}")
        metric_names.append(f"NDCG@{top_k}")

    aggregate: dict[str, float] = {}
    for metric_name in metric_names:
        aggregate[metric_name] = sum(float(query_metrics[metric_name]) for query_metrics in per_query) / len(per_query)
    return aggregate
