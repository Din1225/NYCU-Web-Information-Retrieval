from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.text_processing import normalize_text


@dataclass(frozen=True)
class GFFFusionConfig:
    """GFF reciprocal-rank weighting 參數。"""

    rrf_k: int = 60
    original_query_weight: float = 0.3


@dataclass(frozen=True)
class RankedList:
    label: str
    query_text: str
    scores: tuple[float, ...]
    ranked_doc_indices: tuple[int, ...]
    rank_by_doc_index: dict[int, int]


@dataclass(frozen=True)
class RankedListContribution:
    label: str
    query_text: str
    list_weight: float
    anchor_doc_rank: int
    anchor_doc_id: str


@dataclass(frozen=True)
class FusionResult:
    fused_scores: tuple[float, ...]
    ranked_doc_indices: tuple[int, ...]
    contributions: tuple[RankedListContribution, ...]
    anchor_doc_index: int


def build_ranked_list(
    label: str,
    query_text: str,
    candidates: list[dict[str, Any]],
    scores: list[float],
) -> RankedList:
    if len(candidates) != len(scores):
        raise ValueError("candidates and scores must have the same length")

    ranking_pairs = [
        (int(candidate["doc_index"]), float(score))
        for candidate, score in zip(candidates, scores, strict=True)
    ]
    ranking_pairs.sort(key=lambda item: item[1], reverse=True)
    ranked_doc_indices = tuple(doc_index for doc_index, _ in ranking_pairs)
    rank_by_doc_index = {
        doc_index: rank
        for rank, doc_index in enumerate(ranked_doc_indices, start=1)
    }
    return RankedList(
        label=label,
        query_text=normalize_text(query_text),
        scores=tuple(float(score) for score in scores),
        ranked_doc_indices=ranked_doc_indices,
        rank_by_doc_index=rank_by_doc_index,
    )


def fuse_ranked_lists(
    candidates: list[dict[str, Any]],
    original_ranked_list: RankedList,
    expansion_ranked_lists: list[RankedList],
    config: GFFFusionConfig,
) -> FusionResult:
    """用 reciprocal-rank weighting 融合 expanded queries 與 original query。"""
    if not candidates:
        return FusionResult(
            fused_scores=tuple(),
            ranked_doc_indices=tuple(),
            contributions=tuple(),
            anchor_doc_index=-1,
        )

    anchor_doc_index = int(original_ranked_list.ranked_doc_indices[0])
    rrf_k = max(1, int(config.rrf_k))
    fused_scores = [0.0 for _ in candidates]
    contributions: list[RankedListContribution] = []

    for ranked_list in expansion_ranked_lists:
        anchor_rank = int(ranked_list.rank_by_doc_index.get(anchor_doc_index, len(candidates) + 1))
        list_weight = 1.0 / float(rrf_k + anchor_rank)
        _accumulate_weighted_reciprocal_ranks(
            fused_scores=fused_scores,
            candidates=candidates,
            ranked_list=ranked_list,
            rrf_k=rrf_k,
            list_weight=list_weight,
        )
        contributions.append(
            RankedListContribution(
                label=ranked_list.label,
                query_text=ranked_list.query_text,
                list_weight=float(list_weight),
                anchor_doc_rank=anchor_rank,
                anchor_doc_id=str(_doc_id_for_index(candidates, anchor_doc_index)),
            )
        )

    original_weight = max(float(config.original_query_weight), 0.0)
    _accumulate_weighted_reciprocal_ranks(
        fused_scores=fused_scores,
        candidates=candidates,
        ranked_list=original_ranked_list,
        rrf_k=rrf_k,
        list_weight=original_weight,
    )
    contributions.append(
        RankedListContribution(
            label=original_ranked_list.label,
            query_text=original_ranked_list.query_text,
            list_weight=original_weight,
            anchor_doc_rank=1,
            anchor_doc_id=str(_doc_id_for_index(candidates, anchor_doc_index)),
        )
    )

    ranking_pairs = [
        (int(candidate["doc_index"]), float(fused_scores[position]))
        for position, candidate in enumerate(candidates)
    ]
    ranking_pairs.sort(key=lambda item: item[1], reverse=True)
    ranked_doc_indices = tuple(doc_index for doc_index, _ in ranking_pairs)
    return FusionResult(
        fused_scores=tuple(float(score) for score in fused_scores),
        ranked_doc_indices=ranked_doc_indices,
        contributions=tuple(contributions),
        anchor_doc_index=anchor_doc_index,
    )


def fusion_result_to_dict(result: FusionResult) -> dict[str, Any]:
    return {
        "anchor_doc_index": result.anchor_doc_index,
        "ranked_doc_indices": list(result.ranked_doc_indices),
        "contributions": [
            {
                "label": item.label,
                "query_text": item.query_text,
                "list_weight": item.list_weight,
                "anchor_doc_rank": item.anchor_doc_rank,
                "anchor_doc_id": item.anchor_doc_id,
            }
            for item in result.contributions
        ],
    }


def _accumulate_weighted_reciprocal_ranks(
    fused_scores: list[float],
    candidates: list[dict[str, Any]],
    ranked_list: RankedList,
    rrf_k: int,
    list_weight: float,
) -> None:
    if list_weight <= 0.0:
        return
    for index, candidate in enumerate(candidates):
        doc_index = int(candidate["doc_index"])
        rank = ranked_list.rank_by_doc_index.get(doc_index)
        if rank is None:
            continue
        fused_scores[index] += list_weight * (1.0 / float(rrf_k + rank))


def _doc_id_for_index(candidates: list[dict[str, Any]], target_doc_index: int) -> Any:
    for candidate in candidates:
        if int(candidate["doc_index"]) == target_doc_index:
            return candidate.get("doc_id")
    return None
