# 主要用途：實作 ReFIT 的 inference-time query embedding 更新。

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ReFITConfig:
    """ReFIT distillation hyperparameters."""

    updates: int = 100
    learning_rate: float = 0.005
    temperature: float = 2.0
    use_minmax: bool = True


def optimize_query_embedding(
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    reranker_scores: list[float] | np.ndarray,
    config: ReFITConfig,
) -> np.ndarray:
    """用 reranker relevance feedback 更新單一 query embedding。

    只更新 query embedding；retriever 文件 embedding 與模型參數都不會被更新。
    """
    if candidate_embeddings.shape[0] == 0 or config.updates <= 0:
        return _l2_normalize_numpy(np.asarray(query_embedding, dtype=np.float32))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    document_vectors = torch.as_tensor(candidate_embeddings, dtype=torch.float32, device=device)
    reranker_tensor = torch.as_tensor(reranker_scores, dtype=torch.float32, device=device)
    target_distribution = _score_distribution(
        reranker_tensor,
        temperature=config.temperature,
        use_minmax=config.use_minmax,
    ).detach()

    query_vector = torch.as_tensor(query_embedding, dtype=torch.float32, device=device).clone().detach()
    query_vector = F.normalize(query_vector, p=2, dim=0)
    query_vector.requires_grad_(True)

    optimizer = torch.optim.SGD([query_vector], lr=config.learning_rate)
    for _ in range(config.updates):
        optimizer.zero_grad(set_to_none=True)
        retriever_scores = document_vectors @ query_vector
        retriever_distribution = _score_distribution(
            retriever_scores,
            temperature=1.0,
            use_minmax=config.use_minmax,
        )
        loss = torch.sum(
            target_distribution
            * (
                torch.log(target_distribution.clamp_min(1e-12))
                - torch.log(retriever_distribution.clamp_min(1e-12))
            )
        )
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            query_vector.copy_(F.normalize(query_vector, p=2, dim=0))

    updated = query_vector.detach().cpu().numpy().astype(np.float32)
    return _l2_normalize_numpy(updated)


def _score_distribution(
    scores: torch.Tensor,
    temperature: float,
    use_minmax: bool,
) -> torch.Tensor:
    """將一組分數轉成 softmax 分布。"""
    logits = scores
    if use_minmax:
        logits = _minmax_normalize(logits)
    safe_temperature = max(float(temperature), 1e-6)
    return torch.softmax(logits / safe_temperature, dim=0)


def _minmax_normalize(scores: torch.Tensor) -> torch.Tensor:
    min_score = torch.min(scores)
    max_score = torch.max(scores)
    denominator = max_score - min_score
    if torch.abs(denominator).item() < 1e-12:
        return torch.zeros_like(scores)
    return (scores - min_score) / denominator


def _l2_normalize_numpy(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)
