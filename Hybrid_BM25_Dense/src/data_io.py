# 主要用途：讀取資料集、讀取 query 檔案，以及將檢索結果輸出成 JSON。

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> Any:
    """讀取 UTF-8 JSON 檔案。"""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: Any, path: str | Path) -> None:
    """將 Python 物件輸出成格式化 UTF-8 JSON。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def load_documents(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    """讀取歷史問答資料池，並檢查檢索流程需要的欄位。"""
    raw_documents = load_json(path)
    if not isinstance(raw_documents, list):
        raise ValueError(f"Document file must contain a JSON list: {path}")

    documents: list[dict[str, Any]] = []
    for index, item in enumerate(raw_documents):
        if not isinstance(item, dict):
            raise ValueError(f"Document at index {index} is not a JSON object")
        if "ID" not in item or "Question" not in item or "Answer" not in item:
            raise ValueError(f"Document at index {index} must contain ID, Question, Answer")

        documents.append(
            {
                "doc_index": index,
                "ID": str(item["ID"]),
                "Question": str(item["Question"]),
                "Answer": str(item["Answer"]),
            }
        )
        if limit is not None and len(documents) >= limit:
            break

    return documents


def load_queries(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    """讀取 query 檔案，並檢查 ID 與 Question 欄位。"""
    raw_queries = load_json(path)
    if not isinstance(raw_queries, list):
        raise ValueError(f"Query file must contain a JSON list: {path}")

    queries: list[dict[str, Any]] = []
    for index, item in enumerate(raw_queries):
        if not isinstance(item, dict):
            raise ValueError(f"Query at index {index} is not a JSON object")
        if "ID" not in item or "Question" not in item:
            raise ValueError(f"Query at index {index} must contain ID and Question")

        queries.append(
            {
                "query_index": index,
                "ID": str(item["ID"]),
                "Question": str(item["Question"]),
            }
        )
        if limit is not None and len(queries) >= limit:
            break

    return queries


def load_subqueries_map(path: str | Path) -> dict[str, list[str]]:
    """讀取 sub-query JSON，回傳 query_id -> sub_queries 對照表。"""
    raw_payload = load_json(path)
    if not isinstance(raw_payload, list):
        raise ValueError(f"Sub-query file must contain a JSON list: {path}")

    subqueries_map: dict[str, list[str]] = {}
    for index, item in enumerate(raw_payload):
        if not isinstance(item, dict):
            raise ValueError(f"Sub-query item at index {index} is not a JSON object")

        query_id = item.get("query_id", item.get("ID"))
        if query_id is None:
            raise ValueError(f"Sub-query item at index {index} must contain query_id")

        sub_queries = item.get("sub_queries", item.get("sub_question"))
        if not isinstance(sub_queries, list):
            raise ValueError(
                f"Sub-query item at index {index} must contain sub_queries or sub_question as a JSON list"
            )

        normalized_sub_queries = [str(sub_query).strip() for sub_query in sub_queries if str(sub_query).strip()]
        subqueries_map[str(query_id)] = normalized_sub_queries

    return subqueries_map
