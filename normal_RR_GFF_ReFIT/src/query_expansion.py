from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
import re
from typing import Any

import jieba

from src.generator import GenerationConfig, InstructionGenerationModel
from src.text_processing import normalize_text


_KEYWORD_SPLIT_RE = re.compile(r"[\n,;|]+")
_KEYWORD_PREFIX_RE = re.compile(r"^(keywords?|key phrases?|terms?)\s*:\s*", flags=re.IGNORECASE)
_LEADING_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_CJK_PUNCT_OR_CONNECTOR_RE = re.compile(r"[，。；？！、]|以及|而且|並且|或者|或是|以及|以及|和|與|或|但")
_NON_WORD_TOKEN_RE = re.compile(r"^[\W_]+$", flags=re.UNICODE)
_ALNUM_TOKEN_RE = re.compile(r"[A-Za-z0-9]")
_META_TEXT_STRONG_MARKERS = (
    "thinking process",
    "analyze the request",
    "identify key information",
)
_META_TEXT_PROMPT_MARKERS = (
    "input:",
    "task:",
    "source:",
    "constraint",
    "<question>",
    "<passage>",
    "<keywords>",
)
_GENERATION_LABEL_PREFIXES = (
    "回答段落：",
    "回答：",
    "段落：",
    "關鍵詞：",
    "keywords:",
    "keyword:",
    "passage:",
)

_CHINESE_STOPWORDS = {
    "什麼",
    "是否",
    "需要",
    "可能",
    "進一步",
    "出現",
    "反覆",
    "長期",
    "目前",
    "相關",
    "有關",
    "這些",
    "這種",
    "這個",
    "那個",
    "問題",
    "原因",
    "結果",
    "正常",
    "以及",
    "或者",
    "或是",
    "如果",
    "因此",
    "可以",
    "應該",
    "怎麼",
    "如何",
    "哪裡",
    "什麼樣",
}
_MERGEABLE_SECOND_TOKENS = {
    "檢查",
    "治療",
    "就醫",
    "手術",
    "藥物",
    "藥",
    "疾病",
    "風險",
    "病變",
    "感染",
    "功能",
    "內科",
    "外科",
    "門診",
}


@dataclass(frozen=True)
class GFFKeywordConfig:
    """GFF Query Expansion 參數。"""

    rounds: int = 3
    passages_per_round: int = 2
    keywords_per_passage: int = 15
    top_keywords: int = 3
    passage_generation: GenerationConfig = field(
        default_factory=lambda: GenerationConfig(max_new_tokens=256, temperature=0.8, top_p=0.95)
    )
    keyword_generation: GenerationConfig = field(
        default_factory=lambda: GenerationConfig(max_new_tokens=64, temperature=0.7, top_p=0.9)
    )


@dataclass(frozen=True)
class GeneratedPassage:
    round_index: int
    passage_index: int
    passage: str
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class Q2D2KResult:
    query: str
    selected_keywords: tuple[str, ...]
    keyword_frequencies: dict[str, int]
    generated_passages: tuple[GeneratedPassage, ...]
    candidate_keywords: tuple[str, ...]


def generate_keywords_with_q2d2k(
    query: str,
    generator: InstructionGenerationModel,
    config: GFFKeywordConfig,
) -> Q2D2KResult:
    """依論文 GFF 的 Q2D2K + self-consistency 流程產生 keywords。"""
    normalized_query = normalize_text(query)
    generated_passages: list[GeneratedPassage] = []
    all_candidate_keywords: list[str] = []
    keyword_counter: Counter[str] = Counter()
    canonical_to_display: dict[str, str] = {}

    for round_index in range(config.rounds):
        passage_prompts = [
            _build_q2d_prompt(
                normalized_query,
                diversity_hint=_q2d_diversity_hint(
                    round_index=round_index + 1,
                    passage_index=passage_index + 1,
                ),
            )
            for passage_index in range(config.passages_per_round)
        ]
        passages = generator.generate(passage_prompts, config.passage_generation)

        for passage_index, passage in enumerate(passages):
            cleaned_passage = _sanitize_generated_text(passage)
            if _looks_like_meta_text(cleaned_passage):
                generated_passages.append(
                    GeneratedPassage(
                        round_index=round_index + 1,
                        passage_index=passage_index + 1,
                        passage=cleaned_passage,
                        keywords=tuple(),
                    )
                )
                continue
            keyword_prompt = _build_d2k_prompt(
                query=normalized_query,
                passage=cleaned_passage,
                keyword_count=config.keywords_per_passage,
            )
            keyword_text = generator.generate([keyword_prompt], config.keyword_generation)[0]
            keywords = _parse_keyword_text(keyword_text, limit=config.keywords_per_passage)
            deduplicated_keywords = _deduplicate_preserve_order(keywords)
            filtered_keywords = _filter_query_overlap_keywords(deduplicated_keywords, normalized_query)

            generated_passages.append(
                GeneratedPassage(
                    round_index=round_index + 1,
                    passage_index=passage_index + 1,
                    passage=cleaned_passage,
                    keywords=tuple(filtered_keywords),
                )
            )
            for keyword in filtered_keywords:
                canonical = _canonical_keyword(keyword)
                if not canonical:
                    continue
                canonical_to_display.setdefault(canonical, keyword)
                keyword_counter[canonical] += 1
                all_candidate_keywords.append(keyword)

    keyword_frequencies = {
        canonical_to_display[canonical]: int(frequency)
        for canonical, frequency in sorted(keyword_counter.items(), key=lambda item: (-item[1], item[0]))
    }

    return Q2D2KResult(
        query=normalized_query,
        selected_keywords=tuple(),
        keyword_frequencies=keyword_frequencies,
        generated_passages=tuple(generated_passages),
        candidate_keywords=tuple(all_candidate_keywords),
    )


def build_expanded_queries(query: str, keywords: list[str] | tuple[str, ...]) -> list[dict[str, str]]:
    """將每個 keyword 個別接回原 query。"""
    normalized_query = normalize_text(query)
    expanded_queries: list[dict[str, str]] = []
    for keyword in keywords:
        normalized_keyword = normalize_text(keyword)
        if not normalized_keyword or _is_invalid_keyword_candidate(normalized_keyword):
            continue
        expanded_queries.append(
            {
                "keyword": normalized_keyword,
                "expanded_query": f"{normalized_query} {normalized_keyword}",
            }
        )
    return expanded_queries


def q2d2k_result_to_dict(result: Q2D2KResult) -> dict[str, Any]:
    return {
        "query": result.query,
        "selected_keywords": list(result.selected_keywords),
        "keyword_frequencies": result.keyword_frequencies,
        "candidate_keywords": list(result.candidate_keywords),
        "generated_passages": [
            {
                "round_index": item.round_index,
                "passage_index": item.passage_index,
                "passage": item.passage,
                "keywords": list(item.keywords),
            }
            for item in result.generated_passages
        ],
    }


def unique_candidate_keywords(keywords: list[str] | tuple[str, ...]) -> list[str]:
    """保留候選 keywords 的首次出現順序，並以 canonical form 去重。"""
    return _deduplicate_preserve_order(list(keywords))


def select_keywords_by_frequency(
    result: Q2D2KResult,
    top_k: int,
) -> Q2D2KResult:
    """依 keyword frequency 選出 top keywords，並嚴格排除 query 已出現的相同字串。"""
    limit = max(0, int(top_k))
    sorted_items = sorted(
        result.keyword_frequencies.items(),
        key=lambda item: (-item[1], item[0]),
    )
    query_canonical = _canonical_keyword(result.query)

    selected_keywords = [
        keyword
        for keyword, _ in sorted_items
        if _canonical_keyword(keyword) not in query_canonical
    ][:limit]

    return replace(
        result,
        selected_keywords=tuple(selected_keywords),
    )


def _build_q2d_prompt(query: str, diversity_hint: str | None = None) -> str:
    lines = [
        "請根據問題生成一段簡短且具體的回答段落。",
        "請使用繁體中文作答。",
        "段落內容必須直接回答問題，並包含和問題主題最相關的處置、風險、就醫方式或其他關鍵資訊。",
        "請優先補充問題中未明說、但對就醫與處理有幫助的資訊，例如就醫科別、急診或送醫時機、安全處置、觀察與記錄重點、可能風險、後續治療或家庭支持。",
        "不要只重複問題中的症狀詞，段落中至少要包含一到兩項具體補充資訊。",
        "不要輸出推理過程、分析、標題、條列符號、前言、結語或其他額外格式。",
        "只輸出一段自然語言回答，不要重複題目。",
        "請與其他可能的回答保持合理差異，可從不同但相關的角度切入。",
    ]
    if diversity_hint:
        lines.append(f"本次請優先聚焦：{diversity_hint}")
    lines.append(f"問題：{query}")
    lines.append("回答段落：")
    return "\n".join(lines).strip()


def _build_d2k_prompt(query: str, passage: str, keyword_count: int) -> str:
    lines = [
        f"請根據問題與段落內容，從段落中抽取最多 {keyword_count} 個適合檢索的短關鍵詞或短片語。",
        "請優先使用繁體中文關鍵詞；英文專有名詞或縮寫可保留原文。",
        "關鍵詞必須直接來自段落內容，且要和問題主題高度相關。",
        "請優先抽取問題中未直接出現、但對檢索有幫助的補充資訊詞，例如就醫科別、急診處置、送醫時機、安全措施、自傷或傷人風險、病史紀錄、藥物治療、心理輔導、家庭支持。",
        "除非段落中沒有其他更合適的資訊，否則不要優先輸出只是重複問題症狀的詞。",
        "請盡量保留疾病名稱、檢查、治療、藥物、器官、部位、風險因子與具體處置等資訊詞，避免過於籠統的詞。",
        "不要輸出解釋、推理過程、標題、條列符號、句子或其他額外文字。",
        "只輸出一行，以半形逗號分隔的關鍵詞列表。",
    ]
    lines.append(f"問題：{query}")
    lines.append(f"段落：{passage}")
    lines.append("關鍵詞：")
    return "\n".join(lines).strip()


def _parse_keyword_text(text: str, limit: int) -> list[str]:
    cleaned_text = _sanitize_generated_text(text).replace("<KEYWORDS>:", "")
    cleaned_text = _KEYWORD_PREFIX_RE.sub("", cleaned_text)
    if not cleaned_text or _looks_like_meta_text(cleaned_text):
        return []

    segments = _KEYWORD_SPLIT_RE.split(cleaned_text)
    keywords: list[str] = []
    for segment in segments:
        candidate = _LEADING_BULLET_RE.sub("", segment).strip(" \"'[](){}")
        candidate = normalize_text(candidate)
        if not candidate:
            continue
        if candidate.lower().startswith("passage:") or _is_invalid_keyword_candidate(candidate):
            continue
        parsed_candidates = _expand_candidate_keywords(candidate)
        if not parsed_candidates:
            parsed_candidates = [candidate]
        for parsed_candidate in parsed_candidates:
            if _is_invalid_keyword_candidate(parsed_candidate):
                continue
            keywords.append(parsed_candidate)
            if len(keywords) >= limit:
                return keywords[:limit]
    return keywords


def _deduplicate_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        canonical = _canonical_keyword(item)
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        output.append(item)
    return output


def _filter_query_overlap_keywords(keywords: list[str], query: str) -> list[str]:
    """過濾和 query 中完全相同的 keyword，避免在 candidate 階段就累積重複症狀詞。"""
    query_canonical = _canonical_keyword(query)
    return [
        keyword
        for keyword in keywords
        if _canonical_keyword(keyword) not in query_canonical
    ]


def _canonical_keyword(keyword: str) -> str:
    return normalize_text(keyword).strip(".,!?;:").casefold()


def _sanitize_generated_text(text: str) -> str:
    cleaned = normalize_text(text)
    lowered = cleaned.casefold()
    for prefix in _GENERATION_LABEL_PREFIXES:
        if lowered.startswith(prefix.casefold()):
            cleaned = normalize_text(cleaned[len(prefix) :])
            lowered = cleaned.casefold()
    return cleaned


def _q2d_diversity_hint(round_index: int, passage_index: int) -> str:
    focus_options = [
        "先說明家屬當下應如何維持安全與穩定現場。",
        "先說明陪同就醫、急診處理與尋求專業協助的流程。",
        "先說明家屬與患者溝通時應避免的行為與合適的互動方式。",
        "先說明可記錄哪些症狀、發作時間與誘發因素供醫師判斷。",
        "先說明何時屬於高風險情況，需要立即送醫或求助。",
        "先說明後續治療配合、家庭支持與持續觀察的重點。",
    ]
    hint_index = ((round_index - 1) * 10 + (passage_index - 1)) % len(focus_options)
    return focus_options[hint_index]


def _looks_like_meta_text(text: str) -> bool:
    if not text:
        return False
    normalized = normalize_text(text)
    lowered = normalized.casefold()
    if any(marker in lowered for marker in _META_TEXT_STRONG_MARKERS):
        return True
    prompt_marker_hits = sum(marker in lowered for marker in _META_TEXT_PROMPT_MARKERS)
    if prompt_marker_hits >= 2:
        return True
    if normalized.count("*") >= 4 and prompt_marker_hits >= 1:
        return True
    return False


def _is_invalid_keyword_candidate(candidate: str) -> bool:
    normalized = normalize_text(candidate)
    if not normalized:
        return True
    lowered = normalized.casefold()
    if lowered.startswith(("passage:", "question:", "keyword:", "keywords:", "task:", "input:", "source:")):
        return True
    if normalized.startswith(("問題：", "段落：", "關鍵詞：", "回答段落：", "回答：")):
        return True
    if _looks_like_meta_text(normalized):
        return True
    if not _CJK_RE.search(normalized) and (len(normalized) >= 80 or normalized.count(" ") >= 6):
        return True
    return False


def _expand_candidate_keywords(candidate: str) -> list[str]:
    """把過長中文片語拆成較適合作為 query expansion 的 keywords。"""
    if not _should_segment_chinese_candidate(candidate):
        return [candidate]

    segmented_tokens = [normalize_text(token) for token in jieba.lcut(candidate, cut_all=False)]
    filtered_tokens = [token for token in segmented_tokens if _is_useful_keyword_token(token)]
    if not filtered_tokens:
        return [candidate]

    expanded_keywords: list[str] = []
    index = 0
    while index < len(filtered_tokens):
        current = filtered_tokens[index]
        if index + 1 < len(filtered_tokens):
            merged = _merge_chinese_tokens(current, filtered_tokens[index + 1])
            if merged is not None:
                expanded_keywords.append(merged)
                index += 2
                continue
        expanded_keywords.append(current)
        index += 1

    deduplicated = _deduplicate_preserve_order(expanded_keywords)
    return deduplicated or [candidate]


def _should_segment_chinese_candidate(candidate: str) -> bool:
    if not _CJK_RE.search(candidate):
        return False
    if len(candidate) >= 8:
        return True
    return bool(_CJK_PUNCT_OR_CONNECTOR_RE.search(candidate))


def _is_useful_keyword_token(token: str) -> bool:
    if not token:
        return False
    if _NON_WORD_TOKEN_RE.match(token):
        return False
    if _CJK_RE.search(token):
        if token in _CHINESE_STOPWORDS:
            return False
        return len(token) >= 2
    if not _ALNUM_TOKEN_RE.search(token):
        return False
    return len(token) >= 2


def _merge_chinese_tokens(first: str, second: str) -> str | None:
    if second not in _MERGEABLE_SECOND_TOKENS:
        return None
    merged = f"{first}{second}"
    if len(merged) > 8:
        return None
    return merged
