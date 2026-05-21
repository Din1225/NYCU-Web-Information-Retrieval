from __future__ import annotations

import gc
import json
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.qwen_utils import (
    build_model_load_kwargs,
    canonical_model_reference,
    ensure_cuda_for_4bit,
    resolve_pretrained_source,
)
from src.text_processing import document_text, normalize_text

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_RISK_KEYS = (
    "topic_mismatch_risk",
    "missing_subquery_risk",
    "keyword_only_risk",
    "shallow_answer_risk",
)
_RISK_WEIGHTS = {
    "missing_subquery_risk": 0.35,
    "topic_mismatch_risk": 0.30,
    "keyword_only_risk": 0.20,
    "shallow_answer_risk": 0.15,
}


class QwenRiskDetector:
    """用 Qwen causal LM 估計候選文件的 over-ranking 風險。"""

    _SYSTEM_PROMPT = (
        "你是一個回覆風險偵測器。"
        "不要輸出思考過程、<think> 標記、分析、說明、markdown 或 code fence。"
        "只能輸出單行合法 JSON。"
        "四個欄位必須固定為 "
        "topic_mismatch_risk, missing_subquery_risk, keyword_only_risk, shallow_answer_risk。"
        "每個欄位都必須輸出 0 到 1 之間的小數。"
    )
    _SUCCESS_LOG_LIMIT = 3

    def __init__(
        self,
        model_name_or_path: str | Path,
        model_cache_dir: str | Path | None = None,
        batch_size: int = 4,
        max_length: int = 1024,
        max_new_tokens: int = 256,
        use_4bit: bool = True,
        compute_dtype: str = "float16",
    ) -> None:
        self.model_name_or_path = canonical_model_reference(model_name_or_path)
        self.model_cache_dir = Path(model_cache_dir) if model_cache_dir is not None else None
        self.batch_size = batch_size
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.use_4bit = use_4bit
        self.compute_dtype = compute_dtype
        self._tokenizer = None
        self._model = None
        self._pretrained_source = None
        self._success_log_count = 0
        self._bad_words_ids = None

    @property
    def pretrained_source(self) -> str:
        if self._pretrained_source is None:
            self._pretrained_source = resolve_pretrained_source(self.model_name_or_path, self.model_cache_dir)
        return self._pretrained_source

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.pretrained_source,
                padding_side="left",
                use_fast=False,
            )
            if self._tokenizer.pad_token is None and self._tokenizer.eos_token is not None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            ensure_cuda_for_4bit(self.use_4bit)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.pretrained_source,
                **build_model_load_kwargs(self.use_4bit, self.compute_dtype),
            ).eval()
        return self._model

    def release_resources(self, clear_tokenizer: bool = True) -> None:
        self._model = None
        if clear_tokenizer:
            self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def detect(
        self,
        query: str,
        sub_queries: list[str],
        candidates: list[dict[str, Any]],
        documents: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        prompts = [
            self._build_prompt(
                query=normalize_text(query),
                sub_queries=sub_queries,
                document=document_text(documents[int(candidate["doc_index"])]),
            )
            for candidate in candidates
        ]
        raw_outputs = self._generate_outputs(prompts)
        parsed_outputs: list[dict[str, Any]] = []
        for raw_output in raw_outputs:
            parsed_outputs.append(self._parse_risk_output(raw_output))
        return parsed_outputs

    def _build_prompt(self, query: str, sub_queries: list[str], document: str) -> str:
        if sub_queries:
            sub_query_lines = "\n".join(
                f"{index}. {normalize_text(sub_query)}"
                for index, sub_query in enumerate(sub_queries, start=1)
            )
        else:
            sub_query_lines = "1. 無提供子查詢"

        user_prompt = (
            f"Query：\n{query}\n\n"
            f"Sub-query：\n{sub_query_lines}\n\n"
            f"Answer：\n{normalize_text(document)}\n\n"
            "請評估四個風險分數，範圍都是 0 到 1：\n"
            "topic_mismatch_risk: 主題方向錯誤。\n"
            "missing_subquery_risk: 遺漏重要子問題。\n"
            "keyword_only_risk: 只對到關鍵字但沒有真正回答。\n"
            "shallow_answer_risk: 回答太通用、不夠具體。\n\n"
            "只輸出單行 JSON，不要輸出任何其他文字。\n"
            "JSON schema:\n"
            '{"topic_mismatch_risk": <float>, "missing_subquery_risk": <float>, "keyword_only_risk": <float>, "shallow_answer_risk": <float>}'
        )
        return self._format_chat_prompt(user_prompt)

    def _format_chat_prompt(self, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": self._SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            return prompt
        except Exception:
            return f"System: {self._SYSTEM_PROMPT}\n\nUser: {user_prompt}\n\nAssistant:"

    @torch.no_grad()
    def _generate_outputs(self, prompts: list[str]) -> list[str]:
        if not prompts:
            return []

        outputs_text: list[str] = []
        for start in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[start : start + self.batch_size]
            tokenized = self.tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            tokenized = tokenized.to(self.model.device)
            generated = self.model.generate(
                **tokenized,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.2,
                top_p=0.9,
                bad_words_ids=self._get_bad_words_ids(),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            prompt_length = tokenized["input_ids"].shape[1]
            generated_tokens = generated[:, prompt_length:]
            decoded_outputs = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            outputs_text.extend(decoded_outputs)
        return outputs_text

    def _parse_risk_output(self, raw_output: str) -> dict[str, Any]:
        json_payload = self._extract_json_block(raw_output)
        try:
            parsed = json.loads(json_payload)
            parse_ok = True
        except (TypeError, json.JSONDecodeError):
            parsed = {}
            parse_ok = False
            print("\n=== Risk Detector JSON Parse Failed ===", flush=True)
            print("Raw LLM output:", flush=True)
            print(raw_output.strip(), flush=True)
            print("Extracted JSON candidate:", flush=True)
            print(json_payload, flush=True)
        else:
            self._log_success_sample(raw_output=raw_output, parsed=parsed)

        normalized_risks: dict[str, float] = {}
        for risk_key in _RISK_KEYS:
            normalized_risks[risk_key] = _clamp_risk_value(parsed.get(risk_key, 0.0))

        risk = sum(normalized_risks[risk_key] * _RISK_WEIGHTS[risk_key] for risk_key in _RISK_KEYS)
        normalized_risks["risk"] = float(risk)
        normalized_risks["risk_parse_ok"] = parse_ok
        return normalized_risks

    @staticmethod
    def _extract_json_block(raw_output: str) -> str:
        stripped_output = _sanitize_generation_text(raw_output).strip()
        match = _JSON_BLOCK_RE.search(stripped_output)
        if match is None:
            if stripped_output.startswith('"topic_mismatch_risk"'):
                return "{\n  " + stripped_output
            if stripped_output.startswith("'topic_mismatch_risk'"):
                return '{\n  "topic_mismatch_risk"' + stripped_output[len("'topic_mismatch_risk'") :]
            return stripped_output
        return match.group(0).strip()

    def _log_success_sample(self, raw_output: str, parsed: dict[str, Any]) -> None:
        if self._success_log_count >= self._SUCCESS_LOG_LIMIT:
            return

        self._success_log_count += 1
        print("\n=== Risk Detector JSON Parse Success Sample ===", flush=True)
        print("Raw LLM output:", flush=True)
        print(raw_output.strip(), flush=True)
        print("Parsed JSON:", flush=True)
        print(json.dumps(parsed, ensure_ascii=False, indent=2), flush=True)

    def _get_bad_words_ids(self) -> list[list[int]] | None:
        if self._bad_words_ids is not None:
            return self._bad_words_ids

        blocked_phrases = ("<think>", "</think>", "```json", "```")
        bad_words_ids: list[list[int]] = []
        for phrase in blocked_phrases:
            token_ids = self.tokenizer.encode(phrase, add_special_tokens=False)
            if token_ids:
                bad_words_ids.append(token_ids)

        self._bad_words_ids = bad_words_ids or None
        return self._bad_words_ids


def _sanitize_generation_text(text: str) -> str:
    without_think = _THINK_BLOCK_RE.sub("", text)
    if "<think>" in without_think and "{" in without_think:
        without_think = without_think[without_think.find("{") :]
    return _strip_code_fence(without_think)


def _strip_code_fence(text: str) -> str:
    stripped_text = text.strip()
    if not stripped_text.startswith("```"):
        return text

    lines = stripped_text.splitlines()
    if len(lines) >= 3 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1])
    return text


def _clamp_risk_value(value: Any) -> float:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        numeric_value = 0.0
    return max(0.0, min(1.0, numeric_value))
