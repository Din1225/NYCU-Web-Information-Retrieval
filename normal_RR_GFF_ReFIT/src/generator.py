from __future__ import annotations

from dataclasses import dataclass
import gc
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.qwen_utils import (
    build_model_load_kwargs,
    canonical_model_reference,
    ensure_cuda_for_4bit,
    resolve_pretrained_source,
)


@dataclass(frozen=True)
class GenerationConfig:
    """文字生成模型的推論參數。"""

    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True


class InstructionGenerationModel:
    """用於 Q2D2K 的 instruction-following 生成模型。"""

    def __init__(
        self,
        model_name_or_path: str | Path,
        model_cache_dir: str | Path | None = None,
        batch_size: int = 1,
        use_4bit: bool = True,
        compute_dtype: str = "float16",
        disable_thinking: bool = True,
    ) -> None:
        self.model_name_or_path = canonical_model_reference(model_name_or_path)
        self.model_cache_dir = Path(model_cache_dir) if model_cache_dir is not None else None
        self.batch_size = max(1, int(batch_size))
        self.use_4bit = use_4bit
        self.compute_dtype = compute_dtype
        self.disable_thinking = disable_thinking
        self._tokenizer = None
        self._model = None
        self._pretrained_source = None

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
        """釋放 generator 模型與 tokenizer，避免與 reranker/dense 同時占用顯存。"""
        self._model = None
        if clear_tokenizer:
            self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(self, prompts: list[str], config: GenerationConfig) -> list[str]:
        """對多個 prompts 進行批次生成。"""
        if not prompts:
            return []

        outputs: list[str] = []
        for start in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[start : start + self.batch_size]
            outputs.extend(self._generate_batch(batch_prompts, config))
        return outputs

    def _generate_batch(self, prompts: list[str], config: GenerationConfig) -> list[str]:
        rendered_prompts = [self._render_chat_prompt(prompt) for prompt in prompts]
        inputs = self.tokenizer(
            rendered_prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generation_kwargs = {
            "max_new_tokens": max(1, int(config.max_new_tokens)),
            "temperature": max(float(config.temperature), 1e-5),
            "top_p": min(max(float(config.top_p), 1e-5), 1.0),
            "do_sample": bool(config.do_sample),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if not config.do_sample:
            generation_kwargs["temperature"] = 1.0
            generation_kwargs["top_p"] = 1.0

        with torch.no_grad():
            generated = self.model.generate(**inputs, **generation_kwargs)

        input_length = int(inputs["input_ids"].shape[1])
        texts: list[str] = []
        for sequence in generated:
            new_tokens = sequence[input_length:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            texts.append(text.strip())
        return texts

    def _render_chat_prompt(self, prompt: str) -> str:
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if self.disable_thinking:
                try:
                    return self.tokenizer.apply_chat_template(
                        messages,
                        enable_thinking=False,
                        **template_kwargs,
                    )
                except TypeError:
                    # 舊版 tokenizer 可能不接受 enable_thinking，退回預設 chat template。
                    pass
            return self.tokenizer.apply_chat_template(
                messages,
                **template_kwargs,
            )
        return prompt
