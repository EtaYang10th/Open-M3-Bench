from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoConfig


class BaseLocalDriver:
    def generate_once(self, messages: List[Dict[str, str]]) -> Tuple[str, str]:
        raise NotImplementedError


class QwenLocalDriver(BaseLocalDriver):
    def __init__(self, model_path: str, max_new_tokens: int = 2048) -> None:
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=getattr(torch, "bfloat16", torch.float16),
            trust_remote_code=True,
            device_map="auto",
        )

    def _build_inputs(self, messages: List[Dict[str, str]]):
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        return inputs

    def generate_once(self, messages: List[Dict[str, str]]) -> Tuple[str, str]:
        inputs = self._build_inputs(messages)
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        new_tokens = out[0][inputs["input_ids"].shape[-1]:]
        full = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        visible = full
        return visible, full


class InternVLLocalDriver(BaseLocalDriver):
    def __init__(self, model_path: str, max_new_tokens: int = 2048) -> None:
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

        # Use device_map="auto" by default; advanced split is in save/app_internvl.py
        self.model = AutoModel.from_pretrained(
            model_path,
            dtype=getattr(torch, "bfloat16", torch.float16),
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map="auto",
        ).eval()

        self._gen_kw = dict(max_new_tokens=self.max_new_tokens, do_sample=True, temperature=0.6)

    def generate_once(self, messages: List[Dict[str, str]]) -> Tuple[str, str]:
        # Flatten messages to a single string prompt; InternVL uses model.chat API normally
        question = "\n".join([m.get("content", "") for m in messages if m.get("role") in ("system", "user")])
        # No image here; image handling remains in MCP/tool side; ensure unified MCP contract upstream
        response_text = self.model.chat(
            tokenizer=self.tokenizer,
            pixel_values=None,
            question=question,
            history=None,
            return_history=False,
            generation_config=self._gen_kw,
            num_patches_list=None,
        )
        if isinstance(response_text, tuple):
            response_text = response_text[0]
        visible = response_text
        full = response_text
        return visible, full