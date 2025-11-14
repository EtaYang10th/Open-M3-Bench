import os
from pathlib import Path
from typing import Optional

from .local_drivers import QwenLocalDriver, InternVLLocalDriver
from .api_clients import (
    OpenAIAPIClient,
    DeepseekAPIClient,
    InternAPIClient,
    GeminiAPIClient,
    AnthropicAPIClient,
    TogetherAPIClient,
    ZhipuAPIClient,
    GrokAPIClient,
)


def _is_path_like(model_spec: str) -> bool:
    try:
        # Treat as path-like only if it exists on filesystem.
        # Model names such as "meta-llama/Llama-4-..." include '/', but should NOT be treated as paths.
        p = Path(model_spec)
        return p.exists()
    except Exception:
        return False


def create_model_driver(model_spec: str, max_new_tokens: int = 2048):
    """
    Route to either a local driver (path-like) or an API client (name-like).

    - If `model_spec` is an existing path: load local HF model.
      - If name contains "intern" → InternVLLocalDriver
      - else → QwenLocalDriver
    - Otherwise: choose API by keywords in model name
      - contains gpt/openai → OpenAIAPIClient
      - contains deepseek → DeepseekAPIClient
      - contains intern → InternAPIClient
      - contains claude/anthropic → AnthropicAPIClient
    """
    spec_lower = (model_spec or "").lower()

    if _is_path_like(model_spec):
        if ("intern" in spec_lower) or ("internvl" in spec_lower):
            return InternVLLocalDriver(model_path=model_spec, max_new_tokens=max_new_tokens)
        return QwenLocalDriver(model_path=model_spec, max_new_tokens=max_new_tokens)

    else:
        # Together route: any model name containing '/'
        if ("/" in model_spec):
            return TogetherAPIClient(model_name=model_spec, max_new_tokens=max_new_tokens)
        # ZhipuAI route: any model name containing glm
        if ("glm" in spec_lower):
            return ZhipuAPIClient(model_name=model_spec, max_new_tokens=max_new_tokens)
        if ("gpt" in spec_lower) or ("openai" in spec_lower):
            return OpenAIAPIClient(model_name=model_spec, max_new_tokens=max_new_tokens)
        if "deepseek" in spec_lower:
            return DeepseekAPIClient(model_name=model_spec, max_new_tokens=max_new_tokens)
        if "intern" in spec_lower:
            return InternAPIClient(model_name=model_spec, max_new_tokens=max_new_tokens)
        if ("gemini" in spec_lower) or ("google" in spec_lower):
            return GeminiAPIClient(model_name=model_spec, max_new_tokens=max_new_tokens)
        if ("claude" in spec_lower) or ("anthropic" in spec_lower):
            return AnthropicAPIClient(model_name=model_spec, max_new_tokens=max_new_tokens)
        if ("grok" in spec_lower):
            return GrokAPIClient(model_name=model_spec, max_new_tokens=max_new_tokens)
    
    raise ValueError(f"Invalid model spec: {model_spec}")


