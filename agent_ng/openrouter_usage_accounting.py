# // pragma: allowlist secret
# pragma: allowlist secret
"""OpenRouter usage normalization and optional accounting callback.

Mirrors the cmw-rag pattern: flatten ``usage`` and nested billing detail fields
for logging and downstream aggregation without coupling to Gradio or session context.

--- Future: Polza.ai provider ---
Polza.ai (https://polza.ai/api/v1) is structurally identical to OpenRouter:
  - Same OpenAI-compatible wire protocol and streaming pattern
    (usage arrives in the final SSE chunk, ``choices: []``)
  - ``OpenRouterNativeChatModel`` can be reused as-is
The only difference is cost currency:
  - OpenRouter: ``usage.cost`` → USD
  - Polza.ai:   ``usage.cost_rub`` → RUB  (``usage.cost`` is an alias)
When adding a ``polza`` provider:
  1. Add ``LLMProvider.POLZA = "polza"`` and a config block in llm_configs.py
     (api_key_env="POLZA_API_KEY", api_base_env="POLZA_BASE_URL").
  2. Add ``_initialize_polza_llm`` in LLMManager — identical to
     ``_initialize_openai_llm`` but wired to POLZA_* env vars.
  3. Add ``PolzaUsageAccountingCallback`` here that reads ``cost_rub``
     and either stores it as-is (display "X ₽") or converts to USD via
     a ``POLZA_RUB_TO_USD_RATE`` env var.
     ``normalize_openrouter_usage`` can be reused after remapping
     ``cost_rub`` → ``cost``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain_core.callbacks import BaseCallbackHandler

if TYPE_CHECKING:
    from langchain_core.outputs import LLMResult

_logger = logging.getLogger(__name__)

USAGE_NUMERIC_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "reasoning_tokens",
    "cached_tokens",
    "cache_write_tokens",
    "cost",
    "upstream_cost",
)


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def normalize_openrouter_usage(token_usage: dict[str, Any] | None) -> dict[str, float]:
    """Normalize OpenRouter-style ``usage`` dict to flat floats."""
    if not isinstance(token_usage, dict):
        return dict.fromkeys(USAGE_NUMERIC_FIELDS, 0.0)

    prompt_tokens = _safe_int(token_usage.get("prompt_tokens"))
    completion_tokens = _safe_int(token_usage.get("completion_tokens"))
    total_tokens = _safe_int(
        token_usage.get("total_tokens") or (prompt_tokens + completion_tokens)
    )

    comp_details = token_usage.get("completion_tokens_details") or {}
    if not isinstance(comp_details, dict):
        comp_details = {}
    reasoning_tokens = _safe_int(comp_details.get("reasoning_tokens"))

    prompt_details = token_usage.get("prompt_tokens_details") or {}
    if not isinstance(prompt_details, dict):
        prompt_details = {}
    cached_tokens = _safe_int(prompt_details.get("cached_tokens"))
    cache_write_tokens = _safe_int(prompt_details.get("cache_write_tokens"))

    cost = _safe_float(token_usage.get("cost"))
    cost_details = token_usage.get("cost_details") or {}
    if not isinstance(cost_details, dict):
        cost_details = {}
    upstream_cost = _safe_float(cost_details.get("upstream_inference_cost"))

    return {
        "prompt_tokens": float(prompt_tokens),
        "completion_tokens": float(completion_tokens),
        "total_tokens": float(total_tokens),
        "reasoning_tokens": float(reasoning_tokens),
        "cached_tokens": float(cached_tokens),
        "cache_write_tokens": float(cache_write_tokens),
        "cost": float(cost),
        "upstream_cost": float(upstream_cost),
    }


def summarize_usage_for_logging(token_usage: dict[str, Any] | None) -> dict[str, float]:
    """Stable diagnostic snapshot (same keys always present)."""
    return normalize_openrouter_usage(token_usage)


class OpenRouterUsageAccountingCallback(BaseCallbackHandler):
    """Accumulates normalized usage from ``LLMResult.llm_output['token_usage']``.

    Attach to ``OpenRouterNativeChatModel`` for structured per-call totals without
    touching ``ConversationTokenTracker`` (optional parallel path).
    """

    def __init__(self) -> None:
        super().__init__()
        self._accumulator: dict[str, float] = dict.fromkeys(USAGE_NUMERIC_FIELDS, 0.0)
        self._last_raw: dict[str, Any] | None = None

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:  # type: ignore[override]
        _ = kwargs
        raw_output = response.llm_output
        llm_output = raw_output if isinstance(raw_output, dict) else {}
        token_usage = llm_output.get("token_usage")
        if not isinstance(token_usage, dict):
            return

        self._last_raw = dict(token_usage)
        normalized = normalize_openrouter_usage(token_usage)
        for k in USAGE_NUMERIC_FIELDS:
            self._accumulator[k] = self._accumulator.get(k, 0.0) + normalized[k]

    def flush_turn_summary(self) -> dict[str, float] | None:
        """Return accumulated totals and reset (single QA-turn boundary)."""
        if not any(self._accumulator.values()) and self._last_raw is None:
            return None
        out = dict(self._accumulator)
        self._accumulator = dict.fromkeys(USAGE_NUMERIC_FIELDS, 0.0)
        self._last_raw = None
        return out
