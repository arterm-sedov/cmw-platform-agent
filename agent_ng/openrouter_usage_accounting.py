# // pragma: allowlist secret
# pragma: allowlist secret
"""Usage normalization and accounting callbacks for OpenRouter and Polza.ai.

Mirrors the cmw-rag pattern: flatten ``usage`` and nested billing fields for
logging and downstream aggregation, decoupled from Gradio / session context.

Polza.ai (https://polza.ai/api/v1) is structurally identical to OpenRouter
(same OpenAI-compatible wire protocol, usage in final SSE chunk) with one
difference: billing is in **rubles** (``usage.cost_rub``; ``usage.cost`` is
an alias for the same value).  ``PolzaUsageAccountingCallback`` handles the
RUB-specific normalization; optionally converts to USD via
``POLZA_RUB_TO_USD_RATE`` env var (float, e.g. ``0.011``).
"""

from __future__ import annotations

import contextlib
import logging
import os
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

        # Streaming: llm_output is None; token_usage lives in generation_info.
        if not isinstance(token_usage, dict):
            for gen_row in response.generations or []:
                gens = gen_row if isinstance(gen_row, list) else [gen_row]
                for gen in gens:
                    gi = getattr(gen, "generation_info", None) or {}
                    if isinstance(gi.get("token_usage"), dict):
                        token_usage = gi["token_usage"]
                        break
                if isinstance(token_usage, dict):
                    break

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


# ---------------------------------------------------------------------------
# Polza.ai
# ---------------------------------------------------------------------------

POLZA_USAGE_NUMERIC_FIELDS = (*USAGE_NUMERIC_FIELDS, "cost_rub")

_POLZA_DEFAULT_RATE = 90.0  # RUB per 1 USD — used when env var is absent


def _get_polza_rate(override: float | None = None) -> float:
    """Return the RUB-per-USD rate, logging a warning when falling back to default."""
    if override is not None:
        return override
    env_val = os.getenv("POLZA_RUB_TO_USD_RATE", "").strip()
    if env_val:
        with contextlib.suppress(ValueError):
            return float(env_val)
        _logger.warning(
            "POLZA_RUB_TO_USD_RATE='%s' is not a valid float; "
            "using default rate %.0f RUB/USD",
            env_val,
            _POLZA_DEFAULT_RATE,
        )
    else:
        _logger.warning(
            "POLZA_RUB_TO_USD_RATE is not set; "
            "using default rate %.0f RUB/USD for cost conversion. "
            "Set POLZA_RUB_TO_USD_RATE in .env to suppress this warning.",
            _POLZA_DEFAULT_RATE,
        )
    return _POLZA_DEFAULT_RATE


def normalize_polza_usage(
    token_usage: dict[str, Any] | None,
    rub_to_usd_rate: float | None = None,
) -> dict[str, float]:
    """Normalize Polza.ai ``usage`` dict.

    ``cost_rub`` is the authoritative billing field (rubles).
    Conversion to USD is **always performed** so that ``cost`` is a valid USD
    figure for the shared billing pipeline.  The rate is expressed as
    **RUB per 1 USD** (e.g. ``90`` means 90 ₽ = $1):
    ``cost_usd = cost_rub / rate``.

    Rate resolution order:
      1. *rub_to_usd_rate* argument (for tests / explicit override)
      2. ``POLZA_RUB_TO_USD_RATE`` env var
      3. Built-in default ``_POLZA_DEFAULT_RATE`` (logged as a warning)
    """
    base = normalize_openrouter_usage(token_usage)
    if not isinstance(token_usage, dict):
        return {**base, "cost_rub": 0.0}

    cost_rub = _safe_float(
        token_usage.get("cost_rub") or token_usage.get("cost")
    )

    rate = _get_polza_rate(override=rub_to_usd_rate)
    cost_usd = cost_rub / rate
    return {**base, "cost": cost_usd, "cost_rub": cost_rub}


class PolzaUsageAccountingCallback(BaseCallbackHandler):
    """Accumulates Polza.ai usage from ``LLMResult.llm_output['token_usage']``.

    Always converts ``cost_rub`` → ``cost`` (USD) via ``POLZA_RUB_TO_USD_RATE``
    (RUB per 1 USD).  Falls back to ``_POLZA_DEFAULT_RATE`` with a warning when
    the env var is absent, so ``cost`` is never silently zero.
    """

    def __init__(self, rub_to_usd_rate: float | None = None) -> None:
        super().__init__()
        self._rate = rub_to_usd_rate
        self._accumulator: dict[str, float] = dict.fromkeys(
            POLZA_USAGE_NUMERIC_FIELDS, 0.0
        )
        self._last_raw: dict[str, Any] | None = None

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:  # type: ignore[override]
        _ = kwargs
        raw_output = response.llm_output
        llm_output = raw_output if isinstance(raw_output, dict) else {}
        token_usage = llm_output.get("token_usage")

        # Streaming: llm_output is None; token_usage lives in generation_info.
        if not isinstance(token_usage, dict):
            for gen_row in response.generations or []:
                gens = gen_row if isinstance(gen_row, list) else [gen_row]
                for gen in gens:
                    gi = getattr(gen, "generation_info", None) or {}
                    if isinstance(gi.get("token_usage"), dict):
                        token_usage = gi["token_usage"]
                        break
                if isinstance(token_usage, dict):
                    break

        if not isinstance(token_usage, dict):
            return

        self._last_raw = dict(token_usage)
        normalized = normalize_polza_usage(token_usage, rub_to_usd_rate=self._rate)
        for k in POLZA_USAGE_NUMERIC_FIELDS:
            self._accumulator[k] = (
                self._accumulator.get(k, 0.0) + normalized.get(k, 0.0)
            )

    def flush_turn_summary(self) -> dict[str, float] | None:
        """Return accumulated totals and reset."""
        if not any(self._accumulator.values()) and self._last_raw is None:
            return None
        out = dict(self._accumulator)
        self._accumulator = dict.fromkeys(POLZA_USAGE_NUMERIC_FIELDS, 0.0)
        self._last_raw = None
        return out
