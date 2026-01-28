"""
OpenRouter Pricing Utility
==========================

Lightweight helper for inspecting OpenRouter model pricing and supporting
the dynamic startup pricing in `LLMManager`.

Primary responsibilities:
- Fetch pricing from `/models/{author}/{slug}/endpoints` API and average across endpoints
- Provide library functions used by `LLMManager` at runtime
- Provide a CLI helper (`python -m agent_ng.utils.openrouter_pricing`) that:
  - Fetches pricing via endpoints API for configured models
  - Prints update suggestions to stdout
  - Writes JSON snapshot to `agent_ng/openrouter_pricing.json` for convenience

Notes:
- Runtime pricing fetch is controlled by `OPENROUTER_FETCH_PRICING_AT_STARTUP` env flag
- If runtime fetch fails or is disabled, falls back to pricing defaults in `LLM_CONFIGS`
- The JSON file is optional and used only for manual inspection
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import sys
from typing import Any

from dotenv import load_dotenv
import httpx

# Allow running as a standalone script from any CWD
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _parse_long_number(value: str | float) -> float:
    """Parse OpenRouter BigNumberUnion (string or number) to float."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # Handle scientific notation strings like "1.2e-05"
        try:
            return float(value)
        except ValueError:
            logger.warning("Failed to parse pricing value: %r", value)
            return 0.0
    return 0.0


def fetch_all_models(api_key: str, base_url: str = "https://openrouter.ai/api/v1") -> list[dict[str, Any]]:
    """Fetch all models with official averaged pricing.

    Uses `/models` endpoint which provides official averaged prices across all endpoints.

    Args:
        api_key: OpenRouter API key
        base_url: OpenRouter API base URL

    Returns:
        List of model dictionaries with pricing information
    """
    url = f"{base_url.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and "data" in data:
                return data["data"]
            return []
    except Exception as exc:
        logger.exception("Failed to fetch models: %s", exc)
        return []


def fetch_model_endpoints(
    author: str, slug: str, api_key: str, base_url: str = "https://openrouter.ai/api/v1"
) -> list[dict[str, Any]]:
    """Fetch all endpoints for a specific model.

    Args:
        author: Model author (e.g., "deepseek")
        slug: Model slug (e.g., "deepseek-v3.1-terminus")
        api_key: OpenRouter API key
        base_url: OpenRouter API base URL

    Returns:
        List of endpoint dictionaries with pricing information
    """
    url = f"{base_url.rstrip('/')}/models/{author}/{slug}/endpoints"
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and "data" in data:
                endpoints = data["data"].get("endpoints", [])
                return endpoints
            return []
    except Exception as exc:
        logger.debug("Failed to fetch endpoints for %s/%s: %s", author, slug, exc)
        return []


def extract_pricing_from_model(model: dict[str, Any]) -> tuple[float, float]:
    """Extract prompt and completion pricing from model dict.

    Args:
        model: Model dictionary from OpenRouter API

    Returns:
        Tuple of (prompt_price_per_1k, completion_price_per_1k) in USD
    """
    pricing = model.get("pricing", {})
    if not isinstance(pricing, dict):
        return (0.0, 0.0)

    # OpenRouter pricing is per token, convert to per 1K
    # Example: "0.00003" per token = $30 per 1M = $0.03 per 1K
    # Example: "0.000002" per token = $2 per 1M = $0.002 per 1K
    prompt_per_token = _parse_long_number(pricing.get("prompt", "0"))
    completion_per_token = _parse_long_number(pricing.get("completion", "0"))

    # Convert per token to per 1K: multiply by 1000
    prompt_per_1k = prompt_per_token * 1000.0
    completion_per_1k = completion_per_token * 1000.0

    return (prompt_per_1k, completion_per_1k)


def average_endpoint_pricing(endpoints: list[dict[str, Any]]) -> tuple[float, float]:
    """Calculate average pricing across multiple endpoints.

    Args:
        endpoints: List of endpoint dictionaries

    Returns:
        Tuple of (avg_prompt_price_per_1k, avg_completion_price_per_1k) in USD
    """
    if not endpoints:
        return (0.0, 0.0)

    prompt_prices = []
    completion_prices = []

    for endpoint in endpoints:
        pricing = endpoint.get("pricing", {})
        if not isinstance(pricing, dict):
            continue

        # OpenRouter pricing is per token, convert to per 1K
        prompt_per_token = _parse_long_number(pricing.get("prompt", "0"))
        completion_per_token = _parse_long_number(pricing.get("completion", "0"))

        if prompt_per_token > 0:
            prompt_prices.append(prompt_per_token * 1000.0)  # Convert per token to per 1K
        if completion_per_token > 0:
            completion_prices.append(completion_per_token * 1000.0)  # Convert per token to per 1K

    avg_prompt = sum(prompt_prices) / len(prompt_prices) if prompt_prices else 0.0
    avg_completion = sum(completion_prices) / len(completion_prices) if completion_prices else 0.0

    return (avg_prompt, avg_completion)


def parse_model_slug(model_slug: str) -> tuple[str | None, str | None]:
    """Parse model slug into author and slug components.

    Examples:
        "deepseek/deepseek-v3.1-terminus" -> ("deepseek", "deepseek-v3.1-terminus")
        "deepseek/deepseek-v3.1-terminus:exacto" -> ("deepseek", "deepseek-v3.1-terminus")
        "gpt-4o" -> (None, None)  # Not an OpenRouter slug format

    Returns:
        Tuple of (author, slug) or (None, None) if not parseable
    """
    # Remove tag suffix (e.g., ":exacto", ":free")
    base_slug = model_slug.split(":")[0]

    parts = base_slug.split("/", 1)
    if len(parts) == 2:
        return (parts[0], parts[1])
    return (None, None)


def fetch_pricing_from_models_api(
    model_names: list[str],
    api_key: str,
    base_url: str = "https://openrouter.ai/api/v1",
) -> dict[str, dict[str, float]]:
    """Fetch pricing from `/models` endpoint (official averaged prices).

    Uses `/models` endpoint which provides official averaged pricing across all endpoints.
    Faster than fetching individual endpoints and provides official pricing.

    Args:
        model_names: List of model identifiers to fetch pricing for
        api_key: OpenRouter API key
        base_url: OpenRouter API base URL

    Returns:
        Dictionary mapping model names to pricing dicts
    """
    pricing_map: dict[str, dict[str, float]] = {}

    # Fetch all models once (contains official averaged pricing)
    logger.info("Fetching all models from OpenRouter /models endpoint...")
    all_models = fetch_all_models(api_key, base_url)
    if not all_models:
        logger.warning("No models returned from OpenRouter API")
        return pricing_map

    # Build lookup map: canonical_slug -> model, id -> model
    model_lookup: dict[str, dict[str, Any]] = {}
    for model in all_models:
        model_id = model.get("id", "")
        canonical_slug = model.get("canonical_slug", "")
        if model_id:
            model_lookup[model_id] = model
        if canonical_slug:
            model_lookup[canonical_slug] = model

    # Extract pricing for requested models
    for model_name in model_names:
        # Remove tag suffix (e.g., ":exacto", ":free")
        base_name = model_name.split(":")[0]

        # Try exact match first
        model = model_lookup.get(model_name) or model_lookup.get(base_name)
        if not model:
            # Try to find by prefix match (for dated versions)
            for key, candidate_model in model_lookup.items():
                if key.startswith(base_name) or base_name.startswith(key.split("-")[0]):
                    model = candidate_model
                    break

        if not model:
            logger.debug("Model not found in OpenRouter API: %s", model_name)
            continue

        # Extract pricing from model (official averaged pricing)
        prompt_price, completion_price = extract_pricing_from_model(model)
        if prompt_price > 0 or completion_price > 0:
            pricing_map[model_name] = {
                "prompt_price_per_1k": prompt_price,
                "completion_price_per_1k": completion_price,
            }
            # Also add base name without tags
            if base_name != model_name:
                pricing_map[base_name] = {
                    "prompt_price_per_1k": prompt_price,
                    "completion_price_per_1k": completion_price,
                }
            logger.info(
                "Model %s: prompt=$%.6f/1K, completion=$%.6f/1K (official pricing)",
                model_name,
                prompt_price,
                completion_price,
            )

    return pricing_map


def fetch_pricing_via_endpoints(
    model_names: list[str],
    api_key: str,
    base_url: str = "https://openrouter.ai/api/v1",
) -> dict[str, dict[str, float]]:
    """Fetch pricing using endpoints API directly (legacy method).

    Uses `/models/{author}/{slug}/endpoints` for each model, which provides
    endpoint-specific pricing and averages across endpoints.

    Args:
        model_names: List of model identifiers to fetch pricing for
        api_key: OpenRouter API key
        base_url: OpenRouter API base URL

    Returns:
        Dictionary mapping model names to pricing dicts
    """
    pricing_map: dict[str, dict[str, float]] = {}

    for model_name in model_names:
        # Remove tag suffix (e.g., ":exacto", ":free")
        base_name = model_name.split(":")[0]

        # Parse author/slug
        author, model_slug = parse_model_slug(base_name)
        if not author or not model_slug:
            logger.debug("Cannot parse model name for endpoints API: %s", model_name)
            continue

        # Fetch endpoints for this model
        endpoints = fetch_model_endpoints(author, model_slug, api_key, base_url)
        if not endpoints:
            logger.debug("No endpoints found for %s/%s", author, model_slug)
            continue

        # Average pricing across endpoints
        prompt_price, completion_price = average_endpoint_pricing(endpoints)
        if prompt_price > 0 or completion_price > 0:
            pricing_map[model_name] = {
                "prompt_price_per_1k": prompt_price,
                "completion_price_per_1k": completion_price,
            }
            # Also add base name without tags
            if base_name != model_name:
                pricing_map[base_name] = {
                    "prompt_price_per_1k": prompt_price,
                    "completion_price_per_1k": completion_price,
                }
            logger.info(
                "Model %s: prompt=$%.6f/1K, completion=$%.6f/1K (from %d endpoints)",
                model_name,
                prompt_price,
                completion_price,
                len(endpoints),
            )

    return pricing_map


def fetch_pricing_for_models(
    model_names: list[str],
    api_key: str,
    base_url: str = "https://openrouter.ai/api/v1",
    *,
    use_endpoint_average: bool = False,
) -> dict[str, dict[str, float]]:
    """Fetch pricing only for specified models (optimized for startup).

    Args:
        model_names: List of model identifiers to fetch pricing for
        api_key: OpenRouter API key
        base_url: OpenRouter API base URL
        use_endpoint_average: If True, fetch endpoints and average pricing per model

    Returns:
        Dictionary mapping model slugs to pricing dicts
    """
    pricing_map: dict[str, dict[str, float]] = {}

    # Fetch all models once (needed to find pricing)
    logger.info("Fetching models from OpenRouter...")
    all_models = fetch_user_models(api_key, base_url)
    if not all_models:
        logger.warning("No models returned from OpenRouter API")
        return pricing_map

    # Build lookup map: canonical_slug -> model, id -> model
    model_lookup: dict[str, dict[str, Any]] = {}
    for model in all_models:
        model_id = model.get("id", "")
        canonical_slug = model.get("canonical_slug", "")
        if model_id:
            model_lookup[model_id] = model
        if canonical_slug:
            model_lookup[canonical_slug] = model

    # Fetch pricing only for requested models
    for model_name in model_names:
        # Remove tag suffix (e.g., ":exacto", ":free")
        base_name = model_name.split(":")[0]

        # Try exact match first
        model = model_lookup.get(model_name) or model_lookup.get(base_name)
        if not model:
            # Try to find by prefix match (for dated versions)
            for key, candidate_model in model_lookup.items():
                if key.startswith(base_name) or base_name.startswith(key.split("-")[0]):
                    model = candidate_model
                    break

        if not model:
            logger.debug("Model not found in OpenRouter API: %s", model_name)
            continue

        model_id = model.get("id", "")
        canonical_slug = model.get("canonical_slug", "")
        slug = canonical_slug or model_id

        prompt_price = 0.0
        completion_price = 0.0

        if use_endpoint_average:
            author, model_slug = parse_model_slug(slug)
            if author and model_slug:
                endpoints = fetch_model_endpoints(author, model_slug, api_key, base_url)
                if endpoints:
                    prompt_price, completion_price = average_endpoint_pricing(endpoints)
                    if prompt_price > 0 or completion_price > 0:
                        logger.info(
                            "Model %s: prompt=$%.6f/1K, completion=$%.6f/1K (averaged from %d endpoints)",
                            slug,
                            prompt_price,
                            completion_price,
                            len(endpoints),
                        )

        if prompt_price <= 0.0 and completion_price <= 0.0:
            prompt_price, completion_price = extract_pricing_from_model(model)
            if prompt_price > 0 or completion_price > 0:
                logger.info(
                    "Model %s: prompt=$%.6f/1K, completion=$%.6f/1K",
                    slug,
                    prompt_price,
                    completion_price,
                )

        if prompt_price > 0 or completion_price > 0:
            # Use original model_name as key (preserves tags)
            pricing_map[model_name] = {
                "prompt_price_per_1k": prompt_price,
                "completion_price_per_1k": completion_price,
            }
            # Also add base name without tags
            if base_name != model_name:
                pricing_map[base_name] = {
                    "prompt_price_per_1k": prompt_price,
                    "completion_price_per_1k": completion_price,
                }

    return pricing_map


def update_llm_config_with_pricing(
    models_data: list[dict[str, Any]],
    api_key: str,
    base_url: str = "https://openrouter.ai/api/v1",
    *,
    use_endpoint_average: bool = False,
) -> dict[str, dict[str, float]]:
    """Build pricing map from OpenRouter API data.

    Args:
        models_data: List of model dictionaries from `/models/user`
        api_key: OpenRouter API key (for endpoint fetching if enabled)
        base_url: OpenRouter API base URL
        use_endpoint_average: If True, fetch endpoints and average pricing per model

    Returns:
        Dictionary mapping model slugs to pricing dicts. Includes both:
        - Canonical slugs (e.g., "z-ai/glm-4.7")
        - Dated versions (e.g., "z-ai/glm-4.7-20251222")
        {
            "deepseek/deepseek-v3.1-terminus": {
                "prompt_price_per_1k": 0.0001,
                "completion_price_per_1k": 0.0002
            },
            ...
        }
    """
    pricing_map: dict[str, dict[str, float]] = {}

    for model in models_data:
        model_id = model.get("id", "")
        canonical_slug = model.get("canonical_slug", "")

        # Prefer canonical_slug, fallback to id
        slug = canonical_slug or model_id
        if not slug:
            continue

        # Extract pricing
        prompt_price = 0.0
        completion_price = 0.0

        if use_endpoint_average:
            author, model_slug = parse_model_slug(slug)
            if author and model_slug:
                endpoints = fetch_model_endpoints(author, model_slug, api_key, base_url)
                if endpoints:
                    prompt_price, completion_price = average_endpoint_pricing(endpoints)
                    if prompt_price > 0 or completion_price > 0:
                        logger.info(
                            "Model %s: prompt=$%.6f/1K, completion=$%.6f/1K (averaged from %d endpoints)",
                            slug,
                            prompt_price,
                            completion_price,
                            len(endpoints),
                        )

        # Fallback to model-level pricing
        if prompt_price <= 0.0 and completion_price <= 0.0:
            prompt_price, completion_price = extract_pricing_from_model(model)
            if prompt_price > 0 or completion_price > 0:
                logger.info(
                    "Model %s: prompt=$%.6f/1K, completion=$%.6f/1K",
                    slug,
                    prompt_price,
                    completion_price,
                )

        if prompt_price > 0 or completion_price > 0:
            pricing_map[slug] = {
                "prompt_price_per_1k": prompt_price,
                "completion_price_per_1k": completion_price,
            }

            # Also add entry for canonical_slug if different from slug (for matching)
            if canonical_slug and canonical_slug != slug:
                pricing_map[canonical_slug] = {
                    "prompt_price_per_1k": prompt_price,
                    "completion_price_per_1k": completion_price,
                }

    return pricing_map


def generate_llm_config_update(pricing_map: dict[str, dict[str, float]]) -> str:
    """Generate Python code snippet to update LLM config.

    Args:
        pricing_map: Dictionary mapping model slugs to pricing

    Returns:
        Python code snippet as string
    """
    lines = [
        "# Auto-generated pricing updates from OpenRouter API",
        "# Run: python -m agent_ng.utils.openrouter_pricing",
        "",
    ]

    for model_slug, pricing in sorted(pricing_map.items()):
        prompt = pricing["prompt_price_per_1k"]
        completion = pricing["completion_price_per_1k"]
        lines.append(f"# {model_slug}: prompt=${prompt:.6f}/1K, completion=${completion:.6f}/1K")
        lines.append(
            f'# Add to model config: "prompt_price_per_1k": {prompt}, "completion_price_per_1k": {completion}'
        )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    """CLI entry point for manual pricing inspection.

    This helper:
    - Fetches pricing via endpoints for configured OpenRouter models
    - Prints update suggestions to stdout
    - Writes a JSON snapshot to `agent_ng/openrouter_pricing.json`

    Runtime pricing for the agent is handled separately by `LLMManager`
    via `fetch_pricing_via_endpoints()` at startup; this CLI is optional.
    """
    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY is not set")
        sys.exit(1)

    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    # Simple CLI helper: fetch pricing for configured OpenRouter models (if available)
    try:
        # Add agent_ng to path for proper imports
        agent_ng_path = Path(_REPO_ROOT) / "agent_ng"
        if str(agent_ng_path) not in sys.path:
            sys.path.insert(0, str(agent_ng_path))
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))

        # Import directly (path is set up)
        try:
            from agent_ng.llm_manager import LLMProvider
            from agent_ng.llm_configs import get_default_llm_configs
        except ImportError:
            # Fallback: try importing as module
            import importlib
            llm_manager_module = importlib.import_module("agent_ng.llm_manager")
            llm_configs_module = importlib.import_module("agent_ng.llm_configs")
            LLMProvider = llm_manager_module.LLMProvider
            get_default_llm_configs = llm_configs_module.get_default_llm_configs

        llm_configs = get_default_llm_configs()
        config = llm_configs.get(LLMProvider.OPENROUTER)
        if not config or not config.models:
            logger.error("No OpenRouter models configured in LLM_CONFIGS")
            sys.exit(1)

        model_names = [m.get("model", "") for m in config.models if m.get("model")]
        logger.info("Fetching pricing via endpoints API for %d models (averaging endpoints)...", len(model_names))
        pricing_map = fetch_pricing_via_endpoints(model_names, api_key, base_url)
    except Exception as exc:  # pragma: no cover - CLI helper only
        logger.exception("Failed to fetch pricing: %s", exc)
        sys.exit(1)

    logger.info("Pricing map contains %d models", len(pricing_map))

    # Print pricing update suggestions
    print("\n" + "=" * 80)
    print("PRICING UPDATE SUGGESTIONS")
    print("=" * 80)
    print(generate_llm_config_update(pricing_map))

    # Also write a JSON snapshot for convenience (not used by runtime)
    output_file = Path(_REPO_ROOT) / "agent_ng" / "openrouter_pricing.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(pricing_map, f, indent=2, ensure_ascii=False)
    logger.info("Saved pricing snapshot to %s", output_file)


if __name__ == "__main__":
    main()
