# OpenRouter Pricing & Cost Tracking

## What We Do

- **Pricing source**: For OpenRouter models we use the `/endpoints` API  
  [`GET /models/{author}/{slug}/endpoints`](https://openrouter.ai/docs/api/api-reference/endpoints/list-endpoints)  
  which provides endpoint-specific pricing. We use **interquartile mean** (average of 25th-75th percentile) across all endpoints for each model to get realistic pricing that removes outliers on both ends while better reflecting typical costs.
- **API format**: OpenRouter API returns prices **per token** (e.g., `"0.00003"` = $0.00003 per token).  
  We convert to per 1K tokens: `price_per_1k = price_per_token * 1000`
- **When**: Once per agent run, at startup, inside `LLMManager` (if enabled)
- **Where stored**: In‑memory in `LLM_CONFIGS[LLMProvider.OPENROUTER].models[*]` as:
  - `prompt_price_per_1k` – USD per 1K input tokens
  - `completion_price_per_1k` – USD per 1K output tokens
- **Who uses it**: `ConversationTokenTracker` in `token_counter.py`

## Startup Flow (LLMManager)

1. `LLMManager.__init__()` calls `_update_openrouter_pricing()`
2. Checks `OPENROUTER_FETCH_PRICING_AT_STARTUP` env flag (default: `true`)
3. **Fallback chain** (tries each in order until pricing is found):
   - **Step 1: API fetch** (if enabled):
     - Fetches endpoints for each configured model from `/models/{author}/{slug}/endpoints`
     - Uses interquartile mean pricing (average of 25th-75th percentile) across all endpoints for each model
     - Removes outliers on both ends while reflecting typical costs
     - Updates model configs in memory: `prompt_price_per_1k`, `completion_price_per_1k`
   - **Step 2: JSON snapshot** (if API fails or disabled):
     - Loads pricing from `agent_ng/openrouter_pricing.json` (if exists)
     - This file is created/updated by the CLI utility script
     - Provides offline fallback without API dependency
   - **If neither available**: Models use 0.0 pricing (unknown cost), logs warning but continues startup

This happens **once per process**; the values are then reused for the whole agent run.

## Cost Calculation (Token Counter)

`ConversationTokenTracker` no longer tries to read cost from LangChain / OpenRouter
responses (that data is missing in streaming mode). Instead it:

1. Extracts **input/output/total tokens** from API responses (or tiktoken estimates)
2. Uses configured prices to compute cost:

```python
cost = (input_tokens / 1000.0) * prompt_price_per_1k \
     + (output_tokens / 1000.0) * completion_price_per_1k
```

3. Tracks:
   - Per‑turn cost
   - Per‑conversation cost
   - Session‑level cumulative cost

The UI then displays these values in the token statistics panel.

## Configuration

- **Env flag**: `OPENROUTER_FETCH_PRICING_AT_STARTUP` (default: `true`)
  - `true`: Try API fetch first, fallback to JSON snapshot
  - `false`: Skip API fetch, use JSON snapshot only

## Notes

- **Pricing source**: JSON snapshot (`agent_ng/openrouter_pricing.json`) is the single source of truth for pricing data:
  - Created/updated by the CLI utility script (`python -m agent_ng.utils.openrouter_pricing`)
  - Used as fallback cache if API fetch fails or is disabled
  - Provides offline resilience and faster startup when API is unavailable
  - If missing, models use 0.0 pricing (unknown cost)
- **Utility script**: `agent_ng/utils/openrouter_pricing.py` provides:
  - Library functions (`fetch_pricing_via_endpoints`) used at runtime
  - CLI helper (`python -m agent_ng.utils.openrouter_pricing`) that fetches pricing via endpoints and writes JSON snapshot
- **Fallback behavior**: If API fetch fails (or is disabled), the system gracefully falls back to JSON snapshot. If JSON is also unavailable, models use 0.0 pricing (unknown cost) and startup continues.

