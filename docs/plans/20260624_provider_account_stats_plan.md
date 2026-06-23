# Provider Account Balances on Statistics Tab

## Summary

Implement provider-agnostic account balance and limit reporting on the Statistics tab without adding billing API calls to the existing automatic stats refresh path. The feature should show detailed account data only for API keys explicitly supplied by the current user in the Config tab. For shared keys loaded from `.env`, the UI must show only a coarse health warning driven by configured thresholds, never numeric balance, usage, limit, reset, or package details.

This plan targets a durable implementation in the platform repo and follows the existing `docs/plans/` convention for cross-cutting feature work.

## Key Changes

- Add a provider account reporting module, likely `agent_ng/provider_account.py`, with a normalized result model containing:
  - `provider`
  - `key_source`: `session` or `environment`
  - `status`: `available`, `unsupported`, `missing_key`, `unauthorized`, `error`, `stale`
  - `balance`
  - `currency`
  - `used`
  - `limit`
  - `remaining`
  - `reset_period`
  - `updated_at`
  - `details`
  - `health`: `ok`, `low`, `depleted`, `unavailable`, for display-safe `.env` results.
- Implement provider adapters:
  - OpenRouter: support `/api/v1/credits` for account credits and `/api/v1/key` for per-key limits.
  - Polza: support `/api/v1/balance`, with short timeout, last-good cache, and stale fallback because local retests showed intermittent HTTPS and TLS failures.
  - GigaChat: support OAuth token exchange plus `/api/v1/balance`; treat zero package balances as valid and `403` as not applicable for this account or package.
  - Gemini, Groq, Hugging Face, Mistral, generic OpenAI-compatible: return `unsupported` with a concise dashboard or API-unavailable message.
  - OpenAI: return `unsupported` for normal inference keys; do not implement org usage unless a separate admin key flow is explicitly requested later.
- Reuse existing session-aware key resolution:
  - Use `get_provider_api_key(provider=..., session_id=...)`.
  - Also determine whether the resolved key came from `llm_provider_api_keys[provider]` in session config or from the provider's `.env` variable.
  - Session-provided Config-tab keys always take precedence over `.env` keys.
  - Never display, log, cache, or persist raw keys.
- Add `.env` threshold settings for shared app-owned keys:
  - `OPENROUTER_LOW_BALANCE_THRESHOLD_USD`
  - `POLZA_LOW_BALANCE_THRESHOLD_RUB`
  - `GIGACHAT_LOW_TOKEN_THRESHOLD`
  - Optional future provider thresholds should follow the same explicit provider/currency naming pattern.
- Add a small in-memory cache:
  - Keyed by `session_id`, `provider`, and a non-secret key fingerprint.
  - TTL default: 5 minutes.
  - Store last successful result separately from transient errors.
  - On refresh failure, show stale last-good data when available.
- Update the Statistics tab:
  - Keep `format_stats_display()` purely local and cached.
  - Add a manual `Refresh provider account` button.
  - Render the account block below current model, provider, and pricing details.
  - Show active provider only in v1.
  - Do not query all configured providers automatically.
  - For Config-tab keys, show detailed numeric balance and limit data.
  - For `.env` keys, show only coarse shared-account health: `OK`, `Low`, `Depleted`, or `Status unavailable`.
- Add English and Russian i18n labels for:
  - Account section title.
  - Refresh provider account button.
  - Missing key.
  - Unsupported provider.
  - Stale data.
  - Unauthorized.
  - Temporary provider error.
  - Shared account OK.
  - Shared account low.
  - Shared account depleted.
  - Shared account status unavailable.

## Implementation Details

- Do not perform remote HTTP calls from `StatsTab.format_stats_display()` because it is called by the 15-second UI timer and several end-of-turn refresh paths.
- Add a new refresh handler on `StatsTab`, for example `refresh_provider_account(request)`, wired to the new button and returning the updated `stats_display` markdown.
- The refresh handler should:
  - Resolve `session_id` from the Gradio request.
  - Resolve the active provider from the session agent’s current `llm_info`.
  - Resolve both the effective key and its source: session Config tab versus `.env`.
  - Call the account service for that provider.
  - Update the provider-account cache.
  - Return `format_stats_display(request)`.
- The normal stats refresh should:
  - Read only cached provider-account state.
  - If no cached data exists, show `Not refreshed yet` or `Click refresh to check account limits`.
- Display policy:
  - `key_source=session`: render numeric balance, used, remaining, limit, reset period, and provider-specific package details when available.
  - `key_source=environment`: suppress numeric values and render only threshold-based health text.
  - `key_source=environment` with no provider threshold configured: render `Shared provider account status unavailable` after refresh.
  - `key_source=environment` with a failed refresh and no last-good health state: render `Shared provider account status unavailable`.
  - `key_source=environment` with a failed refresh and cached last-good health state: render stale coarse health only, never stale numeric values.
- HTTP behavior:
  - Use timeout-bounded requests, target 5-10 seconds per provider.
  - Avoid retries in the UI request path except possibly one fast retry for Polza.
  - Log only provider, status, and error class, never headers or keys.
- Polza transport note:
  - Previous scratch testing showed Node `fetch` once succeeded with `amount=470.02686607 RUB`, but later Node and Python checks failed intermittently.
  - Implement the Python adapter defensively; if Python transport remains unreliable, return `error` or `stale` rather than blocking the UI.
- GigaChat behavior:
  - Use existing `GIGACHAT_API_KEY`, `GIGACHAT_SCOPE`, `GIGACHAT_VERIFY_SSL`, and base URL conventions.
  - Convert the authorization key to a temporary access token before calling the balance endpoint.
  - For Config-tab keys, display package token balances as token counts, not currency.
  - For `.env` keys, compare the lowest relevant remaining package token value to `GIGACHAT_LOW_TOKEN_THRESHOLD` and display only coarse shared-account health.

## Test Plan

- Unit tests for account normalization:
  - OpenRouter credits response maps to remaining, used, total credits, and USD.
  - OpenRouter key response with `limit=null` is valid and displayed as no key cap configured.
  - Polza balance response maps `amount` and `spentAmount` to RUB.
  - GigaChat balance response maps package balances to token details.
  - Unsupported providers return deterministic `unsupported` results.
- Unit tests for key-source policy:
  - Session Config-tab key results render numeric details.
  - `.env` key results never render numeric balance, usage, limits, reset periods, or token package details.
  - `.env` key with balance below threshold renders `low`.
  - `.env` key with zero or exhausted balance renders `depleted`.
  - `.env` key with no configured threshold renders `unavailable`.
- Unit tests for failure modes:
  - Missing key returns `missing_key`.
  - `401` and `403` return `unauthorized` or provider-specific not-applicable status.
  - Timeout returns `error`.
  - Timeout with last-good cached value returns `stale`.
  - Malformed JSON returns `error` without raising into the UI.
- Stats tab tests:
  - Normal `format_stats_display()` performs no HTTP calls.
  - Manual account refresh updates cache and returns updated markdown.
  - Cached account data appears in the stats markdown.
  - Numeric cached data appears only for session-supplied keys.
  - Shared `.env` key health messages contain no numeric details.
  - No raw key appears in rendered stats or logs.
- Integration-style tests with mocked HTTP clients:
  - Session A and Session B with different provider keys do not share account cache entries.
  - Switching active provider changes which cached account block is shown.
- Verification commands:
  - `python -m pytest agent_ng/_tests/test_provider_account.py`
  - `python -m pytest agent_ng/_tests/test_stats_tab_overview.py`
  - `ruff check agent_ng/provider_account.py agent_ng/tabs/stats_tab.py agent_ng/i18n_translations.py agent_ng/_tests/test_provider_account.py agent_ng/_tests/test_stats_tab_overview.py`

## Assumptions

- V1 reports only the active provider, not every configured provider.
- V1 uses manual refresh only; no automatic billing refresh timer.
- Full numeric balance and limit details are shown only for keys provided by the user in the Config tab for the current session.
- Keys loaded from `.env` are treated as shared app-owned credentials. The UI shows only coarse health warnings for those keys.
- If a shared `.env` key has no explicit threshold configured, the feature does not infer a low-balance threshold.
- OpenRouter and GigaChat are considered fully supported.
- Polza is considered supported but potentially flaky from this host, so stale and error handling is required.
- OpenAI organization usage is out of scope unless a separate admin-key requirement is added later.
- No secrets or real key fragments should be stored in docs, tests, logs, snapshots, or UI output.
