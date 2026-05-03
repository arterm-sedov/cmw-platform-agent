# Plan: Wire Polza.ai into the VL Pipeline

## Goal

Allow VL calls (image, video, audio) to route through Polza.ai when
`AGENT_PROVIDER=polza`, without duplicating any adapter logic. Polza uses
an identical wire protocol to OpenRouter for multimodal content.

## Contracts (SDD)

### `OpenRouterVisionAdapter.__init__(llm_manager, provider=LLMProvider.OPENROUTER)`
- Accepts an optional `provider` kwarg (default `LLMProvider.OPENROUTER`)
- Stores it as `self.provider`; all existing behaviour unchanged

### `VisionToolManager._init_adapters()`
- Registers `adapters['polza']` using `OpenRouterVisionAdapter(llm_manager, provider=LLMProvider.POLZA)`
- Existing `openrouter` and `gemini` slots unchanged

### `VisionToolManager.__init__()`
- Reads `AGENT_PROVIDER` env var as `self.vl_default_provider` (default `"openrouter"`)
- Normalises to lowercase; used by `get_adapter_for_model`

### `VisionToolManager.get_adapter_for_model(model)`
- `gemini-*` (bare) → `adapters['gemini']` (unchanged — Direct API only)
- plain name (no `/`) → `adapters['gemini']` (unchanged)
- `google/*` or `*/*` → `adapters.get(self.vl_default_provider, adapters['openrouter'])`
  - Falls back to `openrouter` if the provider slot is missing

### `VisionToolManager._resolve_gemini_model()`
- Adds `elif p == "polza":` → `_to_openrouter_gemini_model(base_model)`
  (`polza` and `openrouter` both accept `google/gemini-*` model ids)

## Env vars

| Var | Change |
|-----|--------|
| `AGENT_PROVIDER` | Now also drives VL adapter selection (no new var) |
| `VL_GEMINI_PROVIDER` | Accepts `polza` as alias for `openrouter` re model naming |

### Typical full-Polza setup
```
AGENT_PROVIDER=polza
VL_GEMINI_PROVIDER=polza        # google/ prefix, call via polza adapter
VL_YOUTUBE_GEMINI_PROVIDER=google  # YouTube still needs Gemini Direct
```

## Tasks

- [ ] `docs/plans/polza_vl_routing.md` — this file
- [ ] Tests in `agent_ng/_tests/test_vl_integration.py` (TDD — write first)
  - `test_polza_adapter_registered`
  - `test_polza_adapter_has_polza_provider`
  - `test_adapter_uses_polza_when_agent_provider_polza` (slash model)
  - `test_adapter_uses_polza_for_google_prefixed_when_agent_provider_polza`
  - `test_vl_gemini_provider_polza_normalizes_to_google_prefix`
  - `test_default_openrouter_adapter_unaffected` (regression)
- [ ] `agent_ng/vision_adapters/openrouter_adapter.py` — parameterise `provider`
- [ ] `agent_ng/vision_tool_manager.py` — register slot, read AGENT_PROVIDER, extend routing
- [ ] `.env.example` — document AGENT_PROVIDER effect on VL
- [ ] Lint (`ruff check` on changed files)
- [ ] All tests green

## Verification

```bash
python -m pytest agent_ng/_tests/test_vl_integration.py -v
ruff check agent_ng/vision_adapters/openrouter_adapter.py agent_ng/vision_tool_manager.py
```
