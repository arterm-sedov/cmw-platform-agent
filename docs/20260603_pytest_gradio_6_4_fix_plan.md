# Pytest fix plan — branch `20250124_gradio_6_4`

## Failures (target)

| Area | Fix strategy |
|------|----------------|
| `test_langchain_agent` | `pyproject.toml` pytest-asyncio `asyncio_mode=auto` + `@pytest.mark.asyncio` |
| `test_browser_automation` | Patch `playwright.async_api.async_playwright` |
| `test_config_tab_llm_override` | Patch `CMW_USE_DOTENV=false` in save passthrough test |
| `test_error_handling_improvements` | Align `test_no_answer_scenarios` with `ensure_valid_answer` strip semantics |
| `test_image_engine` flux | Set `supports_image_config=False` on flux.2-flex/pro; update `test_image_models` |
| Image sizing flags | `supports_image_config` = OpenRouter `image_config` only; `supports_polza_sizing` = Polza `aspect_ratio`/`quality`; Flux: OR false, Polza true |

### Image sizing decouple (2026-06-03)

- **OpenRouter:** `supports_image_config` gates `image_config` in `OpenRouterProvider._build_image_config`.
- **Polza:** `supports_polza_sizing` gates `aspect_ratio` / `quality` in `PolzaProvider._build_payload`.
- **Flux 2 flex/pro:** `supports_image_config=False`, `supports_polza_sizing=True` (Polza docs; OR has no `image_config`).
- **Verify:** `pytest agent_ng/_tests/test_image_engine.py agent_ng/_tests/test_image_models.py -v`
| `test_langsmith_integration` | Match `cmw-agent` default; remove obsolete API tests |
| `test_logging_rotation` | `pip install concurrent-log-handler` (already in `requirements.txt`) |
| `test_vl_integration` audio | Mock `VisionProviderAdapter.invoke` — no live OpenRouter |
| `test_platform_tools` | Rename helper `test_tool` → `run_tool_check` |
| `test_streaming_agent_behavior` | Accept `turn_complete` text events (Gradio 6 streaming contract) |

## TDD / verification

```powershell
.venv\Scripts\Activate.ps1
pip install concurrent-log-handler==0.9.29
python -m pytest agent_ng/_tests/ -v --tb=short -q
ruff check <modified files>
```

## Checkpoint

- All listed tests green; full suite exit code 0 (or document env-only skips).
