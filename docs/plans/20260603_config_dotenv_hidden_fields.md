# Config Dotenv Hidden Fields Plan

## Goal

Fix Gradio 6.10 UI startup when `CMW_USE_DOTENV=true`.

## Finding

`ConfigTab._create_config_interface()` does not create `platform_url`, `username`, or `password` components in dotenv mode, but `_connect_events()` always wires those components. This raises `KeyError: 'platform_url'` and causes the app to serve the initialization fallback UI.

## Tasks

1. Keep existing dotenv notice behavior.
2. Always create the platform credential components.
3. Hide those components when dotenv mode is active.
4. Add a focused test for dotenv-mode component presence.
5. Run focused tests and restart the single `.venv64` UI instance.

## Verification

```powershell
python -m pytest agent_ng/_tests/test_config_tab_use_dotenv.py agent_ng/_tests/test_artifacts_export.py
ruff check agent_ng/tabs/config_tab.py agent_ng/_tests/test_config_tab_use_dotenv.py
```
