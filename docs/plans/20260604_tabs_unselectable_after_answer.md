# Tabs Unselectable After Answer Plan

## Goal

Find and fix the Gradio 6.10 UI state that makes tabs unselectable after the agent finishes an answer.

## Tasks

1. Reproduce in the live `.venv64` app with agent-browser.
2. Inspect DOM state after completion for stuck overlays, disabled controls, selected tabs, or active loading classes.
3. Check backend logs for callback output mismatches or event-chain exceptions.
4. Apply the smallest fix that preserves the existing Gradio 6 queue/event discipline.
5. Add/adjust focused tests where the behavior is testable without a browser.
6. Restart the app and verify tab selection after an answer.

## Verification

```powershell
.venv64\Scripts\python.exe -m pytest agent_ng/_tests/test_config_tab_use_dotenv.py agent_ng/_tests/test_artifacts_export.py
```

Manual browser verification:

1. Send a short chat prompt.
2. Wait for completion.
3. Click each tab and confirm selection changes.
