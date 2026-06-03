# Download Artifacts ZIP Plan

## Goal

Add a third Downloads tab button that provides a ZIP package containing all files registered in the current session agent file registry.

## References

- Existing Downloads tab exposes `gr.DownloadButton` components and receives updates from `UIManager`.
- Existing generated/uploaded/fetched files are tracked through `CmwAgent.file_registry`.
- Gradio `DownloadButton` accepts a single local file path, so a ZIP file matches the current download model.
- Gradio file access should stay scoped to generated temp/cache files; do not scan arbitrary temp directories.

## Scope

- Include only current-session file registry entries whose paths still exist and are regular files.
- Preserve logical registered names inside the ZIP under `artifacts/`.
- Add `manifest.json` with logical name, ZIP path, source path, size, and SHA-256.
- Hide the ZIP button when no registered files are available.
- Clear/hide the ZIP button when chat is cleared.

## Tasks

1. Add a small artifact export helper with registry filtering, safe ZIP names, collision handling, manifest generation, and ZIP creation.
2. Add focused tests for empty registry, session filtering, missing files, duplicate names, and manifest content.
3. Add a `Download artifacts (ZIP)` translation key.
4. Add the third button to `DownloadsTab`.
5. Wire the third button in `UIManager` using the current Gradio request/session.
6. Run focused tests and lint for modified Python files.

## Verification Commands

```powershell
python -m pytest agent_ng/_tests/test_artifacts_export.py
ruff check agent_ng/artifacts_export.py agent_ng/tabs/downloads_tab.py agent_ng/tabs/chat_tab.py agent_ng/ui_manager.py agent_ng/i18n_translations.py agent_ng/_tests/test_artifacts_export.py
```
