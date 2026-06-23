# Plan: Sidebar column + LLM controls in Config tab

## Scope

- Restore **persistent left sidebar column** (not a tab): quick actions, progress, token budget, status — **same behavior as current post-`4bc927b` wiring**.
- Move **LLM selection block** (provider/model dropdown, fallback checkbox + dropdown, compression checkbox) from sidebar tab into **Config tab**, merged visually under **«Выбор LLM»** row alongside existing **«Подключение к LLM»** (API keys table).
- Preserve **event handlers** (`Sidebar._connect_sidebar_events`, `connect_quick_action_dropdown`, `ui_manager` chat tails, `demo.load` sync) — only **relocate components**.

## TDD checkpoints

1. **Unit:** `ConfigTab` save/load/clear handlers receive trailing BrowserState plus N LLM key inputs **and** the relocated LLM controls (signature length stable).
2. **Smoke import:** `Sidebar.create_sidebar_column` + `ConfigTab` with dummy sidebar mount does not raise when `CMW_USE_DOTENV=true` (config hidden — stubs only).

## Implementation tasks

1. Refactor `Sidebar`: extract `_mount_llm_section(parent_ctx)` + `_connect_llm_events()`; add `create_sidebar_column()` returning `(components)` without `TabItem`; keep `create_tab()` thin or remove usage.
2. `ConfigTab`: optional `sidebar_instance`; call `sidebar_instance.mount_llm_into_config(parent)` before platform URL block; extend save/load/clear `inputs`/`outputs`.
3. `UIManager`: `gr.Row` → left `Sidebar.create_sidebar_column()`, right `gr.Column` + `gr.Tabs`; drop sidebar-as-tab block; pass `sidebar_instance` into `ConfigTab` via closure or `set_sidebar_for_config` hook.
4. `app_ng_modular`: after building `ConfigTab`, `config_tab.set_sidebar_instance(sidebar_ref)` if needed.
5. Docs: update `AGENTS.md` operational UI note (sidebar column + LLM in config).

## Verification

```bash
.venv/bin/python -m pytest agent_ng/_tests/test_config_tab_llm_mount.py -q
.venv/bin/ruff check agent_ng/tabs/sidebar.py agent_ng/tabs/config_tab.py agent_ng/ui_manager.py agent_ng/app_ng_modular.py
```
