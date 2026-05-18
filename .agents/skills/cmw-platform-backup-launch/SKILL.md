---
name: cmw-platform-backup-launch
description: >-
  Launch an existing Comindware Platform configuration backup from the admin UI
  (Settings → Backup → Configurations). Use when the user or migration plan
  requires a post-change backup, configuration backup, FR rollback snapshot,
  Settings/Backup, run backup, or after-batch safety backup. Agnostic for any
  instance via CMW_BASE_URL. Does not create backup configurations unless the
  user explicitly asks.
---

# CMW Platform — Launch configuration backup (UI)

Start an **existing** configuration backup on any Comindware instance. There is **no** reliable public API documented in `cmw_open_api` for starting configuration backups — use **browser automation** (cursor-ide-browser MCP or agent-browser).

**Do not use** backup/restore to clone data between instances; this skill is for **rollback snapshots** on the target host only.

## UI layout (Configurations tab)

Typical page: breadcrumb **Administration** → title **Backup**; tabs **Configurations** (active) and **Log**. The configurations grid has a **checkbox** column, then **ID**, **Name**, and other columns. Rows are often named like **Backup по умолчанию** (default backup configuration) — use an **existing** row; do not add a new one.

**Toolbar after selection:** Checking a row’s checkbox enables toolbar actions. The primary action is **Start backup** (play icon) — **not** labeled “Run” on current builds. **Delete** may also appear when a row is selected; **do not** click Delete unless the user explicitly asks to remove a configuration.

Reference screenshot (workspace, for maintainers): `assets/c__Users_ased_AppData_Roaming_Cursor_User_workspaceStorage_empty-window_images_image-1b10a211-6af1-4c18-8490-ef4d83eafe46.png` — describes the above layout; agents should rely on live snapshots, not the image file at runtime.

## Configuration

Load from `cmw-platform-agent/.env` with `CMW_USE_DOTENV=true`. **Never log or commit** secrets.

Wrong host → [cmw-platform-instance-switch](../cmw-platform-instance-switch/SKILL.md) (`CMW_BASE_URL`, verify) before opening backup UI.

| Variable | Purpose |
|----------|---------|
| `CMW_BASE_URL` | Target instance root, e.g. `https://{your-host}/` |
| `CMW_LOGIN` / `CMW_PASSWORD` | Credentials for UI login if prompted |

Backup URL (typical): `{CMW_BASE_URL}#Settings/Backup/Configurations`

## Critical rules

1. **Do NOT create new backup configurations** unless the user **explicitly** asks to add one.
2. Use **existing** entries in the configurations list only (e.g. default row **Backup по умолчанию**).
3. **Selection is via checkbox** — check the box next to one **existing** configuration row first.
4. **Start backup is hidden until a row is checked** — the **Start backup** control does not appear in the toolbar until at least one configuration checkbox is selected. Do not search for Start backup before checking a box; take a fresh snapshot after the checkbox click.
5. Do not click Add / New / **Create** configuration controls unless the user requested a new config.
6. Do not click **Delete** unless the user explicitly asks to remove a backup configuration.

## Workflow (browser)

```text
Navigate → Login (if needed) → Configurations tab → Checkbox on existing row → Start backup visible → Start backup → Verify acknowledged
```

### 1. Open Configurations

1. Set browser to `{CMW_BASE_URL}#Settings/Backup/Configurations` (or navigate Settings → Backup → **Configurations** tab).
2. `browser_lock` after navigation if using cursor-ide-browser; take `browser_snapshot` before interacting.

### 2. Select existing configuration (checkbox first)

1. In the configurations list, locate an **existing** backup row (e.g. **Backup по умолчанию** / default backup for that host).
2. **Check the checkbox** in the first column on that row (not merely highlighting the row or clicking the name).
3. **Re-snapshot** — confirm the row is selected and the toolbar now shows **Start backup** (play icon). Labels may be localized (e.g. RU UI on an EN host); older docs may say “Run” — treat **Start backup** as the correct control. If it is still missing, the checkbox was not toggled; try the row checkbox again (grid cells may not expose `role=checkbox` in accessibility trees — use coordinates from a fresh screenshot if needed).

### 3. Start backup (only after checkbox)

1. Click **Start backup** in the toolbar. It is **not** available before step 2.
2. Confirm any benign confirmation dialog if shown (accept only when intent is to start backup).

### 4. Verify

- UI shows the job was accepted: progress indicator, success toast, status change on the row, or entry on the **Log** tab — **confirm Start backup was acknowledged**.
- If the UI gives no clear signal, note in project progress: `fr_backup_launched: true` with `fr_backup_note` describing what was observed.
- Record optional `fr_backup_configuration_name` (display name of the **existing** config used, e.g. `Backup по умолчанию`) in project progress JSON — not secrets.

## MCP preference

| Tool | When |
|------|------|
| **cursor-ide-browser** | Default in Cursor: snapshot → checkbox → re-snapshot (Start backup visible) → Start backup |
| **agent-browser** | Headless/scripted runs outside IDE |

Follow each MCP server's lock/navigate/snapshot rules; re-snapshot after every click.

## Project-repo logging

Instance-specific progress files (e.g. `localization/migration_progress/*.json`) live in the **migration project repo**, not cmw-platform-agent. After backup:

- `meta.fr_backup_status`: `required` or `launched`
- `meta.fr_backup_launched`: `true` when Start backup succeeded
- `meta.fr_backup_url`: backup Configurations deep link
- Optional: `meta.fr_backup_configuration_name` (existing config display name)

## When to run

| Situation | Action |
|-----------|--------|
| Multi-change batch (several accounts, roles, records) | **Required** after commit, per project master plan |
| Single small change | Often `recommended_not_required` — follow project plan |
| User says "take a backup" / "run configuration backup" | Run this skill |

## Related skills

- [cmw-platform-account-bootstrap](../cmw-platform-account-bootstrap/SKILL.md) — account create/update; points here after batch account work
- [cmw-platform](../cmw-platform/SKILL.md) — general platform operations

## Maintaining this skill

When CMW backup UI changes (control labels, layout), update the checkbox → **Start backup** steps only; keep the **no new configurations** and **no Delete** rules unless product policy changes.
