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

Run an **existing** configuration backup on any Comindware instance. There is **no** reliable public API documented in `cmw_open_api` for starting configuration backups — use **browser automation** (cursor-ide-browser MCP or agent-browser).

**Do not use** backup/restore to clone data between instances; this skill is for **rollback snapshots** on the target host only.

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
2. Use **existing** entries in the configurations list only.
3. **Selection is via checkbox** — check the box next to one **existing** configuration row first.
4. **Run is hidden until a row is checked** — the **Run** control does not appear in the toolbar until at least one configuration checkbox is selected. Do not search for Run before checking a box; take a fresh snapshot after the checkbox click.
5. Do not click Add / New / **Create** configuration controls unless the user requested a new config.

## Workflow (browser)

```text
Navigate → Login (if needed) → Backup/Configurations → Checkbox on existing row → Run visible → Run → Verify acknowledged
```

### 1. Open Configurations

1. Set browser to `{CMW_BASE_URL}#Settings/Backup/Configurations` (or navigate Settings → Backup → Configurations).
2. `browser_lock` after navigation if using cursor-ide-browser; take `browser_snapshot` before interacting.

### 2. Select existing configuration (checkbox first)

1. In the configurations list, locate an **existing** backup row (project docs may name a preferred config; otherwise pick the standard/default configuration already on the instance, e.g. default/manual backup for that host).
2. **Check the checkbox** in the first column on that row (not merely highlighting the row or clicking the name).
3. **Re-snapshot** — confirm the row is selected and the toolbar now shows **Run** (English UI). If **Run** is still missing, the checkbox was not toggled; try the row checkbox again (grid cells may not expose `role=checkbox` in accessibility trees — use coordinates from a fresh screenshot if needed).

### 3. Run (only after checkbox)

1. Click **Run** in the toolbar (label may be localized; English UI: **Run**). It is **not** available before step 2.
2. Confirm any benign confirmation dialog if shown (accept only when intent is to start backup).

### 4. Verify

- UI shows the job was accepted: progress indicator, success toast, status change on the row, or backup history entry — **confirm Run was acknowledged**.
- If the UI gives no clear signal, note in project progress: `fr_backup_launched: true` with `fr_backup_note` describing what was observed.
- Record optional `fr_backup_configuration_name` (display name of the **existing** config used) in project progress JSON — not secrets.

## MCP preference

| Tool | When |
|------|------|
| **cursor-ide-browser** | Default in Cursor: snapshot → checkbox → re-snapshot (Run visible) → Run |
| **agent-browser** | Headless/scripted runs outside IDE |

Follow each MCP server's lock/navigate/snapshot rules; re-snapshot after every click.

## Project-repo logging

Instance-specific progress files (e.g. `localization/migration_progress/*.json`) live in the **migration project repo**, not cmw-platform-agent. After backup:

- `meta.fr_backup_status`: `required` or `launched`
- `meta.fr_backup_launched`: `true` when Run succeeded
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

When CMW backup UI changes (control labels, layout), update the checkbox → **Run** steps only; keep the **no new configurations** rule unless product policy changes.
