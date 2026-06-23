---
name: cmw-platform-backup-launch
description: >-
  Launch an existing Comindware Platform configuration backup via Web API
  (preferred) or admin UI fallback. Use when the user or migration plan requires
  a post-change backup, configuration backup, FR rollback snapshot, or
  after-batch safety backup between major themed migration edits. Agnostic for any
  instance via CMW_BASE_URL. Does not create backup configurations unless the
  user explicitly asks.
---

# CMW Platform — Launch configuration backup (API preferred)

Start an **existing** configuration backup on any Comindware instance.

**Workflow order:** OpenAPI (`cmw_open_api/web_api_v1.json`) → agent tools / HTTP → **browser last resort** (admin UI). Same order as [cmw-platform](../cmw-platform/SKILL.md) § Workflow order.

**Prefer the Web API** (`Backup_CreateSession`); use **browser automation** only when API is unavailable or the user requests UI.

**Do not use** backup/restore to clone data between instances; this skill is for **rollback snapshots** on the target host only.

**Instance migration progress:** record `meta.fr_backup_*` in `{instance_progress_dir}/localization/migration_progress/` and the instance roadmap doc — not in cmw-platform-agent.

**OpenAPI source of truth:** [`cmw_open_api/web_api_v1.json`](../../../cmw_open_api/web_api_v1.json) — operations `Backup_ListConfigurations`, `Backup_CreateSession`, `Backup_GetSession`, `Backup_ListSessions`.

## When to run

Run a configuration backup **after meaningful batches**, not after every single field change.

| Situation | Action |
|-----------|--------|
| **Between major themed migration edits** — accounts/groups batch, employee ↔ account linking, a record phase (e.g. all Phase 0 roles, a Phase 1 template group, Phase 2 hub) | **Required** — launch backup before starting the next theme |
| Multi-change batch (`meta.changes_in_batch` ≥ 2, multi-entity Phase 0, Phase 2 hubs) | **Required** after commit, per project master plan |
| Single small change (one UAT account, one attribute tweak) | Often `recommended_not_required` — follow project plan; document in `meta.notes` |
| User says "take a backup" / "run configuration backup" | Run this skill |

**Cadence rule:** Think in **migration themes** (security, accounts, employees, record phases) — one backup checkpoint between themes, not per tool call or per row.

## Configuration

Load from `cmw-platform-agent/.env` with `CMW_USE_DOTENV=true`. **Never log or commit** secrets.

Wrong host → [cmw-platform-instance-switch](../cmw-platform-instance-switch/SKILL.md) (`CMW_BASE_URL`, verify) before any backup call or UI.

| Variable | Purpose |
|----------|---------|
| `CMW_BASE_URL` | Target instance root, e.g. `https://{your-host}/` (trailing slash optional) |
| `CMW_LOGIN` / `CMW_PASSWORD` | HTTP Basic auth for Web API |

## Prohibitions

1. **Do NOT create new backup configurations** (`POST /webapi/Backup/Configuration`, UI **Add/New**) unless the user **explicitly** asks.
2. Use **existing** configurations only (e.g. display name **Backup по умолчанию** / default row on that host).
3. **Do NOT delete** configurations or sessions (`DELETE` endpoints, UI **Delete**) unless the user explicitly asks.
4. Do not use backup/restore to copy data between TR and FR hosts.

## Preferred: API workflow

Auth: **HTTP Basic** with `CMW_LOGIN` and `CMW_PASSWORD`. Base URL: `{CMW_BASE_URL}` (normalize: strip trailing slash before appending paths).

```text
Verify CMW_BASE_URL → GET configurations → pick existing configurationId → POST create session → (optional) poll session until terminal status
```

### 1. Instance and credentials

1. Confirm [cmw-platform-instance-switch](../cmw-platform-instance-switch/SKILL.md) — correct `CMW_BASE_URL` for the target (e.g. FR during migration).
2. Load `.env`; never echo passwords in logs or progress JSON.

### 2. List configurations

- **GET** `{CMW_BASE_URL}/webapi/Backup/Configuration`
- **OperationId:** `Backup_ListConfigurations`
- Response: `WebApiResponse` with a list of `BackupConfigurationModel` — each item has `id`, `description`, `fileName`, flags (`withStreams`, `withScripts`, …).

Pick an **existing** row (commonly the default **Backup по умолчанию**). Record `configurationId` = model `id`.

### 3. Create backup session

- **POST** `{CMW_BASE_URL}/webapi/Backup/Session/{configurationId}`
- **OperationId:** `Backup_CreateSession`
- Path parameter `configurationId` = id from step 2 (not the display name).
- Response: `BackupSessionModel` with `id` (session id), `sessionStatus`, timestamps.

### 4. Optional: poll until complete

- **GET** `{CMW_BASE_URL}/webapi/Backup/Session/{sessionId}`
- **OperationId:** `Backup_GetSession`
- Poll until `sessionStatus` is terminal: `Completed`, `Failed`, `Aborted`, or other failure enum (see OpenAPI `BackupSessionModel.sessionStatus`).
- For history/audit: **POST** `{CMW_BASE_URL}/webapi/Backup/Session` — `Backup_ListSessions` with filter body per `BackupFilterModel`.

### Example (curl — placeholders only)

```bash
export CMW_BASE_URL="https://example-fr.test/"
export CMW_LOGIN="admin@example.test"
export CMW_PASSWORD="your-secret-from-env"

# List configurations
curl -sS -u "${CMW_LOGIN}:${CMW_PASSWORD}" \
  "${CMW_BASE_URL%/}/webapi/Backup/Configuration"

# Create session (replace CONFIGURATION_ID from list response)
curl -sS -u "${CMW_LOGIN}:${CMW_PASSWORD}" -X POST \
  "${CMW_BASE_URL%/}/webapi/Backup/Session/CONFIGURATION_ID"

# Poll session (replace SESSION_ID from create response)
curl -sS -u "${CMW_LOGIN}:${CMW_PASSWORD}" \
  "${CMW_BASE_URL%/}/webapi/Backup/Session/SESSION_ID"
```

### Example (Python — env vars, no secrets in code)

```python
import os
import time
import requests
from requests.auth import HTTPBasicAuth

base = os.environ["CMW_BASE_URL"].rstrip("/")
auth = HTTPBasicAuth(os.environ["CMW_LOGIN"], os.environ["CMW_PASSWORD"])

configs = requests.get(f"{base}/webapi/Backup/Configuration", auth=auth, timeout=60)
configs.raise_for_status()
body = configs.json()
items = body.get("result") or body.get("data") or body  # unwrap WebApiResponse if needed
configuration_id = items[0]["id"]  # prefer known default by description/fileName in real runs

session = requests.post(
    f"{base}/webapi/Backup/Session/{configuration_id}",
    auth=auth,
    timeout=60,
)
session.raise_for_status()
session_id = session.json().get("result", session.json()).get("id")

terminal = {"Completed", "Failed", "Aborted"}
for _ in range(120):
    st = requests.get(f"{base}/webapi/Backup/Session/{session_id}", auth=auth, timeout=60)
    st.raise_for_status()
    status = st.json().get("result", st.json()).get("sessionStatus")
    if status in terminal:
        break
    time.sleep(5)
```

Adapt response unwrapping to the instance’s `WebApiResponse` shape (`result` vs nested `data`).

**Maintained CLI:** [backup_configuration_session.py](../cmw-platform/scripts/backup_configuration_session.py) — list configs, create session, `--poll`. Index: [scripts_index.md](../cmw-platform/references/scripts_index.md).

### API verify / logging

- Success: `sessionStatus` → `Completed` (or job accepted if you stop after POST when project plan only requires launch).
- Log in migration project JSON (no secrets): `meta.fr_backup_status`, `meta.fr_backup_launched`, optional `meta.fr_backup_configuration_name`, `meta.fr_backup_session_id`.

## Fallback: UI workflow

Use when API returns errors, auth blocks automation, or the user insists on UI verification.

**Deep link:** `{CMW_BASE_URL}#Settings/Backup/Configurations`

Typical page: **Administration** → **Backup** → tab **Configurations**. Grid: checkbox column, **ID**, **Name**, … Rows often include **Backup по умолчанию**.

```text
Navigate → Login (if needed) → Configurations tab → Checkbox on existing row → Start backup visible → Start backup → Verify acknowledged
```

1. Open Configurations tab (`browser_lock` / snapshot per MCP rules).
2. **Check the checkbox** on one **existing** row (not only row highlight).
3. **Re-snapshot** — toolbar shows **Start backup** (play icon); localized labels may differ; older docs say "Run".
4. Click **Start backup**; confirm dialog only when starting backup.
5. Verify: progress, toast, **Log** tab entry, or row status change.

| Tool | When |
|------|------|
| **cursor-ide-browser** | Default in Cursor |
| **agent-browser** | Headless/scripted outside IDE |

Reference layout (maintainers): `assets/c__Users_ased_AppData_Roaming_Cursor_User_workspaceStorage_empty-window_images_image-1b10a211-6af1-4c18-8490-ef4d83eafe46.png` — agents use live snapshots at runtime.

## Project-repo logging

Progress files (e.g. `localization/migration_progress/*.json`) live in the **migration project repo**, not cmw-platform-agent. After backup:

| Field | Meaning |
|-------|---------|
| `meta.fr_backup_status` | `required`, `launched`, or `recommended_not_required` |
| `meta.fr_backup_launched` | `true` when session created / backup acknowledged |
| `meta.fr_backup_url` | Configurations deep link (UI) or note `api:Backup_CreateSession` |
| `meta.fr_backup_configuration_name` | Display name of existing config used |
| `meta.fr_backup_session_id` | Optional API session id |

## Related skills

- [cmw-platform-instance-switch](../cmw-platform-instance-switch/SKILL.md) — correct host before backup
- [cmw-platform-account-bootstrap](../cmw-platform-account-bootstrap/SKILL.md) — account batches; points here after themed account work
- [cmw-platform](../cmw-platform/SKILL.md) — general platform operations

## Maintaining this skill

When OpenAPI or UI changes, update API paths/operationIds from `cmw_open_api/web_api_v1.json` first; adjust UI checkbox → **Start backup** steps only as needed. Keep **no new configurations** and **no Delete** unless product policy changes.
