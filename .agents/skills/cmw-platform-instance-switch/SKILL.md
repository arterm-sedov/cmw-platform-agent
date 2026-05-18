---
name: cmw-platform-instance-switch
description: >-
  Switch Comindware Platform target instance (host/tenant) via CMW_BASE_URL and
  dotenv. Use when the user asks to switch instance, change host, connect to a
  different server, update CMW_BASE_URL, work on mz-tr vs mz-fr, edit .env for
  another tenant, or verify credentials before bulk platform work. Agnostic for
  any instance pair. Does not duplicate general platform CRUD — see
  cmw-platform skill.
---

# CMW Platform — Instance switching

Point scripts, tools, and browser automation at a **different** Comindware host without changing workflows. Pair examples: reference TR vs target FR (`mz-tr` / `mz-fr` placeholders only — use your real hosts in `.env`).

**Related:** [cmw-platform](../cmw-platform/SKILL.md) (operations), [cmw-platform-account-bootstrap](../cmw-platform-account-bootstrap/SKILL.md), [cmw-platform-backup-launch](../cmw-platform-backup-launch/SKILL.md).

## When to use

- User says switch instance, change host, other server, different tenant
- Compare two instances (schema, accounts, localization) — **read-only first**
- Before a migration batch: confirm you are on the **intended** target
- After editing `.env` / `CMW_BASE_URL` — re-verify connection

**Out of scope:** instance-specific progress JSON, TR→FR matrices, inventories — keep in the owning project repo (e.g. **my-building** `localization/migration_progress/`), not in `cmw-platform-agent`.

## Configuration (never commit secrets)

Edit `cmw-platform-agent/.env` locally. Document placeholders only in `.env.example`:

| Variable | Per instance? | Notes |
|----------|---------------|--------|
| `CMW_BASE_URL` | **Yes** | Root URL with trailing slash, e.g. `https://{your-host}/` |
| `CMW_LOGIN` / `CMW_PASSWORD` | Often same, may differ | Confirm auth on **each** host after switch |
| `CMW_USE_DOTENV` | Repo default | `.env.example` uses `false` for Gradio Config tab — **override for one-off scripts** |
| `CMW_TIMEOUT` | Usually shared | Optional |

**Never** log passwords, commit `.env`, or paste real hosts/credentials into skills, commits, or chat artifacts.

## Switch workflow

```text
Set CMW_BASE_URL → load dotenv (override) → CMW_USE_DOTENV=true → verify (read-only) → proceed
```

### 1. Set target instance

In `.env` (or session env for a single run):

```env
CMW_BASE_URL=https://{target-host}/
```

Trailing slash is conventional; tools typically normalize with `rstrip("/")`.

Optional: different `CMW_LOGIN` / `CMW_PASSWORD` if the target uses other API users.

### 2. Scripts and one-off Python

Gradio may run with `CMW_USE_DOTENV=false` (Config tab). **One-off scripts** must force dotenv:

```python
import os
from dotenv import load_dotenv

load_dotenv(override=True)  # pick up edited .env
os.environ["CMW_USE_DOTENV"] = "true"

from tools.applications_tools.tool_list_applications import list_applications

result = list_applications.invoke({})
```

PowerShell (same session):

```powershell
$env:CMW_USE_DOTENV = "true"
python .agents/skills/cmw-platform/scripts/diagnose_connection.py
```

### 3. Verify before bulk work

Run **one** lightweight check; do not start migrations until it passes.

| Check | Command / action |
|-------|------------------|
| Config + auth | `python .agents/skills/cmw-platform/scripts/diagnose_connection.py` (exit `0`) |
| Tool path | `list_applications.invoke({})` → `success: true` |
| Browser | Navigate to `{CMW_BASE_URL}`; login if prompted; confirm expected instance (title, admin URL) |

On **401**: fix credentials for **this** host, not the previous instance.

### 4. Comparing two instances

1. **Read-only** on both: list apps/templates, export schemas, diff — no writes until target is confirmed.
2. Switch `CMW_BASE_URL` + verify between sides; do not assume record IDs or UI entity IDs (`oa.{N}`) match across hosts.
3. Write batches only on the **target** after explicit user/plan approval.
4. Post-change backup on target → [cmw-platform-backup-launch](../cmw-platform-backup-launch/SKILL.md).

## What changes vs what stays

| Changes per instance | Stays the same (typical) |
|----------------------|---------------------------|
| `CMW_BASE_URL`, browser base URL, backup deep links | API path patterns (`webapi/...`, `api/public/system/...`) |
| Display names (EN vs RU UI), localized app titles | Application **system name** / alias (e.g. `Volga`) |
| Record IDs, template instance IDs, hash-route entity IDs | Tool names and invoke patterns in `tools/` |
| Host-specific seed data, accounts, groups | Skill workflows (account bootstrap, backup launch) |
| Project progress JSON path (see below) | `cmw-platform-agent` skills and agent code |

**Volga note:** The solution **display name** may be Russian or English per instance; the **system name** alias (`Volga`) is what tools use — always resolve via `list_applications` after a switch.

## Repo boundaries

| Artifact | Repository |
|----------|------------|
| Skills, `diagnose_connection.py`, generic tools | `cmw-platform-agent` |
| `migration_progress/*.json`, inventories, TR→FR matrices | **my-building** (or other migration project) |
| Per-instance backup launch notes (`fr_backup_*` meta) | Migration project progress files |

When switching hosts, update **project** progress `meta` with the active `CMW_BASE_URL` (host only, no secrets) — not committed platform-agent docs.

## Checklist

- [ ] `CMW_BASE_URL` points to the intended host (reference vs target)
- [ ] `load_dotenv(override=True)` and `CMW_USE_DOTENV=true` for scripts
- [ ] `diagnose_connection.py` or `list_applications` succeeds
- [ ] For comparisons: read-only pass complete before writes
- [ ] Migration logs updated in **my-building**, not this repo

## Maintaining this skill

Update only instance-switch mechanics (env, dotenv override, verify, compare policy). Keep CRUD, browser, and backup detail in **cmw-platform** and sibling skills.
