---
name: cmw-platform-staff-account-link
description: >-
  Links an existing platform login to a Staff (Sotrudniki) account-template employee
  row on Comindware Platform. Use when attaching accounts, Staff, Sotrudniki, account
  template records, John Demonstrator, employee login link, Phase 0 staff personas,
  or mz-fr/mz-tr employee seeding. Instance-agnostic via CMW_BASE_URL. Prefer OpenAPI
  and agent tools before browser; UI Attach account modal is last resort on mz-fr when
  IncludeInContainer returns 500.
---

# CMW Platform — Staff account link

Link a **platform security account** (`account.N`) to an **employee row** on the Volga **Staff** account template (`Sotrudniki`, template id often `aa.2` on a host — resolve per instance).

**Workflow order:** OpenAPI (`cmw_open_api/`) → agent `tools/` → browser (last resort). See [cmw-platform](../cmw-platform/SKILL.md) § Workflow order.

**Do not** rename usernames during attach. Account creation is [cmw-platform-account-bootstrap](../cmw-platform-account-bootstrap/SKILL.md).

**Instance migration progress (Volga TR→FR):** log outcomes in **my-building** — [`docs/20260519_migration_status_and_roadmap.md`](file:///D:/Repo/my-building/docs/20260519_migration_status_and_roadmap.md), `localization/migration_progress/` (not this repo).

## Discovery (find the link field)

1. Switch host: [cmw-platform-instance-switch](../cmw-platform-instance-switch/SKILL.md).
2. Open a **linked** reference row (e.g. John Demonstrator / `demo` on FR) and an **unlinked** row in the same template.
3. Compare payloads:

| Source | Endpoint | Linked signal |
|--------|----------|----------------|
| Records list | `GET webapi/Records/AccountTemplate@{App}.Sotrudniki` | `username` non-empty; `fullName` / `mbox` populated from account profile |
| Single record | `GET webapi/Record/{id}` | Same fields (`username`, `fullName`, …) |
| TeamNetwork | `POST api/public/system/TeamNetwork/ObjectService/Get` body `{"objectId":"<id>","accountTemplateId":"aa.2"}` | `cmw.account.username`, `cmw.account.fullName`, … |

**Field names (verified on Volga / Sotrudniki):**

| Layer | System name | Type | Example (John Demonstrator / demo) |
|-------|-------------|------|-----------------------------------|
| Web API Records | `username` | string | `demo` (when linked) |
| TeamNetwork Get | `cmw.account.username` | string | `demo` |
| Display | `fullName` / `cmw.account.fullName` | string | `John Demonstrator` |

**Employee record id:** On mz-fr Phase 0 rows, list/API ids are **numeric strings** (`182`, `183`, …), not `account.182`. Form/card links may use `#form/.../account.{N}` only after attach. Do not assume progress JSON `account.182` works as `webapi/Record/account.182`.

Attribute metadata: `GET webapi/Attribute/List/Template@{App}.Sotrudniki` → `cmw_account_username` (system string; `create_edit_record` skips system fields).

## Read path (verify link)

```http
GET webapi/Records/AccountTemplate@Volga.Sotrudniki
```

Row is linked when `username` (or list column **Full name** / account profile fields) is populated.

Optional:

```http
POST api/public/system/TeamNetwork/ObjectService/GetPropertyValues
```

Body: `objects: ["183"]`, `propertiesByAlias: ["username", "cmw_account_username"]`.

## Write path (API — try in order)

### 1. Direct record PUT (works only when uniqueness allows)

```http
PUT webapi/Record/{employeeId}
Content-Type: application/json

{"username": "engineer"}
```

- `{employeeId}` = numeric id (`183`), not `account.183`.
- On mz-fr, may return **200** with error: `triple uniqueness - {id} cmw.account.username {value}` if the login is already reserved in the template without a proper Include.
- `create_edit_record` **does not** set `cmw_account_username` (system field skipped).

### 2. TeamNetwork Include (per-row — preferred in OpenAPI, blocked on mz-fr)

```http
POST api/public/system/TeamNetwork/ObjectService/IncludeInContainer
```

```json
{"accountId": "account.4", "containerId": "183"}
```

OpenAPI: `system_core_api.json` → `TeamNetworkObjectServiceIncludeInContainerParameters`.

**mz-fr (2026-05):** HTTP **500** for `containerId` = numeric employee id, `account.{id}`, or `aa.{id}`. Do not use as primary on FR until fixed.

### 3. IncludeInContainer1 (template-level only)

```http
POST api/public/system/TeamNetwork/ObjectService/IncludeInContainer1
```

```json
{"accountIds": ["account.3", "account.4"], "accountTemplateId": "aa.2"}
```

Returns **200** but does **not** set `username` on pre-created empty rows `182`–`191`; may surface accounts in filtered Staff lists (UI **Full name** from account profile).

## Write path (UI — last resort on mz-fr)

1. Employees / Staff list: `#data/aa.2/lst.{M}` (verify `lst` id on host; FR often `lst.279`).
2. Select employee row → toolbar **Attach account** (user command kind **Include**).
3. Modal: search platform login → confirm.

Observed SPA calls:

- `POST /User/GetAccountsToInclude`
- `POST /UserCommandExecution/PerformUserAction`

Capture payload from browser network if automating; not in `web_api_v1.json` snapshot in repo.

→ UI details: [employee_account_attach.md](../cmw-platform/references/employee_account_attach.md)

## Phase 0 persona table (FR reference)

| Username | Platform account id | Employee row id (API) |
|----------|---------------------|------------------------|
| dispatcher01 | account.3 | 182 |
| engineer | account.4 | 183 |
| techniktest | account.5 | 184 |
| clean_manager | account.8 | 185 |
| manager | account.9 | 186 |
| ingeneer | account.10 | 187 |
| expluatation | account.12 | 188 |
| smirnova | account.13 | 189 |
| serov | account.14 | 190 |
| isaeva | account.15 | 191 |

## Related

- [cmw-platform](../cmw-platform/SKILL.md)
- [employee_account_attach.md](../cmw-platform/references/employee_account_attach.md)
- [cmw-platform-backup-launch](../cmw-platform-backup-launch/SKILL.md) — backup after a successful attach batch
