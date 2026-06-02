---
name: cmw-platform-staff-account-link
description: >-
  Links an existing platform login to a Staff (Sotrudniki) account-template employee
  row on Comindware Platform. Use when attaching accounts, Staff, Sotrudniki, account
  template records, employee login link, or UAT persona attach after account bootstrap.
  Instance-agnostic via CMW_BASE_URL. Prefer OpenAPI and agent tools before browser;
  UI Attach account modal is last resort when IncludeInContainer fails on a host.
---

# CMW Platform — Staff account link

Link a **platform security account** (`account.N`) to an **employee row** on the **Staff** account template (`Sotrudniki` system name is common — resolve template id per host).

**Workflow order:** OpenAPI (`cmw_open_api/`) → agent `tools/` → browser (last resort). See [cmw-platform](../cmw-platform/SKILL.md) § Workflow order.

**Do not** rename usernames during attach. Account creation is [cmw-platform-account-bootstrap](../cmw-platform-account-bootstrap/SKILL.md).

**Instance migration progress:** log outcomes in `{instance_progress_dir}` — instance roadmap under `docs/`, `localization/migration_progress/` (not this repo).

## Discovery (find the link field)

1. Switch host: [cmw-platform-instance-switch](../cmw-platform-instance-switch/SKILL.md) — set `CMW_BASE_URL` to `{CMW_BASE_URL}`.
2. Open a **linked** reference row and an **unlinked** row in the same Staff template.
3. Compare payloads:

| Source | Endpoint | Linked signal |
|--------|----------|----------------|
| Records list | `GET webapi/Records/AccountTemplate@{App}.{StaffTemplate}` | `username` non-empty; `fullName` / `mbox` populated from account profile |
| Single record | `GET webapi/Record/{employeeId}` | Same fields (`username`, `fullName`, …) |
| TeamNetwork | `POST api/public/system/TeamNetwork/ObjectService/Get` body `{"objectId":"<id>","accountTemplateId":"<templateId>"}` | `cmw.account.username`, `cmw.account.fullName`, … |

**Field names (typical Staff / Sotrudniki):**

| Layer | System name | Type | Example (linked) |
|-------|-------------|------|------------------|
| Web API Records | `username` | string | `{login_name}` |
| TeamNetwork Get | `cmw.account.username` | string | `{login_name}` |
| Display | `fullName` / `cmw.account.fullName` | string | `{display_name}` |

**Employee record id:** Often a **numeric string** (`{employee_row_id}`), not `account.{N}`. Form/card links may use `#form/.../account.{N}` only after attach. Do not assume progress JSON `account.{N}` works as `webapi/Record/account.{N}`.

Attribute metadata: `GET webapi/Attribute/List/Template@{App}.{StaffTemplate}` → `cmw_account_username` (system string; `create_edit_record` skips system fields).

## Read path (verify link)

```http
GET webapi/Records/AccountTemplate@{Application}.{StaffTemplate}
```

Row is linked when `username` (or list column **Full name** / account profile fields) is populated.

Optional:

```http
POST api/public/system/TeamNetwork/ObjectService/GetPropertyValues
```

Body: `objects: ["{employee_row_id}"]`, `propertiesByAlias: ["username", "cmw_account_username"]`.

## Write path (API — try in order)

### 1. Direct record PUT (works only when uniqueness allows)

```http
PUT webapi/Record/{employeeId}
Content-Type: application/json

{"username": "{login_name}"}
```

- `{employeeId}` = employee row id from list/Get, not `account.{N}`.
- May return **200** with error: `triple uniqueness - {id} cmw.account.username {value}` if the login is already reserved in the template without a proper Include.
- `create_edit_record` **does not** set `cmw_account_username` (system field skipped).

### 2. TeamNetwork Include (per-row — preferred in OpenAPI when supported)

```http
POST api/public/system/TeamNetwork/ObjectService/IncludeInContainer
```

```json
{"accountId": "account.{N}", "containerId": "{employee_row_id}"}
```

OpenAPI: `system_core_api.json` → `TeamNetworkObjectServiceIncludeInContainerParameters`.

**Host caveat:** Some instances return HTTP **500** for certain `containerId` shapes (numeric employee id, `account.{id}`, or `aa.{id}`). Verify on `{CMW_BASE_URL}` before batch automation; fall back to UI.

### 3. IncludeInContainer1 (template-level only)

```http
POST api/public/system/TeamNetwork/ObjectService/IncludeInContainer1
```

```json
{"accountIds": ["account.{N}"], "accountTemplateId": "{account_template_id}"}
```

Returns **200** on some hosts but may **not** set `username` on pre-created empty employee rows; may surface accounts in filtered Staff lists (UI **Full name** from account profile).

## Write path (UI — last resort)

1. Employees / Staff list: `#data/aa.{N}/lst.{M}` (verify `lst` id on host via `list_datasets` or UI).
2. Select employee row → toolbar **Attach account** (user command kind **Include**).
3. Modal: search platform login → confirm.

Observed SPA calls (not always in `web_api_v1.json` snapshot):

- `POST /User/GetAccountsToInclude`
- `POST /UserCommandExecution/PerformUserAction`

Capture payload from browser network if automating.

→ UI details: [employee_account_attach.md](../cmw-platform/references/employee_account_attach.md)

## Example mapping table (instance — replace placeholders)

Store real ids in `{instance_progress_dir}/localization/migration_progress/` — do not copy from another host.

| Username | Platform account id | Employee row id (API) |
|----------|---------------------|------------------------|
| `{user_a}` | `account.{id_a}` | `{row_a}` |
| `{user_b}` | `account.{id_b}` | `{row_b}` |

## Related

- [cmw-platform](../cmw-platform/SKILL.md)
- [employee_account_attach.md](../cmw-platform/references/employee_account_attach.md)
- [cmw-platform-backup-launch](../cmw-platform-backup-launch/SKILL.md) — backup after a successful attach batch
