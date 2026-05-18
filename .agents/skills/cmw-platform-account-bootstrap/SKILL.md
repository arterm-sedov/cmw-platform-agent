---
name: cmw-platform-account-bootstrap
description: >-
  Create or update platform login accounts via System Core AccountService —
  Create, Edit, SetAccountPassword, AccountGroupService IncludeMembers, optional
  Volga role. Triggers on seed users, administration accounts, AccountService,
  SetAccountPassword, SetUsername, account rename, Mbox, FullName, UAT login
  bootstrap. Agnostic for any CMW instance (CMW_BASE_URL). Does not cover batch
  migration workflows (those live in the project repo).
---

# CMW Platform — Account bootstrap (create and update)

Create or update **one** platform login account on any Comindware instance using **System Core** HTTP APIs. For OpenAPI field detail, see [../cmw-platform/references/account_bootstrap_api.md](../cmw-platform/references/account_bootstrap_api.md).

**OpenAPI source of truth:** `cmw_open_api/system_core_api.json` — paths under `Base/AccountService/*` and `Base/AccountGroupService/*`.

## When to use

- Seed a UAT or admin-test login on a target instance
- Fix **Username**, **FullName**, or **Mbox** on an existing account (prefer **Edit**)
- Script account + password + group membership
- Verify account exists after migration

**Do not use this skill for:** multi-account batch migration, instance-specific progress JSON, or host-specific inventories — keep those in the owning project repo (e.g. Volga TR→FR migration in **my-building**: [`docs/20260519_migration_status_and_roadmap.md`](file:///D:/Repo/my-building/docs/20260519_migration_status_and_roadmap.md), `localization/migration_progress/`).

**Post-batch configuration backup (UI):** After several account or security changes, use [cmw-platform-backup-launch](../cmw-platform-backup-launch/SKILL.md) — do not duplicate backup steps here.

## Configuration

Load from `cmw-platform-agent/.env` with `CMW_USE_DOTENV=true`. **Never log or commit** secrets.

Switching reference vs target host → [cmw-platform-instance-switch](../cmw-platform-instance-switch/SKILL.md) first.

| Variable | Purpose |
|----------|---------|
| `CMW_BASE_URL` | Target instance root, e.g. `https://{your-host}/` |
| `CMW_LOGIN` / `CMW_PASSWORD` | Basic auth for API calls |
| `CMW_USE_DOTENV` | `true` for scripted runs |
| `UAT_ACCOUNT_PASSWORD_PREFIX` | Optional; composed password below |

**Password (test env):** `final_password = {UAT_ACCOUNT_PASSWORD_PREFIX}{CMW_PASSWORD}` — record only `password_set: true` in project progress files; never log prefix, base password, or composed value.

## Create workflow

```text
Create → SetAccountPassword → IncludeMembers (group) → [optional app role] → List/Get verify
```

### 1. Create account

`POST {CMW_BASE_URL}api/public/system/Base/AccountService/Create`

Body (adjust placeholders):

```json
{
  "account": {
    "Username": "{username}",
    "Mbox": "{username}@example.test",
    "FullName": "{display_name}",
    "IsSystemAdministrator": false,
    "IsAnonymous": false,
    "IsActive": true,
    "Role": "User"
  },
  "withTimeout": false,
  "checkPersonalDataProcessingConfirmation": false
}
```

Response: account id string (e.g. `account.N`).

**Field rules at create (EN / screenshot-ready targets):**

| Field | Rule |
|-------|------|
| **Username** | Latin only; match reference-instance login when seeding from TR (e.g. keep `engineer`, not Cyrillic) |
| **FullName** | US facility-management English on EN instances — persona label (Tenant, Engineer on Duty, …); **never** copy Cyrillic display names from a RU reference instance |
| **Mbox** | `{username}@example.test` or an existing latin email; **no Cyrillic** |

### 2. Set password

`POST .../Base/AccountService/SetAccountPassword`

```json
{"id": "<account_id>", "password": "<final_password>"}
```

### 3. Add to group

`POST .../Base/AccountGroupService/IncludeMembers`

```json
{"groupId": "<group.id>", "memberIds": ["<account_id>"]}
```

Discover groups: `POST .../Base/AccountGroupService/List` (empty body).

### 4. Optional application role

Assign app roles via application tools or admin UI after the account exists.

## Update workflow (existing account)

**Prefer `Edit` over delete/recreate.** Do **not** delete `admin`, `demo`, or other protected accounts unless the user asks.

```text
Get (by id) → merge fields → Edit → List/Get verify
```

### 1. Load current account

`POST .../Base/AccountService/Get` — body `{"id": "<account_id>"}`

Or find id: `POST .../Base/AccountService/FindByUsername` — body `{"user": "<username>"}`

### 2. Edit account

`POST .../Base/AccountService/Edit` — body is the full `ComindwarePlatformApiDataAccount` object (PascalCase keys), including **`Id`**, with updated fields.

Typical updates:

| Field | Guidance |
|-------|----------|
| **Username** | Latin only; align with reference login when fixing FR seeds (e.g. `dispetcher` → `dispatcher01` only when project matrix says so) |
| **FullName** | Whole-phrase US FM English on EN instances (e.g. `Engineer on Duty`, `Tenant`) — project repo defines persona labels |
| **Mbox** | `{username}@example.test` after any username change; no Cyrillic, no RU freemail domains on EN demo hosts |

Preserve required booleans from **Get** (`IsSystemAdministrator`, `IsAnonymous`, `Role`, etc.) — do not strip fields unless OpenAPI marks them optional and you know defaults.

### 3. Password / group (only if needed)

- Password unchanged → skip `SetAccountPassword`
- Group membership unchanged → skip `IncludeMembers`

### When to update vs recreate

| Situation | Action |
|-----------|--------|
| Wrong FullName / Mbox / Username | **Edit** |
| Account exists but not in group | **IncludeMembers** only |
| Account missing entirely | **Create** path |
| Duplicate broken account | Ask user before **Delete** |

## Verify

- `POST .../Base/AccountService/List` — find username, confirm Mbox/FullName
- `POST .../Base/AccountService/Get` — body `{"id": "<account_id>"}`

## Related AccountService paths (OpenAPI)

| Action | Path |
|--------|------|
| List | `Base/AccountService/List` |
| Get | `Base/AccountService/Get` |
| FindByUsername | `Base/AccountService/FindByUsername` |
| Edit | `Base/AccountService/Edit` |
| Enable / Disable | `Base/AccountService/Enable`, `Disable` |

Full catalog: `cmw_open_api/system_core_api.json`.

## Checklist (create)

1. Set `CMW_BASE_URL` to the **target** instance.
2. Confirm target **group** exists.
3. `Create` with latin Username, EN FullName, `{username}@example.test` Mbox.
4. `SetAccountPassword` → `IncludeMembers` → optional app role.
5. List/Get verify; log `password_set: true` in project progress only.
6. If project plan requires post-batch backup → [cmw-platform-backup-launch](../cmw-platform-backup-launch/SKILL.md).

**Staff employee link (separate step):** Creating the login does **not** attach it to the **Employees** (Staff / `Sotrudniki`) template. After the account exists, use the app **Attach account** modal on an employee row, or the Object **Include** API — see [employee_account_attach.md](../cmw-platform/references/employee_account_attach.md).

## Checklist (update)

1. `Get` / `FindByUsername` → capture full account payload.
2. `Edit` with merged fields (Username / FullName / Mbox per rules above).
3. List/Get verify; log `original` → `became` per field in project progress JSON.
4. Post-batch backup per project plan → backup-launch skill.

## Project-repo artifacts (out of scope here)

Progress JSON, `used_accounts_inventory.json`, and TR→FR matrices belong in the migration project (e.g. `localization/migration_progress/`). See that repo’s master plan for batch size, git commit, and FR backup policy.

## Maintaining this skill

When `system_core_api.json` gains account fields or new `AccountService` actions, update [account_bootstrap_api.md](../cmw-platform/references/account_bootstrap_api.md) first, then mirror create/update steps here. Keep backup UI out of this file — only link to **cmw-platform-backup-launch**.
