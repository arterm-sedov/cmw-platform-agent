# Account bootstrap (System Core API)

Create or update platform **accounts** on any CMW instance via **System Core** services (`api/public/system/...`).

**Agent skills:**

- [cmw-platform-instance-switch/SKILL.md](../../cmw-platform-instance-switch/SKILL.md) — switch `CMW_BASE_URL` / verify target host
- [cmw-platform-account-bootstrap/SKILL.md](../../cmw-platform-account-bootstrap/SKILL.md) — create/update workflow
- [cmw-platform-backup-launch/SKILL.md](../../cmw-platform-backup-launch/SKILL.md) — post-batch configuration backup (UI)

**OpenAPI:** `cmw_open_api/system_core_api.json` — `Base/AccountService/*`, `Base/AccountGroupService/*`.

## Configuration

From `.env` (never log or commit secrets):

| Variable | Purpose |
|----------|---------|
| `CMW_BASE_URL` | Target instance, e.g. `https://{host}/` |
| `CMW_LOGIN` / `CMW_PASSWORD` | Basic auth for API calls |
| `CMW_USE_DOTENV` | Set `true` for scripted runs |
| `UAT_ACCOUNT_PASSWORD_PREFIX` | Optional prefix for new account passwords |

**UAT password pattern (test env):** `final = {UAT_ACCOUNT_PASSWORD_PREFIX}{CMW_PASSWORD}` — log `password_set: true` in project progress only.

## HTTP pattern

All calls: `POST {CMW_BASE_URL}api/public/system/{Service}/{Action}` with JSON body and Basic auth.

Direct HTTP (recommended for scripts — avoids heavy `tools` package import):

```python
import base64, json, os, requests
from dotenv import load_dotenv

load_dotenv()
base = os.environ["CMW_BASE_URL"].rstrip("/")
auth = base64.b64encode(
    f'{os.environ["CMW_LOGIN"]}:{os.environ["CMW_PASSWORD"]}'.encode()
).decode()
headers = {"Authorization": f"Basic {auth}", "Content-Type": "application/json"}

def post(path: str, body: dict | None = None):
    r = requests.post(f"{base}/api/public/system/{path}", headers=headers, json=body or {}, timeout=60)
    r.raise_for_status()
    return r.json() if r.text.strip() else []
```

## Create account

**Path:** `Base/AccountService/Create`

**Body:**

```json
{
  "account": {
    "Username": "uat_user01",
    "Mbox": "uat_user01@example.test",
    "FullName": "UAT User",
    "IsSystemAdministrator": false,
    "IsAnonymous": false,
    "IsActive": true,
    "Role": "User"
  },
  "withTimeout": false,
  "checkPersonalDataProcessingConfirmation": false
}
```

**Response:** account id string (e.g. `account.N`).

## Edit account (update)

**Path:** `Base/AccountService/Edit`

**Body:** full `ComindwarePlatformApiDataAccount` (PascalCase), including **`Id`**.

1. `POST .../Base/AccountService/Get` with `{"id": "<account_id>"}`.
2. Merge `Username`, `FullName`, `Mbox` (and other fields as needed).
3. `POST .../Base/AccountService/Edit` with the merged object.

| Field | Typical rule (EN target instance) |
|-------|-----------------------------------|
| Username | Latin; match reference login when applicable |
| FullName | US FM English persona label; no Cyrillic from RU reference |
| Mbox | `{username}@example.test`; no Cyrillic |

There is no separate `SetUsername` path in OpenAPI — username changes go through **Edit**.

## Set password

**Path:** `Base/AccountService/SetAccountPassword`

**Body:** `{"id": "<account_id>", "password": "<final_password>"}`

## Add to group

**Path:** `Base/AccountGroupService/IncludeMembers`

**Body:** `{"groupId": "<group.id>", "memberIds": ["<account_id>"]}`

Discover groups: `POST .../Base/AccountGroupService/List` (empty body).

**Get group detail:** `POST .../Base/AccountGroupService/Get` with body `{"groupId": "<id>"}`.

## Verify

- `POST .../Base/AccountService/List`
- `POST .../Base/AccountService/Get` — `{"id": "<account_id>"}`
- `POST .../Base/AccountService/FindByUsername` — `{"user": "<username>"}`

## Post-change backup (UI only)

After account/security batches on the target host, launch an **existing** configuration backup via UI — see [cmw-platform-backup-launch/SKILL.md](../../cmw-platform-backup-launch/SKILL.md) (checkbox on existing row → **Start backup**; do not create or delete backup configurations unless the user asks).

## Instance-specific migration artifacts

Progress JSON and host-specific inventories belong in the **project repo** that owns the migration (not in cmw-platform-agent).
