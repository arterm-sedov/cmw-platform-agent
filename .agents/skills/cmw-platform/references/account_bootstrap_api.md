# Account bootstrap (System Core API)

Create platform **accounts** on any CMW instance via **System Core** services (`api/public/system/...`). Use when admin UI is slow or for scripted UAT seeding.

**OpenAPI:** `cmw_open_api/system_core_api.json` — `Base/AccountService/*`, `Base/AccountGroupService/*`.

## Configuration

From `.env` (never log or commit secrets):

| Variable | Purpose |
|----------|---------|
| `CMW_BASE_URL` | Target instance, e.g. `https://{host}/` |
| `CMW_LOGIN` / `CMW_PASSWORD` | Basic auth for API calls |
| `CMW_USE_DOTENV` | Set `true` for scripted runs |
| `UAT_ACCOUNT_PASSWORD_PREFIX` | Optional prefix for new account passwords |

**UAT password pattern (test env):** `final = {UAT_ACCOUNT_PASSWORD_PREFIX}{CMW_PASSWORD}` — log `password_set: true` in project progress only; never log the composed password.

## HTTP pattern

All calls: `POST {CMW_BASE_URL}api/public/system/{Service}/{Action}` with JSON body and Basic auth.

In Python (cmw-platform-agent):

```python
from tools import requests_ as requests_

requests_._post_request(body, "api/public/system/Base/AccountService/Create")
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

## Set password

**Path:** `Base/AccountService/SetAccountPassword`

**Body:** `{"id": "<account_id>", "password": "<final_password>"}`

Requires admin-capable credentials.

## Add to group

**Path:** `Base/AccountGroupService/IncludeMembers`

**Body:** `{"groupId": "<group.id>", "memberIds": ["<account_id>"]}`

Discover groups: `POST .../Base/AccountGroupService/List` (empty body). Resolve `groupId` by `name` on the target instance.

**Get group detail:** `POST .../Base/AccountGroupService/Get` with body `{"groupId": "<id>"}` (not `id`).

## Verify

- `POST .../Base/AccountService/List` — count and username
- `POST .../Base/AccountService/Get` — body `{"id": "<account_id>"}`

## Checklist

1. Set `CMW_BASE_URL` to target instance.
2. Confirm target **group** exists (create via admin or `AccountGroupService/Create` if needed).
3. `Create` → capture returned id.
4. `SetAccountPassword`.
5. `IncludeMembers` for business group(s).
6. List/Get to verify.

## Instance-specific migration artifacts

Progress JSON, inventories, and host-specific harvest files belong in the **project repo** that owns that migration (not in cmw-platform-agent). This reference stays generic.
