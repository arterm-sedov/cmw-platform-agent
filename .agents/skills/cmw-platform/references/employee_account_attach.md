# Employee record ↔ platform account (Attach account)

> **Entity recipe pattern:** Browser steps for this modal belong here (and in [browser_automation.md](browser_automation.md)); future entity-specific UI flows should follow the same reference-first pattern — see [cmw-platform SKILL §9](../SKILL.md#9-growing-platform-skills).

Link an existing **platform login** to an **employee row** in the **Staff** account template (system name varies by solution, e.g. `Sotrudniki`). This is **not** account creation — use [account_bootstrap_api.md](account_bootstrap_api.md) / [cmw-platform-account-bootstrap](../../cmw-platform-account-bootstrap/SKILL.md) for `AccountService/Create` first.

**Workflow order:** OpenAPI (`cmw_open_api/`) → agent `tools/` → browser last resort. Full attach recipe (field names, PUT/Include, host-specific blockers): [cmw-platform-staff-account-link](../../cmw-platform-staff-account-link/SKILL.md).

## UI workflow (instance-agnostic)

1. Open the **Employees** list (hash pattern `#data/aa.{account_template_id}/lst.{list_id}` — dataset is usually `defaultList` on the Staff account template; **UI entity ids differ per host**).
2. Select an employee row (or open the record), or use the list toolbar action **Attach account** (CTF user command `include`, kind **Include**; legacy export name may show as “Follow record”).
3. In the **modal**, pick one or more **existing** platform accounts (created earlier via administration or `AccountService`).
4. Confirm — the employee record is now linked; account profile fields (`fullName`, `mbox`, `username`, …) surface on the employee card.

**Unlink:** toolbar **Unfollow record** / `excludeAccount` (kind **Exclude**).

## vs AccountService bootstrap

| Step | Mechanism | Skill |
|------|-----------|--------|
| Create login, password, groups | System Core `Base/AccountService/*`, `AccountGroupService/IncludeMembers` | cmw-platform-account-bootstrap |
| Link login → employee row | App **Attach account** modal (Include) or Object API below | this doc |

Do **not** assume `AccountService/Create` attaches the account to Staff — that only creates the security principal.

## Phase ordering (migration projects)

| When | Action |
|------|--------|
| **Phase 0** (optional) | After UAT accounts exist on FR: create **employee rows** (if missing), then **Attach account** per persona — enables assignee/Staff UX before bulk employee seeding |
| **Phase 1+** | Employee seeding from TR may create rows first; run **Attach account** (or API Include) when usernames are known |

If employee seeding is deferred, you can still link Phase 0 accounts as soon as a minimal employee row exists (create empty row → Attach account).

## Link field (Staff account template)

| Read/write | Name |
|------------|------|
| `GET webapi/Records/...` / `PUT webapi/Record/{id}` | `username` |
| `ObjectService/Get` + `accountTemplateId` | `cmw.account.username` |
| List UI column | **Full name** (`fullName` from account profile when linked) |

**Employee row id:** numeric string on the target host, distinct from `account.{N}` platform login ids. Demo persona rows show **Full name** in `#data/aa.{account_template_id}/lst.{list_id}` — **resolve list ids per host**; card URL may use `account.{N}`.

## API (when field names unknown)

Prefer **read** via records list:

```http
GET webapi/Records/AccountTemplate@{Application}.{StaffTemplateSystemName}
```

Example: `AccountTemplate@{App}.{StaffTemplateSystemName}` (display name may differ from system name after localization).

**Linked record heuristics** (verify on your instance):

- Record `id` equals `account.{N}` when the platform account is included, and/or
- `username` populated on the employee record.

**Programmatic attach** (write — use only when approved):

```http
POST api/public/system/TeamNetwork/ObjectService/IncludeInContainer1
```

Body shape (OpenAPI): `accountIds` (array of account ids), `accountTemplateId` (Staff template id, e.g. `aa.2` on a given host — **resolve per instance**, do not copy from another host).

Single-account variant: `IncludeInContainer` with `accountId` + `containerId` (employee numeric id). On some hosts, per-row Include returns **500**; `IncludeInContainer1` alone may not set `username` on pre-created rows — verify on `{CMW_BASE_URL}`.

When list/get payloads are unclear, use **browser MCP** (cmw-platform § Browser) on the Employees list and Attach account modal — SPA: `User/GetAccountsToInclude`, `UserCommandExecution/PerformUserAction`.

## Verification checklist

- [ ] `AccountTemplate@…Sotrudniki` row count on FR (and TR reference) via GET above
- [ ] For each UAT username: employee row exists **and** `id` / `username` shows link
- [ ] Progress JSON in the **migration project** repo does **not** log attach unless performed — do not infer from `security_create` alone

## Related

- [cmw-platform-staff-account-link](../../cmw-platform-staff-account-link/SKILL.md) — discovery, API order, persona table
- [cmw-platform-instance-switch](../../cmw-platform-instance-switch/SKILL.md) — switch `CMW_BASE_URL` before TR vs FR compare
- [browser_automation.md](browser_automation.md) — SPA hash routes, snapshots
