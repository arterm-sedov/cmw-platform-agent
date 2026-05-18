# Employee record ↔ platform account (Attach account)

> **Entity recipe pattern:** Browser steps for this modal belong here (and in [browser_automation.md](browser_automation.md)); future entity-specific UI flows should follow the same reference-first pattern — see [cmw-platform SKILL §9](../SKILL.md#9-growing-platform-skills).

Link an existing **platform login** to an **employee row** in the Volga **Staff** account template (`Sotrudniki` system name). This is **not** account creation — use [account_bootstrap_api.md](account_bootstrap_api.md) / [cmw-platform-account-bootstrap](../../cmw-platform-account-bootstrap/SKILL.md) for `AccountService/Create` first.

## UI workflow (instance-agnostic)

1. Open the **Employees** list in the Volga app (hash pattern `#data/aa.{N}/lst.{M}` — dataset is usually `defaultList` on the Staff account template; **UI entity ids differ per host**, e.g. FR `lst.279` / TR `lst.399`).
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

## API (when field names unknown)

Prefer **read** via records list:

```http
GET webapi/Records/AccountTemplate@{Application}.{StaffTemplateSystemName}
```

Example: `AccountTemplate@Volga.Sotrudniki` (system name often unchanged after EN display rename to “Staff”).

**Linked record heuristics** (verify on your instance):

- Record `id` equals `account.{N}` when the platform account is included, and/or
- `username` populated on the employee record.

**Programmatic attach** (write — use only when approved):

```http
POST api/public/system/TeamNetwork/ObjectService/IncludeInContainer1
```

Body shape (OpenAPI): `accountIds` (array of account ids), `accountTemplateId` (Staff template id, e.g. `aa.2` on a given host — **resolve per instance**, do not copy from another host).

Single-account variant: `IncludeInContainer` with `accountId` + `containerId`.

When list/get payloads are unclear, use **browser MCP** (cmw-platform § Browser) on the Employees list and Attach account modal.

## Verification checklist

- [ ] `AccountTemplate@…Sotrudniki` row count on FR (and TR reference) via GET above
- [ ] For each UAT username: employee row exists **and** `id` / `username` shows link
- [ ] Progress JSON in the **migration project** repo does **not** log attach unless performed — do not infer from `security_create` alone

## Related

- [cmw-platform-instance-switch](../../cmw-platform-instance-switch/SKILL.md) — switch `CMW_BASE_URL` before TR vs FR compare
- [browser_automation.md](browser_automation.md) — SPA hash routes, snapshots
