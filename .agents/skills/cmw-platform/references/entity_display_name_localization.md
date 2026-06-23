# Entity display name localization (platform-generic)

Translate **user-visible** labels (names, titles, column headers, button labels) on any Comindware tenant while keeping **system names** unchanged: `globalAlias`, technical aliases, `cmw.container.alias`, filter attribute keys, and BPMN technical ids.

**Instance playbooks** (concrete designer URLs, first-wave targets, tenant batch notes): `{instance_progress_dir}/.agents/skills/cmw-platform/references/en_template_ru_leftover_cleanup.md`.

**Related:** [localization.md](localization.md) Workflow B · [process_model_template_localization.md](process_model_template_localization.md) (`doc.XXXX` trees) · [platform_usage_discoveries.md](platform_usage_discoveries.md) (dataset PUT, `globalAlias`) · [ui_components.md](ui_components.md) · [instance_repo_documentation_boundary.md](instance_repo_documentation_boundary.md).

---

## Core rule: aliases must not change

| Translate (display) | Do not change (unless Workflow A alias-rename wave) |
|---------------------|-----------------------------------------------------|
| `name`, `title`, `displayName`, column `name`, button `name`, toolbar item `name` | `globalAlias.alias`, dataset/template **system names**, attribute **aliases** in columns/filters |
| `systemFilterExpression` **literal** user-facing words embedded in expressions | Stable ids referenced by calculations unless a dedicated remap wave |
| Ontology `cmw.object.name`, `cmw.dataset.name`, `cmw.eventTrigger.name` | `cmw.container.alias` value used in Web API paths |

Use [localization.md](localization.md) **Workflow A** only when the migration explicitly renames system identifiers.

---

## Entity types and API surfaces

| Entity kind | Designer / id hint | Primary localization surface | Tools / endpoints |
|-------------|-------------------|-------------------------------|-------------------|
| **Record template** (`oa.*`) | `#RecordType/oa.{N}/…` | `name`, description | `edit_or_create_record_template` → `PUT webapi/RecordTemplate/{app}` |
| **Process model template** (`doc.*`) | `#RecordType/doc.{N}/…` | Display via ontology or RecordTemplate PUT | `edit_or_create_record_template` with `globalAlias.type: DocumentationTemplate` **or** `AddStatement` on `cmw.object.name` — see [process_model_template_localization.md](process_model_template_localization.md) |
| **Dataset / table** | Lists tab `lst.{M}` | `name`, column `name`, filters (display literals) | `get_dataset` → merge → `edit_or_create_dataset` → `PUT webapi/Dataset/{app}` (preserve `globalAlias`) |
| **Toolbar** | Linked to dataset/list | Toolbar `name`, item display names | `list_toolbars` / `edit_or_create_toolbar` |
| **Button (user command)** | Operations tab | `name`, `description` | `list_buttons` / `edit_or_create_button` → `PUT webapi/UserCommand/…` |
| **Form / card** | `#form/…` under template | Titles, section labels (non-calculated) | `list_forms`, form edit tools |
| **Context / registry rows** | Solution dataset grids (`cmw.container.dataset.*`) | Personal/shared table display names | Often **ontology** on `lst.*` (`cmw.dataset.name`) — not always `edit_or_create_dataset` |
| **Role / org catalog templates** | `ra.*`, `os.*` in TemplateService | Template display name | `AddStatement` `cmw.object.name` on subject id |
| **Ontology predicates** | Any entity id | Single-field display override | `POST …/OntologyService/AddStatement` + verify `GetAxioms` |

Resolve `sln.{N}` → application system name (e.g. `CMW_FM`) via `GetAxioms` / `resolve_entity` before `webapi/*` calls. Resolve `doc.{N}` → `cmw.container.alias` (template system name) via `GetAxioms` — enumerate **all** `doc.*` process model templates; do not assume a single id covers the family.

---

## Pattern A — Ontology display names (`AddStatement`)

When Web API PUT is blocked (system-solution guard) or only one label field needs change:

```http
POST {base}/api/public/system/Base/OntologyService/AddStatement
{"subject": "<entity_id>", "predicate": "<predicate>", "value": "<EN display>"}
```

| Predicate | Typical subjects | Use for |
|-----------|------------------|---------|
| `cmw.object.name` | `doc.{N}`, `oa.{N}`, `os.{N}`, `ra.{N}` | Template / catalog display titles |
| `cmw.dataset.name` | `lst.{M}` (personal or shared table id) | List/table title in designer |
| `cmw.eventTrigger.name` | `event.{M}` | Operations button label when `edit_or_create_button` PUT returns `InterceptedException` |

Verify: `POST …/GetAxioms` with raw body = entity id (same as `resolve_entity`).

Map `event.{id}` to parent process model template via `cmw.eventTrigger.container` = `doc.{N}` when renaming Operations buttons under `#RecordType/doc.{N}/Operations`.

---

## Pattern B — Web API GET → merge → PUT

Preferred for bulk, repeatable edits when PUT succeeds:

1. **Dataset:** `get_dataset` → merge `name` + `columns` keyed by **attribute alias** → `edit_or_create_dataset` with full body and `globalAlias` ([platform_usage_discoveries.md](platform_usage_discoveries.md)).
2. **Toolbar:** `get_toolbar` → merge `name` + item display fields → raw `PUT webapi/Toolbar/{app}` with `globalAlias` and owner type **`DocumentationTemplate`** for `doc.*` parents (not `RecordTemplate`).
3. **Button:** `list_buttons` → `edit_or_create_button` with merged GET body.
4. **Form:** `list_forms` → GET form → **minimal** title PUT only when repairing breakage — see [Form PUT safe patterns](#form-put-safe-patterns) in [process_model_template_localization.md](process_model_template_localization.md#form-put-safe-patterns).
5. **Context attributes:** `list_attributes` on template system name → PUT attribute `displayName` / `name` when host allows; else ontology on attribute subject if exposed.
6. **Process model template (`doc.*`):** `edit_or_create_record_template` with `application_system_name` = app alias (not solution id `sln.*`), `system_name` from `cmw.container.alias` (API `globalAlias.type` may be `DocumentationTemplate`).

**Pitfall:** PUT `webapi/Dataset/{app}/Dataset@…` may return **405** — PUT to `webapi/Dataset/{app}` with injected `globalAlias` instead.

### Pattern B′ — PUT accepted but display unchanged (no-op)

Common on **system-solution** toolbars and guarded metadata:

| Signal | Meaning | Next step |
|--------|---------|-----------|
| HTTP **200** + `success: true` but re-GET shows same Cyrillic | Server **ignored** body (guard / invalid inner alias) | Do not retry identical PUT in a loop |
| `InterceptedException`, `Alias  is invalid`, `RequestedOperationFailedException` | Toolbar/button PUT blocked | Ontology `AddStatement` on `cmw.eventTrigger.name` (buttons) or designer UI for toolbar titles |
| Dataset PUT OK, designer list tab still RU | Stale UI cache or personal `lst.*` title via ontology | Re-GET API; `GetAxioms` on `lst.*` for `cmw.dataset.name` |

Always **re-GET** after PUT; do not trust status code alone ([errors.md](errors.md#pascalcase-alias-no-op-http-200-field-unchanged)).

---

## Pattern C — Designer hash segments (agnostic)

| Hash segment (pattern) | Entity kind | Typical work |
|------------------------|-------------|--------------|
| `#RecordType/doc.{N}/Operations` | Process model template + Operations buttons | Pattern A on `event.*` or Pattern B on buttons |
| `#RecordType/doc.{N}/Lists/lst.{M}` | Table on process model template | Dataset PUT |
| `#RecordType/oa.{N}/Lists/lst.{M}` | Table on record template | Dataset PUT |
| `#solutions/sln.{N}/templates/showall/cmw.container.dataset.*` | Registry / solution dataset grid | Ontology on `lst.*` or catalog template names |

Enumerate **all** ids in a family (`doc.1` … `doc.{N}`) — not a single example id. Per-template checklist: [process_model_template_localization.md](process_model_template_localization.md).

---

## Cyrillic scan and verification

1. Walk GET JSON from datasets, buttons, toolbars, forms; flag `[\u0400-\u04FF]` in user-visible fields.
2. Re-GET after PUT; confirm no unintended Cyrillic in `name` / column titles.
3. `GetAxioms` for ontology-updated ids — EN `cmw.object.name` / `cmw.dataset.name`.
4. Do **not** rewrite `globalAlias`, filter attribute keys, or calculated field system names unless a separate migration task requires it.

---

## Browser vs API

```
Need EN display strings on a template?
├─ Dataset / button / toolbar / form — GET ok, PUT ok → API (edit_or_create_*)
├─ System-guarded button PUT fails → AddStatement on cmw.eventTrigger.name
├─ Personal registry row (owner resolution error on dataset PUT) → AddStatement on lst.*
└─ FilterTree / BPMN diagram — API may not persist → designer UI fallback
```

**Order:** OpenAPI → `tools/` → skills → browser ([SKILL.md §9](../SKILL.md#9-growing-platform-skills)).

---

## Instance progress

After a tenant batch, append `operations[]` in `{instance_progress_dir}/localization/migration_progress/*.json` with `meta.host` (no secrets) — not in cmw-platform-agent.

---

## Related platform docs

- [process_model_template_localization.md](process_model_template_localization.md) — enumerate all `doc.XXXX`, five-area checklist, form PUT safety
- [record_instance_field_localization.md](record_instance_field_localization.md) — **data rows** in `#data/…/lst.M/` (Records API), not list metadata
- `{instance_progress_dir}/.agents/skills/cmw-platform/references/en_template_ru_leftover_cleanup.md` — instance playbook (EN target RU leftovers)
- [browser_automation.md](browser_automation.md) — `#RecordType/…` hash patterns
