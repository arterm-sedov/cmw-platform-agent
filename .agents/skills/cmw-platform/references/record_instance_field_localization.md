# Record instance field localization (platform-generic)

Translate **user-visible Russian (or wrong-language) values on data rows** shown in a list view (`#data/{template_id}/lst.{M}/`), without changing record ids, template aliases, or attribute system names.

**Not in scope here:** list **metadata** (dataset title, column headers) — use [entity_display_name_localization.md](entity_display_name_localization.md) and [process_model_template_localization.md](process_model_template_localization.md). **Not:** process designer BPMN, toolbar definitions, or form layout metadata.

**Instance batch logs:** `{instance_progress_dir}/localization/migration_progress/*.json` with `change_kind: data_migrate` or dedicated `operations[]` per row.

---

## UI hash → API mapping

| UI segment | Meaning | Resolve before Records API |
|------------|---------|----------------------------|
| `#data/doc.{N}/lst.{M}/` | **Instance rows** for process model template `doc.{N}` in list `lst.{M}` | `GetAxioms(doc.{N})` → `cmw.container.alias` (template **system name**); map `lst.{M}` → backing **dataset** (`defaultList` or named alias) via `list_datasets` |
| `#data/oa.{N}/lst.{M}/` | Instance rows on **record template** `oa.{N}` | Same pattern with `oa` axioms |

Designer list id (`lst.6`) may differ from dataset alias (`defaultList`) on the same tab — always confirm via `list_datasets` + GET dataset body, not hash alone.

**Application:** resolve solution → app system name (e.g. `CMW_FM`) via `GetAxioms` / `resolve_entity` — never pass `sln.*` as `webapi/{app}`.

---

## OpenAPI and live swagger

| Source | Use |
|--------|-----|
| `cmw_open_api/web_api_v1.json` | Committed Web API v1 paths and models |
| `{target_host}/docs` | Live swagger on the tenant — confirm pagination query params and Record vs Records path variants |

Core record endpoints ([api_endpoints.md](api_endpoints.md)):

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `webapi/Attribute/List/Template@{app}.{tmpl}` | GET | Attribute aliases, types, `isSystem`, calculated flags |
| `webapi/Records/Template@{app}.{tmpl}` | GET | List instances (pagination) |
| `webapi/Records/Template@{app}.{tmpl}` | POST | Create row |
| `webapi/Record/{record_id}` | GET | Single row (if exposed on host) |
| `webapi/Record/{record_id}` | PUT | Update writable fields on existing row |

Prefer agent tool `create_edit_record` (`operation: edit`) when attribute metadata is needed for coercion — same PUT contract as raw HTTP.

---

## Discovery workflow

1. **Schema:** `list_attributes` on `Template@{app}.{template_system_name}` — note **PascalCase** aliases for PUT; GET list responses often use **lowercase** keys ([process_record_demo_fill.md](process_record_demo_fill.md)).
2. **List rows:** GET `webapi/Records/Template@{app}.{tmpl}` with pagination (`sk`, `t` / host-specific page size — verify in swagger).
3. **Cyrillic scan:** walk string fields in each row; skip ids, guids, and known system-only keys.
4. **Classify each hit:**

| Class | Action |
|-------|--------|
| **Writable string / text** | PUT only changed keys; GET verify after |
| **Calculated** (`*_calc`, expression-backed, `isSystem` read-only) | **Skip** — display may mirror calc; changing alias or type breaks lists/KPIs |
| **Record / lookup** | Value may be **reference id**; fixing display may require target catalog row or lookup label elsewhere — do not PUT raw id as “translation” |
| **Choice / enum** | Map to valid choice id on **target** host; read-only compare `{source_host}` for meaning only |

5. **Sample then batch:** scan first page(s); if thousands of RU rows, fix a capped sample, ship a rerun script under `{instance_progress_dir}/localization/scripts/`, document totals in progress JSON.

---

## PUT safety

- **Merge:** GET row → change only intended fields → PUT. Do not strip unrelated keys if the host requires full body (tool handles coercion).
- **Alias casing:** lowercase PUT keys are often **no-ops** (HTTP 200, unchanged) — see [errors.md](errors.md#pascalcase-alias-no-op-http-200-field-unchanged).
- **No** mass-delete, alias rename, or record id rewrite in a display-localization wave.
- **Pagination:** persist `sk` offset in script; stop on empty page or `--limit`.
- Log each update: `record_id`, field alias, `original`, `became`, `change_kind: data_migrate` in instance JSON.

---

## Browser vs API

```
RU text visible in #data/... list grid?
├─ Column header / list title → dataset PUT (metadata layer)
├─ Cell value on instance row → Records API PUT (this doc)
├─ Calculated column → skip or separate expression migration
└─ Lookup shows RU catalog label → fix catalog row or reference target, not display string on wrong id
```

**Order:** OpenAPI → `tools/` (`create_edit_record`, query helpers) → skills → browser ([SKILL.md §9](../SKILL.md#9-growing-platform-skills)).

---

## Related

- [entity_display_name_localization.md](entity_display_name_localization.md) — templates, datasets, buttons, forms (metadata)
- [process_model_template_localization.md](process_model_template_localization.md) — `doc.XXXX` five-area checklist
- [cmw-platform-process-record-fill](../../cmw-platform-process-record-fill/SKILL.md) — running process instances, demo fill
- [instance_repo_documentation_boundary.md](instance_repo_documentation_boundary.md) — progress JSON lives in instance repo only
