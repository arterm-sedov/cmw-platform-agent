# EN target — Russian leftover cleanup

Platform-generic patterns for removing **Cyrillic display text** on an English US FM target (`{target_host}`) while keeping `{source_host}` read-only for translation context.

**Not in scope here:** Workflow A alias renames ([localization.md](localization.md)) · long Volga id maps (instance repo only).

**Process model family:** multiple templates `doc.XXXX` — full tree checklist in [process_model_template_localization.md](process_model_template_localization.md).

**Repo boundary:** [instance_repo_documentation_boundary.md](instance_repo_documentation_boundary.md)

---

## Mandatory workflow order

1. [cmw-platform-instance-switch](../../cmw-platform-instance-switch/SKILL.md) — set `{target_host}`, verify GET.
2. OpenAPI / KB MCP → `tools/` (`get_dataset`, `edit_or_create_dataset`, `list_buttons`, `get_ontology_objects`).
3. Browser only when API cannot persist (FilterTree, designer-only process areas).

Never POST/PUT to `{source_host}`.

---

## Target area catalog (patterns)

Replace `{target_host}` and ids with tenant values in **instance** progress JSON — do not commit host tables here.

| Area | URL pattern (UI) | Entity layer | Primary API |
|------|------------------|--------------|-------------|
| **Process model — Operations** | `#RecordType/doc.{N}/Operations` | User commands (buttons) under process model template | `webapi/UserCommand/List/Template@{app}.{tmpl}` · `edit_or_create_button` |
| **Process model — list** | `#RecordType/doc.{N}/Lists/lst.{M}` | List → backing **table** (dataset) | `get_dataset` / `edit_or_create_dataset` |
| **Solution org structure apps** | `#solutions/sln.{S}/templates/showall/cmw.container.dataset.orgStructureApps` | Solution-level dataset template registry | `GET webapi/Dataset/{app}/Dataset@cmw.container.dataset.orgStructureApps` (+ personal variants via ontology) |
| **Solution role apps** | `#solutions/sln.{S}/templates/showall/cmw.container.dataset.roleApps?...solution filter...` | Role-app dataset grid (often filtered by solution column) | Same dataset GET/PUT; verify `systemFilterExpression` / solution filter |

**Clarification:** `doc.{N}` is a **process model template** id (RecordType scope), not a generic document container. Enumerate **all** `doc.*` on the host — see [process_model_template_localization.md](process_model_template_localization.md).

---

## orgStructureApps and roleApps (solution dataset grids)

These containers appear under **solution templates → showall** for system datasets `cmw.container.dataset.orgStructureApps` and `cmw.container.dataset.roleApps`.

| Check | Action |
|-------|--------|
| Dataset **name** / **title** Cyrillic | GET full dataset → translate → PUT `webapi/Dataset/{app}` with **`globalAlias`** |
| Column **name** Cyrillic | Merge column array from GET; PUT same endpoint |
| **Personal** dataset clones (ontology `cmw.dataset.baseDataset` points at org/role base) | `get_ontology_objects` `types: ["Dataset"]` + GetAxioms per id — translate `cmw.dataset.name` when Cyrillic |
| **roleApps** solution filter | After EN labels, confirm filter still scopes to `{solution_id}` (e.g. `solutionColumnDS eq sln.{S}`) — remap ids if import used `{source_host}` literals |

**Investigation helper (instance scratch only):** one-off Cyrillic scan script pattern — ontology walk + `GET webapi/Dataset/.../orgStructureApps` — lives under `{instance_progress_dir}/docs/_scratch/`, not cmw-platform-agent.

---

## Operations under `doc.XXXX`

| Step | Tool |
|------|------|
| Resolve `doc.{N}` → `{app}` + template system name | GetAxioms(`doc.{N}`) or TemplateService row |
| List buttons | `list_buttons` / UserCommand List |
| Edit labels | `edit_or_create_button` (`operation: "edit"`) — preserve `kind`, `create_form`, context |
| Toolbar override | If label unchanged in UI, fix **toolbar item** `name` ([ui_components.md](ui_components.md)) |

Re-fetch Operations UI or list API to verify.

---

## List `lst.*` under `doc.XXXX`

Typical first list in a wave: `lst.6` (example from operator URL — **id varies by host**).

| Step | Action |
|------|--------|
| Open list metadata | Dataset behind `lst.{M}` via template list config or resolver |
| Columns + filters | Full GET → translate RU headers/expressions → PUT with `globalAlias` |
| Empty list after EN fix | Check **`systemFilterExpression`** and **USER()** / catalog **Code_calc** — not only display names ([platform_usage_discoveries.md](platform_usage_discoveries.md)) |

---

## Common blockers

| Symptom | Likely cause | Next step |
|---------|--------------|-----------|
| PUT 400 `globalAlias` null | Missing alias injection | `build_global_alias` + PUT parent app endpoint |
| 200 PUT but filter unchanged | FilterTree API strip | Designer **Tables** tab |
| Button EN in API, RU in UI | Toolbar item name override / cache | `get_button` truth; hard refresh |
| HTTP 500 on dataset GET | Host/template mismatch | Confirm `CMW_BASE_URL` and template resolution |

---

## Documentation flush

| Scope | Where |
|-------|--------|
| Repeatable API lesson | This file or [platform_usage_discoveries.md](platform_usage_discoveries.md) |
| `doc.XXXX` enumeration + per-template checklist | [process_model_template_localization.md](process_model_template_localization.md) |
| Wave ids, `operations[]`, verification | `{instance_progress_dir}/localization/migration_progress/YYYYMMDD_docxxxx_ru_localization.json` |

---

## Related

- [process_model_template_localization.md](process_model_template_localization.md)
- [ui_components.md](ui_components.md)
- [platform_usage_discoveries.md](platform_usage_discoveries.md)
- [localization_instruction.md](localization_instruction.md)
