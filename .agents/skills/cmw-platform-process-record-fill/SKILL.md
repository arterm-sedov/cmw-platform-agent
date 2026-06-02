---
name: cmw-platform-process-record-fill
description: >-
  Fills or updates running process instance records (work orders, Sobytiya, form.oa)
  on Comindware Platform via Records API or create_edit_record. Use for process instance,
  work order, Sobytiya, form.oa, running process, demo fill, Obrabotkasobytiya, or
  placeholder message text on in-progress rows. Instance-agnostic via CMW_BASE_URL.
  Prefer OpenAPI and GET/PUT before browser; browser only for workflow buttons/transitions.
---

# CMW Platform — Process record demo fill

Update **existing** process-owned template rows (e.g. Volga **Sobytiya** / Work Orders) with US FM demo text **in place**. Do not clone TR record ids, workflow state, or Russian copy onto FR unless a spec explicitly requires new instances via ProcessObjectService.

**Workflow order:** OpenAPI → `get` record tools → `Attribute/List` → PUT/`create_edit_record` → verify GET → browser only for workflow buttons/transitions. See [cmw-platform](../cmw-platform/SKILL.md) § Workflow order.

**Instance progress:** log in `{instance_progress_dir}` — `localization/migration_progress/YYYYMMDD_phase1_process_demo_fill.json`, findings in `docs/YYYYMMDD_phase1_process_demo_fill_findings.md`. Do not store instance batch JSON in cmw-platform-agent.

## How it works (technical)

1. **Process instances are records** on a template (e.g. Volga `Sobytiya`) bound to a process (e.g. `Obrabotkasobytiya`). UI URL: `#form/oa.{appId}/form.{formId}/{recordId}` (example: `#form/oa.104/form.219/166`).
2. **Discover aliases:** `GET webapi/Attribute/List/Template@{App}.{Template}` → use **PascalCase** API aliases (`Soobschenie`) in PUT/`create_edit_record`, **not** lowercase keys from `GET webapi/Record/{id}` (`soobschenie`). Wrong casing often returns 200 with no field change.
3. **GET** full record first — read current state, process binding, status refs, assignees. Preserve workflow-related ids and status unless intentionally advancing the process.
4. **PUT/PATCH** via Records API or `create_edit_record` (`operation=edit`, numeric `record_id`) with **minimal** changed fields + preserved ids/status.
5. **Do not clone TR process state** — edit existing FR rows; optional TR list harvest is **read-only** for message patterns. Create new instances via ProcessObjectService only when the spec requires it.
6. **Indicator widgets** (e.g. `dwc.237` Active count) are **query-driven** on template rows (e.g. status not complete/cancelled). Filling in-progress rows updates dashboards **without** editing widget JSON.

## Discovery checklist

| Step | Action |
|------|--------|
| URL → template | `#form/oa.*` → object app; `GET webapi/Record/{id}` for template/process hints (lowercase in body) |
| Writable fields | `GET webapi/Attribute/List/Template@{App}.{Template}` → PascalCase aliases |
| KPI / indicator | `#indicator/id=dwc.{N}` → backup `WidgetConfigs/*.json` or export: `QueryRule`, `Template`, `AggregationMethod` |
| TR patterns (optional) | `GET webapi/Records/Template@{App}.{Template}` or `list_template_records` — patterns only |

## Edit pattern

```http
GET webapi/Record/{recordId}
GET webapi/Attribute/List/Template@Volga.Sobytiya
PUT webapi/Record/{recordId}
Content-Type: application/json

{ "Soobschenie": "...", "Soobscheniedlyaispolnitelya": "...", "Kommentariydispetchera": "..." }
```

**Tool path (preferred):**

```text
create_edit_record(
  operation=edit,
  application_system_name=Volga,
  template_system_name=Sobytiya,
  record_id=<numeric>,
  values={ <PascalCase aliases> }
)
```

**Verify:** `GET webapi/Record/{id}` — compare lowercase keys in response to intended text.

## US FM content

- **Tone:** Executive, operational English for tenant-facing and dispatcher fields (complaint, assignee instructions, dispatcher comment).
- **Scenarios:** Synthetic US FM (HVAC, parking lighting, restroom exhaust, SLA overdue) — not translated TR Russian.
- **Personas:** Leave existing requester/assignee refs unless the batch spec changes them.
- **Picklists:** Keep existing status/type refs (e.g. In progress `100`) unless advancing workflow is in scope.

**Example field set (Volga Sobytiya):**

| PascalCase alias | Role |
|------------------|------|
| `Soobschenie` | Requester / main message |
| `Soobscheniedlyaispolnitelya` | Assignee instructions |
| `Kommentariydispetchera` | Dispatcher comment |
| `Ssylkanazadachudlyadispetchera2` | Task link title (may mirror `Soobschenie` after save) |
| `Raspolozhenie` | Location text — often empty after PUT while process active; defer to building refs (`Zdaniya`) |

Do not edit computed/UI-only fields (`indikatorotkrytoyzayavki`, `indikatorStatus` HTML).

## Pitfalls

| Issue | Mitigation |
|-------|------------|
| PUT 200 but fields unchanged | PascalCase from Attribute/List, not lowercase GET keys |
| `_put_request` argument order | Body first, endpoint second |
| Location empty after PUT | Likely derived or blocked while process active; use `Zdaniya` hub refs in a later batch |
| Advancing workflow | May require UI user commands — API fill is data-only by default |
| TR→FR ids | Do not map TR record ids onto FR; edit pre-existing FR rows |

## Indicator KPIs

Usually **no API action** on `dwc.*` — populate underlying `Sobytiya` rows so filters match (e.g. status not complete/cancelled). Edit widget config only when label or query rule is wrong.

## Verification

1. `GET webapi/Record/{id}` for each edited id.
2. Open form URLs and optional `#indicator/id=dwc.{N}` for visual proof.
3. Append `fr_edits[]` and `verification_urls` to instance progress JSON.

## Scripts and references

| Resource | Location |
|----------|----------|
| Slim API cheat sheet | [references/process_record_demo_fill.md](../cmw-platform/references/process_record_demo_fill.md) |
| One-off batch example (scratch) | `{instance_progress_dir}/docs/_scratch/phase1_process_demo_fill.py` (not production; pattern reference) |
| Harvest / seed index | [references/scripts_index.md](../cmw-platform/references/scripts_index.md) |

## Cross-links

- [cmw-platform](../cmw-platform/SKILL.md) — workflow order, terminology, growing skills §9
- [cmw-platform-instance-switch](../cmw-platform-instance-switch/SKILL.md) — `CMW_BASE_URL`
- [cmw-platform-backup-launch](../cmw-platform-backup-launch/SKILL.md) — post-batch configuration backup
