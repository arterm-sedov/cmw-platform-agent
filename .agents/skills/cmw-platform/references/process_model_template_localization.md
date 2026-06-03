# Process model template localization (`doc.XXXX`)

Platform-generic workflow for localizing **process model templates** on an EN target instance (`{target_host}`). UI navigation uses **`#RecordType/doc.{N}/…`** — treat **`doc.XXXX` as a family** (`doc.1`, `doc.2`, …), not a single template.

**Instance scope:** concrete `doc.*` ids, wave results, and `operations[]` live in `{instance_progress_dir}/localization/migration_progress/` — see [instance_repo_documentation_boundary.md](instance_repo_documentation_boundary.md).

**Related:** [en_template_ru_leftover_cleanup.md](en_template_ru_leftover_cleanup.md) (solution-level dataset grids + first-wave targets) · [platform_usage_discoveries.md](platform_usage_discoveries.md) (dataset PUT, filters) · [ui_components.md](ui_components.md) · [localization.md](localization.md) Workflow B.

---

## What `doc.XXXX` means

| Term | Meaning |
|------|---------|
| **Process model template** | BPMN / process-documentation container in the **RecordType** admin tree — **not** a generic document file template |
| **`doc.{N}`** | Platform **template id** in URLs (`#RecordType/doc.1/Operations`) — enumerate **all** `doc.*` on the target host |
| **`pa.{N}`** | Separate id family for **process templates** in TemplateService (`Type: Process`) — do not assume `doc.N` equals `pa.N` without GetAxioms |
| **`oa.{N}`** | **Record template** (business data templates) — different scope; see boundary doc |

**Rule:** Batch JSON and operator notes must record `meta.record_type_id: "doc.1"` (or `doc.2`, …) when the fix is process-model scoped — do not file under unrelated `oa.*` batches unless the wave explicitly spans both.

---

## Enumerate all `doc.XXXX` on `{target_host}`

Use **read-only** discovery first (`CMW_USE_DOTENV=true`, `{target_host}` in `.env`). Never write to `{source_host}` (RU reference).

### 1. TemplateService scan (preferred API)

`POST {target_host}/api/public/system/Solution/TemplateService/List` with body `{"Type": "<Type>"}` for each of: `Record`, `Process`, `Role`, `OrgStructure`.

Collect every item where `id` matches `doc.\d+` (regex `^doc\.(\d+)$`).

```python
# Pattern (agnostic) — run from cmw-platform-agent with active venv + dotenv
import ast, re
from tools import requests_ as r_

DOC_RE = re.compile(r"^doc\.(\d+)$")
found: list[dict] = []
for tpl_type in ("Record", "Process", "Role", "OrgStructure"):
    res = r_._post_request(
        {"Type": tpl_type},
        "api/public/system/Solution/TemplateService/List",
    )
    if not res.get("success"):
        continue
    raw = res["raw_response"]
    items = ast.literal_eval(raw) if isinstance(raw, str) else raw
    for item in items:
        if isinstance(item, dict) and DOC_RE.match(str(item.get("id", ""))):
            found.append(item)
```

Sort by numeric suffix; localize **each** id independently.

### 2. Resolve `doc.{N}` → application + template system name

Child APIs (`webapi/Dataset/List/Template@{app}.{template}`, buttons, forms) need **system names**, not only `doc.N`.

`POST {target_host}/api/public/system/Base/OntologyService/GetAxioms` with body = raw entity id string (e.g. `doc.1`). Read axioms for solution alias / container alias (same pattern as dataset ontology walks in harvest tooling).

Optional: `get_platform_entity_url` / `platform_entity_resolver` when the id is indexed — confirm `application` + `parent_system_name` before list/edit calls.

### 3. Ontology listing (supplement)

`get_ontology_objects` with `types: ["ProcessTemplate", "UserCommand", "Dataset", "Form", "Toolbar"]` and `application_system_name: "<app>"` — filter results whose template parent resolves to a `doc.*` id from step 1. Use for Cyrillic scans at scale ([localization.md](localization.md) Workflow B tools).

### 4. UI / browser (when API id set is incomplete)

Navigate `#RecordType/doc.{N}/Administration` per discovered id; note sibling areas (**Operations**, **Lists**). Use browser only to **discover** missing ids or verify labels — prefer API PUT for bulk renames.

**Do not** commit host-specific id tables to this repo; flush discovered `doc.*` list to instance progress JSON `meta.process_model_templates[]`.

---

## Per-template localization checklist

For **each** `doc.XXXX`, walk the tree in this order (re-verify with GET after each wave):

| Area | UI hash (pattern) | API / tools | What to translate |
|------|-------------------|-------------|-------------------|
| **Administration** | `#RecordType/doc.{N}/Administration` | Template metadata via GetAxioms / template GET | Display name, description fields with Cyrillic |
| **Operations** | `#RecordType/doc.{N}/Operations` | `webapi/UserCommand/List/Template@{app}.{tmpl}` · `edit_or_create_button` | Button **name**, **description**, toolbar item **name** overrides |
| **Lists** | `#RecordType/doc.{N}/Lists/lst.{M}` | Dataset GET/PUT per list’s backing table | Column **name**, dataset **name**, filters |
| **Linked tables** | Resolver / dataset aliases under template | `get_dataset` → merge → `edit_or_create_dataset` | Column headers, `systemFilterExpression`, scalar `filter` |
| **Toolbars** | Linked to datasets/lists | `list_toolbars` / `edit_or_create_toolbar` | Toolbar title, item display names |
| **Forms / cards** | `#form/...` under template | `list_forms`, form edit tools | Titles, section labels (non-calculated) |

**Cyrillic scan:** walk JSON from GET responses; flag strings matching `[\u0400-\u04FF]`. Skip **system names** unless the migration wave explicitly renames aliases (Workflow A — [localization.md](localization.md)).

---

## RU → EN rules (display text)

| Do translate | Do not change (unless alias-rename wave) |
|--------------|----------------------------------------|
| `name`, `title`, `displayName`, column `name`, button `name`, toolbar item `name` | **`Code_calc`** and calculated codes used in filters/indicators |
| `systemFilterExpression` **RU literals** (status words, labels embedded in SPARQL) | Stable **system names** / `globalAlias.alias` unless Workflow A |
| Stale **filter.value** catalog ids from `{source_host}` | Read-only **calculated** attribute definitions (change breaks expressions) |

Use US FM facility-management English on `{target_host}`; `{source_host}` read-only for meaning only.

---

## Dataset PUT pitfalls (`globalAlias`, filters)

Process-model lists almost always bind to **Dataset** entities — same rules as record templates:

| Pitfall | Remediation |
|---------|-------------|
| PUT `webapi/Dataset/{app}/Dataset@…` returns **405** | PUT full body to **`webapi/Dataset/{app}`** with injected **`globalAlias`** |
| GET returns **`globalAlias: null`** | Set via `build_global_alias("Dataset", template_system_name, dataset_system_name)` before PUT |
| **`systemFilterExpression`** still references RU status words or `{source_host}` ids | GET status/catalog rows on **target** → remap expression literals and ids |
| **`FilterTree` `children: []`** after PUT | Some hosts **strip** FilterTree children via API — use **designer UI** for list tab filters ([platform_usage_discoveries.md](platform_usage_discoveries.md) § Cancellation / Terminate lists) |
| Partial PUT drops columns/toolbar | Merge **full** GET body; change only intended keys |

→ Detail: [platform_usage_discoveries.md](platform_usage_discoveries.md#dataset-put-endpoint-globalalias-and-full-body)

---

## Read-only / designer-only

| Item | Why |
|------|-----|
| **Calculated fields** (`*_calc`, expressions, indicators) | Renaming or retyping breaks lists and KPIs — translate **display** on non-calculated attributes only unless a dedicated expression migration wave |
| **BPMN diagram** layout, gateway conditions in designer | Often no safe public PUT — browser verification only |
| **FilterTree children** when API returns empty after PUT | Tables → tab filter UI |
| **Scenario / route** wiring inside process designer | Extend [browser_automation.md](browser_automation.md) when API lacks coverage |

---

## Browser vs API decision tree

```
Need to fix RU display strings on doc.XXXX?
├─ Dataset / button / toolbar / form metadata?
│  ├─ GET succeeds, PUT with globalAlias succeeds → API only (edit_or_create_* )
│  └─ PUT 4xx/5xx or FilterTree won't persist → browser designer fallback + document blocker
├─ Enumerate all doc.* ids?
│  ├─ TemplateService lists doc.* → API
│  └─ Missing from API → RecordType Administration UI pass
└─ Process diagram / transition guard?
   └─ Browser (API last)
```

**Order:** OpenAPI → `tools/` → skills → browser ([SKILL.md §9](../SKILL.md#9-growing-platform-skills)).

---

## Parallel workers

Independent **`doc.{N}`** trees (no shared dataset writes) may run in **parallel subagents** — one template id per agent. Serialize PUTs on the same dataset. One configuration backup per milestone ([cmw-platform-backup-launch](../../cmw-platform-backup-launch/SKILL.md)).

---

## Verification

1. Re-GET each edited dataset/button; confirm Cyrillic gone from user-visible fields.
2. Open UI list/Operations tabs under each `doc.XXXX` (or snapshot) for regressions.
3. Flush instance `operations[]` / `meta.process_model_templates[]` — not platform `docs/_scratch/`.

---

## Related platform docs

- [instance_repo_documentation_boundary.md](instance_repo_documentation_boundary.md) — `doc.*` vs `oa.*`, scratch, promotion
- [en_template_ru_leftover_cleanup.md](en_template_ru_leftover_cleanup.md) — orgStructureApps, roleApps, first-wave URLs
- [browser_automation.md](browser_automation.md) — `#RecordType/…` hash patterns
- [api_endpoints.md](api_endpoints.md) — Template list endpoints
