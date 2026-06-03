# Documentation template localization (`doc.XXXX`)

Platform-generic workflow for localizing **documentation templates** on an EN target instance (`{target_host}`). UI navigation uses **`#RecordType/doc.{N}/…`** — treat **`doc.XXXX` as a family** (`doc.1`, `doc.2`, …), not a single template.

**Instance scope:** concrete `doc.*` ids, wave results, and `operations[]` live in `{instance_progress_dir}/localization/migration_progress/` — see [instance_repo_documentation_boundary.md](instance_repo_documentation_boundary.md).

**Related:** [en_template_ru_leftover_cleanup.md](en_template_ru_leftover_cleanup.md) (solution-level dataset grids + first-wave targets) · [platform_usage_discoveries.md](platform_usage_discoveries.md) (dataset PUT, filters) · [ui_components.md](ui_components.md) · [localization.md](localization.md) Workflow B.

---

## What `doc.XXXX` means

| Term | Meaning |
|------|---------|
| **Documentation template** | `DocumentationTemplate` container in the **RecordType** admin tree (`#RecordType/doc.{N}/…`) — **not** a generic document file template and **not** a BPMN process model alias |
| **`doc.{N}`** | Platform **template id** in URLs (`#RecordType/doc.1/Operations`) — enumerate **all** `doc.*` on the target host |
| **`pa.{N}`** | Separate id family for **process templates** in TemplateService (`Type: Process`) — do not assume `doc.N` equals `pa.N` without GetAxioms |
| **`oa.{N}`** | **Record template** (business data templates) — different scope; see boundary doc |

**Rule:** Batch JSON and operator notes must record `meta.record_type_id: "doc.1"` (or `doc.2`, …) when the fix is **documentation-template** scoped — do not file under unrelated `oa.*` batches unless the wave explicitly spans both.

---

## Enumerate all `doc.XXXX` on `{target_host}`

Use **read-only** discovery first (`CMW_USE_DOTENV=true`, `{target_host}` in `.env`). Never write to `{source_host}` (RU reference).

Sort by numeric suffix (`doc.1` … `doc.N`); localize **each** id independently. On a typical FM EN target, expect **~69** documentation templates (`doc.1`–`doc.69`) — not a single `doc.1`.

### 1. Ontology alias scan (preferred — complete `doc.*` set)

`POST {target_host}/api/public/system/Base/OntologyService/GetWithMultipleValues` with body `{"predicate": "cmw.container.alias", "min": 1, "max": 10000}`.

Keep keys matching `^doc\.\d+$`. For each id, `POST …/GetAxioms` with raw body = `doc.{N}` to read `cmw.object.name` (display) and `cmw.container.alias` (Web API **system name**, e.g. `template_ProcessModelNone_systemSolution`).

```python
# Pattern (agnostic) — venv + dotenv + CMW_BASE_URL={target_host}
import re
import requests
from tools import requests_ as r_

DOC_RE = re.compile(r"^doc\.(\d+)$")
cfg = r_._load_server_config()
base = cfg.base_url.rstrip("/")
session = requests.Session()
session.auth = (os.environ["CMW_LOGIN"], os.environ["CMW_PASSWORD"])
headers = {"Content-Type": "application/json", "Accept": "application/json"}
resp = session.post(
    f"{base}/api/public/system/Base/OntologyService/GetWithMultipleValues",
    headers=headers,
    json={"predicate": "cmw.container.alias", "min": 1, "max": 10000},
    timeout=60,
)
for item in resp.json():
    doc_id = re.sub(r"\s*:.*", "", str(item.get("key") or item.get("id") or ""))
    if not DOC_RE.match(doc_id):
        continue
    # GetAxioms(doc_id) → name + cmw.container.alias for PUT below
```

`webapi/DocumentationTemplate/List/{app}` may **404** on some hosts; do not rely on it alone.

### 2. TemplateService scan (supplement)

`POST {target_host}/api/public/system/Solution/TemplateService/List` with body `{"Type": "<Type>"}` for each of: `Record`, `Process`, `Role`, `OrgStructure`.

Collect every item where `id` matches `doc.\d+`. Merge with step 1; prefer ontology when counts disagree.

### 3. Resolve `doc.{N}` → application + template system name

Child APIs (`webapi/Dataset/List/Template@{app}.{template}`, buttons, forms) need **system names**, not only `doc.N`.

**UI id vs API alias (critical on FM EN targets):**

| Surface | Example for process catalog root |
|---------|----------------------------------|
| Hash **RecordType id** | `doc.1` |
| Administration **system name** field | `template_ProcessModelNone_systemSolution` |
| Toolbar/dataset **owner** in GET body | `ProcessModelNone` |
| `container.type` in Web API | `DocumentationTemplate` (not `RecordTemplate`) |

List/get children with **`template_ProcessModelNone_systemSolution`** (or per-item alias from GetAxioms). PUT bodies must keep **`DocumentationTemplate`** on `container` — `edit_or_create_toolbar` defaults to `RecordTemplate` and fails.

`GetAxioms` on `doc.{N}` returns `cmw.container.alias` (e.g. `template_StartEventNone_systemSolution`) and `cmw.solution` (often `sln.1` → app **`CMW_FM`** for Web API).

**Display-name PUT (batch-friendly):** `edit_or_create_record_template` with `application_system_name: "CMW_FM"`, `operation: "edit"`, `system_name` = value from `cmw.container.alias`, `name` = EN string. Under the hood: `PUT webapi/RecordTemplate/CMW_FM` with `globalAlias.type: DocumentationTemplate`. **Do not** pass `sln.1` as application — PUT fails with “Cannot get id for solution by solution alias: sln.1”.

Alternative for single fields: `AddStatement` on `cmw.object.name` ([en_template_ru_leftover_cleanup.md](en_template_ru_leftover_cleanup.md) Pattern A).

Optional: `get_platform_entity_url` / `platform_entity_resolver` when the id is indexed — confirm `application` + `parent_system_name` before list/edit calls.

### 4. Ontology listing (supplement)

`get_ontology_objects` with `types: ["ProcessTemplate", "UserCommand", "Dataset", "Form", "Toolbar"]` and `application_system_name: "<app>"` — filter results whose template parent resolves to a `doc.*` id from step 1. Use for Cyrillic scans at scale ([localization.md](localization.md) Workflow B tools).

### 5. UI / browser (when API id set is incomplete)

Navigate `#RecordType/doc.{N}/Administration` per discovered id; note sibling areas (**Operations**, **Lists**, **Forms**, **Context**). Use browser only to **discover** missing ids or verify labels — prefer API PUT for bulk renames.

**Do not** commit host-specific id tables to this repo; flush discovered `doc.*` list to instance progress JSON `documentation_templates[]` (or `meta.documentation_template_count`).

---

## Per-template localization checklist

For **each** `doc.XXXX`, walk the tree in this order (re-verify with GET after each wave):

| Area | UI hash (pattern) | API / tools | What to translate |
|------|-------------------|-------------|-------------------|
| **Administration** | `#RecordType/doc.{N}/Administration` | Template metadata via GetAxioms / template GET | Display name, description fields with Cyrillic |
| **Operations** | `#RecordType/doc.{N}/Operations` | `webapi/UserCommand/List/Template@{app}.{tmpl}` · `edit_or_create_button` | Button **name**, **description**, toolbar item **name** overrides |
| **Lists** | `#RecordType/doc.{N}/Lists/lst.{M}` | Dataset GET/PUT per list’s backing table | Column **name**, dataset **name**, filters |
| **Linked tables** | Resolver / dataset aliases under template | `get_dataset` → merge → `edit_or_create_dataset` | Column headers, `systemFilterExpression`, scalar `filter` |
| **Toolbars** | Linked to datasets/lists | `get_toolbar` + raw `PUT webapi/Toolbar/{app}` (preserve `DocumentationTemplate`) | Toolbar title, item display names — **system-solution toolbars may ignore PUT** (verify GET); use designer if labels stay RU |
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

Documentation-template lists almost always bind to **Dataset** entities — same rules as record templates:

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
| **Embedded BPMN diagram** layout, gateway conditions in designer | Often no safe public PUT — browser verification only (separate from `doc.*` template metadata) |
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
└─ Process designer / transition guard (BPMN runtime)?
   └─ Browser (API last) — not the same as doc.* template localization
```

**Order:** OpenAPI → `tools/` → skills → browser ([SKILL.md §9](../SKILL.md#9-growing-platform-skills)).

---

## Parallel workers

Independent **`doc.{N}`** trees (no shared dataset writes) may run in **parallel subagents** — one template id per agent. Serialize PUTs on the same dataset. One configuration backup per milestone ([cmw-platform-backup-launch](../../cmw-platform-backup-launch/SKILL.md)).

---

## Verification

1. Re-GET each edited dataset/button; confirm Cyrillic gone from user-visible fields.
2. Open UI list/Operations tabs under each `doc.XXXX` (or snapshot) for regressions.
3. Flush instance `operations[]` / `documentation_templates[]` — not platform `docs/_scratch/`.

---

## Related platform docs

- [instance_repo_documentation_boundary.md](instance_repo_documentation_boundary.md) — `doc.*` vs `oa.*`, scratch, promotion
- [en_template_ru_leftover_cleanup.md](en_template_ru_leftover_cleanup.md) — stub → instance playbook (orgStructureApps, roleApps, first-wave URLs)
- [browser_automation.md](browser_automation.md) — `#RecordType/…` hash patterns
- [api_endpoints.md](api_endpoints.md) — Template list endpoints
