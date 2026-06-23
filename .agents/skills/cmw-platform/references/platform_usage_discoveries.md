# Platform usage discoveries (agnostic)

Repeatable lessons from FM-style migration and demo-fill work on Comindware Platform. **Platform-generic only** — use placeholders `{CMW_BASE_URL}`, `{source_host}`, `{target_host}`, `{property_id}`; never commit tenant hosts, record ids, or account literals in this file.

**Instance plans, harvest JSON, and wave checklists:** see `{instance_progress_dir}` (migration progress, operator runbooks, gap analyses).

**Mandatory wave-end documentation:** After any batch/wave that discovers **new** API or UI manipulation patterns, append an agnostic subsection here (or extend the matching reference) **before** ending the session wave or claiming done. Use placeholders only (`{CMW_BASE_URL}`, `{source_host}`, `{target_host}`, `{instance_progress_dir}`, `{property_id}`) — **no** tenant hosts, numeric record ids, or account literals in this file. Instance `operations[]` / `map[]` flush under `{instance_progress_dir}/localization/migration_progress/` remains mandatory in parallel ([cmw-platform SKILL §9](../SKILL.md#9-growing-platform-skills)).

---

## Cross-instance record IDs (critical)

Record ids on `{source_host}` and `{target_host}` are **different namespaces**. A numeric id or `Record/account.{id}` that resolves on `{source_host}` is **not valid** on `{target_host}` unless harvest `map[]` proves the pairing.

| Never | Always |
|-------|--------|
| PUT lookup / Record-ref values using `{source_host}` numeric ids on `{target_host}` | Resolve on `{target_host}` via **Code_calc**, catalog **system name**, or GET list on the **target** catalog template |
| Assume the same id means the same catalog row (status, TechnicalSystem, meter type, property) | Match **business code** on `{source_host}` read-only, then GET the matching row on `{target_host}`; persist in progress `map[]` (TR→EN) |
| Use `{source_host}` `account.{id}` in assignee / filter literals | Use `{target_host}` **AccountService** ids only — assignee fields expect `Record/account.{target_id}` after staff-account link |
| Verify writes with list API alone when refs look empty | **GET `webapi/Record/{id}`** on `{target_host}` — response keys are often **lowercase** (`workStatus_calc`, `assignee_calc`); `GetPropertyValues` with wrong alias casing can look empty while the row still holds stale refs |

**Typical stale-id symptoms:** subsystem list tabs empty while Records API returns rows (dataset `filter.value` still holds `{source_host}` catalog id); KPI / status pickers show wrong catalog entity (same numeric id, different template on EN); assignee shows orphan or wrong persona (staff row id copied into `Assignee_calc` as `account.{staff_id}`, or `{source_host}` account id on `{target_host}`).

**Remediation order:** (1) dataset filters GET → merge → PUT, (2) catalog rows via target GET, (3) dependent record PUT with PascalCase aliases (`WorkStatus_calc`, `Assignee_calc`, `TechnicalSystem_calc`), (4) flush `operations[]` under `{instance_progress_dir}/localization/migration_progress/`.

→ Harvest/seed: [record_harvest_seed.md#cross-instance-lookup-ids-never-copy-source-numeric-ids](record_harvest_seed.md#cross-instance-lookup-ids-never-copy-source-numeric-ids) · Growing skills: [cmw-platform SKILL §9](../SKILL.md#9-growing-platform-skills) · Dataset filters: [List dataset filters with stale source catalog ids](#list-dataset-filters-with-stale-source-catalog-ids) · Assignees: [Resolve assignees from target Administration account list](#resolve-assignees-from-target-administration-account-list)

---

## Resolve assignees from target Administration account list

Assignee and account-reference fields must use **platform login ids** that exist on `{target_host}` — not `{source_host}` account ids, not **Staff** template row ids mistaken for accounts, and not ids that pass a list scan but fail **Record GET**.

| Step | Action |
|------|--------|
| 1 — Harvest catalog | Open **Account Administration** (`#RecordType/{account_template}/Administration`, e.g. `aa.2`) or `POST …/Base/AccountService/List` on `{target_host}`. Build `account.{id} → {username, fullName}`. |
| 2 — Verify each candidate | `POST …/AccountService/Get` **and** `GET webapi/Record/account.{id}`. Reject ids where Record GET returns *“object does not exist”* even if a prior migration batch wrote them. |
| 3 — Map Staff ↔ login | After staff-account link, persist `{staff_row_id} → account.{platform_id}` in progress JSON — **do not** PUT `account.{staff_row_id}`. |
| 4 — Field shape | `Assignee_calc` / `ResponsibleStaff_calc` (account type) → `Record/account.{id}` string on PUT. Some inspection attributes (`AssignedAssignee_calc`) store **numeric account id only** — match attribute type from GET, not a single convention. |
| 5 — Audit templates | Scan WorkOrders, MaintenanceExecution, PM plans, Inspections, Staff, Annual plans for assignee aliases; fix stand rows to role-matched catalog accounts (primary PM persona from catalog username, e.g. `engineer`). |
| 6 — Flush | `operations[]` + `bad_to_good_map` under `{instance_progress_dir}/localization/migration_progress/`; merge into stand-priority batch when applicable. |

**Never copy** `{source_host}` assignee literals or reuse `{target_host}` ids from another migration wave without re-checking Administration — numeric collisions (staff row vs account id) produce silent orphans in UI pickers.

→ Staff link: [cmw-platform-staff-account-link](../../cmw-platform-staff-account-link/SKILL.md) · Employee attach: [employee_account_attach.md](employee_account_attach.md)

---

## Indicator widgets (count / list KPIs)

| Fact | Action |
|------|--------|
| Indicator widgets (`dwc.*`) are **query-driven** on an underlying **template** plus list/dataset filters. | Read widget config (`QueryRule`, `Template`, `AggregationMethod`) from export/`WidgetConfigs` or `#indicator/id=dwc.{N}` — do not treat the widget as a data store. |
| Filters often reference **status catalog** attributes via **Code_calc** (or equivalent calculated code fields), e.g. payment or work-order status codes. | When a KPI shows **zero** but lists look populated, fix **underlying rows** (status codes, refs, property scope) — not widget JSON. |
| Populating rows that match the widget’s filter updates the KPI **without** editing widget configuration. | Prefer record PUT/`create_edit_record` on the driving template; edit widget config only when the rule or label is wrong. |

→ Detail: [cmw-platform-process-record-fill](../../cmw-platform-process-record-fill/SKILL.md) § Indicator KPIs · [process_record_demo_fill.md](process_record_demo_fill.md)

---

## Work orders and conflicting list filters

One **record** cannot simultaneously satisfy **conflicting** dataset or indicator pool filters (e.g. “needs attention” vs “in progress” vs a dedicated indicator slice).

| Situation | Approach |
|-----------|----------|
| Multiple dashboards/lists need different status pools | Use **separate records** with disciplined status/refs, or advance status so only one pool applies — do not expect one row to appear everywhere. |
| Demo fill for KPIs | Align **WorkOrderStatus** / process status **Code_calc** (and related refs) with each target list’s filter before blaming list or widget config. |

→ Process-owned rows: [cmw-platform-process-record-fill](../../cmw-platform-process-record-fill/SKILL.md)

---

## WorkOrders Completed list (`ServiceRequestCompleted`) — USER() and Code_calc

The **Completed** tab (`ServiceRequestCompleted` / legacy `Moivypolnennyezayavki`) often ships with **SPARQL** `systemFilterExpression`: `_creator = cmw:currentUser` **and** status catalog `Code_calc` equal to a **complete** literal from export (frequently `Complete_calc`).

| Symptom | Cause | Fix |
|---------|-------|-----|
| **0 rows** for admin/dispatcher while Records API shows many `WorkOrders_calc` rows | Signed-in user did not create the stand rows **and/or** EN `WorkOrderStatuses_calc` uses a different **Code_calc** than export (e.g. id `{status_id}` → `complete`, not `Complete_calc`) | GET dataset → verify live `Code_calc` on `{target_host}` → PUT remapped `systemFilterExpression` |
| Prior batch edited status/property but list still empty | Filter still gates on **creator**, not assignee | For stand/demo on `{property_id}`: remap to `db->` expression scoped to `Property_calc` + `Status_calc->Code_calc` (read literal from GET catalog row); keep USER() only for true “my completed” personas |
| `?dataset=ServiceRequestCompleted` row count matches whole template | That query may **not** apply dataset filters | Use **list API** with `status_calc` + `property_calc` filters or UI row count after filter PUT |

**Verify:** `GET webapi/Dataset/{app}/Dataset@{template}.ServiceRequestCompleted` (unwrap `response`), merge, inject `globalAlias`, **`PUT webapi/Dataset/{app}`**. Align rows: `Status_calc` → complete status id, `Property_calc` / `Building_calc` / `Floor_calc` stand links, `Assignee_calc` → `Record/account.{target_account_id}` (never `{source_host}` / orphan `account.{staff_row_id}`).

---

## Code_calc and status catalogs

**Code_calc** (and similar calculated code fields on status picklists) are the stable join key between lists, indicators, and invoices/payments — not display text.

| Rule | Why |
|------|-----|
| **Never rename** a Code_calc value during gap-fill or localization touch-up. | Breaks filters, indicators, and cross-template expressions that key on code id. |
| Match the **source catalog by code id** when seeding or aligning target rows. | Display names differ by locale; code id is what queries use. |
| Seed or verify **status catalog rows** before dependent templates (invoices, payments, work orders). | Empty or mismatched catalogs produce empty lists and zero indicators even when parent records exist. |

---

## Gap-fill policy (target locale rows)

When filling demo or migration data on `{target_host}`:

| Condition | Action |
|-----------|--------|
| Target template already has rows (even empty display fields) | **Edit in place** (`create_edit_record` edit / PUT) — match by stable business key from harvest `map[]`. |
| Target template is **truly empty** (harvest count 0, no pre-seeded ids) | **POST** create a small bounded subset; document `harvest_mode: source_empty_pattern_seed` (or instance schema equivalent) in progress JSON. |
| Never | POST with **source** record ids in the body; never map `{source_host}` ids onto `{target_host}` creates. |

→ Contract: [record_harvest_seed.md](record_harvest_seed.md) · Script: [scripts_index.md](scripts_index.md) (`seed_records_from_harvest.py`)

---

## List API cast errors → per-record fallback

Bulk list endpoints (`GET webapi/Records/Template@…`) may fail for the whole page when **one row** has a type the serializer cannot cast (common on mixed legacy + calculated fields).

| Step | Action |
|------|--------|
| 1 | Note failing list offset/id from error or binary search on `limit`/`offset`. |
| 2 | **GET webapi/Record/{id}** or **GetPropertyValues** / single-record read for survivors. |
| 3 | Fix or skip the bad row; retry list — do not assume the template is empty. |

→ Retry patterns: [errors.md](errors.md)

---

## InvoicesPayments and interconnected status catalogs

Templates for **invoices**, **payments**, and **work orders** often share **PaymentStatus**, **WorkOrderStatus**, or sibling picklists.

| Order | Action |
|-------|--------|
| 1 | Harvest or verify status catalog rows on `{source_host}` (code + Code_calc). |
| 2 | Ensure matching catalog rows exist on `{target_host}` **before** invoice/payment line items. |
| 3 | Then seed parent/child records so list filters and indicators see consistent codes. |

Empty payment or invoice lists with populated work orders usually mean **catalog or Code_calc mismatch**, not a broken list API.

---

## Source form-first (replication subagents)

For **any** subagent copying business meaning from `{source_host}` to `{target_host}`:

| Step | Rule |
|------|------|
| 0 | Open the matching **source record form** (read-only UI) — breadcrumbs, status, embedded grids, links — **before** list APIs, export, or datamodel-only reads. |
| 1+ | Then source API read → map to target attribute aliases → target write → target verify → flush progress JSON under `{instance_progress_dir}`. |

Schema export and bulk list endpoints **do not** replace source forms for business logic.

→ [ralph_loop_goal_autonomy.md](ralph_loop_goal_autonomy.md#source-form-first-replication-subagents) · [cmw-platform SKILL §9](../SKILL.md#9-growing-platform-skills)

---

## Mandatory `operations[]` JSON flush

Cross-instance replay and Ralph-style waves require **durable** operation logs — not chat memory.

| Field | Role |
|-------|------|
| `operations[]` | Append-only log of harvest/seed/edit steps (`template`, `source_id`, `target_id`, verb, timestamp) for `seed_records_from_harvest.py` and audit. |
| `map[]` | Idempotent source→target id map; skip create when target id present. |
| `meta.*` | `status`, `errors`, `backup_pending`, `agent_wave`, paths to harvest files. |

Flush under `{instance_progress_dir}/localization/migration_progress/` after each wave; optional progress report under `{instance_progress_dir}/docs/localization/progress_reports/`.

→ [record_harvest_seed.md](record_harvest_seed.md) · [ralph_loop_goal_autonomy.md](ralph_loop_goal_autonomy.md#progress-flush-instance-migration)

---

## Staff, PM settings, and assignee links

Persona and routing UX depends on **account ↔ employee row** links and assignee refs on process/PM templates — not username alone.

| Pattern | Where |
|---------|--------|
| Platform login → **Staff** employee row | [cmw-platform-staff-account-link](../../cmw-platform-staff-account-link/SKILL.md) · [employee_account_attach.md](employee_account_attach.md) |
| Account create / password / groups (before attach) | [cmw-platform-account-bootstrap](../../cmw-platform-account-bootstrap/SKILL.md) |
| Assignee / PM settings attributes on work-order or PM templates | Resolve via `Attribute/List`; preserve existing refs on PUT unless batch spec changes them — link field discovery in staff-account-link skill |

Attach accounts **before** bulk assignee-dependent demo fill when UAT personas must appear in assignee pickers.

### Staff row id vs platform account (IncludeInContainer)

**Staff_calc row id** and **platform login id** are **different namespaces** on `{target_host}`.

| Trap | Detail |
|------|--------|
| **`Assignee_calc` = staff row id** | PUT may return 200; **`GET webapi/Record/account.{staff_id}`** returns **404** — USER()-filtered lists show **0 rows** |
| **Orphan `account.{id}` on assignee** | Progress JSON may carry a stale **`account.{orphan_id}`** while the real login is **`account.{valid_id}`** — **`GET webapi/Record/account.{orphan_id}`** 404; **`AccountService/Get`** with numeric id confirms the real Administration account |
| **Confusing similar numbers** | Engineer login may be **`account.{N}`** while stand progress JSON cites staff row **`{N}`** — verify **Administration → Accounts** (`AccountService/Get`), not Staff list id alone |
| **ResponsibleStaff vs Assignee** | Some templates accept staff-row refs on one field and **`Record/account.{id}`** on another — read Attribute/List owner template per field |

**Remediation:** (1) **IncludeInContainer** (or Attach account UI) to link platform login → Staff row — [cmw-platform-staff-account-link](../../cmw-platform-staff-account-link/SKILL.md); (2) PUT **`Assignee_calc`** with validated **`Record/account.{target_AccountService_id}`**; (3) re-verify list under **that user's session** when dataset uses USER().

Never use `{source_host}` `account.{id}` literals on `{target_host}` assignee fields.

---

## SLN1 / solution template catalog (tier dependency)

Linked templates (datasets, toolbars, nested record templates) depend on **solution-level** catalog entries (often **SLN1** or the application’s solution template list).

| Rule | Action |
|------|--------|
| Import or replicate **parent solution / SLN1 tier** before child templates that reference it. | Missing tier → broken links, empty nav, or failed include on `{target_host}`. |
| After CTF or application import | Grep model for hardcoded `account.{id}` in filters (see [localization.md](localization.md)) — remap on target; role-based assignees are usually safer. |

Instance-specific tier maps and export paths: `{instance_progress_dir}` only.

---

## Calculated location vs writable location text

FM templates often expose **two location concepts** — do not conflate them.

| Field pattern | Writable? | How to populate |
|---------------|-----------|-----------------|
| **`Location` / `Location_calc`** (Equipment, some asset rows) | **No** — calculated from hierarchy | Set **`Property_calc`**, **`Building_calc`**, **`Floor_calc`**, **`Space_calc`** links; verify computed `location` on GET after edit — never PUT `Location` directly |
| **`Location4it`** (WorkOrders_calc and similar) | **Yes** — free-text / IT location | Gap-fill with stand-scoped display strings when lists or forms show empty location columns |

**Equipment location chain:** Property → Building → Floor → Space drives the calculated location breadcrumb. Rows can have Property/Building set but still show empty Location until **Floor_calc** and/or **Space_calc** resolve on `{target_host}`.

**InspectionRoutes_calc** uses a **shorter chain** — see [Inspection routes: Property + Space only](#inspection-routes-property--space-only) below.

Instance gap-fill examples: `{instance_progress_dir}`.

---

## List dataset filters with stale source catalog ids

After RU→EN (or any `{source_host}`→`{target_host}`) migration, **list dataset filters** may still reference **record ids from the source host** (e.g. TechnicalSystem catalog rows that do not exist on `{target_host}`).

| Symptom | Cause | Fix |
|---------|-------|-----|
| Subsystem tab / filtered list shows **zero rows** while template Records API returns data | Dataset `filter.value` still holds **source-era catalog id** | Remap filter to matching **target catalog row** (match by business code / harvest `map[]`, not numeric id carry-over) |
| Records exist but wrong subsystem slice | Filter points at wrong TechnicalSystem (or sibling catalog) | GET dataset → edit `filter.value` only → PUT full body |

**Procedure:** `get_dataset` → preserve full payload → change `filter.value` (or nested filter tree) → **GET → merge → PUT** ([edit_or_create.md](edit_or_create.md#dataset-filters-account-id-literals-and-other-filter-json)). Document before/after in batch `operations[]`. Browser-verify the list shows expected row counts.

**Do not** assume row-level edits alone fix visibility — stale filters hide rows even when record refs are valid.

---

## Dataset PUT: endpoint, `globalAlias`, and full body

Dataset filter edits require **GET → merge → PUT** on the **application** endpoint — not the per-dataset `@` path.

| Mistake | Result | Correct path |
|---------|--------|--------------|
| `PUT webapi/Dataset/{app}/Dataset@{template}.{dataset}` | **405 Method Not Allowed** | **`PUT webapi/Dataset/{app}`** with the **full merged body** (same contract as `edit_or_create_dataset` internal edit) |
| Partial body / missing keys | 400 or silent drop | Preserve columns, toolbar, paging, grouping, `filter`, `systemFilterExpression`, … from GET |
| `globalAlias: null` on GET | **`Value cannot be null. Parameter name: key`** | Inject **`globalAlias`** via **`build_global_alias("Dataset", template_system_name, dataset_system_name)`** before PUT |

| Step | Action |
|------|--------|
| 1 | GET — `get_dataset` or `GET webapi/Dataset/{app}/Dataset@{template}.{dataset}` |
| 2 | Merge filter / `systemFilterExpression` changes only |
| 3 | If `globalAlias` is null or missing, set it from `build_global_alias(...)` |
| 4 | PUT full body to **`webapi/Dataset/{app}`** |

`edit_or_create_dataset` (`operation: "edit"`) already follows this path — raw HTTP must mirror it. Reinforcing **`globalAlias`** here is intentional (cross-ref with [errors.md](errors.md#405-dataset-put-wrong-endpoint)).

→ [edit_or_create.md](edit_or_create.md#dataset-filters-account-id-literals-and-other-filter-json) · `tools/templates_tools/tools_dataset.py`

### MyMaintenance and USER()-scoped maintenance lists

Lists with **`MyMaintenance`** (or similar) dataset plus **`systemFilterExpression`** matching **`Assignee_calc` to current USER()** need **three** checks:

| Check | Detail |
|-------|--------|
| **Status filter literal** | Scalar `filter.value` may still hold a **`{source_host}` catalog id** that maps to a **KPI / meter-type** row on `{target_host}` — remap to target **JobStatuses** (or correct work-status catalog) id by **Code_calc**, not numeric carry-over |
| **Assignee = platform account** | Rows must use a **real** Administration account id (`Record/account.{id}` on PUT; confirm with **`AccountService/Get`** when `webapi/Record/account.{id}` returns an empty wrapper) — see [Staff row id vs platform account](#staff-row-id-vs-platform-account-includeincontainer) |
| **Session context** | Admin session may show **0 rows** while engineer login shows rows — USER() filter is **by design**; verify under the persona that owns PMSettings / assignee |

Dataset filter remap for these lists: [Dataset PUT: endpoint, globalAlias, and full body](#dataset-put-endpoint-globalalias-and-full-body) above.

### Staff utilization (`Staff_calc` / `StaffUtilization` dataset)

The **Staff utilization** list (`#data/{account_template}/lst.{N}` on `{target_host}`, often paired with a **Routes** tab on `defaultList`) can show **0 rows** while `list_template_records` / Administration still returns staff rows.

| Symptom | Likely cause |
|---------|----------------|
| UI **No data to show**, API staff count > 0 | **`StaffUtilization.systemFilterExpression`** resolves `Property_calc` via **`PMSettings_calc` where `RequesterAccount_calc = USER()`** — admin or wrong persona has no PMSettings row |
| Only assignees visible when leaf filter present | Leaf filter **`WorkAssignee_calc eq true`** hides non-assignee staff even after expression fix |

**Remediation:** GET `StaffUtilization` → clear or replace `systemFilterExpression` (stand demo: scope to target **`Property_calc`** id from harvest, not `{source_host}` ids) → optionally remove leaf filter for full roster → inject **`globalAlias`** → PUT **`webapi/Dataset/{app}`**. Re-verify UI under the operator persona that owns PMSettings when USER() scope must stay.

### AccountTemplate staff lists: cmw_account_* empty without bind (safe PUT)

When **`Records API` returns rows** but **`#data/{account_template}/lst.{N}`** shows **0** and columns are **`System#` `cmw_account_fullName`** (staff **`username` null**, **`IncludeInContainer` HTTP 500**):

| Approach | Result on `{target_host}` |
|----------|---------------------------|
| In-place remap `cmw_account_*` **propertyPath** → `Staff_calc` `fullName`/`title`/… in one full-body PUT | **Wipes** dataset (`GET` → **0 columns**) |
| `edit_or_create_dataset` on **`Staff_calc`** without checking **`container.type`** | Often **wipes** — tool defaults **`RecordTemplate`**; live body uses **`AccountTemplate`** |
| **Restore snapshot** → **`systemFilterExpression=null`**, **`filter=null`** → **prepend** visible **`RecordLink_calc`** → **`isHidden=true`** on `cmw_account_*` | **UI rows without account bind** (demo: N names on staff default list / utilization list) |

**Safe sequence:** GET `webapi/Dataset/{app}/Dataset@Staff_calc.{defaultList|StaffUtilization}` → save JSON → clear filters only → add/highlight **`RecordLink_calc`** (staff row has HTML link + name) → hide account columns; **do not** delete the `cmw_*` column entries in the same PUT that rebuilds the array. Re-verify with CDP name-link count on `#data/{account_template}/lst.{staff_list_id}` and paired utilization list routes.

### form.333 platform account cards (Staff utilization form on Administration ids)

**Staff utilization** form **`form.{N}`** on **`#form/{account_template}/form.{N}/account.{id}`** edits **platform login rows** from Account Administration — **not** the numeric **`Staff_calc`** employee ids (`182`–`191`). Same template (`Staff_calc`) accepts both id shapes; do not conflate them.

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| GET **`webapi/Record/account.{id}`** returns null / *Parameter name: key* | Account records are not readable via Record GET on some hosts | Gap-fill with **PUT `webapi/Record/account.{id}`** (lowercase keys: `fullName`, `property_calc`, `department`, …) + **`create_edit_record`** (`Staff_calc`, PascalCase aliases). Verify on **form UI**, not GET. |
| **AccountService/List** still shows generic FullName while form title shows persona-specific name | PUT updates form-bound fields; List reads AccountService profile | **AccountService/Edit** for FullName may HTTP **500** when Username is included — target form UI is authoritative for demo fill. |
| Property / Department / Space / Floor pills show **–** (readonly) on form.333 | Org hierarchy refs are often bound to the **linked employee row** (`form.588/{staff_row_id}`), not the platform account card | Fill hierarchy on the **numeric staff row** when lists/filters need Property/Department; keep platform account card for identity, contact, position. |
| Assignee fields expect **`account.{platform_id}`** | Staff row id **≠** platform account id | Resolve via Administration catalog + staff-account link map in progress JSON — never PUT `account.{staff_row_id}`. |
| PUT returns **triple uniqueness** on `cmw.account.mbox` | Re-sending an existing **mbox** on `webapi/Record/account.{id}` collides with the account triple store | **Omit `mbox`** from gap-fill PUT when List/Get already shows the login email; update `fullName`, `title`, `phone`, and hierarchy refs only. |

→ Staff link: [cmw-platform-staff-account-link](../../cmw-platform-staff-account-link/SKILL.md)

---

## Composite list filters (TechnicalSystem + Space→Floor OR)

Equipment and subsystem tabs often use **nested filter trees**, not a single `filter.value`:

| Pattern | Meaning |
|---------|---------|
| `AND(TechnicalSystem_calc, OR(...))` | Subsystem slice **and** one of several floor scopes |
| `Space_calc` → `Floor_calc` branch | List resolves floor via **space** link, not direct `Floor_calc` on equipment row |

After `{source_host}`→`{target_host}` migration, **remap every stale id** in the tree — TechnicalSystem catalog id, each Space id, and each Floor id in OR branches. Fixing row-level `TechnicalSystem_calc` alone leaves the list **empty** when the dataset filter still references source-era ids.

**Procedure:** GET dataset → walk nested `filter` / `filters` / `conditions` → replace all catalog and hierarchy ids using target GET + harvest `map[]` → inject `globalAlias` if null → PUT.

Instance wave detail: `{instance_progress_dir}` only.

---

## WorkStatus_calc vs KPI meter catalog (numeric id collision)

On `{target_host}`, the **same numeric record id** can refer to **different templates** than on `{source_host}`. A common FM trap: **`WorkStatus_calc`** on **MaintenanceExecution_calc** vs a **KPI / meter value type** row sharing the id (e.g. water-consumption type vs **JobStatuses** “preplanned”).

| Symptom | Fix |
|---------|-----|
| Status shows wrong label or list filter misses rows | Resolve status via **JobStatuses_calc** (or correct catalog) on **target** using **Code_calc** / harvest map — never assume source numeric id |
| PUT 200 but status unchanged | PascalCase **`WorkStatus_calc`**; verify catalog template on GET |

Always verify the **catalog template** for the id on `{target_host}` before PUT.

---

## Contract lists: All vs filtered datasets (empty sibling lists)

**Contracts_calc** (or similar) may expose:

| List role | Typical dataset | Symptom when broken |
|-----------|-----------------|---------------------|
| “All contracts” | `All` / unfiltered | Rows visible |
| Active / default / cancellation | `defaultList`, `Cancellation`, grouped Active | **Empty** while “All” shows rows |

**Cause:** `defaultList` **Status_calc** (and sometimes **MainDocument_calc**) filter literals still hold **`{source_host}` status record ids**; stand rows use **target** status ids.

**Fix:** GET each affected dataset → remap **all** status (and boolean) filter values to target catalog ids → PUT with **`globalAlias`** if GET returned null. Re-verify **lst.272** (cancellation) separately from **lst.275** (active/default).

### Cancellation / Terminate lists: systemFilterExpression vs FilterTree PUT

Some **Contracts_calc** sibling datasets use **`systemFilterExpression`** (e.g. `LinkedDocuments_calc->ContractType_calc` traversal) instead of scalar **`filter.value`** literals:

| Symptom | Likely cause |
|---------|----------------|
| Terminate tab **0 rows** while All shows contracts and termination addenda exist | Expression traversal fails on `{target_host}` after import; or empty **`FilterTree` with `children: []`** left on dataset blocks all rows |
| PUT `filter` with new **`FilterTree` children** returns 200 but GET shows **`children: []`** | **`webapi/Dataset/{app}` PUT strips FilterTree children** — regular filter not persistable via API on some hosts |
| Hardcoded `OR(a->id=="…")` in expression shows **over-broad** row set | Malformed/unsupported expression syntax may be **ignored** (fallback list), not fail-closed |

**Remediation order:** (1) confirm addendum rows + parent **`LinkedDocuments_calc`** / **`MainContract_calc`** on `{target_host}`, (2) PUT **`systemFilterExpression`** with **`filter` key removed** (not empty tree), (3) if still empty → **designer UI** (Tables → Cancellation) — API cannot set FilterTree children, (4) re-verify **lst.275** unchanged.

---

## AnnualPlans_calc: year / assignee hub and PlanWorks

**AnnualPlans_calc** is a **planning container**, not a work-order row:

| Field / area | Role |
|--------------|------|
| **Year**, **Assignee**, date window | Scope which **WorkTasks_calc** appear in embedded Gantt / plan works |
| **PlanWorks_calc** / work Gantt | Often **empty** until child tasks link via **`AnnualPlan_calc`** (or equivalent back-ref) |

**Order:** (1) seed or create annual plan rows on `{target_host}` when template is empty (`harvest_mode: source_empty_pattern_seed`), (2) link **WorkTasks_calc** with **`AnnualPlan_calc`** in a follow-up wave, (3) optional **AnnualPlansByAssignee_calc** child rows.

RU list empty + EN create exception is valid when the source had no rows but the form defines the business hub.

---

## InspectionPoints_calc: POST with Inspection_calc back-link

Inspection **grid** rows on **`Inspections_calc`** forms are **`InspectionPoints_calc`** children:

| Rule | Action |
|------|--------|
| Create path | **POST** point row with **`Inspection_calc`** (parent inspection id) set |
| UI grid | Parent form embed uses **`inspectionPoints_calc`** (alias casing per Attribute/List) |
| Space on point | Prefer route **`Space_calc`** from linked **InspectionRoutes_calc**; fallback stand space ids from progress map |

Verify on parent GET / form grid after POST — point list API alone may lag.

---

## Equipment list filters: TechnicalSystem id remap pattern

When remapping subsystem equipment tabs, treat **TechnicalSystem_calc** filter id and **Space→Floor** OR branches as one unit:

| Remap | From (typical) | To (pattern) |
|-------|----------------|--------------|
| HVAC / ventilation TS | Source-era TS catalog id (e.g. legacy **773**) | Target **HVAC** row id from target GET (**Code_calc** / system name) |
| Floor scope in OR branch | Source floor ids in filter tree | Target **Floor_calc** ids from hierarchy `map[]` |

Row-level equipment may already show correct **`TechnicalSystem_calc`** on GET while the **list stays empty** until the **dataset filter tree** is updated.

---

## MaintenanceExecution_calc: refs, lookups, and API writes

| Pattern | Detail |
|---------|--------|
| **Hierarchy on execution rows** | Property / Building / Floor on `MaintenanceExecution_calc` are often **derived from linked Equipment_calc** — fix equipment refs first; direct Property/Building/Floor PUT may **not round-trip** on GET (`property_calc` stays null while form UI may show breadcrumbs) |
| **Lookup popup vs API** | Browser **lookup popup Search** may return no rows; **type display text into the main readonly lookup field** (autocomplete) often succeeds where popup Search fails — when blocked, API `create_edit_record` with validated target ids is acceptable fallback |
| **Assignee refs** | Use **`Assignee_calc`** with **`Record/account.{id}`** (platform login row), not Staff row id — see [Staff row id vs platform account](#staff-row-id-vs-platform-account-includeincontainer) |
| **PascalCase on PUT** | `create_edit_record` **`values` keys must match Attribute/List PascalCase aliases** (e.g. `WorkStatus_calc`, `Assignee_calc`, `Notes_calc`) — lowercase keys can return HTTP 200 but **do not persist** |
| **Work notes text** | Gap-fill PM task descriptions via **`Notes_calc`** (GET lowercase `notes_calc`) — not `Description_calc` |
| **Cross-template id reuse** | A numeric id on `{target_host}` may belong to a **different template** than on `{source_host}` (e.g. `{source_host}` status id **341** = KPI temperature on EN while **JobStatuses** “assigned” is a different id) — never copy source numeric ids without harvest map verification |

→ Assignee bootstrap: [cmw-platform-staff-account-link](../../cmw-platform-staff-account-link/SKILL.md) · Process rows: [cmw-platform-process-record-fill](../../cmw-platform-process-record-fill/SKILL.md)

### MaintenanceExecution_calc: grouped month lists vs Records API count

Month PM lists on `MaintenanceExecution_calc` combine **three layers** — do not conflate them:

| Layer | What it is | Example |
|-------|------------|---------|
| **Dataset `systemFilterExpression`** | Month (`PlannedPeriod_calc->MonthAux_calc`), **Year** + **Property** from **`PMSettings_calc` where `RequesterAccount_calc = USER()`**, often **Equipment_calc → PMPlans_calc → Property_calc** | `May_calc`, `defaultList` (“All PM”) |
| **Dataset `grouping` (API)** | Often **TechnicalSystem_calc** + **PMPlans_calc** via equipment — from GET `webapi/Dataset/{app}/Dataset@MaintenanceExecution_calc.{dataset}` | Not the same as URL `g=ds.*` |
| **List URL `g=ds.{id}`** | **UI list** grouping (commonly **Property_calc** + **Building_calc** on `lst.158`) | `…/lst.158/…&g=ds.197790 Asc false,ds.197791 Asc false` |

**List id map (typical FM PM hub):** month tab **“5. May”** → `lst.146` + **`May_calc`**; **“6. June”** → `lst.153` + **`June_calc`**; **“7. July”** → `lst.152` + **`July_calc`**; **“All PM”** → `lst.158` + **`defaultList`**; **“My maintenance”** → `lst.145` + **`MyMaintenance`** (assignee = current user). Opening bare `#data/oa.59/lst.158` may land on **My maintenance** (0 rows for admin) — not the May grouped view. An **empty month tab** with no stale scalar filter usually means **zero rows with that month’s `PlannedPeriod_calc` id** on `{target_host}` — gap-fill period + equipment + status before remapping dataset filters.

| Symptom | Likely cause | First fix |
|---------|--------------|-----------|
| Records API / template list returns **N rows**, month or grouped UI shows **0** | **`systemFilterExpression`** + **USER()** PMSettings (year/property) or wrong **tab** (My maintenance vs May vs All PM) | Confirm **PMSettings_calc** for operator; gap-fill **`PlannedPeriod_calc`**, **`Year_calc`**, **`Equipment_calc`** + PM plan property chain; open **May** tab (`lst.146`) or grouped **`lst.158`** with `g=ds.*` |
| Scalar **`filter.value`** empty but list still empty | Filter is **`systemFilterExpression`**, not stale TR ids in `filter.value` | Do not only remap `filter.value` — verify expression + row fields + PMSettings |
| Grouped **Property+Building** URL still empty after row fill | **`property_calc` / `building_calc`** often **null on GET** (equipment-derived); UI grouping columns empty | Fix **equipment → PMPlans → Property** chain; use **ungrouped** month list or **All PM** + expand groups as workaround |
| **`globalAlias: null`** on dataset GET | PUT without alias may fail on some hosts | Inject **`build_global_alias("Dataset", template, dataset)`** before filter/grouping PUT ([Dataset PUT: globalAlias](#dataset-put-globalalias-when-get-returns-null)) |

**Workaround URLs (pattern):** `{target_host}/#data/oa.59/lst.158/…&g=ds.{propertyGroup} Asc false,ds.{buildingGroup} Asc false` (All PM / grouped hub); `{target_host}/#data/oa.59/lst.146` (May month slice). Instance row counts and ids: `{instance_progress_dir}` only.
| Skeleton rows reject API writes | Orphan month stubs (HTTP 200, GET unchanged) | Exclude from create resume; recreate with full equipment + period + status refs |

**Do not** assume Administration / template list count equals grouped list visibility — reconcile **template Records API**, **month slice list**, and **grouped list** separately.

Instance wave detail: `{instance_progress_dir}` only.

---

## PMPlans_calc: equipment pool and list filters

| Constraint | Action |
|------------|--------|
| **Equipment multi-link** | PM plans reference many equipment rows on `{source_host}`; `{target_host}` may have a **small equipment pool** — equipment multi-link stays empty until enough Equipment_calc rows exist and are mapped; do not force multi-link until pool is seeded |
| **PMSettings scope** | **`PMSettings_calc.Property_calc`** must align with plan **`Property_calc`** — user-scoped PM lists (plans filtered by current user's property settings) show **empty** when settings property ≠ plan property |
| **List USER() filters** | PM plan lists with **SystemFilter / USER()** on PMSettings property require matching PMSettings rows for the operator's property context — fix settings before blaming plan row data |
| **PMCode discipline** | Never rename stable **PMCode** during gap-fill — match harvest business keys |

Seed equipment and PMSettings catalogs before bulk PM plan equipment linking.

### PMPlans_calc: seed-driven equipment create-and-link

When `{target_host}` equipment pool is far smaller than `{source_host}` plan equipment lists:

| Step | Action |
|------|--------|
| 1 | **RU form first** on one representative plan — harvest equipment **business codes** / technical-system meaning (read-only) |
| 2 | Match existing EN **Equipment_calc** by code / system name where possible — link in place |
| 3 | **Bounded POST create** for missing codes on stand property/building/floor — preserve ASCII equipment ids; link **`Property_calc`**, **`Building_calc`**, **`Floor_calc`**, **`TechnicalSystem_calc`** |
| 4 | **PUT `Equipment_calc` multi-ref** on plan row — verify GET after each batch; sibling plans defer until pool grows |
| 5 | Document **`equipment_created`**, **`equipment_linked`**, **`equipment_gaps`** in progress JSON — expect large remaining gap counts until full equipment seed wave |

Prefer **one seed plan per wave** (complete create+link) over partial multi-link across all sibling plans.

---

## Operations_calc and SOP bidirectional links

| Fact | Action |
|------|--------|
| **RU template alias differs** | Source app may use a localized template name while `{target_host}` uses **`Operations_calc`** — resolve via export / object-app list, not assumed alias parity |
| **SOP ↔ operation linkage** | **`Related` / `soP_calc` on operation GET** are often **empty** even when the SOP side is linked — treat **`SOPs_calc.Operations_calc`** (multi-ref on the SOP row) as **source of truth**; verify in SOP list UI columns |
| **Small EN pool** | `{target_host}` may have far fewer operation rows than `{source_host}` — gap-fill **existing** rows only unless template is truly empty ([gap-fill policy](#gap-fill-policy-target-locale-rows)) |
| **OperationID** | Stable **`OperationID`** (business code) — never rename during localization touch-up |

When linking SOPs to operations, prefer editing **SOP row `Operations_calc`** and confirm list UI over relying on operation GET alone.

---

## WorkOrders_calc: Location4it and status pools

Extends [Work orders and conflicting list filters](#work-orders-and-conflicting-list-filters) and [Indicator widgets](#indicator-widgets-count--list-kpis).

| Field / rule | Detail |
|--------------|--------|
| **`Location4it`** | Writable location text for work-order demo fill — empty list columns often need **`Location4it`**, not calculated `Location_calc` (no-op on PUT) |
| **Status pool discipline** | One row cannot satisfy **Open**, **In progress**, and **indicator slice** filters simultaneously — assign **`WorkOrderStatus` / `Code_calc`** deliberately per target list or widget |
| **Assignee** | Same as maintenance rows — **`Assignee_calc` → `Record/account.{id}`** after staff-account link |
| **Advisory calculated fields** | Read-only SLA/display calc fields may reference legacy app names in expressions — treat as **advisory**; fix writable refs first |

→ [cmw-platform-process-record-fill](../../cmw-platform-process-record-fill/SKILL.md)

---

## WorkOrders_calc: multi-list variety and status pools

FM object apps often expose **many sibling lists** on the same template (Take attention, Review, In progress, Completed, service-type slices). Demo fill is **orchestration**, not one bulk PUT.

| Pattern | Action |
|---------|--------|
| **One batch JSON per list** | Parallel subagents each own **one** `lst.*` + dataset filter — no shared mutable progress file ([ralph_loop_goal_autonomy.md](ralph_loop_goal_autonomy.md#abuse-guardrails-mandatory)) |
| **Discover filter before edit** | Per list: `get_dataset` (or list URL) → required `Status_calc` / `Type_calc` / `Priority_calc` / property scope — map to catalog **Code_calc** on `{source_host}` forms, then target catalog GET |
| **Status pool discipline** | One row cannot satisfy **conflicting** list filters **and** indicator widgets (`dwc.*`) at once — assign status per **list role** (attention vs review vs in-progress vs completed vs open slice); use **separate rows** or **new creates** for missing variety instead of retargeting anchor ids |
| **Indicator carve-outs** | Before changing stand anchor rows, read which **indicator** pools depend on them (Open, Review, emergency slice) — gap-fill empty fields on anchors; create **new** stand-scoped rows for missing type/status variety |
| **Service-request variety** | Cover **Type_calc** / work-type picklists (operations, cleaning, construction repair, etc.) via Administration gap matrix + **create form** when list lacks a type — not only status edits on existing rows |
| **Administration + lists** | `#RecordType/oa.{appId}/Administration` lists **all** template rows for cross-list gap matrix; per-list batches still record list-specific `operations[]` |

**Reconcile pass:** when a list fill moves anchor rows off an in-progress status, run a **minimal** follow-up on the sibling list (or document intentional split in instance progress JSON) — do not assume one status edit satisfies every list.

→ Conflicting filters: [Work orders and conflicting list filters](#work-orders-and-conflicting-list-filters) · Process fill: [cmw-platform-process-record-fill](../../cmw-platform-process-record-fill/SKILL.md)

---

## Record Administration sweep (template-wide gap matrix)

`#RecordType/oa.{appId}/Administration` lists **every** row on a record template — use for **coverage audits** before or after list-scoped waves.

| Step | Action |
|------|--------|
| 1 | Enumerate all rows (Administration UI or Records API with pagination fallback) |
| 2 | Build gap matrix: empty vs semi-empty vs valid refs per attribute group |
| 3 | Gap-fill in place first ([gap-fill policy](#gap-fill-policy-target-locale-rows)); create only for thin coverage |
| 4 | **GET `webapi/Record/{id}`** after each edit — list columns and forms can lag API |

### PMPlans_calc (Administration equipment bulk-link)

Extends [PMPlans_calc: equipment pool and list filters](#pmplans_calc-equipment-pool-and-list-filters).

| Constraint | Action |
|------------|--------|
| **Equipment_calc multi-link** | Link only equipment rows that **exist on `{target_host}`** — match by equipment code / technical-system business key, not `{source_host}` numeric id |
| **Bulk-link sizing** | Prefer **bounded batches** (sibling plans in one wave) — verify GET after each multi-link PUT; empty multi-link until equipment pool is seeded is expected |
| **PMSettings gate** | Align **`PMSettings_calc.Property_calc`** with plan property before Administration shows “full” rows in user-scoped PM lists |
| **PMCode** | Never rename **PMCode** during Administration touch-up |
| **SaveAssignee** | Template attribute is **Boolean** (“save assignee” flag) — PUT `true`/`false`, not `Record/account.{id}`; GET returns `saveAssignee` bool. Staff login ids (`account.4`, etc.) belong on other FM rows (e.g. `Assignee_calc`), not this field |

### MaintenanceExecution_calc (Administration sweep)

Extends [MaintenanceExecution_calc: refs, lookups, and API writes](#maintenanceexecution_calc-refs-lookups-and-api-writes).

| Constraint | Action |
|------------|--------|
| **WorkStatus catalog** | **`WorkStatus_calc`** must reference the **maintenance/work-status** catalog row on `{target_host}` — numeric ids shared with **KPI / meter-type** templates on the host are a common false positive (PUT 200, wrong semantics) |
| **Equipment-first** | Set **`Equipment_calc`** to stand equipment pool rows; Property/Building/Floor often **derive** — fix equipment before direct hierarchy PUT |
| **Semi-empty rows** | Administration sweep: edit semi-empty rows in place; **create** only when execution coverage is thin — pair with list `lst.*` batches for operator-visible slices |
| **Assignee** | **`Assignee_calc` → `Record/account.{id}`** on `{target_host}` after staff-account link — verify assignee GET, not AccountService id alone |

---

## Inspections_calc: empty form UI vs populated API

Inspection and similar FM forms often show **"–"** on tabs while **Records API** or **list dataset columns** already hold values.

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| Form field blank, GET has value | **Calculated** or **list-only** column (e.g. name from datetime pair) | Fill the **driving** writable fields (`ActualStartDateTime_calc`, `ActualEndDateTime_calc`, `Route_calc`, `Status_calc`) — then re-GET; do not chase form labels that are not persisted attributes |
| List **Name** empty, times set | **`Name_calc`** derived from start/end | Set actual start/end datetimes; verify list column via GET, not form tab alone |
| Status shows orphan code | Stale catalog id from `{source_host}` | Remap **`Status_calc`** to target inspection-status catalog; verify GET Record |
| Embedded grids empty | Child template / points not linked | Fill **`InspectionPoints_calc`** (or sibling) on EN ids; route → property/space scope via **`Route_calc`** |
| **Status tab shows "–"**, **`Status_calc` populated on GET** | **`StatusBoost_calc`** (or sibling) expression references **legacy source app** via **`OBJECT('{source_app}', …)`** — evaluates null on `{target_host}` | Treat as **advisory UI blocker**; **`Status_calc`** is the writable/list driver — expression migration is a separate localization task |

**Rule:** gap matrix = **union** of form walk, list visible columns, and **GET Record** — browser-only audit under-reports populated rows.

→ Routes: [Inspection routes: Property + Space only](#inspection-routes-property--space-only)

---

## Inspection routes: Property + Space only

**`InspectionRoutes_calc`** location scope differs from Equipment / hierarchy templates:

| Attribute | On InspectionRoutes_calc? |
|-----------|---------------------------|
| **`Property_calc`** | Yes — primary site scope |
| **`Space_calc`** (multi or single) | Yes — route stops / covered spaces |
| **`Building_calc` / `Floor_calc`** | **Typically absent** — do not expect full Property→Building→Floor→Space chain on route rows |

Gap-fill routes with **Property + Space + ResponsibleStaff_calc**; derive building/floor context from linked spaces when needed for display, not as direct route attributes.

---

## Floors_calc, Staff_calc, WorkTasks_calc (recent waves)

| Template | Generic lesson |
|----------|----------------|
| **Floors_calc** | Same hierarchy as FM seed: **`Building_calc` → Property**; status via **`Status_calc` → SpaceStatuses_calc` by `Code_calc`** — pair floor/space occupancy codes when demo-fill requires consistency |
| **Staff_calc** | Employee rows (often account-template app): **`Property_calc`**, **`Space_calc`**, department/counterparty links — attach platform login via [staff-account-link](../../cmw-platform-staff-account-link/SKILL.md) before assignee-dependent fill |
| **WorkTasks_calc** | Tasks link to **WorkOrders_calc**, **Operations_calc**, staff, PM context — audit **all sibling rows** in object app; preserve task codes; stand-scoped refs same as parent work order |

Instance wave JSON: `{instance_progress_dir}` only.

---

## Spaces_calc: semi-empty hierarchy gap-fill

After hierarchy seed or import, **Spaces_calc** rows often exist with **title / SpaceID** populated but **missing link attrs** — lists and equipment location chains stay empty.

| Semi-empty signal | Writable fix |
|-------------------|--------------|
| **`Floor_calc` / `Building_calc` / `Property_calc` null** on GET | Edit in place — set full stand chain from `{instance_progress_dir}` hierarchy `map[]`; never POST duplicate space when row id exists |
| **`Status_calc` null** | Link to **SpaceStatuses_calc** on `{target_host}` by **`Code_calc`** (occupancy / lease state) |
| Cyrillic or legacy chars in **`SpaceID_calc` / number fields** | ASCII normalization gap-fill (e.g. Latin letter variants) — preserve stable business key semantics |
| Equipment / rental lists empty downstream | Fix **space → floor → building → property** on space row first — calculated **Location** on equipment derives from chain |

**Audit path:** `#RecordType/oa.{appId}/Administration` or grouped space list → gap matrix (empty vs semi-empty vs valid) → bounded edit batch → GET verify.

→ Location calc chain: [Calculated location vs writable location text](#calculated-location-vs-writable-location-text)

---

## ServiceRequestTypes_calc: cross-app SLA catalog remap

Work-order **type / service-request type** catalogs migrated from a legacy app may still hold **SLA picklist refs** pointing at **`{source_host}` numeric ids**.

| Rule | Action |
|------|--------|
| Match types by **`Code_calc`** (stable business code from source form) | Never rename **Code_calc** during gap-fill |
| **`SLA` / `Slatime_calc` refs** | GET target SLA catalog rows only — remap stale `{source_host}` ids to `{target_host}` SLA ids (often a small pool on EN) |
| Calculated **`Note`** / HTML banner fields | **Advisory** — may stay empty on EN; do not block batch on readonly calc gaps |
| List **`defaultList`** with no filter | Empty list usually means **row field gaps**, not dataset filter — still audit SLA refs on each row |

Use instance **`sla_map`** / progress `map[]` when present; otherwise GET SLA catalog on `{target_host}` and match by code or display key from source read-only harvest.

Instance harvest JSON: `{instance_progress_dir}` only.

---

## WorkOrders_calc.SLAStatus_calc: OBJECT() solution name on EN

**`SLAStatus_calc`** on work orders is a **calculated, readonly** Instance ref to **`Slastatus_calc`**. Per-record PUT does not persist; list/GET show null until the attribute expression resolves.

| Symptom | Cause | Fix |
|---------|-------|-----|
| All target rows **`slaStatus_calc` null** after migration gap-fill | Expression still uses **`OBJECT("{source_app}", "Slastatus_calc", …)`** with the **source** solution name while `{target_host}` only exposes **`{target_app}`** | **Edit attribute expression** — replace solution literal in `OBJECT()` with `{target_app}`; catalog `instanceGlobalAlias.solution` may already be correct |
| PUT **`SLAStatus_calc`** returns 200, GET still null | Field is calculated | Do not gap-fill via record PUT; fix expression or upstream **`SLAOverdue_calc`** drivers |
| All rows show **Good** only | **`SLAOverdue_calc`** false on stand data | Expected until overdue + in-progress status/time; **Bad** id comes from same catalog by **`Code_calc`** |

**Catalog on `{target_host}`:** GET **`Slastatus_calc`** — match **`Code_calc`** (`Bad` / `Good`); use **target numeric ids** only (small pool). **`SLAOverdue_calc`** expression keys off **`Status_calc->Code_calc`**, **`PlannedCompletionTime_calc`**, **`CompletionDateTime_calc`** — preserve status pools when testing Bad.

**Verify:** `GET webapi/Attribute/{target_app}/Attribute@WorkOrders_calc.SLAStatus_calc` → `expression`; then `GET webapi/Records/Template@{target_app}.WorkOrders_calc` — all rows should carry valid catalog ids.

→ Related: [ServiceRequestTypes_calc: cross-app SLA catalog remap](#servicerequesttypes_calc-cross-app-sla-catalog-remap)

---

## Browser automation: FM forms and lookups

| Caveat | Action |
|--------|--------|
| **License banner** | “All licenses in use” banner may **block Save** on some forms (e.g. certain service-request forms) while **other forms on the same host still save** — document blocker per form id; use API create fallback only when browser is fully blocked |
| **Lookup popup Search empty** | Prefer **typing into the main lookup input** (autocomplete) over popup Search when popup grid returns no rows |
| **List Create vs record Create** | Toolbar **Create** on some lists opens a **batch generator form**, not a single-row create — confirm URL form id before fill workflow |
| **Re-snapshot** | Standard browser discipline ([browser_automation.md](browser_automation.md)) — refs invalidate after navigation |

→ License-sensitive flows: try alternate form or API; do not assume one banner blocks the whole app.

---

## Quick decision matrix

| Symptom | Likely cause | First fix |
|---------|--------------|-----------|
| Indicator shows 0 | Rows don’t match Code_calc filter | Fix status/refs on driving template |
| Row missing from list A but in list B | Conflicting filters | Separate records or status discipline |
| List API 500 / cast error | One bad row | Per-record GET; fix/skip row |
| Gap-fill created duplicates | POST when rows existed | Edit-in-place policy |
| EN invoices empty | Status catalog not seeded | Catalogs before line items |
| Subagent “got datamodel wrong” | Skipped source form | Step 0 form-first |
| Replay lost steps | No `operations[]` flush | Write migration_progress JSON |
| Equipment list empty, records exist | Stale dataset filter id from `{source_host}` | Remap dataset filter on `{target_host}` |
| Location empty on equipment | Missing Floor/Space links | Set hierarchy refs; never PUT Location |
| Work order location column empty | Wrong field | Fill **`Location4it`**, not Location_calc |
| PUT 200 but field unchanged | Lowercase alias in `values` | Use PascalCase from Attribute/List |
| PM plan list empty for user | PMSettings property mismatch | Align PMSettings_calc.Property_calc |
| Operation shows no SOP link | Wrong GET surface | Check **SOPs_calc.Operations_calc** |
| Inspection route missing floor link | Wrong template model | Use Property + Space only |
| PUT ok but ref unchanged on GET | `{source_host}` id on `{target_host}` or lowercase alias | Target catalog GET + `map[]`; PascalCase; verify via **GET Record** |
| Assignee wrong / orphan on EN | Stale `account.{source_id}` | Remap to `{target_host}` AccountService id in `Record/account.{id}` form |
| Status / TechnicalSystem nonsense | Numeric id collision across templates | Match **Code_calc** on target; never copy source numeric id |
| Dataset PUT fails “key null” | Missing `globalAlias` on body | `build_global_alias("Dataset", template, dataset)` then PUT to **`webapi/Dataset/{app}`** |
| Dataset PUT **405** | Wrong endpoint (`Dataset@{template}.{dataset}`) | Full body **`PUT webapi/Dataset/{app}`** — [Dataset PUT: endpoint](#dataset-put-endpoint-globalalias-and-full-body) |
| ME grouped list 0, API has rows | Missing property/building on rows + grouping | Equipment + period + status; try month slice list |
| MyMaintenance empty (admin) | USER() assignee filter | Real `Record/account.{id}` + login as that user |
| Assignee PUT ok, list 0 | Staff row id used as account | IncludeInContainer + remap to platform account id |
| Inspection status tab "–" | StatusBoost legacy OBJECT expr | Fix **Status_calc**; expression migration separate |
| ServiceRequestTypes SLA empty | Stale `{source_host}` SLA ids | Remap SLA refs by target catalog GET |
| WO **SLA status** null on target, PUT ignored | `OBJECT("{source_app}",…)` in calc expression | Edit **`SLAStatus_calc`** expression → `{target_app}` |
| PM plan equipment empty | Small EN pool | Seed-driven create-and-link from RU codes |
| Space list gap | Semi-empty hierarchy attrs | Edit Floor/Building/Property/Status in place |
| Equipment tab empty, rows OK on GET | Composite filter + stale TS/floor ids | Remap full AND/OR filter tree; not row refs only |
| Contracts “All” full, Active empty | Status filter still source ids | Remap per-dataset `Status_calc` literals on target |
| Annual plan Gantt empty | No WorkTasks link | POST tasks with `AnnualPlan_calc` back-ref |
| Inspection grid empty | Points not POSTed | POST `InspectionPoints_calc` + `Inspection_calc` parent link |
| Wrong maintenance status label | WorkStatus id = KPI row on target | JobStatuses_calc + Code_calc on target host |
| WO missing from list after fill | Status pool / filter mismatch | Per-list dataset filter + dedicated row or status per list role |
| **aa.2 Staff list 0 rows, Records API 10** | Columns use **`cmw_account_*`**; **`username` null** (IncludeInContainer 500) | Prepend **`RecordLink_calc`** + hide `cmw_account_*`; or account bind when username required — **not** in-place `cmw_*`→`fullName` path remap (wipes dataset) |
| Staff **`department` null on GET** | Org unit on **`serviceDepartment`** on some hosts | Gap-fill **`ServiceDepartment`** + **`Department`** via `create_edit_record`; verify both on GET |
| Indicator broke after WO edit | Shared anchor row | Carve-out anchors; new rows for new pools |
| PM plan Administration still empty | Equipment pool / PMSettings | Seed equipment; align PMSettings property |
| ME row wrong status semantics | KPI id used as WorkStatus | Target maintenance catalog GET — not colliding numeric id |
| Inspection form "empty", API full | Calculated / list-driven field | Fill drivers; GET Record + list columns |

---

## Related

- [cmw-platform SKILL §9](../SKILL.md#9-growing-platform-skills) — where to save platform-generic vs instance findings
- [record_harvest_seed.md](record_harvest_seed.md) — harvest/seed JSON patterns
- [ralph_loop_goal_autonomy.md](ralph_loop_goal_autonomy.md) — loops, verification, form-first
- [cmw-platform-instance-switch](../../cmw-platform-instance-switch/SKILL.md) — `{source_host}` / `{target_host}` env switching
- [edit_or_create.md](edit_or_create.md) — dataset filter GET → merge → PUT
- [browser_automation.md](browser_automation.md) — FM lookup and license-banner caveats
