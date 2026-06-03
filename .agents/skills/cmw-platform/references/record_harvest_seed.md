# Source → target record harvest and seed (leaf templates)

Generic dual-host pattern: read a bounded subset on the **source** host, create or update rows on the **target** host, persist id maps in the **instance repository** (`{instance_progress_dir}`) — not in cmw-platform-agent.

**Instance schema and tenant checklists:** `{instance_progress_dir}/localization/migration_progress/README.md` and instance skills under `{instance_progress_dir}/.agents/skills/`.

## JSON source of truth (required)

| Rule | Detail |
|------|--------|
| **Durable state** | `{instance_progress_dir}/localization/migration_progress/YYYYMMDD_phaseN_{template}.json` — not agent chat memory |
| **Before done** | Update batch JSON: `meta.status`, `meta.template`, root `map[]`, `meta.errors`, `meta.backup_pending`, `meta.retry_count`, `meta.agent_wave`, `started_at` / `updated_at` |
| **Resume** | Read existing `map[]`; skip creates when target id already present (idempotent) |
| **Failure** | `meta.status`: `partial` or `failed`; retry agents use JSON only |
| **Scratch** | `harvest_template_records.py` output under `{instance_progress_dir}/docs/_scratch/` — link via `meta.harvest_path`; promote into `migration_progress/` before closing batch |

Field names like `source_record_id` / `target_record_id` (or legacy `tr_record_id` / `fr_record_id`) are conventional — follow the instance README.

## Workflow

```text
CMW_BASE_URL = source → list_template_records (harvest count + field keys; no PII in git)
Transform → target-locale display values per instance spec
CMW_BASE_URL = target → create_edit_record (create missing rows OR edit pre-existing)
Write/update localization/migration_progress/YYYYMMDD_phaseN_{template}.json BEFORE claiming done
Optional: cmw-platform-backup-launch between themed batches; set meta.backup_pending until coordinator clears
```

Use [cmw-platform-instance-switch](../../cmw-platform-instance-switch/SKILL.md) when source and target are different hosts (`CMW_BASE_URL`, or paired vars such as `CMW_BASE_URL_SOURCE` / `CMW_BASE_URL_TARGET`).

## Tools

| Step | Tool / API |
|------|------------|
| Harvest | `list_template_records` or `GET webapi/Records/Template@{App}.{Template}` |
| Attributes | `list_attributes` when Record refs need resolution |
| Target write | `create_edit_record` — **PascalCase** aliases from Attribute/List |
| Verify | `GET webapi/Record/{target_id}` — response keys often lowercase |

## Rules

1. **Never** POST source record ids on target create bodies.
2. **Never** PUT lookup values using `{source_host}` numeric ids on `{target_host}` — resolve via **Code_calc** / catalog system name / GET on target, or progress `map[]`. Account links must use `{target_host}` AccountService ids (`Record/account.{target_id}`). See [platform_usage_discoveries.md](platform_usage_discoveries.md#cross-instance-record-ids-critical) and [cmw-platform SKILL §9](../SKILL.md#9-growing-platform-skills).
3. Maintain `map[]`: source id, target id, plus stable business key (year, code, hours).
4. When target rows pre-exist empty: match by sorted business key, then **edit** in place.
5. Progress JSON: field name inventory only; no personal data.
6. **Account verify:** use `AccountService/Get`, not `List` alone (List may show stale `Username`).
7. **Target locale:** user-visible strings on the target host follow the instance migration spec — source rows are **pattern reference** when they exist.
8. **Empty source template:** when harvest returns **0** on source, document `harvest_mode: source_empty_pattern_seed` (or instance-specific value) and create a small target subset from template attributes; record `source_record_id: null` in `map[]` when applicable.
9. **Backup cadence:** one configuration backup between **themed** batches (accounts, groups, record phase hubs), not per row. Record backup session id on the batch file when your instance schema defines it.
10. **Dataset filter remap:** after locale migration, grep target datasets for `{source_host}` catalog ids in `filter.value`, **`systemFilterExpression`**, and nested composite trees; **PUT** merged full body to **`webapi/Dataset/{app}`** (not `Dataset@{template}.{dataset}` — **405**); inject **`globalAlias`** when GET returns null — see [platform_usage_discoveries.md](platform_usage_discoveries.md).
11. **SOP ↔ operation links:** when seeding Operations_calc, verify **`SOPs_calc.Operations_calc`** on the SOP row — operation GET may not show `soP_calc` / `Related` even when linked.
12. **Verify target writes:** after PUT, confirm refs with **GET `webapi/Record/{target_id}`** on `{target_host}` (lowercase field keys in body); do not rely on list-only reads or wrong-casing `GetPropertyValues` aliases.
13. **Assignee / USER() lists:** **`Assignee_calc`** must be **`Record/account.{target_AccountService_id}`** that passes **`GET webapi/Record/account.{id}`** — staff row id ≠ platform account; link via IncludeInContainer first.
14. **Cross-app catalog refs:** SLA and sibling picklists on migrated templates may still hold `{source_host}` numeric ids — remap via target catalog GET + **Code_calc**.
15. **PMPlans equipment pool:** when `{target_host}` pool is small, use **seed-driven create-and-link** from source equipment codes before bulk multi-ref PUT.
16. **Grouped maintenance lists:** template Records API count ≠ grouped month list UI — fix equipment + period + PMSettings; use month slice list when Property+Building grouping hides rows.

### Cross-instance lookup ids (never copy source numeric ids)

Same as rule 2 — expanded checklist for replay agents:

| Lookup type | `{source_host}` read | `{target_host}` write |
|-------------|----------------------|------------------------|
| Status / picklist | **Code_calc** (or stable business code) | GET catalog on target; use **target** row id in PUT |
| TechnicalSystem / sibling catalogs | Code or name from source form/API | Target id from progress `map[]` — not the source numeric id |
| Property / Building / Floor / Space / Equipment | Business keys in harvest | Target ids from `{instance_progress_dir}` hierarchy maps only |
| Assignee / ResponsibleStaff | Persona meaning from source form | `Record/account.{target_AccountService_id}` after [cmw-platform-staff-account-link](../../cmw-platform-staff-account-link/SKILL.md) — **verify GET Record/account** exists |
| Dataset list filters | Document source filter intent | Remap `filter.value` / `systemFilterExpression`; **PUT `webapi/Dataset/{app}`** full body + `globalAlias` |
| SLA / cross-app picklists | Code from source type row | Target SLA catalog GET — not `{source_host}` numeric id |

→ [platform_usage_discoveries.md](platform_usage_discoveries.md#cross-instance-record-ids-critical) · [cmw-platform SKILL §9](../SKILL.md#9-growing-platform-skills)

## Backup API unwrap

`GET /webapi/Backup/Configuration` returns `{ "response": [ BackupConfigurationModel, ... ], "success": true }` — not always `result`. Then `POST /webapi/Backup/Session/{configurationId}`.

## CLI scripts (agnostic)

See [scripts_index.md](scripts_index.md) — `harvest_template_records.py` (read-only), `seed_records_from_harvest.py` (apply `operations` / `map`), `backup_configuration_session.py` between themed batches. Pass `--base-url` or set `CMW_BASE_URL`; store JSON under `{instance_progress_dir}`, not cmw-platform-agent.

## Related

- [process_record_demo_fill.md](process_record_demo_fill.md) — process-owned form rows
- [platform_usage_discoveries.md](platform_usage_discoveries.md) — gap-fill, Code_calc, `operations[]`, list fallback, dataset PUT, assignees
- [cmw-platform-backup-launch](../../cmw-platform-backup-launch/SKILL.md)
- [ralph_loop_goal_autonomy.md](ralph_loop_goal_autonomy.md) — loops, verification, form-first
