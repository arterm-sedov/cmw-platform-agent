# Source → target record harvest and seed (leaf templates)

Generic dual-host pattern: read a bounded subset on the **source** host, create or update rows on the **target** host, persist id maps in the **instance repository** (`{instance_progress_dir}`) — not in cmw-platform-agent.

## JSON source of truth (required)

| Rule | Detail |
|------|--------|
| **Durable state** | `{instance_progress_dir}/localization/migration_progress/YYYYMMDD_phaseN_{template}.json` — not agent chat memory |
| **Before done** | Update batch JSON: `meta.status`, `meta.template`, root `map[]`, `meta.errors`, `meta.backup_pending`, `meta.retry_count`, `meta.agent_wave`, `started_at` / `updated_at` |
| **Resume** | Read existing `map[]`; skip creates when target id already present (idempotent) |
| **Failure** | `meta.status`: `partial` or `failed`; retry agents use JSON only |
| **Scratch** | `harvest_template_records.py` output under `{instance_progress_dir}/docs/_scratch/` — link via `meta.harvest_path`; promote into `migration_progress/` before closing batch |

Schema: `{instance_progress_dir}/localization/migration_progress/README.md` (field names like `tr_record_id` / `fr_record_id` are conventional **source** / **target** ids).

## Workflow

```text
CMW_BASE_URL = source → list_template_records (harvest count + field keys; no PII in git)
Transform → target-locale display values per instance spec
CMW_BASE_URL = target → create_edit_record (create missing rows OR edit pre-existing)
Write/update localization/migration_progress/YYYYMMDD_phaseN_{template}.json BEFORE claiming done
Optional: cmw-platform-backup-launch between themed batches; set meta.backup_pending until coordinator clears
```

Use [cmw-platform-instance-switch](../../cmw-platform-instance-switch/SKILL.md) when source and target are different hosts (`CMW_BASE_URL`, or paired vars such as `CMW_BASE_URL_RU` / `CMW_BASE_URL_EN`).

## Tools

| Step | Tool / API |
|------|------------|
| Harvest | `list_template_records` or `GET webapi/Records/Template@{App}.{Template}` |
| Attributes | `list_attributes` when Record refs need resolution |
| Target write | `create_edit_record` — **PascalCase** aliases from Attribute/List |
| Verify | `GET webapi/Record/{target_id}` — response keys often lowercase |

## Rules

1. **Never** POST source record ids on target create bodies.
2. Maintain `map[]`: source id, target id (`fr_record_id` or `target_record_id` per instance schema), plus stable business key (year, code, hours).
3. When target rows pre-exist empty: match by sorted business key, then **edit** in place.
4. Progress JSON: field name inventory only; no personal data.
5. **Account verify:** use `AccountService/Get`, not `List` alone (List may show stale `Username`).
6. **Target locale:** user-visible strings on the target host follow the instance migration spec (tone, language, addressing) — source rows are **pattern reference** when they exist.
7. **Empty source template:** when harvest returns **0** on source, document `harvest_mode: source_empty_pattern_seed` (or instance-specific value) and create a small target subset from template attributes; record `tr_record_id: null` in `map[]` when applicable.
8. **Backup cadence:** one configuration backup between **themed** batches (accounts, groups, record phase hubs), not per row. Record `meta.fr_backup_session_id` or `meta.target_backup_session_id` on the batch file when your instance schema defines it.

## Backup API unwrap

`GET /webapi/Backup/Configuration` returns `{ "response": [ BackupConfigurationModel, ... ], "success": true }` — not always `result`. Then `POST /webapi/Backup/Session/{configurationId}`.

## CLI scripts (agnostic)

Use [scripts_index.md](scripts_index.md) — `harvest_template_records.py` (read-only), `seed_records_from_harvest.py` (apply `operations` / `map`), `backup_configuration_session.py` between themed batches. Pass `--base-url` or set `CMW_BASE_URL`; store JSON under `{instance_progress_dir}`, not cmw-platform-agent.

## Related

- [process_record_demo_fill.md](process_record_demo_fill.md) — process-owned form rows
- [cmw-platform-backup-launch](../../cmw-platform-backup-launch/SKILL.md)
- Instance FM / hierarchy workflows: `{instance_progress_dir}/.agents/skills/cmw-platform-fm-hierarchy-seed/SKILL.md`
