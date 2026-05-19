# TR → FR record harvest and seed (leaf templates)

Generic pattern for Volga-style demo data migration: read subset on TR, create or edit on FR, store id map in the **project repo** (not cmw-platform-agent).

## JSON source of truth (required)

| Rule | Detail |
|------|--------|
| **Durable state** | [**my-building**](file:///D:/Repo/my-building) `localization/migration_progress/YYYYMMDD_phaseN_{template}.json` — not agent chat memory |
| **Before done** | Update batch JSON: `meta.status`, `meta.template`, root `map[]`, `meta.errors`, `meta.backup_pending`, `meta.retry_count`, `meta.agent_wave`, `started_at` / `updated_at` |
| **Resume** | Read existing `map[]`; skip creates when `fr_record_id` already present (idempotent) |
| **Failure** | `meta.status`: `partial` or `failed`; retry agents use JSON only |
| **Scratch** | `harvest_template_records.py` output under project `docs/_scratch/` — link via `meta.harvest_path`; promote into `migration_progress/` before closing batch |

Schema: [my-building `migration_progress/README.md`](file:///D:/Repo/my-building/localization/migration_progress/README.md).

## Workflow

```text
CMW_BASE_URL = TR → list_template_records (harvest count + field keys, no PII in git)
Transform → US FM EN display values (whole phrase; serious-only)
CMW_BASE_URL = FR → create_edit_record (create missing rows OR edit pre-existing)
Write/update localization/migration_progress/YYYYMMDD_phaseN_{template}.json (meta.status, map[], errors) BEFORE claiming done
Optional: cmw-platform-backup-launch between themed batches; set meta.backup_pending until coordinator clears
```

## Tools

| Step | Tool / API |
|------|------------|
| Harvest | `list_template_records` or `GET webapi/Records/Template@{App}.{Template}` |
| Attributes | `list_attributes` when Record refs need resolution |
| FR write | `create_edit_record` — **PascalCase** aliases from Attribute/List |
| Verify | `GET webapi/Record/{fr_id}` — response keys often lowercase |

## Rules

1. **Never** POST TR record ids on FR create bodies.
2. Maintain `map[]`: `tr_record_id`, `fr_record_id`, plus stable business key (year, hours, code).
3. When FR rows pre-exist empty: match by sorted business key (e.g. year), then **edit** in place.
4. Progress JSON: field name inventory only; no personal data.
5. **Account verify:** use `AccountService/Get`, not `List` alone (List may show stale `Username`).
6. **US FM tone:** user-visible EN in elevated facilities/real-estate executive register — clear, confident, US addresses and building names (no translit, no Cyrillic). TR is **pattern source only** when rows exist.
7. **TR empty templates:** when `list_template_records` / Records GET returns **0** on TR, document `harvest_mode: tr_empty_us_fm_pattern_seed` and create a small FR subset (3–5 rows) from template attributes — still record `tr_record_id: null` in `map[]`.
8. **Backup cadence:** one configuration backup between **themed** batches (accounts, groups, record phase hubs), not per row. Record `meta.fr_backup_session_id` on the batch progress file (e.g. `backupSession.39` after Zdaniya).

## Backup API unwrap

`GET /webapi/Backup/Configuration` returns `{ "response": [ BackupConfigurationModel, ... ], "success": true }` — not always `result`. Then `POST /webapi/Backup/Session/{configurationId}`.

## CLI scripts (agnostic)

Use [scripts_index.md](scripts_index.md) — `harvest_template_records.py` (read-only), `seed_records_from_harvest.py` (apply `operations` / `map`), `backup_configuration_session.py` between themed batches. Pass `--base-url` or set `CMW_BASE_URL`; store JSON under the **project repo**, not cmw-platform-agent.

## Related

- [process_record_demo_fill.md](process_record_demo_fill.md) — process-owned form rows
- [cmw-platform-backup-launch](../../cmw-platform-backup-launch/SKILL.md)
- Instance progress: **my-building** `localization/migration_progress/`
