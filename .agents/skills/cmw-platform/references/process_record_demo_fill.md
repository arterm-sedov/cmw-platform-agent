# Process-owned record demo fill (API)

Use when FR **process form** rows exist but user-facing text is placeholder (`test`, empty location).

## Discovery

1. URL `#form/oa.{N}/form.{F}/{recordId}` → resolve object app / template via Volga export or `GET webapi/Record/{id}` (lowercase keys in body).
2. `GET webapi/Attribute/List/Template@{App}.{Template}` → **PascalCase** aliases for `create_edit_record` / PUT body.
3. Dashboard `#indicator/id=dwc.{N}` → read widget config in backup (`WidgetConfigs/*.json`): `QueryRule`, `Template`, `AggregationMethod`.

## Edit pattern

```text
TR (optional): list_template_records / Records GET — patterns only, no id clone
FR: create_edit_record operation=edit, record_id=<numeric>, values={ PascalCase aliases }
Verify: GET webapi/Record/{id} — compare lowercase keys in response
```

## Pitfalls

| Issue | Mitigation |
|-------|------------|
| PUT returns 200 but fields unchanged | Use PascalCase from Attribute/List, not lowercase GET keys |
| `_put_request(body, endpoint)` | Argument order: body first, endpoint second |
| Process rows | Prefer edit existing instances; advancing workflow may need UI user commands |
| Location empty after PUT | May need building (`Zdaniya`) refs first; document and defer |

## Indicator KPIs

Usually **query-driven** on template rows (e.g. active = status not complete/cancelled). Populate underlying records; do not edit `dwc.*` unless display name/filter is wrong.

## Progress artifact

**Instance repo (my-building):** `localization/migration_progress/YYYYMMDD_phase1_process_demo_fill.json` with `fr_edits[]`, `verification_urls`, `tr_to_fr_mapping`. Roadmap: [`docs/20260519_migration_status_and_roadmap.md`](file:///D:/Repo/my-building/docs/20260519_migration_status_and_roadmap.md). Do not store instance progress JSON in cmw-platform-agent.
