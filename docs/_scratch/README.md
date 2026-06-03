# Scratch (platform repo)

**Do not** store instance-specific migration artifacts here (harvest JSON, id maps, phase runners).

Use `{instance_progress_dir}/docs/_scratch/` for one-off harvest outputs and runners. Link from `localization/migration_progress/*.json` via `meta.harvest_path` / `meta.seed_path`.

Reusable automation belongs in `.agents/skills/cmw-platform/scripts/` (see [scripts_index](../.agents/skills/cmw-platform/references/scripts_index.md)).
