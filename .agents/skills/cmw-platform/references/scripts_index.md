# Platform maintenance CLI scripts

Agnostic scripts under [`.agents/skills/cmw-platform/scripts/`](../scripts/). Load credentials from `cmw-platform-agent/.env` (`CMW_BASE_URL`, `CMW_LOGIN`, `CMW_PASSWORD`). Use `--base-url` to override the host. **Do not commit** harvest outputs or progress JSON with PII — write them in the **project repo** (e.g. my-building `localization/migration_progress/`).

## Script-only environment variables

Not listed in the root agent `.env.example` (Gradio app does not use them). Set in the shell, a local `.env` for scripted runs (`CMW_USE_DOTENV=true`), or via CLI flags where supported.

| Variable | Scripts | Default | Purpose |
|----------|---------|---------|---------|
| `CMW_EMAIL_DOMAIN` | [account_update_mbox_batch.py](../scripts/account_update_mbox_batch.py) | `facility-demo.example` | Synthetic Mbox domain: `{username}@{CMW_EMAIL_DOMAIN}` |

| Script | Purpose | Env vars | Example |
|--------|---------|----------|---------|
| [backup_configuration_session.py](../scripts/backup_configuration_session.py) | List backup configs, POST session, optional poll | `CMW_BASE_URL`, `CMW_LOGIN`, `CMW_PASSWORD` | `python .agents/skills/cmw-platform/scripts/backup_configuration_session.py --poll` |
| [account_update_mbox_batch.py](../scripts/account_update_mbox_batch.py) | Batch `Mbox` → `{username}@{domain}` | above + optional `CMW_EMAIL_DOMAIN` (default `facility-demo.example`) | `.../account_update_mbox_batch.py --dry-run` |
| [verify_account_usernames.py](../scripts/verify_account_usernames.py) | List vs Get username staleness | above | `.../verify_account_usernames.py --only-stale` |
| [harvest_template_records.py](../scripts/harvest_template_records.py) | Read-only template record harvest | above | `.../harvest_template_records.py --app YourApp --template YourTpl --output ./harvest.json` |
| [seed_records_from_harvest.py](../scripts/seed_records_from_harvest.py) | Apply `operations[]` or `map[]` from JSON | above | `.../seed_records_from_harvest.py --file ./seed_ops.json --apply` |
| [diagnose_connection.py](../scripts/diagnose_connection.py) | Config + auth smoke test | above | `.../diagnose_connection.py` |

Shared helper: [`_cmw_cli.py`](../scripts/_cmw_cli.py) (import only; not run directly).

## JSON contracts

- **Harvest output:** `meta`, `hosts.primary.records`, `field_inventory` — see [tr_fr_record_harvest_seed.md](tr_fr_record_harvest_seed.md).
- **Seed input:** `operations` array with `operation` (`create`/`edit`), `values` (PascalCase attribute aliases), optional `record_id`; or `map[]` with `fr_record_id` + `values`.

## Related skills

- [cmw-platform-backup-launch](../../cmw-platform-backup-launch/SKILL.md)
- [cmw-platform-account-bootstrap](../../cmw-platform-account-bootstrap/SKILL.md)
- [cmw-platform-instance-switch](../../cmw-platform-instance-switch/SKILL.md)

Instance migration progress stays in the **project repo**, not here.
