# Instance repo vs platform repo — documentation boundary

**cmw-platform-agent** is the agnostic platform agent: reusable API/browser skills, OpenAPI-aligned references, and CLI scripts. **Instance repositories** (e.g. `my-building`, any `{instance_progress_dir}`) own migration state, tenant playbooks, harvest JSON, and operator checklists.

Use placeholder `{instance_progress_dir}` in platform docs — never hardcode tenant hosts, passwords, or numeric record ids in committed platform skill bodies.

---

## Decision table

| Artifact | Platform repo (`cmw-platform-agent`) | Instance repo (`{instance_progress_dir}`) |
|----------|--------------------------------------|----------------------------------------|
| Harvest/seed **contract** (JSON shape, scratch rules, idempotent replay) | `references/record_harvest_seed.md` | `localization/migration_progress/README.md` + instance playbook e.g. `references/tr_fr_record_harvest_seed.md` |
| Themed batch JSON, `map[]`, `operations[]`, `meta.harvest_path` | — | `localization/migration_progress/*.json` |
| One-off harvest output, phase runners, batch scripts | — | `docs/_scratch/` (link from batch `meta`) |
| Volga/mz-tr/mz-fr audits, TR→FR maps, gap analyses, roadmaps | — | `docs/`, `docs/localization/`, `localization/AGENTS.md` |
| US FM replication, hierarchy seed order, Ralph instance consumer | Stub only (see below) | `.agents/skills/cmw-platform/references/us_fm_ru_to_en_replication.md`, `fm_hierarchy_ru_to_us_seed.md`, `cmw-platform-fm-hierarchy-seed/`, `ralph-loop-instance/` |
| Repeatable API/UI lesson (any tenant) | `references/platform_usage_discoveries.md` or extend an existing reference | One-line pointer + link to platform ref after wave flush |
| CLI scripts (harvest, seed, backup) | `.agents/skills/cmw-platform/scripts/` | Invoke from instance; **write outputs** under instance paths only |
| OpenAPI, tool inventory, browser hash patterns | `cmw_open_api/`, `references/*` | Cite platform paths; do not fork OpenAPI into instance |
| Localization of Volga CTF / translation memory | — | `localization/`, `comindware-l10n` skills in instance |
| Credentials | `.env` / `.env.example` in platform (execution) | Optional paired vars in platform `.env` only; **no secrets in instance commits** |

**Promote pattern to platform, keep data in instance:** After a wave, flush instance JSON first, then add **one agnostic paragraph** to a platform reference if the manipulation pattern applies on any tenant. Do **not** copy instance audits, id maps, or host-specific tables into platform `docs/`.

---

## Scratch (`docs/_scratch/`)

| Repo | Rule |
|------|------|
| **cmw-platform-agent** | Keep **empty** of instance migration artifacts (no harvest JSON, id maps, themed batch results). Reusable automation belongs in `.agents/skills/cmw-platform/scripts/`. See [docs/_scratch/README.md](../../../../docs/_scratch/README.md). |
| **Instance** | All one-off harvest files, `phase*_*.py`, and wave runners under `{instance_progress_dir}/docs/_scratch/`. Reference from `localization/migration_progress/*.json` via `meta.harvest_path` / `meta.seed_path`. |

---

## `localization/migration_progress/` (instance only)

- Lives only under `{instance_progress_dir}/localization/migration_progress/`.
- **Never** commit instance progress JSON to cmw-platform-agent.
- Schema and filename conventions: `{instance_progress_dir}/localization/migration_progress/README.md`.
- Operator context: `{instance_progress_dir}/localization/AGENTS.md`.

---

## Linking from platform skills to instance repos

1. Use `{instance_progress_dir}` as the root placeholder (e.g. `D:\Repo\my-building` on a developer machine — **not** in committed platform markdown except as a non-secret example in stubs).
2. Point to **relative paths inside the instance repo**, not to sibling repo absolute paths in platform skill bodies.
3. For moved instance playbooks, keep a **one-paragraph stub** in platform `references/` with canonical instance path — do not duplicate full content.

| Platform stub (redirect only) | Canonical instance path |
|------------------------------|-------------------------|
| [tr_fr_record_harvest_seed.md](tr_fr_record_harvest_seed.md) | `{instance_progress_dir}/.agents/skills/cmw-platform/references/tr_fr_record_harvest_seed.md` |
| [us_fm_ru_to_en_replication.md](us_fm_ru_to_en_replication.md) | `{instance_progress_dir}/.agents/skills/cmw-platform/references/us_fm_ru_to_en_replication.md` |
| [fm_hierarchy_ru_to_us_seed.md](fm_hierarchy_ru_to_us_seed.md) | `{instance_progress_dir}/.agents/skills/cmw-platform/references/fm_hierarchy_ru_to_us_seed.md` |
| [cmw-platform-fm-hierarchy-seed/SKILL.md](../../cmw-platform-fm-hierarchy-seed/SKILL.md) | `{instance_progress_dir}/.agents/skills/cmw-platform-fm-hierarchy-seed/SKILL.md` |

---

## Process model vs record template scope (`RecordType` / `doc.*`)

When documenting UI or API fixes in an **instance** repo, state the **template class** so agents do not apply a fix to the wrong entity:

| Hash / id prefix | Typical meaning | Example doc scope |
|------------------|-----------------|-------------------|
| `oa.{N}` | Record template (datasets, forms, lists on a business template) | Property, Space, Work order templates |
| `doc.{N}` | **Process model template** (BPMN/process definition container) | e.g. `doc.1` on a target host — process-model-scoped docs and fixes |
| `event.{M}` / Operations under `oa` or process | User command (button), scenario step | Button/filter fixes tied to parent template |
| `aa.{N}` | Account template | Administration / account lists |

**Rule:** Process-model-scoped documentation (transitions, process-owned forms, scenario wiring) belongs with the **`doc.*`** process model template id in the instance playbook or progress JSON `meta.template` — not mixed into unrelated `oa.*` record-template batches unless the batch explicitly spans both.

Platform-generic hash patterns: [browser_automation.md](browser_automation.md) (`#RecordType/oa.{N}/…`). Instance playbooks name the concrete `oa.*` / `doc.*` ids for that tenant.

---

## Workflow for agents

1. **Read** instance state from disk: `{instance_progress_dir}/localization/migration_progress/*.json` and instance skills — not from prior chat.
2. **Execute** API/browser using cmw-platform-agent tools and scripts.
3. **Write** durable artifacts only in the instance repo (JSON, scratch, progress reports).
4. **Promote** repeatable lessons to platform references (placeholders only).
5. **Commit** instance repo for migration waves; platform repo only when generic skills/docs change.

---

## Related platform docs

- [record_harvest_seed.md](record_harvest_seed.md) — agnostic harvest/seed contract
- [process_model_template_localization.md](process_model_template_localization.md) — `doc.XXXX` enumeration and per-template checklist
- [en_template_ru_leftover_cleanup.md](en_template_ru_leftover_cleanup.md) — RU leftovers on EN target (Operations, solution dataset grids)
- [ralph_loop_goal_autonomy.md](ralph_loop_goal_autonomy.md) — platform-generic Ralph loop
- [SKILL.md §9](../SKILL.md#9-growing-platform-skills) — growing skills policy
- [AGENTS.md](../../../../AGENTS.md) — short “Where findings belong” summary

## Related instance docs (example: my-building)

- `{instance_progress_dir}/localization/AGENTS.md`
- `{instance_progress_dir}/.agents/skills/README.md`
- `{instance_progress_dir}/docs/20260519_volga_tr_to_fr_data_migration_master_plan.md` (repo split table)
