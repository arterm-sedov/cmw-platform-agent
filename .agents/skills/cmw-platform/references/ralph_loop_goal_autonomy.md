# Ralph Loop — goal autonomy (platform-generic)

Use when an agent must **repeat the same scoped goal** until **external verification** passes — not when the model says it is done.

**Instance example (non-normative):** `{instance_progress_dir}/docs/localization/ralph_loop_us_fm_autonomous_runbook.md`  
**Instance consumer skill:** `{instance_progress_dir}/.agents/skills/ralph-loop-instance/SKILL.md`  
**Emulation scratchpad template:** `{instance_progress_dir}/docs/localization/ralph_scratchpad.example.md`

External references: [ghuntley.com/ralph](https://ghuntley.com/ralph/), [Cursor Directory — Ralph Loop](https://cursor.directory/plugins/ralph-loop), [forum guide](https://forum.cursor.com/t/ralph-cursor-guide/149998), [ralph-cursor](https://github.com/danielsinewe/ralph-cursor).

## Safe Ralph (one sentence)

**Re-run the same bounded prompt only while disk-backed checks fail, with a hard iteration cap, no parallel writers on the same artifact, and `<promise>` only after jq/tests pass — never because the model claims done.**

## Source-form-first (replication subagents)

**For ANY subagent** spawned to replicate demo data from a **read-only source host** to a **write target host**:

| Rule | Detail |
|------|--------|
| **Step 0** | Open the matching **source record form** in the UI (read-only) and capture business meaning — breadcrumbs, labels, links, status, embedded grids — **before** `GetPropertyValues`, list APIs, or export/datamodel. |
| **Insufficient alone** | API schema, template system names, configuration export, and bulk list endpoints do **not** replace source forms for business logic. |
| **Spawn prompts** | Parent coordinators must paste Step 0 as the **first numbered step** in every per-scope worker prompt (one template, floor, or list batch per agent). |
| **After Step 0** | Source API read → map to target attribute aliases → target write → target form verify → flush progress JSON in `{instance_progress_dir}`. |

Instance-specific checklist text (tenant hosts, template names): `{instance_progress_dir}/.agents/skills/cmw-platform/references/us_fm_ru_to_en_replication.md`.

## When to use

| Signal | Why Ralph fits |
|--------|----------------|
| Multi-wave goal | Property → building → floor → spaces, themed migration batches, N independent templates |
| Disk-backed state | `{instance_progress_dir}/localization/migration_progress/*.json`, `map[]`, `operations[]` |
| External verification | `jq`, pytest, API GET spot-check, `meta.status` terminal values |
| Coordinator + optional workers | Parent reconciles JSON; up to 6 parallel subagents on **independent** scopes |
| Fresh context per wave | New subagent / new turn; ids live in JSON, not chat |
| Long-running CMW fill | Harvest/seed/backup CLI in background shell + poll JSON |

## When NOT to use

| Situation | Use instead |
|-----------|-------------|
| Single-shot edit, one record, one attribute | Normal agent turn; no loop |
| Secrets, credentials, password rotation | Explicit user step; no autonomous retry loop |
| Destructive ops (delete tree, force-push, mass unpublish) | **User approval** each time; no Ralph until approved |
| Infinite or unbounded retry | Cap iterations; document blocker in JSON + roadmap |
| Two agents writing **same** batch file or **same** EN record ids | One writer per batch JSON; one scope per parallel worker |
| Plugin stop-hook **while** Multitask parallel workers active | **Mode A** coordinator only; pause workers before **Mode B** plugin |
| Authoritative state only in chat | Flush JSON first — Ralph requires disk memory |
| Platform-generic recipe in instance repo only | Promote to cmw-platform-agent skill; keep ids/maps in instance repo |

## Abuse guardrails (mandatory)

| Guardrail | Rule |
|-----------|------|
| **Max loop iterations** | Default `max_iterations: 30` per coordinator session (scratchpad YAML or `meta.agent_wave`); stop and record blocker when exceeded |
| **Completion promise** | Output `<promise>…</promise>` **only** after external verification (jq, tests, terminal `meta.status`) — never on model self-assessment |
| **Parallel batch files** | **Separate** `migration_progress/YYYYMMDD_<scope>.json` per parallel subagent; no shared mutable file |
| **Plugin vs Multitask** | Do **not** run Ralph Loop **plugin** stop-hook on the same workspace while parallel subagents are in flight |
| **Scope boundary** | **Platform-generic** patterns → cmw-platform-agent skills; **instance** ids, maps, harvest → `{instance_progress_dir}` only |
| **Same-step failures** | Stop loop after **4** failed attempts on the **same** step without new evidence; append `meta.errors[]` |
| **Secrets** | Never commit `.env`, passwords, or real tenant credentials; placeholders in `.env.example` only |
| **Destructive / git force** | User must explicitly approve force-push, hard reset, mass delete, production config wipe |
| **RU write ban** | Migration loops: reads on source host only; writes only on target host env (`CMW_BASE_URL_EN` or agreed target) |
| **Backup serialization** | One configuration backup per **milestone** after wave completes — not per subagent in parallel |
| **CLI bash loop** | Optional `agent -p` loop: **one** atomic scope per iteration; do not overlap IDE subagents on same batch file |

## Execution modes

| Mode | What it is | Default? | Parallel worker risk |
|------|------------|----------|----------------------|
| **A — Coordinator emulation** | Parent Multitask: read JSON → verify → spawn ≤6 workers → flush → repeat “next loop” prompt | **Yes** in IDE Multitask | **Low** — no stop hook |
| **B — Ralph Loop plugin** | Cursor plugin: `.cursor/ralph/scratchpad.md`, stop hook re-feeds same prompt, `completion_promise` | Only when user installs plugin and **pauses** parallel workers | **High** if combined with active Multitask |
| *(optional)* **Timed `/loop`** | Cursor `/loop 15m` on parent with coordinator-only body (reconcile + spawn, no duplicate in-flight work) | Coordinator-only heartbeat | Low if body does not duplicate subagents |

**Rule:** Do not start **Mode B** on the same workspace while **Mode A** subagents are running.

### Mode A — coordinator loop (recommended)

Each iteration:

1. Read `{instance_progress_dir}/localization/migration_progress/*.json` and instance roadmap.
2. Run **external verification** (jq / Python / GET checks).
3. If incomplete: spawn **up to 6** background subagents (independent templates/floors only).
4. Poll background shells / JSON until wave settles.
5. **Mandatory progress flush** — JSON + progress report MD in instance repo; commit instance repo when policy requires.
6. One target backup per milestone ([cmw-platform-backup-launch](../../cmw-platform-backup-launch/SKILL.md)).
7. Increment `meta.agent_wave` or scratchpad `iteration`.

Stop when verification passes, `max_iterations` reached, or documented blocker.

### Mode B — plugin

- Install from [Cursor Directory — Ralph Loop](https://cursor.directory/plugins/ralph-loop).
- Copy `{instance_progress_dir}/docs/localization/ralph_scratchpad.example.md` → `.cursor/ralph/scratchpad.md` (local, not committed).
- Subagents **cannot** install plugins; user action required.
- Pause parallel workers before enabling stop-hook loop.

## External verification (trust files)

| Check | Example pattern |
|-------|-----------------|
| Batch terminal | `jq -e '.meta.status == "done"' …/migration_progress/YYYYMMDD_*.json` |
| Map complete | Required keys present in `map[]` / level maps for wave scope |
| Errors empty | `jq -e '(.meta.errors // []) \| length == 0' …` |
| Pending scope | No unmapped source ids for current wave in priority JSON |
| Write target | Operations log only targets agreed EN host env |

Do not paste full `map[]` into chat; verify in instance working tree.

## Progress flush (instance migration)

Flush to **`{instance_progress_dir}`** after each wave; harvest complete (`meta.harvest_path`); every N records; wave timeout (`partial` + `retry_count`); **before session end**.

Required JSON shape: `meta`, `map[]` (or level-keyed map), `operations[]`, `errors[]`, counts. Instance FM/US FM detail: `{instance_progress_dir}/.agents/skills/cmw-platform/references/us_fm_ru_to_en_replication.md`. Generic contract: [tr_fr_record_harvest_seed.md](tr_fr_record_harvest_seed.md).

## Environment placeholders (no absolute paths in git)

| Placeholder / var | Role |
|-------------------|------|
| `{instance_progress_dir}` | Instance repo root — `localization/migration_progress/` |
| `CMW_BASE_URL_EN` | Target writes |
| `CMW_BASE_URL_RU` | Source read-only harvest (when applicable) |
| `CMW_BASE_URL` | Set to write target immediately before record tools |

## Parent: next loop vs spawn workers

| Condition | Parent action |
|-----------|---------------|
| External verify **fails**, scoped work remains, under iteration cap | **Next loop** — same checklist prompt; optionally spawn ≤6 workers for **independent** slices |
| Verify **passes** for current milestone | Stop loop; output `<promise>` only if using plugin and promise defined |
| Same step failed 4× | Stop; document blocker in JSON + roadmap; no blind retry |
| Long harvest/seed/backup | **Background shell** + poll JSON — parent does not block chat |
| Workers already running on a scope | Do **not** spawn duplicate workers; reconcile after wave |

## Instance fill agreements (not in this repo)

Tenant-specific goals (US FM hierarchy, stand ids, naming, RU-form-first rules) live in the **instance repository** — not here. Start with:

- `{instance_progress_dir}/.agents/skills/ralph-loop-instance/SKILL.md`
- `{instance_progress_dir}/localization/AGENTS.md` (operating agreements)
- `{instance_progress_dir}/docs/localization/ralph_loop_us_fm_autonomous_runbook.md` (one example runbook)
- `{instance_progress_dir}/docs/plans/20260602_ralph_loop_fm_fill_process.md` (instance process plan)

## Related

- [ralph_loop_autonomous_execution.md](ralph_loop_autonomous_execution.md) — short index
- [cmw-platform SKILL §9](../SKILL.md#9-growing-platform-skills) — parallel subagents, JSON over memory
- [tr_fr_record_harvest_seed.md](tr_fr_record_harvest_seed.md) — harvest/seed JSON contract
