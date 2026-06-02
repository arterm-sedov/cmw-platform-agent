# Ralph-style autonomous execution (platform-generic)

Companion for long-running CMW work when agents should **iterate until externally verified done**, not stop when the model says “complete.”

**Full guardrails:** [ralph_loop_goal_autonomy.md](ralph_loop_goal_autonomy.md)  
**Instance consumer:** `{instance_progress_dir}/.agents/skills/ralph-loop-instance/SKILL.md`  
**Example instance runbook:** `{instance_progress_dir}/docs/localization/ralph_loop_us_fm_autonomous_runbook.md`  
**Instance process plan:** `{instance_progress_dir}/docs/plans/20260602_ralph_loop_fm_fill_process.md`

## Core idea (Ralph Wiggum)

1. **Same task prompt** each iteration (or coordinator re-prompts subagents with the same wave spec).
2. **Fresh context** — new subagent / new chat turn; do not rely on chat for `map[]` or record ids.
3. **Persistent memory on disk** — git-tracked skills/docs + **instance** `migration_progress/*.json` (not conversation).
4. **External verification** — loop stops when JSON counts / `meta.status` / jq checks pass, not when the LLM claims done.

References: [ghuntley.com/ralph](https://ghuntley.com/ralph/), [Cursor Directory — Ralph Loop](https://cursor.directory/plugins/ralph-loop), [forum guide](https://forum.cursor.com/t/ralph-cursor-guide/149998), [ralph-cursor](https://github.com/danielsinewe/ralph-cursor).

## Three execution modes (pick one)

| Mode | When | Risk to parallel workers |
|------|------|---------------------------|
| **A. Coordinator emulation** (recommended) | Parent Multitask + up to 6 background subagents | **Low** — no stop hook |
| **B. Cursor `/loop`** | Timed heartbeat for parent coordinator only | **Low** if body only reconciles + spawns waves |
| **C. Ralph Loop plugin** | Single-agent iterative task in IDE | **High** if combined with active parallel subagents |

**Rule:** Do not start mode **C** on the same workspace while mode **A** subagents are running.

## Coordinator emulation (mode A)

Parent agent each “loop”:

1. Read `{instance_progress_dir}/localization/migration_progress/*.json` and instance roadmap.
2. Run **external verification** (jq / Python / GET — see instance runbook).
3. If incomplete: spawn **up to 6** parallel background subagents (independent scopes only).
4. **Mandatory progress flush** after wave — JSON + MD in instance repo.
5. One target backup per milestone ([cmw-platform-backup-launch](../../cmw-platform-backup-launch/SKILL.md)).
6. Increment `meta.agent_wave` or scratchpad `iteration`.

## External verification (mandatory)

Trust **files**, not chat. Run checks in the **instance repo** working tree; do not paste full maps into chat.

## Scratchpad (plugin or emulation)

**Plugin path:** `.cursor/ralph/scratchpad.md` (local).

**Emulation template:** `{instance_progress_dir}/docs/localization/ralph_scratchpad.example.md`

## Progress flush triggers

Flush to **`{instance_progress_dir}`** after property/building/floor/space waves, harvest complete, every N records, before session end, and on wave timeout (`meta.status: partial`, `retry_count++`).

Never store authoritative record ids only in chat.

## Related

- [cmw-platform SKILL §9](../SKILL.md#9-growing-platform-skills) — parallel subagents, JSON over memory
- `{instance_progress_dir}/localization/AGENTS.md` — autonomous migration execution (instance)
