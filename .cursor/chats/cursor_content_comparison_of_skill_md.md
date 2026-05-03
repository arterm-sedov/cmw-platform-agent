# Content comparison of SKILL.md
_Exported on 5/2/2026 at 00:53:09 GMT+3 from Cursor (3.2.16)_

---

**User**

@.agents/skills/cmw-platform/SKILL.md 
How much useful content did it loose comparing to @.agents/skills/cmw-platform/SKILL.md.backup  @SKILL.md (1-806) 

Also check git commits changing this file

---

**Cursor**

Now let me check the git history for this file.

Now I have enough information. Let me also do a quick `diff --stat` to quantify the changes.

I have all the information needed. Let me present the comparison and findings.

---

# Comparison: `SKILL.md` vs `SKILL.md.backup`

## Provenance

The backup was created on **2026-04-22** by commit `d4e9462` ("docs: add cmw-platform skill reorganization plan") — `arterm-sedov` snapshotted the old `SKILL.md` as `SKILL.md.backup` and rewrote `SKILL.md` into a new section-numbered structure. Subsequent commits then added new content (browser automation, button kinds, etc.) to the new file.

## Size Snapshot

- `SKILL.md.backup` — **623 lines / 21,890 bytes** (frozen on 2026-04-22)
- `SKILL.md` — **806 lines / 30,524 bytes** (last touched 2026-04-27)
- Raw diff: **+570 / −388 lines** between the two files

The current file is bigger overall, but that's because ~340 lines of brand-new browser-automation content was added. The reorg itself dropped real material.

## Useful Content That Was Lost or Reduced

The following were in the backup but are **gone or substantially shrunk** in the current file:

### Fully removed
- **"Knowledge Base" section** (backup lines 88–108) — instructions to use the `cmw_platform_knowledge-base` MCP (`get_knowledge_base_articles`) for platform expertise, with a code example and "when to use" criteria. Also the explicit warning **"Do NOT use `ask_comindware`"**. Not present anywhere in the current file.
- **"System Prompt Alignment" section** (backup lines 416–439) — precedence rules between `agent_ng/system_prompt.json` (PRIMARY, agentic behavior) and `AGENTS.md` (SECONDARY, project guidelines), with a "for platform operations" 4-step checklist. Completely dropped.
- **"Required Fields for Create" table** (backup lines 498–511) — per-attribute-type required field map (Decimal → `number_decimal_places`, Enum → `display_format`+`enum_values`, DateTime, Document, Image, Duration, Account, Record), plus the note that Text/String needs no type-specific fields. Gone.
- **"Edit Tool Validation Pattern" section** (backup lines 513–557) — full table of editable fields per tool type (Dataset / Button / Toolbar), the "identifiers always required" list, and the **"⚠️ WARNING: System Buttons"** note about `create`/`edit`/`archive`/`delete`/`unarchive` only being safe to rename. Gone.
- **"How Partial Updates Work" numbered explanation** (backup lines 559–575) — 4-step description of fetch → merge → preserve plus a worked dataset-name-only edit example. Gone.
- **"Working Files" full guidance** (backup lines 578–621) — the *Fetch-and-Save Immediately* pattern with a real Python example writing `cmw-platform-workspace/rentlots_schema_20260415.json`, and the **File Naming Convention** (`{entity}_schema_BEFORE.json` / `_AFTER.json` / `_changes.json`). The current file has only a 5-line "Save Before Edit" stub at the top.
- **"Guiding Principles" four-pointer block** (backup lines 25–35) — *Persist Context*, *Read Before Write*, *Idempotent Operations*, *Explicit Over Implicit*. Only "Save Before Edit" survives.
- **`archive_unarchive_button` code example** (backup lines 298–304) — referenced in the tool table but the working snippet was removed.

### Substantially reduced
- **Inline "Core Tools" table** (backup lines 67–86) — the at-a-glance 18-row tool catalog was replaced with a pointer to `references/tool_inventory.md`. Loss of one-page discoverability.
- **Dataset-specific toolbar 3-step workflow** (backup lines 211–249) — the worked "Step 1 create toolbar → Step 2 add items → Step 3 link to dataset" code example became a single warning sentence (current line 599).
- **"Toolbar Item Names Override Button Names" WRONG vs CORRECT examples** (backup lines 251–273) — collapsed to one warning line (current line 621). The contrasting code blocks are gone.
- **Detailed dataset edit options** (backup lines 371–382) — the column-mutation menu (`Name`, `isHidden`, `_delete`/`null`, add via `propertyPath`) and the full `sorting`/`grouping`/`fields`+`aggregationMethod` spec was dropped. Current file shows only `Name` + `isHidden` in one example.
- **"How edit_or_create Tools Work" walk-through** (backup lines 449–496) — the worked `Ploschad`/`Volga`/`RentLots` partial-update example explaining `remove_values()` → fetch → patch → API was reduced to a 3-row table (current lines 769–775).

## Content Gained in Current `SKILL.md`

For balance — the rewrite isn't pure loss:
- ~340 lines on **browser automation** (4 options: agent-browser MCP, playwright MCP, agent-browser CLI, playwright-cli) with decision matrix, refs, and CMW SPA URL map.
- **Pagination helper** (`fetch_all` with `offset`).
- **Localization workflow** (harvest → build → apply → CSV).
- **Create-kind button details** (`create_form` requirement, `context`, UI cache caveat).
- **Toolbar-Dataset linking** semantics (`IsDefaultForLists` flag).
- Expanded reference index (9 vs 4 entries).

## Bottom Line

Roughly **8 substantive blocks** of platform-API lore are no longer in `SKILL.md`. The most operationally painful losses are:

1. The **Knowledge Base / `get_knowledge_base_articles` MCP usage**.
2. The **"Required Fields for Create" attribute-type table**.
3. The **Edit Tool Validation Pattern** (per-tool editable fields + system-buttons warning).
4. The **"Fetch-and-Save Immediately" pattern + file naming convention**.
5. The **System Prompt Alignment precedence rules**.

These are concrete, hard-to-rederive checklists. The new file leans more on `references/*.md` files, but unless those references mirror the dropped tables (worth verifying), the agent has lost ready-at-hand guardrails.

## Git History (commits touching `SKILL.md`)

Reachable from `origin/main` (the local `main` is 147 commits behind):

- **2026-04-30** `5cf2f48` Dagdaf — Merge `origin/main` into `20260430_agentic_skills_marat_main_merge`
- **2026-04-29** `d746b4b` / `26344f1` Dagdaf — Updated extractions / aliases extract + helper scripts
- **2026-04-28** `21464b5` arterm-sedov — merge: reconcile CMW skill and tool exports
- **2026-04-28** `cd8efe3` / `9c468cc` / `eacb585` / `aa12104` / `148872f` arterm-sedov — `resolve_entity` unification, generic URLs, task entity resolution, human-friendly terms, `resolve_entity` tool
- **2026-04-28** `bfe8fdc` Dagdaf — Update localization tool and skills
- **2026-04-27** `44658f7` arterm-sedov — Russian names for all 29 button kinds
- **2026-04-27** `3ceab86` / `495ae15` / `3bd5b5e` arterm-sedov — button kind optional, button kind validator, button context + Create-kind support
- **2026-04-27** `32f9d5d` Dagdaf — ontology object tools, localization helpers, skill update
- **2026-04-23** `15365b0` / `1e1f1bf` arterm-sedov — Browser automation plan + URL patterns
- **2026-04-22** `d4e9462` arterm-sedov — **docs: add cmw-platform skill reorganization plan** ← this is the commit that produced `SKILL.md.backup` and rewrote `SKILL.md` (-388 / +577)
- **2026-04-21** `00fc115` Dagdaf — Updated skill, fixed problems with new tool commits
- **2026-04-20** Dagdaf — series of dataset / toolbar fixes (`4ad8a34`, `f7ca91f`, `65ad960`, `5c61aaf`, `7f77056`, `0f21401`, `ad30b49`)
- **2026-04-15** arterm-sedov — toolbar-item override clarification (`a95b19e`), toolbar/button tools (`011f615`), `edit_or_create_form` (`08c6a3d`), `batch_edit_attributes` refactor (`bb8e53a`), several `docs(skill)` clarifications, ruff cleanup (`d4a81ac`), and the **initial creation** `76438f3` "feat(skill): add cmw-platform autonomous platform skill"

---

**Want me to draft a recovery plan?** I can propose a concise plan to fold the lost-but-still-useful blocks (knowledge base usage, required-fields-for-create, validation pattern, fetch-and-save pattern, system-prompt precedence) back into the new section-numbered structure without breaking the new layout — confirm and I'll prepare it via `CreatePlan`.

---

**User**

was deleted material per git diffs dropped or moved and streamlined to @.agents/skills/cmw-platform/references ?

---

**Cursor**

# Answer: Mostly **dropped**, not moved

Hard evidence from git: the three reference files (`tool_inventory.md`, `api_endpoints.md`, `errors.md`, `workflow_sequences.md`) were all created on **2026-04-15** in commit `76438f3` — **a week before** the reorg commit `d4e9462` (2026-04-22). Their last commit is also `2026-04-15` (`bb8e53a` / `d4a81ac`). The reorg commit only added a new `localization.md` to references; it did **not** push any content from the old `SKILL.md` into the existing reference files.

So the references existed in *parallel* with the backup `SKILL.md`, not as the *destination* of the reorg.

## Per-block fate (verified by grepping the whole skill folder)

| Backup section | Fate | Where it ended up |
|---|---|---|
| Core Tools table | **Moved & streamlined** | [`references/tool_inventory.md`](.agents/skills/cmw-platform/references/tool_inventory.md) (already existed) |
| "How edit_or_create Tools Work" walk-through (`Ploschad`/`Volga`/`RentLots`, `remove_values()` mechanics) | **Partially moved & streamlined** | [`references/workflow_sequences.md`](.agents/skills/cmw-platform/references/workflow_sequences.md) §7 has a small table + 1 example, prose internals dropped |
| "How Partial Updates Work" 4-step explanation | **Partially moved** | Same place — `workflow_sequences.md` §7 |
| Working Files / Fetch-and-Save / File Naming Convention | **Streamlined inline** | Reduced to 5-line "Save Before Edit" stub at top of new `SKILL.md` (lines 84–95). **Not** in references. |
| Toolbar Item Names Override Button Names (WRONG vs CORRECT) | **Streamlined inline** | One warning sentence (`SKILL.md` line 621). Code examples gone. **Not** in references. |
| Dataset-specific toolbar 3-step workflow | **Streamlined inline** | One warning sentence (`SKILL.md` line 599). Worked example gone. **Not** in references. |
| Knowledge Base / `get_knowledge_base_articles` MCP usage + "Do NOT use `ask_comindware`" | **Dropped** | Grep finds it **only** in `SKILL.md.backup`. Not in any reference. |
| System Prompt Alignment (precedence: `system_prompt.json` PRIMARY vs `AGENTS.md` SECONDARY) | **Dropped** | Only in backup. |
| Required Fields for Create table (per attribute type) | **Dropped** | Only in backup. (`number_decimal_places` appears once in `workflow_sequences.md` line 197 but only as an inline comment, not the table.) |
| Edit Tool Validation Pattern (per-tool editable fields, System Buttons warning) | **Dropped** | Only in backup. Keywords `view_type`, `is_default_for_forms`, `kind: UserEvent` exist nowhere else. |
| Guiding Principles (Persist Context / Read Before Write / Idempotent / Explicit Over Implicit) | **Dropped** | Only in backup. |
| `archive_unarchive_button` code example | **Dropped** | Only in backup. (Tool name appears nowhere else.) |
| Detailed dataset options (`aggregationMethod`, grouping `fields`, sorting `direction`/`nullValuesOnTop`) | **Dropped** | `aggregationMethod` keyword exists only in backup. |

## TL;DR

- **2 blocks genuinely moved/streamlined to references** (Core Tools table → `tool_inventory.md`; partial-update mechanics → `workflow_sequences.md` §7).
- **3 blocks streamlined inline** in the new `SKILL.md` (kept as one-liners, lost the worked examples).
- **7 blocks were straight dropped** with no replacement — most painfully: Knowledge Base MCP usage, Required Fields for Create, Edit Tool Validation Pattern, System Prompt Alignment, and the Fetch-and-Save / file-naming pattern.

If you want, I can prepare a `CreatePlan` to restore the 7 dropped blocks — either by re-inserting them into the section-numbered `SKILL.md` or by extending the existing reference files (e.g., a new `references/edit_validation.md` and an enrichment of `tool_inventory.md` / `workflow_sequences.md`). Tell me which target you prefer (inline in `SKILL.md`, references, or split) and I'll draft the plan.

---

**User**

can we move these into @.agents/skills/cmw-platform/references in a very elegant way?

---

**Cursor**

Good question. Let me design this elegantly. Before I write up the plan, one quick decision affects the structure:

> **How would you like the relocated content organized in references/?**
> - Two focused new files: principles.md (agentic discipline + working files) + edit_or_create.md (validation/required-fields/partial-update deep dive). Enrich the existing tool_inventory.md / workflow_sequences.md with the rest. Cleanest semantic split.
> - No new files — fit everything into existing references (tool_inventory.md, workflow_sequences.md, errors.md). Smallest blast radius.
> - One file per topic: principles.md, working_files.md, edit_or_create.md, knowledge_base.md, system_prompt_alignment.md. Most discoverable but more files. ✔

> **What should remain inline in SKILL.md after the move?**
> - Keep one-line warnings/headers + '→ See also: references/X.md' links. Lean entrypoint, all detail in references. ✔
> - Keep short summaries (3–5 lines per topic) + links. More self-contained SKILL.md.
> - Move everything; leave only links in SKILL.md. Most aggressive.

---

**User**

Continue

@.agents/skills/cmw-platform/SKILL.md current version have been updated in latest commits, make sure to review the git diff and compare to our conversation and the backup

---

**Cursor**

I've reviewed the new state. Here's what I found:

## Current State of `SKILL.md`

- **Local branch**: `main` is now at `5cf2f48` (caught up to `origin/main`).
- **File size**: 1243 lines committed (working tree differs by 4 lines — see issue below). The IDE banner showing 1555 was apparently stale.
- **Diff vs previous HEAD**: +798 / −45 lines.

### What the new commits added (none of it was the "lost" content)

| Addition | Section | Origin commit(s) |
|---|---|---|
| Expanded browser automation guidance | §2 | `15365b0`, `1e1f1bf` |
| **New §3.5 Import/Export Applications** (CTF, `export_application` / `import_application`) | new | `98e4f20` |
| **Button Kinds table — all 29 kinds with Russian names** | §4 (UI Components) | `44658f7`, `5f4d6dd` |
| **Toolbar-Dataset Linking** subsection | §4 | `dbdf2d2`, `3bd5b5e` |
| **Massive Localization (System Names) workflow** (Phases 1–9, type mappings, `tool_localize`, step scripts, expression patterns, aliasLocked logic) | §6 (entirely rewritten) | `26344f1`, `d746b4b`, `bfe8fdc`, `32f9d5d` |
| Extra terminology rows (`list`, `solution`, `scheme`, `trigger`, `workspace`, `card`, `task`) | §1 | `aa12104`, `eacb585` |
| `resolve_entity` mention indirectly via terminology | §1 | `148872f` |

## Re-verification: the 10 lost blocks remain lost

Searched the entire `.agents/skills/cmw-platform/` tree for the canonical markers from each lost block:

| Lost block | Marker grepped | Status in new file |
|---|---|---|
| Guiding Principles | `Persist Context\|Read Before Write\|Idempotent\|Explicit Over Implicit` | **Still gone** |
| Knowledge Base MCP | `get_knowledge_base\|ask_comindware` | **Still gone** |
| System Prompt Alignment | `system_prompt\.json\|System Prompt Alignment` | **Still gone** |
| Required Fields for Create | `Required Fields for Create` | **Still gone** |
| Edit Tool Validation Pattern | `view_type\|is_default_for_forms\|Editable Fields\|System Buttons` | **Still gone** |
| Working Files / Fetch-and-Save | `Fetch and Save\|File Naming` | **Still 5-line stub at lines 91–104** |
| `archive_unarchive_button` example | `archive_unarchive_button` | **Still gone** |
| Detailed dataset options | `aggregationMethod` | **Still gone** |
| Toolbar Item Override (WRONG/CORRECT) | full code blocks | **Still 1-line warning at line 843** |
| Dataset-specific Toolbar 3-step | full 3-step example | **Still 1-line warning at line 821** |

## Two structural issues I noticed in the new file

1. **The committed file at HEAD `5cf2f48` contains unresolved merge conflict markers** in the footer (lines 1551–1555):

```1551:1555:.agents/skills/cmw-platform/SKILL.md
<<<<<<< HEAD
*End of SKILL.md - Updated 2026-04-29 with localization workflow fixes*
=======
*End of SKILL.md - Updated 2026-04-27 with import/export application tools, unified browser automation guidance, and localization workflow updates*
>>>>>>> 8ad0b47587a708b13f67d95d8a6091138df87915
```

The working tree has these resolved (kept the 2026-04-27 line, dropped the 2026-04-29 line) — that's the `M` you see in `git status`. This is technically a separate fix and unrelated to our relocation plan.

2. **Broken section numbering** in the new file:
   - `## 3. Exploration` → `## 4. Data Operations` → `## 3.5. Import/Export Applications` (out-of-order) → `## 4. UI Components` (duplicate of §4) → `## 6. Localization` (skips §5).

## Plan Status

**Our plan is still 100% valid** — none of the lost content was restored by the recent commits. Only minor mechanical updates needed:

- Line numbers shift: "Dataset-Specific Toolbars" warning is now **line 821** (was 599); "Toolbar Item Names Override" warning is now **line 843** (was 621); "Save Before Edit" stub is now **lines 91–104** (was 84–95); "Safe Attribute Translation" mini-table is now **lines 1516–1524** (was 769–775).
- Section labels change: the moves now land in **§1 Core Concepts** (Guiding Principles, Knowledge Base, System Prompt Alignment stubs), **§4 UI Components** (the duplicate-numbered one), and **§7 Troubleshooting**.
- We can optionally bundle two pre-existing fixes into the same patch: (a) clean the conflict markers in the footer, (b) renumber the broken sections (`3.5` → `5`, fix the duplicate `4`, restore `5`).

Let me update the plan file to reflect the new line numbers and offer the two structural fixes as opt-in extras.

I'll update the plan to reflect the new line numbers and add the two structural fixes I discovered.

Plan updated. Summary of what changed in the plan:

- **New "State Verified" callout** at the top — confirms 1243-line HEAD, lists what the recent commits added, and re-affirms that **none of the 10 lost blocks were restored**.
- **Refreshed line numbers**: Dataset-Specific Toolbars warning **821** (was 599), Toolbar Item Override warning **843** (was 621), Safe Attribute Translation table **1516–1524** (was 769–775).
- **Section label fix**: the duplicate-numbered "Section 4: UI Components" instead of the old "Section 5".
- **New "Pre-Flight Cleanup" section** (opt-in, included by default) covering the two structural issues I found:
  1. Unresolved merge conflict markers in the footer (lines 1551–1555 of HEAD).
  2. Broken section numbering (`3.5` out-of-order, duplicate `4`, missing `5`).
- **Two new todos** for those cleanups (`cleanup_conflict_markers`, `cleanup_section_numbering`) — easy to drop if you'd rather handle them separately.

Plan file: [`relocate_lost_skill.md_content_d9bc3cc5.plan.md`](C:/Users/webma/.cursor/plans/relocate_lost_skill.md_content_d9bc3cc5.plan.md)

Want me to:
- **Proceed with execution** (I'll need to switch to agent mode), or
- **Drop the pre-flight cleanups** from scope first, or
- **Tweak something else** in the plan?

