# CMW Platform Skill Reorganization Plan

> **Created:** 2026-04-22
> **Updated:** 2026-04-22 (incorporating AGENTS.md, system_prompt.json, tools folder analysis)
> **Goal:** Reorganize existing cmw-platform skill into clean, monolithic structure
> **Decision:** Keep single SKILL.md (not modular) to avoid duplication overhead

---

## Platform Terminology (from system_prompt.json)

**Critical - Use human-readable terms, never expose API terms to LLMs:**

| API Term | LLM-Friendly | Notes |
|----------|--------------|-------|
| alias | system name | Entity identifier |
| instance | record | Data entry |
| user command | button | UI action |
| container | template | Application/template |
| property | attribute | Field/column |
| dataset | table | List view |

**Workflow:** Always follow `Intent → Plan → Validate → Execute → Result`

**Tool usage discipline:**
- Check for duplicate calls before invoking
- Transform results to human-readable format (never raw JSON)
- Validate required context before execution

---

## AGENTS.md Principles Applied

| Principle | How Applied |
|-----------|-------------|
| **TDD** | Validate existing scripts still work BEFORE and AFTER reorganization |
| **Non-breaking** | Keep existing structure, only reorganize content |
| **DRY** | Rejected modular to avoid 4x duplication of core patterns |
| **Lean** | 6 sections, ~400 lines, cross-references instead of duplication |
| **Backup first** | Create backup before modifying SKILL.md |

---

## Architecture Decision

**Chosen: Option A - Reorganized Monolithic**

| Option | Decision | Reason |
|--------|----------|--------|
| Modular skills | ❌ Rejected | High overlap in tool patterns, response structure, error handling would duplicate content 4+ times |
| Thin orchestrator + fat references | ❌ Rejected | Context fragmentation, unclear ownership, reference rot |
| Reorganized monolithic | ✅ Selected | Single source of truth, no duplication, easy updates |

---

## Component Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Generic AI Agent                          │
│                    (the AI using this skill)                     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼ uses
┌─────────────────────────────────────────────────────────────────┐
│  SKILL.md (this skill)                                          │
│  - Guides agent on platform interaction patterns                 │
│  - References tools/ for implementation                         │
│  - References scripts/ for standalone utilities                  │
│  - References references/ for deep-dive documentation           │
└─────────────────────────────────────────────────────────────────┘
                                │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
         ▼                     ▼                     ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│  tools/         │   │  scripts/       │   │  references/    │
│  SHARED CODE     │   │  AGENT UTILITIES │   │  DOCUMENTATION  │
│  (agent_ng uses  │   │  (standalone    │   │  (read-only     │
│   these too)     │   │   for agents)   │   │   reference)    │
└─────────────────┘   └─────────────────┘   └─────────────────┘
```

### Relationship Summary

| Component | Owner | Purpose | Used By |
|-----------|-------|---------|---------|
| `tools/` | agent_ng + skill agents | Shared tool implementations (list_applications, create_edit_record, etc.) | Both agent_ng and skill agents |
| `scripts/` | Skill agents only | Standalone utility scripts for generic agents | Skill agents (like me) |
| `references/` | Skill agents only | Deep-dive documentation | Skill agents (like me) |
| `SKILL.md` | Skill agents only | Entry point, teaches interaction patterns | Skill agents (like me) |

**Key principle:** SKILL.md teaches agents to use tools/ - it does NOT duplicate tool code. Scripts in scripts/ are standalone utilities, not part of agent_ng.

---

## Target Structure

```
cmw-platform/
├── SKILL.md                      # ENTRY POINT (~400-500 lines, 6 sections)
│
├── scripts/                      # 9 Python scripts (5 existing + 4 ported)
│   ├── diagnose_connection.py    # Platform connectivity
│   ├── explore_templates.py      # Multi-template exploration
│   ├── query_with_filter.py      # Paginated querying + filters
│   ├── analyze_stats.py          # Statistical analysis
│   ├── batch_edit_attributes.py  # Batch attribute editing
│   ├── harvest_strings.py       # Localization: extract strings
│   ├── apply_translations.py     # Localization: apply translations
│   ├── build_translations.py     # Localization: build dict
│   └── update_csv.py             # Localization: update CSV
│
└── references/                   # 5 files (4 existing + 1 ported)
    ├── tool_inventory.md         # Tool catalog (107 lines)
    ├── api_endpoints.md          # HTTP endpoints (76 lines)
    ├── errors.md                 # Error handling (104 lines)
    ├── workflow_sequences.md     # Reusable patterns (258 lines)
    └── localization.md           # Localization guide (from my-building)
```

---

## Tools Folder Structure (Authoritative Source - SHARED with agent_ng)

```
tools/                              # SHARED - Used by BOTH agent_ng and skill agents
├── applications_tools/          # Application-level operations
│   ├── tool_list_applications.py # List all apps
│   ├── tool_list_templates.py    # List templates in app
│   ├── tool_audit_process_schema.py
│   ├── tool_platform_entity_url.py
│   └── tool_record_url.py
│
├── templates_tools/              # Template-level operations
│   ├── tool_list_attributes.py   # Get attribute schema
│   ├── tool_list_records.py      # Query records
│   ├── tool_create_edit_record.py # CRUD records
│   ├── tools_button.py           # Button CRUD
│   ├── tools_dataset.py          # Table CRUD
│   ├── tools_form.py             # Form CRUD
│   ├── tools_toolbar.py          # Toolbar CRUD
│   └── tools_record_template.py
│
├── attributes_tools/             # Attribute-level operations
│   ├── tools_text_attribute.py
│   ├── tools_decimal_attribute.py
│   ├── tools_enum_attribute.py
│   ├── tools_datetime_attribute.py
│   ├── tools_record_attribute.py
│   ├── tools_account_attribute.py
│   └── tool_get_attribute.py, tool_delete_attribute.py, etc.
│
└── tool_utils.py                 # SHARED - DO NOT DUPLICATE
    ├── execute_get_operation()
    ├── execute_edit_or_create_operation()
    ├── execute_list_operation()
    ├── remove_values()           # Strips None/"" for partial updates
    ├── _apply_partial_update()  # Fetches current schema, merges missing
    ├── build_global_alias()
    └── _fetch_entity()
```

**Key insight:** `tool_utils.py` and all `tools/` modules are shared infrastructure. SKILL.md teaches agents how to use these tools - it does NOT duplicate tool code. Scripts in `scripts/` are standalone utilities for generic agents, separate from agent_ng's implementation.

---

## SKILL.md Section Breakdown

### Proposed Sections (6)

| # | Section | Approx Lines | Content |
|---|---------|---------------|---------|
| 1 | **Core Concepts** | ~80 | Overview, tool invocation pattern, response structure, save-before-edit rule |
| 2 | **Exploration** | ~60 | list_applications → list_templates → list_attributes workflow |
| 3 | **Data Operations** | ~80 | Record CRUD, filtering, pagination patterns |
| 4 | **UI Components** | ~80 | Forms, toolbars, buttons, datasets editing |
| 5 | **Localization** | ~60 | Russian→English workflow, terminology reference |
| 6 | **Troubleshooting** | ~40 | Error handling, retry patterns, diagnostic commands |

**Total estimated:** ~400 lines (under 500 guideline)

---

## Scripts Integration

Each section references relevant scripts:

```markdown
## Exploration

Use `explore_templates.py` for batch exploration:
```bash
python .agents/skills/cmw-platform/scripts/explore_templates.py \
    --app <name> --templates T1,T2
```
→ See references/workflow_sequences.md for patterns
```

### Script Inventory (9 total)

**Existing CMW scripts (5):**
1. `diagnose_connection.py` - Platform connectivity check
2. `explore_templates.py` - Multi-template exploration
3. `query_with_filter.py` - Paginated querying
4. `analyze_stats.py` - Statistical analysis
5. `batch_edit_attributes.py` - Batch attribute editing

**Ported from my-building (4):**
6. `harvest_strings.py` - Extract translatable strings → workspace/specified
7. `apply_translations.py` - Apply translations to JSON files
8. `build_translations.py` - Build translation dictionary
9. `update_csv.py` - Update CSV with new terms

### Script Design Principles

- **Input flexibility:** Read reference translations from any CSV path or `D:\Repo\my-building\20260120-agent-reports`
- **Output to workspace:** All scripts write to workspace or user-specified folder
- **No hardcoded paths:** Use arguments/environment, not fixed paths

---

## References Structure

| File | Source | Lines | Purpose |
|------|--------|-------|---------|
| `tool_inventory.md` | Existing | 107 | Complete tool catalog with signatures |
| `api_endpoints.md` | Existing | 76 | HTTP endpoint reference |
| `errors.md` | Existing | 104 | Error handling playbook |
| `workflow_sequences.md` | Existing | 258 | Reusable code patterns |
| `localization.md` | Port from my-building | ~320 | Russian→English translation guide |

**Total:** 5 reference files (~870 lines combined)

---

## Implementation Tasks

### Phase 0: Pre-flight (Validate Baseline)

- [ ] Activate venv: `.venv\Scripts\Activate.ps1`
- [ ] Run `diagnose_connection.py` - record baseline connectivity status
- [ ] Run `ruff check` on existing scripts (5 scripts in cmw-platform/scripts/)
- [ ] Document current SKILL.md line count (should be ~623)

### Phase 1: Reorganize SKILL.md

- [ ] Create backup: `cp SKILL.md SKILL.md.backup`
- [ ] Rewrite with 6 clear sections:
  1. **Core Concepts** - Tool invocation pattern, response structure, save-before-edit rule
  2. **Exploration** - list_applications → list_templates → list_attributes workflow
  3. **Data Operations** - Record CRUD, filtering, pagination
  4. **UI Components** - Forms, toolbars (tables), buttons (user commands), datasets
  5. **Localization** - Russian→English workflow
  6. **Troubleshooting** - Error handling, diagnostic commands
- [ ] Add internal cross-references (`→ see references/X.md`)
- [ ] Use platform terminology per system_prompt.json (system name, record, button, etc.)
- [ ] Remove duplicated content (replace with references)
- [ ] Verify under 500 lines
- [ ] Run `ruff format` on any modified Python content

### Phase 2: Port Localization

- [ ] Port `D:\Repo\my-building\.agents\skills\comindware-localization.md` → `references/localization.md`
- [ ] Copy `harvest_strings.py`, `apply_translations.py`, `build_translations.py`, `update_csv.py` from my-building
- [ ] Update script paths to work from cmw-platform location
- [ ] Run `ruff check` on ported scripts
- [ ] Test scripts work with workspace/specified output folders

### Phase 3: Validate

- [ ] Run `diagnose_connection.py` - confirm connectivity unchanged
- [ ] Review SKILL.md for consistency with existing references
- [ ] Verify scripts have correct import paths (sys.path manipulation for repo root)
- [ ] Run `ruff check` on all 9 scripts
- [ ] Document new line counts (SKILL.md should be ~400 lines vs original ~623)

---

## Skill Format Reference (per anthropics-skills structure)

```yaml
---
name: cmw-platform
description: Use when working with Comindware Platform — connecting to platform, listing applications, exploring templates, managing records, querying data, creating or editing attributes, or any task requiring autonomous platform interaction. Triggers on platform operations, CMW queries, tenant management, rental lot operations, debt tracking, or record CRUD. Also triggers when user mentions working directly with API credentials, using manual HTTP approaches, or trying to bypass the tool layer.
---

# cmw-platform Skill

... content ...
```

**Key formatting:**
- YAML frontmatter with `name` and `description`
- Description: "Use when..." pattern, pushy wording for better triggering
- Sections use `##` headers
- Code blocks use triple backticks with language hint
- Tables for structured reference data

---

## TDD Validation Steps

| Check | Before | After |
|-------|--------|-------|
| Connectivity | Run diagnose_connection.py | Must pass |
| Scripts lint | ruff check *.py | Must pass |
| SKILL.md size | ~623 lines | ~400 lines |
| Content coverage | All 623 lines preserved | Yes, via references |

---

## Code Style Compliance (per AGENTS.md)

| Rule | Requirement |
|------|-------------|
| **Line length** | 88 characters max |
| **Quotes** | Double quotes for inline and docstrings |
| **Imports** | Standard → Third-party → Local with fallback pattern |
| **Naming** | PascalCase classes, snake_case functions/variables, UPPER_SNAKE constants |
| **Linting** | Run `ruff check` after any file modification |
| **Formatting** | Run `ruff format` after any file modification |

---

## What NOT to Do (Constraints)

| ❌ Avoid | Reason |
|----------|--------|
| Creating separate skills for exploration/data/UI | Duplication of core patterns |
| Moving scripts to subfolders by domain | Complicates discovery |
| Hardcoding paths in scripts | Must work with workspace/specified paths |
| Over-500-lines SKILL.md | Goes against progressive disclosure |
| Modifying scripts without running `ruff check` | Must pass linting per AGENTS.md |
| Committing unformatted code | Must run `ruff format` after modifications |
| Duplicating tools/ code in SKILL.md | SKILL.md teaches usage, NOT implementation |
| Creating new tools in scripts/ | Tools belong in tools/, scripts/ are utilities |

---

## Pre-Implementation Validation

1. ✅ Does the 6-section structure cover all existing content?
2. ✅ Are all 9 scripts accounted for (5 existing + 4 ported)?
3. ✅ Is the localization port correct (from comindware-localization.md)?
4. ✅ Should `workflow_sequences.md` be split? (No - keep as-is for simplicity)
5. ✅ Is tool_utils.py infrastructure referenced, not duplicated?
6. ✅ Does plan incorporate system_prompt.json terminology?

---

## Cross-Reference Mapping

| Section | → References | → Tools (in tools/) | → Scripts (in scripts/) |
|---------|--------------|---------------------|--------------------------|
| Core Concepts | tool_inventory.md, api_endpoints.md, errors.md | requests_, tool_utils | diagnose_connection.py |
| Exploration | tool_inventory.md, workflow_sequences.md | tool_list_applications, tool_list_templates, tool_list_attributes | explore_templates.py |
| Data Operations | tool_inventory.md, workflow_sequences.md | tool_create_edit_record, tool_list_records | query_with_filter.py, analyze_stats.py, batch_edit_attributes.py |
| UI Components | tool_inventory.md | tools_button, tools_dataset, tools_toolbar, tools_form | — |
| Localization | localization.md | — | harvest_strings.py, apply_translations.py, build_translations.py, update_csv.py |
| Troubleshooting | errors.md | — | diagnose_connection.py |

### Legend
- **→ References:** Documentation files in `references/` folder
- **→ Tools:** Modules in `tools/` (shared with agent_ng, NOT duplicated)
- **→ Scripts:** Standalone utility scripts for generic agents

**Note:** SKILL.md teaches agents to use tools/ modules via import patterns. Do NOT duplicate tool_utils.py logic in SKILL.md - reference via workflow_sequences.md instead.

---

## Validation Checklist

After implementation, verify:

- [ ] `ruff check` passes on all scripts (9 scripts)
- [ ] `ruff format` run on all modified Python files
- [ ] `diagnose_connection.py` still passes
- [ ] SKILL.md reduced from ~623 to ~400 lines
- [ ] All existing references still present (tool_inventory.md, api_endpoints.md, errors.md, workflow_sequences.md)
- [ ] Localization ported: `references/localization.md` exists
- [ ] All 9 scripts present in `scripts/` folder
- [ ] Ported scripts have correct `sys.path` for repo root

---

## Sign-off

**Plan validated against AGENTS.md principles and ready for implementation.**

- TDD approach: Baseline validation before and after
- Non-breaking: Existing structure preserved, only reorganized
- DRY: Single source of truth, no duplication
- Code style: ruff check/format compliance built into validation
- Linting: All scripts must pass before completion

---

*End of plan*