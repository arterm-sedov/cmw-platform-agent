# Localization

This skill supports **two distinct localization workflows**. Choose the right one before starting.

| Goal | Workflow | Key tools |
|------|----------|-----------|
| Rename system names / aliases safely | **Workflow A — Alias Rename** | `export_application`, `get_ontology_objects`, `update_object_property`, `tool_localize` |
| Translate Russian UI strings to English | **Workflow B — UI Text Translation** | `harvest_strings.py`, `build_translations.py`, `apply_translations.py`, `update_csv.py` |

⚠️ **These are different operations.** Alias rename changes internal identifiers; UI text translation changes user-facing labels. Never mix them.

---

## Workflow A — System-Name (Alias) Rename

Rename platform system names (aliases) safely across a full application.
**Multi-step interactive process** — show tables and wait for user confirmation at Phases 4, 5, 6, and 9.

### Phase Overview

| Phase | Action | Show Table | User Confirm |
|-------|--------|------------|--------------|
| 1 | Export CTF (if not provided) | — | — |
| 2 | Collect aliases from JSON, tag by object type | ✅ | — |
| 3 | Verify IDs via `get_ontology_objects` | ✅ | — |
| 4 | Analyze Expression fields, assign suffixes | ✅ | **Yes** |
| 5 | Rename via `update_object_property` | ✅ | **Yes** |
| 6 | Request platform restart | — | **Yes** |
| 7 | Re-export CTF | — | — |
| 8 | Replace dangerous aliases in JSON | ✅ | — |
| 9 | Import modified CTF | — | **Yes** |

**Table columns:** `type`, `systemName`, `jsonPath`, `id`, `renamedSystemName`

**Suffix rules:**
- `dangerous` — alias appears in Expression fields → suffix `_calc`, rename everywhere (platform + JSON)
- `safe` — alias only in `GlobalAlias.Alias` context → suffix `_sv`, rename in platform only

### Phase 1: Export CTF

```python
from tools.transfer_tools.tool_export_application import export_application

result = export_application.invoke({
    "application_system_name": "my_app",
    "save_to_file": True,
})
if result["success"]:
    ctf_path = result["ctf_file_path"]
```

### Phase 2: Collect Aliases from JSON

Traverse the extracted CTF folder. For each JSON file, extract `"Alias"` values and tag with the object type inferred from the parent folder:

```
RecordTemplates/  → RecordTemplate     Datasets/      → Dataset
ProcessTemplates/ → ProcessTemplate    Toolbars/      → Toolbar
Attributes/       → Attribute          Forms/         → Form
Workspaces/       → Workspace          UserCommands/  → UserCommand
Pages/            → Page               Roles/         → RoleTemplate
Accounts/         → AccountTemplate
```

Build: `{alias: {"type": obj_type, "json_path": path}}`. Show table to user.

### Phase 3: Verify Aliases via get_ontology_objects

```python
from tools.applications_tools.tool_get_ontology_objects import get_ontology_objects

result = get_ontology_objects.invoke({
    "application_system_name": "my_app",
    "types": ["RecordTemplate", "ProcessTemplate", "Dataset", "Form",
              "Toolbar", "UserCommand", "Attribute", "Workspace", "Page"],
    "parameter": "alias",
    "min_count": 1,
    "max_count": 10000,
})
```

Only include aliases found in platform. Build verified dict: `{alias: object_id}`. Show updated table.

### Phase 4: Analyze Expression Fields

```python
import re
from pathlib import Path

EXPRESSION_KEYS = {"Expression", "Code", "ValueExpression", "ValidationScript"}

def check_dangerous_aliases(json_folder: str, aliases: set[str]) -> dict[str, bool]:
    dangerous = {a: False for a in aliases}
    for json_file in Path(json_folder).rglob("*.json"):
        content = open(json_file).read()
        for alias in aliases:
            if dangerous[alias]:
                continue
            for key in EXPRESSION_KEYS:
                if re.search(rf'"{key}"\s*:\s*"[^"]*{re.escape(alias)}[^"]*"', content):
                    dangerous[alias] = True
                    break
    return dangerous
```

Show complete table with assigned suffixes. **Ask user to confirm before proceeding.**

### Phase 5: Apply Renames via update_object_property

```python
from tools.applications_tools.tool_update_object_property import update_object_property

for alias, new_alias in rename_map.items():
    result = update_object_property.invoke({
        "object_id": verified_aliases[alias],
        "object_type": TYPE_MAPPING[alias_type],
        "new_value": new_alias,
    })
```

Aliases must be without spaces — use CamelCase or underscores. Show results table. **Ask user to confirm before restart.**

### Phase 6: Request Platform Restart

> "System names have been renamed. Please restart the Comindware Platform service. Confirm when ready."

**Wait for user confirmation.**

### Phase 7: Re-Export CTF

```python
result = export_application.invoke({
    "application_system_name": "my_app",
    "save_to_file": True,
})
```

### Phase 8: Replace Dangerous Aliases in JSON

```python
for alias, new_alias in dangerous_renames.items():
    safe_pattern = re.escape(alias)
    for json_file in Path(json_folder).rglob("*.json"):
        content = open(json_file).read()
        content = re.sub(r'"Alias"\s*:\s*"' + safe_pattern + r'"',
                         '"Alias": "' + new_alias + '"', content)
        for key in EXPRESSION_KEYS:
            content = re.sub(rf'"{key}"\s*:\s*"[^"]*{safe_pattern}[^"]*"',
                             lambda m: m.group(0).replace(alias, new_alias), content)
        open(json_file, "w").write(content)
```

Only dangerous aliases are replaced in JSON. Safe aliases are renamed in platform only (Phase 5). Show updated table.

### Phase 9: Import Modified CTF

```python
from tools.transfer_tools.tool_import_application import import_application

result = import_application.invoke({
    "application_system_name": "my_app",
    "ctf_file_path": "/path/to/modified.ctf",
    "update_existing": True,
})
```

Save the final table to workspace files before importing. **Ask user to confirm before importing.**

```python
import json
from pathlib import Path

output_dir = Path("cmw-platform-workspace")
(output_dir / "localization_table.json").write_text(json.dumps(table_data, indent=2))
md_lines = ["| type | systemName | jsonPath | id | renamedSystemName |",
            "|------|------------|----------|----|-------------------|"]
for row in table_data:
    md_lines.append(f"| {row['type']} | {row['systemName']} | {row['jsonPath']} | {row['id']} | {row['renamedSystemName']} |")
(output_dir / "localization_table.md").write_text("\n".join(md_lines))
```

### Type-Folder and Predicate Mapping

```python
TYPE_FOLDER_MAPPING = {
    "RecordTemplate": "RecordTemplates",    "ProcessTemplate": "ProcessTemplates",
    "RoleTemplate": "Roles",               "AccountTemplate": "Accounts",
    "OrgStructureTemplate": "OrgStructure", "MessageTemplate": "MessageTemplates",
    "Workspace": "Workspaces",             "Page": "Pages",
    "Attribute": "Attributes",             "Dataset": "Datasets",
    "Toolbar": "Toolbars",                 "Form": "Forms",
    "UserCommand": "UserCommands",         "Card": "Cards",
    "Cart": "Carts",                       "Trigger": "Triggers",
    "Role": "Roles",                       "WidgetConfig": "WidgetConfigs",
}

TYPE_PREDICATE_MAPPING = {
    "RecordTemplate": "cmw.container.alias",   "ProcessTemplate": "cmw.container.alias",
    "RoleTemplate": "cmw.container.alias",     "AccountTemplate": "cmw.container.alias",
    "OrgStructureTemplate": "cmw.container.alias", "MessageTemplate": "cmw.message.type.alias",
    "Workspace": "cmw.alias",                  "Page": "cmw.desktopPage.alias",
    "Attribute": "cmw.object.alias",           "Dataset": "cmw.alias",
    "Toolbar": "cmw.alias",                    "Form": "cmw.alias",
    "UserCommand": "cmw.alias",                "Card": "cmw.alias",
    "Cart": "cmw.cart.alias",                  "Trigger": "cmw.trigger.alias",
    "Role": "cmw.role.alias",                  "WidgetConfig": "cmw.form.alias",
}
```

**Role objects:** support `cmw.role.alias` (direct) and `cmw.role.aliasProperty` (indirect → contains attribute ID, resolve via `GetAxiomsByPredicate`). ID prefix: `role.`

**WidgetConfig objects:** ID prefix `fw.`

### tool_localize — Automated Workflow Tool

Function: `localize_aliases` (`tools.localization_tools.tool_localize`)

**Capabilities:** Orchestrates the CTF alias pipeline (extract → platform cache → verify → dangerous scan → finalize → schema JSON), optional `_tr` translation file creation, interactive per-alias edits, `apply_renames` on platform, and expression fixes — matching the implementation in `tool_localize.py`.

**Parameters** (see `LocalizeSchema` in-repo):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `application_system_name` | str | required | Application system name |
| `json_folder` | str | required | Path to the folder whose **parent** is used as `--extract-dir` for scripts (typically the inner JSON root inside the CTF tree) |
| `output_dir` | str \| `None` | `None` | Output directory; defaults to `json_folder` when omitted |
| `dry_run` | bool | `True` | If **`True`**, the scripted pipeline (steps 1–6: extract through schema export) **does not run**. If **`False`**, runs `tool_extract_aliases` → `tool_collect_platform` → per-folder verify → `tool_find_dangerous` → `tool_finalize` → schema file write |
| `create_tr` | bool | `False` | Build `{domain}_{app}_aliases_tr.json` from existing aliases output (requires prior artifacts on disk) |
| `translate_one` | str \| `None` | `None` | Interactive edit for a single `aliasOriginal` in the `_tr` file |
| `resume` | bool | `False` | Resume interactive translation from saved state |
| `apply_renames` | bool | `False` | Run `apply_renames.py` to push renames to the platform |
| `fix_expressions` | bool | `False` | Rewrite `_calc` expression strings inside the `_tr` JSON on disk (also runs inside `create_tr` when combined with that flag) |
| `dangerous_suffix` | str | `"_calc"` | Suffix for dangerous (expression-backed) aliases |
| `safe_suffix` | str | `"_sv"` | Suffix for safe aliases |

```python
from tools.localization_tools.tool_localize import localize_aliases

# Run the full extract→finalize pipeline (writes under output_dir)
result = localize_aliases.invoke({
    "application_system_name": "MyApp",
    "json_folder": "/path/to/ctf_json_root",
    "output_dir": "/path/to/workdir",
    "dry_run": False,
})

# After schema / verified files exist: create _tr copy (pipeline skipped if dry_run True)
result = localize_aliases.invoke({
    "application_system_name": "MyApp",
    "json_folder": "/path/to/ctf_json_root",
    "output_dir": "/path/to/workdir",
    "create_tr": True,
    "dry_run": True,
})
```

**Return structure** (always present):

```python
{
    "success": bool,
    "phase": str,       # e.g. "started", "create_tr", "translate", "apply_renames", "fix_expressions", "complete", "error"
    "actions": list,    # human-readable log lines
    "errors": list,     # error strings when success is False
}
```

**Optional keys** after a successful `dry_run=False` pipeline (when verification file existed and finalize produced entries):

- `output_file` — path to `{domain}_{app}_aliases.json`
- `total_entries` — count of entries written
- `locked_entries` — count with `aliasLocked`
- `dangerous_count` — length of dangerous-alias list when `*_dangerous_aliases.json` exists

**Important:**

- Default call (`dry_run=True`, other flags off) **does not** run extraction; you get `phase: "complete"` and empty `actions` unless you set another mode (`create_tr`, `translate_one`, etc.) that uses existing files.
- Manual Phases 1–9 above remain the conceptual workflow; `localize_aliases` wraps the same scripts under `dry_run=False` and optional later stages.
- Display-name–vs–alias semantics for CTF import vs API still follow Workflow A (Phases 5 vs 9) in this document.

### Step Scripts — Large Application Workflow

For apps with 5000+ CTF JSON files, use resumable batched scripts:

| Step | Script | Purpose | Output |
|------|--------|---------|--------|
| 1 | `tool_extract_aliases.py` | Extract aliases per folder | `{app}_{folder}_aliases.json` |
| 2 | `tool_collect_platform.py` | Query platform types (parallel) | `{app}_platform_cache.json` |
| 3 | `tool_verify_aliases.py` | Verify aliases per folder | `{app}_{folder}_verified.json` |
| 4 | `tool_find_dangerous.py` | Scan expression patterns | `{app}_dangerous_aliases.json` |
| 5 | `tool_finalize.py` | Merge + set aliasLocked | `{app}_verified_complete.json` |

```bash
source .venv/bin/activate

# Run all steps in one command
python .agents/skills/cmw-platform/scripts/tool_analyze_all.py \
    --app Volga --extract-dir /tmp/cmw-transfer/Volga --output-dir /tmp/cmw-transfer/Volga_tr

# Or individually
python .agents/skills/cmw-platform/scripts/tool_extract_aliases.py \
    --app Volga --extract-dir /tmp/cmw-transfer/Volga --output-dir /tmp/cmw-transfer/Volga_tr
python .agents/skills/cmw-platform/scripts/tool_collect_platform.py \
    --app Volga --output-dir /tmp/cmw-transfer/Volga_tr
python .agents/skills/cmw-platform/scripts/tool_verify_aliases.py \
    --app Volga --folder RecordTemplates --output-dir /tmp/cmw-transfer/Volga_tr
python .agents/skills/cmw-platform/scripts/tool_find_dangerous.py \
    --app Volga --extract-dir /tmp/cmw-transfer/Volga --output-dir /tmp/cmw-transfer/Volga_tr
python .agents/skills/cmw-platform/scripts/tool_finalize.py \
    --app Volga --output-dir /tmp/cmw-transfer/Volga_tr
```

**Expression patterns detected:** `${alias}`, `->{alias}`, `{alias}->`, `"alias"`

**aliasLocked:** `true` = has displayName, NOT dangerous → skip; `false` = will be renamed.

**Final JSON schema per alias entry:**

```json
{
  "type": "RecordTemplate",
  "ids": ["container.42"],
  "parent_template": "Sotrudniki",
  "aliasOriginal": "SomeAlias",
  "aliasRenamed": "",
  "displayNameOriginal": "Some Display Name",
  "displayNameRenamed": "",
  "jsonPathOriginal": ["Volga/RecordTemplates/Sotrudniki/SomeAlias.json"],
  "jsonPathRenamed": [],
  "expressions": [
    {
      "jsonPathOriginal": "Volga/RecordTemplates/Sotrudniki/Attributes/Count.json#Expression",
      "jsonPathRenamed": "",
      "expressionOriginal": "COUNT(from a in $SomeAlias where ...)",
      "expressionRenamed": ""
    }
  ],
  "aliasLocked": false
}
```

**State files:** `{app}_extraction_state.json` (folder progress), `{app}_master_state.json` (overall progress for resume).

---

## Workflow B — RU→EN UI Text Translation

Translate user-facing Russian strings to English without touching system names or aliases.

**Core rule:** ALWAYS update the localization CSV with any new terms discovered during translation.

### What to Translate

| Item Type | Examples | JSON Fields |
|-----------|----------|-------------|
| Template Names | "Здания" → "Buildings" | `"Name"` |
| Attribute Names | "Название" → "Name" | `"Name"` |
| Form Labels | "Основная форма" → "Main Form" | `"Text"`, `"DisplayName"` |
| Dataset Columns | "Статус" → "Status" | `"Name"` |
| Buttons | "Создать" → "Create" | `"Name"` |
| Toolbar Names | "Область кнопок" → "Toolbar" | `"Name"` |
| Enum Variants | "Свободно" → "Vacant" | `"Ru"`/`"En"` in LocalizedTextModel |
| Workspace Navigation | "Рабочий стол" → "Dashboard" | `"Name"` |
| Widget Display Names | "Новости" → "News" | `"Name"` |
| Role Names | "Руководитель" → "Manager" | `"Name"` |
| Route Names | "пуш" → "push" | `"Name"`, `"MessageName"` |

### What NOT to Translate

- System aliases (`Zdaniya`, `Pomescheniya`, etc.)
- JSON paths and property references
- Expression logic in calculated attributes
- GUIDs/UUIDs and internal IDs
- Technical field names (`Alias`, `GlobalAlias`, etc.)
- Mathematical or code expressions

### Step 1: Harvest Strings

```bash
python .agents/skills/cmw-platform/scripts/harvest_strings.py \
    "path/to/Workspaces" --output harvested.json
```

### Step 2: Build Translation Dictionary

```bash
python .agents/skills/cmw-platform/scripts/build_translations.py \
    harvested.json --output translations.json
```

Edit `translations.json` manually or use LLM to translate.

### Step 3: Apply Translations

```bash
python .agents/skills/cmw-platform/scripts/apply_translations.py \
    "path/to/Workspaces" translations.json
```

### Step 4: Update CSV Reference

```bash
python .agents/skills/cmw-platform/scripts/update_csv.py \
    translations.json translations.csv
```

### CTF File Structure Reference

```
Application/
├── RecordTemplates/          # Business entities
│   └── TemplateName/         # One folder per template
│       ├── Attributes/       # Field definitions
│       ├── Forms/            # UI forms
│       ├── Datasets/         # List views
│       ├── UserCommands/     # Actions (Create, Edit, etc.)
│       └── Toolbars/         # Button bars
├── WidgetConfigs/            # Dashboard widgets
├── Workspaces/               # Role workspaces
├── Roles/                    # Role configurations
├── Routes/                   # Communication routes
└── Pages/                    # Pages
```

### CSV Format

```
исходное название (RU);Системное имя (RU);Английское название (EN);Системное имя (EN);Исходный JSON-Path
```
