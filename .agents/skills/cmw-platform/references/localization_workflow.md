# Localization Workflow — System Name (Alias) Rename

Full 9-phase workflow for renaming system names (aliases) in Comindware Platform applications.
This is a **multi-step interactive process** — the agent must show tables and wait for user confirmation at Phases 4, 5, 6, and 9.

## Phase Overview

| Phase | Action | Show Table | User Confirm |
|-------|--------|------------|--------------|
| 1 | Export CTF (if not provided) | — | — |
| 2 | Collect aliases from JSON, tag by object type | **Yes** | — |
| 3 | Verify IDs via get_ontology_objects | **Yes** | — |
| 4 | Analyze Expression fields, suggest suffixes | **Yes** | **Yes** |
| 5 | Rename via update_object_property | **Yes** | **Yes** |
| 6 | Ask user to restart platform | — | **Yes** |
| 7 | Re-export CTF | — | — |
| 8 | Replace dangerous aliases in JSON | **Yes** | — |
| 9 | Import modified CTF | — | **Yes** |

**Table columns:** `type`, `systemName`, `jsonPath`, `id`, `renamedSystemName`

**Suffix rules:** `dangerous` aliases used in expressions usually get `_calc`; `safe` aliases used only in alias fields usually get `_sv`.

## Key Tools

| Purpose | Tool / Reference |
|---------|------------------|
| Export / re-export CTF | `export_application` |
| Verify aliases in platform | `get_ontology_objects` |
| Apply alias rename in platform | `update_object_property` |
| RU→EN UI text translation workflow | [localization.md](localization.md) |
| Large-app batched scripts | `.agents/skills/cmw-platform/scripts/tool_*.py` |

---

## Phase 1: Export CTF

```python
from tools.transfer_tools.tool_export_application import export_application

result = export_application.invoke({
    "application_system_name": "my_app",
    "save_to_file": True,
})
if result["success"]:
    ctf_path = result["ctf_file_path"]
```

---

## Phase 2: Collect Aliases from JSON

Traverse the exported JSON folder. For each file, extract `"Alias"` values and tag with the **object type inferred from the parent folder name:**

```
RecordTemplates/   → RecordTemplate
ProcessTemplates/  → ProcessTemplate
Datasets/          → Dataset
Forms/             → Form
Toolbars/          → Toolbar
UserCommands/      → UserCommand
Attributes/        → Attribute
Workspaces/        → Workspace
Pages/             → Page
Roles/             → RoleTemplate
Accounts/          → AccountTemplate
```

Build a lookup dict: `{alias: {"type": obj_type, "json_path": path}}`.

**Show table to user:**

| type | systemName | jsonPath | id | renamedSystemName |
|------|------------|----------|----|-------------------|
| Form | MyForm | Forms/MyForm.json#$.GlobalAlias.Alias | — | — |
| RecordTemplate | MyRecord | RecordTemplates/MyRecord.json#$.GlobalAlias.Alias | — | — |

---

## Phase 3: Verify Aliases via get_ontology_objects

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

Compare results with Phase 2 aliases. **Only include aliases found in platform** — silently exclude unmatched. Build verified dict: `{alias: object_id}`.

**Show table to user (id column now filled).**

---

## Phase 4: Analyze Expression Fields

Scan all JSON files for `"Expression"` fields containing alias references **outside of `"Alias"` context**.

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
                pattern = rf'"{key}"\s*:\s*"[^"]*{re.escape(alias)}[^"]*"'
                if re.search(pattern, content):
                    dangerous[alias] = True
                    break
    return dangerous
```

Assign suffixes:

| Category | Suffix | Condition |
|----------|--------|-----------|
| **Dangerous** | `_calc` | Mentioned in Expression — rename everywhere |
| **Safe** | `_sv` | Only in `GlobalAlias.Alias` context |

**Show complete table to user and ask to confirm the rename plan before proceeding.**

---

## Phase 5: Apply Renames via update_object_property

```python
from tools.applications_tools.tool_update_object_property import update_object_property

for alias, new_alias in rename_map.items():
    object_id = verified_aliases[alias]
    object_type = TYPE_MAPPING[alias_type]

    result = update_object_property.invoke({
        "object_id": object_id,
        "object_type": object_type,
        "new_value": new_alias,
    })
```

**Important:** Alias values must be without spaces — use CamelCase or underscores (e.g. `myAlias_calc`).

**Show rename results table and ask user to confirm** before proceeding to restart.

---

## Phase 6: Request Platform Restart

Inform the user:

> "System names have been renamed. Please restart the Comindware Platform service now. Once restarted, confirm to proceed with re-export."

**Wait for user confirmation** that restart is complete.

---

## Phase 7: Re-Export CTF

```python
result = export_application.invoke({
    "application_system_name": "my_app",
    "save_to_file": True,
})
```

---

## Phase 8: Replace Dangerous Aliases in JSON

In the newly exported JSON files, replace all occurrences of **dangerous** aliases (in both `Alias` and `Expression` fields) with their new suffixed names:

```python
for alias, new_alias in dangerous_renames.items():
    safe_pattern = re.escape(alias)
    for json_file in Path(json_folder).rglob("*.json"):
        content = open(json_file).read()
        content = re.sub(
            r'"Alias"\s*:\s*"' + safe_pattern + r'"',
            '"Alias": "' + new_alias + '"',
            content
        )
        for key in EXPRESSION_KEYS:
            content = re.sub(
                rf'"{key}"\s*:\s*"[^"]*{safe_pattern}[^"]*"',
                lambda m: m.group(0).replace(alias, new_alias),
                content
            )
        open(json_file, "w").write(content)
```

**Note:** Only dangerous aliases are replaced in JSON. Safe aliases are renamed only in the platform (via Phase 5).

**Show table to user** with updated jsonPaths.

---

## Phase 9: Import Modified CTF (Update Existing)

```python
from tools.transfer_tools.tool_import_application import import_application

result = import_application.invoke({
    "application_system_name": "my_app",
    "ctf_file_path": "/path/to/modified_ctf.ctf",
    "update_existing": True,
})
```

**Use `update_existing: True`** to update the existing application, not create a new one.

Save the final table to workspace files:

```python
import json
from pathlib import Path

table_data = [...]  # Final rename table rows

output_dir = Path("cmw-platform-workspace")
(output_dir / "localization_table.json").write_text(json.dumps(table_data, indent=2))

md_lines = ["| type | systemName | jsonPath | id | renamedSystemName |",
            "|------|------------|----------|----|-------------------|"]
for row in table_data:
    md_lines.append(
        f"| {row['type']} | {row['systemName']} | {row['jsonPath']} | {row['id']} | {row['renamedSystemName']} |"
    )
(output_dir / "localization_table.md").write_text("\n".join(md_lines))
```

**Show final table and ask user to confirm** before importing.

---

## Type-Folder and Predicate Mapping Reference

```python
TYPE_FOLDER_MAPPING = {
    "RecordTemplate": "RecordTemplates",
    "ProcessTemplate": "ProcessTemplates",
    "RoleTemplate": "Roles",
    "AccountTemplate": "Accounts",
    "OrgStructureTemplate": "OrgStructure",
    "MessageTemplate": "MessageTemplates",
    "Workspace": "Workspaces",
    "Page": "Pages",
    "Attribute": "Attributes",
    "Dataset": "Datasets",
    "Toolbar": "Toolbars",
    "Form": "Forms",
    "UserCommand": "UserCommands",
    "Card": "Cards",
    "Cart": "Carts",
    "Trigger": "Triggers",
    "Role": "Roles",
    "WidgetConfig": "WidgetConfigs",
}

TYPE_PREDICATE_MAPPING = {
    "RecordTemplate": "cmw.container.alias",
    "ProcessTemplate": "cmw.container.alias",
    "RoleTemplate": "cmw.container.alias",
    "AccountTemplate": "cmw.container.alias",
    "OrgStructureTemplate": "cmw.container.alias",
    "MessageTemplate": "cmw.message.type.alias",
    "Workspace": "cmw.alias",
    "Page": "cmw.desktopPage.alias",
    "Attribute": "cmw.object.alias",
    "Dataset": "cmw.alias",
    "Toolbar": "cmw.alias",
    "Form": "cmw.alias",
    "UserCommand": "cmw.alias",
    "Card": "cmw.alias",
    "Cart": "cmw.cart.alias",
    "Trigger": "cmw.trigger.alias",
    "Role": "cmw.role.alias",
    "WidgetConfig": "cmw.form.alias",
}
```

**Role objects:** support both `cmw.role.alias` (direct) and `cmw.role.aliasProperty` (indirect, contains attribute ID). Use `GetAxiomsByPredicate` to resolve. ID prefix: `role.`

**WidgetConfig objects:** ID prefix `fw.`

---

## tool_localize Reference

Function name: `localize_aliases`

**Capabilities:**
- Collect aliases (system names) from CTF JSON
- Collect display names (Name property) from CTF JSON
- Verify aliases via API (`GetWithMultipleValues`)
- Apply alias renames via API (`OntologyService/AddStatement`)
- Generate reports

**Important:**
- Both alias and displayName collection are optional and independent
- Aliases are applied via API (step 5 in workflow)
- DisplayNames are applied via CTF import (step 9 in workflow)
- DisplayNames are NOT verified via API (CTF-based workflow)

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `application_system_name` | str | required | App system name |
| `json_folder` | str | required | Path to extracted CTF JSON |
| `collect_aliases` | bool | `True` | Collect alias data |
| `collect_display_names` | bool | `True` | Collect Name property data |
| `dry_run` | bool | `True` | Preview without applying |
| `dangerous_suffix` | str | `"_calc"` | Suffix for dangerous aliases |
| `safe_suffix` | str | `"_sv"` | Suffix for safe aliases |

**Usage:**

```python
from tools.localization_tools.tool_localize import localize_aliases

# Collect both (default)
result = localize_aliases.invoke({
    "application_system_name": "MyApp",
    "json_folder": "/path/to/ctf",
    "collect_aliases": True,
    "collect_display_names": True,
    "dry_run": True
})

# Aliases only
result = localize_aliases.invoke({
    "application_system_name": "MyApp",
    "json_folder": "/path/to/ctf",
    "collect_aliases": True,
    "collect_display_names": False,
})

# Display names only
result = localize_aliases.invoke({
    "application_system_name": "MyApp",
    "json_folder": "/path/to/ctf",
    "collect_aliases": False,
    "collect_display_names": True,
})
```

**Return Structure:**

```python
{
    "success": bool,
    "aliases_collected": int,
    "display_names_collected": int,
    "aliases_verified": int,
    "aliases_missing": list,
    "dangerous_aliases": list,
    "safe_aliases": list,
    "collect_aliases": bool,
    "collect_display_names": bool,
    "errors": list
}
```

---

## Step Scripts — Large Application Workflow

For large applications (5000+ CTF JSON files), use the step scripts for resumable, batched processing.

**Location:** `.agents/skills/cmw-platform/scripts/tool_*.py`

**Folder Structure:**

```
/tmp/cmw-transfer/
  {app}/                     # CTF extracted content
    metadata.json
    {app}/                   # Actual app content
      RecordTemplates/
      ProcessTemplates/
      ...
  {app}.ctf                  # CTF file
  {app}_tr/                  # Output folder
```

**Workflow:**

| Step | Script | Purpose | Output |
|------|--------|---------|--------|
| 1 | `tool_extract_aliases.py` | Extract aliases per folder | `{app}_{folder}_aliases.json` |
| 2 | `tool_collect_platform.py` | Query platform types (parallel) | `{app}_platform_cache.json` |
| 3 | `tool_verify_aliases.py` | Verify aliases per folder | `{app}_{folder}_verified.json` |
| 4 | `tool_find_dangerous.py` | Scan for expression patterns | `{app}_dangerous_aliases.json` |
| 5 | `tool_finalize.py` | Merge and set aliasLocked | `{app}_verified_complete.json` |

**Usage:**

```bash
source .venv/bin/activate

# Run all steps
python .agents/skills/cmw-platform/scripts/tool_analyze_all.py \
    --app Volga \
    --extract-dir /tmp/cmw-transfer/Volga \
    --output-dir /tmp/cmw-transfer/Volga_tr

# Or run individual steps
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

**Expression Patterns Detected:**

- `${alias}` — variable reference
- `->{alias}` — method call on alias
- `{alias}->` — alias as method target
- `"alias"` — string-quoted alias

**aliasLocked Logic:**

- `true` — matches skip pattern, has displayName, NOT dangerous → safe to skip
- `false` — normal, OR matches skip pattern BUT dangerous → will be renamed

**Final JSON Schema:**

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

**State Files:**

- `{app}_extraction_state.json` — tracks completed folders
- `{app}_master_state.json` — tracks overall workflow progress (for resume)
