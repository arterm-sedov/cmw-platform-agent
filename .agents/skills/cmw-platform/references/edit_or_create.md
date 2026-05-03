# edit_or_create Tools Reference

The `edit_or_create_*` attribute tools have **smart partial update support**.

## Behavior Matrix

| Operation | Behavior | Mechanism |
|-----------|----------|-----------|
| **Create** | Requires ALL type-specific fields | Model validator raises error if missing |
| **Edit - partial** | Missing fields fetched from API and merged | `tool_utils.py` patch fills gaps |
| **Edit - explicit** | Provided fields override existing values | User intent respected |

## How the Patch Works

When editing with missing fields:

```python
# Agent calls edit with only name
edit_or_create_numeric_attribute.invoke({
    "operation": "edit",
    "name": "New Name",
    "system_name": "Ploschad",
    "application_system_name": "Volga",
    "template_system_name": "RentLots"
    # number_decimal_places NOT provided
})
```

**What happens internally:**

1. `remove_values()` strips `None` fields → `number_decimal_places` not in body
2. `tool_utils.py` fetches current `Ploschad` schema from API
3. Patch merges: adds `decimalPlaces: 2` from current
4. API receives complete schema with `decimalPlaces: 2` preserved ✅

## Edit with Explicit Values

```python
# Agent provides value - THIS WILL OVERRIDE
edit_or_create_numeric_attribute.invoke({
    "operation": "edit",
    "name": "New Name",
    "system_name": "Ploschad",
    "number_decimal_places": 3  # ← Explicit value overrides existing
})
```

**Result:** API receives `decimalPlaces: 3` — existing value is overridden.

## Required Fields for Create

| Attribute Type | Required Fields |
|----------------|----------------|
| Decimal | `number_decimal_places` |
| Enum | `display_format`, `enum_values` |
| DateTime | `display_format` |
| Document | `display_format` |
| Image | `rendering_color_mode` |
| Duration | `display_format` |
| Account | `related_template_system_name` |
| Record | `related_template_system_name` |

**Note:** Text/String attributes require **no type-specific fields** — they work with defaults. Only provide `display_format` (e.g., `PlainText`) when you need a specific format.

## Edit Tool Validation Pattern

All `edit_or_create` tools (datasets, buttons, toolbars, forms, attributes) follow the same validation pattern:

| Operation | Required Fields | Optional Fields |
|-----------|----------------|-----------------|
| **Create** | `name` + identifiers | All editable fields |
| **Edit** | identifiers only | All editable fields (partial update) |

### Identifiers (Always Required)

- `operation`: `"create"` or `"edit"`
- `application_system_name`: App system name
- `template_system_name`: Template system name
- `{entity}_system_name`: The specific entity (`button_system_name`, `toolbar_system_name`, etc.)

### Editable Fields per Tool

**Dataset (`edit_or_create_dataset`):**
- `name` — display name
- `view_type` — `Undefined`, `General`, `SplitVertical`, `SplitHorizontal`
- `is_default` — set as default dataset
- `show_disabled` — show disabled records
- `toolbar_system_name` — link toolbar
- `columns` — add/remove/rename columns
- `sorting`, `grouping`, `totals` — view configuration

**Button (`edit_or_create_button`):**
- `name` — display name
- `description` — button description
- `kind` — `UserEvent`, `Create`, `Edit`, `Delete`, `Archive`, `Unarchive`, `Test`
- `context` — `Record`, `List`
- `multiplicity` — `OneByOne`, `Many`
- `result_type` — `DataChange`, `Redirect`
- `has_confirmation` — show confirmation dialog
- `navigation_target` — `SameForm`, `NewForm`, `Undefined`

**Toolbar (`edit_or_create_toolbar`):**
- `name` — display name
- `is_default_for_forms` — default for forms
- `is_default_for_lists` — default for lists
- `is_default_for_task_lists` — default for task lists
- `items` — add/remove toolbar items (buttons)

## ⚠️ WARNING: System Buttons

Buttons with system names `create`, `edit`, `archive`, `delete`, `unarchive` are **platform defaults**. Only modify `name` and `description` for these — other changes may cause unexpected behavior.

## How Partial Updates Work

1. Edit call provides only the fields to change.
2. Tool fetches current entity state from API.
3. Tool merges current state with provided fields.
4. Missing fields are preserved from fetched state.

```python
# Example: Edit only dataset name — all other fields preserved
edit_or_create_dataset.invoke({
    "operation": "edit",
    "application_system_name": "supportTest",
    "template_system_name": "LegalEntity",
    "dataset_system_name": "testDataset",
    "name": "New Dataset Name"
    # Other fields omitted - will be fetched and preserved
})
```
