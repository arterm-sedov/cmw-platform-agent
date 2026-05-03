# UI Components

Datasets, Toolbars, and Buttons are **separate API entities** with different endpoints:

| Entity | Tool to Get | Tool to Edit |
|--------|-------------|--------------|
| Dataset | `get_dataset` | `edit_or_create_dataset` |
| Toolbar | `get_toolbar` | `edit_or_create_toolbar` |
| Button | `get_button` | `edit_or_create_button` |

## List and Edit Toolbars

```python
from tools.templates_tools.tools_toolbar import list_toolbars, get_toolbar, edit_or_create_toolbar

toolbars = list_toolbars.invoke({
    "application_system_name": "<app>",
    "template_system_name": "<template>"
})
for tb in toolbars["data"]:
    print(f"{tb['globalAlias']['alias']}: {tb['name']}")

toolbar = get_toolbar.invoke({
    "application_system_name": "<app>",
    "template_system_name": "<template>",
    "toolbar_system_name": "<toolbar>"
})
for item in toolbar.get("items", []):
    print(f"  - {item['name']} ({item['action']['alias']})")

edit_or_create_toolbar.invoke({
    "operation": "edit",
    "application_system_name": "<app>",
    "template_system_name": "<template>",
    "toolbar_system_name": "<toolbar>",
    "name": "<New Name>",
    "items": [
        {"button_system_name": "create", "display_name": "<Label>", "item_order": 0},
    ]
})
```

**⚠️ Dataset-Specific Toolbars:** If a dataset shares a toolbar with other datasets, editing that toolbar affects ALL linked datasets. Create a NEW toolbar for dataset-specific buttons.

→ See also: [workflow_sequences.md](workflow_sequences.md#8-dataset-specific-toolbars-3-step-workflow)

## List and Edit Buttons

```python
from tools.templates_tools.tools_button import list_buttons, edit_or_create_button

buttons = list_buttons.invoke({
    "application_system_name": "<app>",
    "template_system_name": "<template>"
})

edit_or_create_button.invoke({
    "operation": "edit",
    "application_system_name": "<app>",
    "template_system_name": "<template>",
    "button_system_name": "<button>",
    "name": "<Name>",
    "description": "<Description>"
})
```

**⚠️ Toolbar Item Names Override Button Names:** Toolbar items have their own `name` field that overrides the button's display name.

→ See also: [workflow_sequences.md](workflow_sequences.md#9-toolbar-item-names-override-button-names)

## Create-Kind Buttons

For buttons with `kind='Create'`, you **MUST** specify `create_form` (and optionally `create_template`):

```python
edit_or_create_button.invoke({
    "operation": "edit",
    "application_system_name": "<app>",
    "template_system_name": "<template>",
    "button_system_name": "<button>",
    "name": "Create New Record",
    "kind": "Create",
    "context": "Record",  # or "List" depending on where button appears
    "create_form": "defaultForm",  # REQUIRED for Create buttons
    "create_template": "<target_template>",  # Optional, defaults to current template
})
```

**Key requirements:**
- `create_form` is **mandatory** for `kind='Create'` buttons
- `context` must match where the button appears (`"Record"` for record forms, `"List"` for list toolbars)
- `create_template` defaults to the current template if omitted
- The API builds `relatedAction.formGlobalAlias` from these parameters

**⚠️ UI Cache Issue:** The platform UI may display stale context values (e.g., `#List` instead of `Record`) even after successful API updates. Always verify via `get_button` API call, not the UI. Hard refresh (Ctrl+F5) or clearing browser cache may be required to see correct values in the UI.

## Button Kinds (Action Types)

CMW Platform supports 29 button action types. The tool accepts LLM-friendly terms and maps them to API values.

### Common Button Kinds

| LLM Term | API Term | Description | Russian (RU) |
|----------|----------|-------------|--------------|
| Trigger scenario | UserEvent | Execute custom scenario (default) | Вызвать событие «Нажата кнопка» |
| Create | Create | Create new record (requires `create_form`) | Создать |
| Edit | Edit | Edit existing record | Редактировать |
| Delete | Delete | Delete record | Удалить |
| Archive | Archive | Archive record | Архивировать |
| Unarchive | Unarchive | Restore archived record | Разархивировать |
| Script | Script | Execute C# script | С# скрипт |

### All 29 Valid Button Kinds

| # | API Term | English | Russian (RU) |
|---|----------|---------|--------------|
| 1 | Undefined | No specific action | - |
| 2 | Create | Create new record (requires `create_form`) | Создать |
| 3 | Edit | Edit existing record | Редактировать |
| 4 | Delete | Delete record | Удалить |
| 5 | Archive | Archive record | Архивировать |
| 6 | Unarchive | Restore archived record | Разархивировать |
| 7 | ExportObject | Export single object | Экспорт записи |
| 8 | ExportList | Export list of objects | Экспорт таблицы |
| 9 | CreateRelated | Create related record | Создать связанную запись |
| 10 | CreateToken | Create token | Создать токен |
| 11 | RetryTokens | Retry tokens | Перезапустить токены |
| 12 | Migrate | Migrate data | Мигрировать |
| 13 | StartCase | Start case | - |
| 14 | StartLinkedCase | Start linked case | - |
| 15 | StartProcess | Start process | Запустить процесс |
| 16 | StartLinkedProcess | Start linked process | Запустить процесс по связанному шаблону |
| 17 | CompleteTask | Complete task | Завершить задачу |
| 18 | ReassignTask | Reassign task | Переназначить |
| 19 | Defer | Defer action | Отложить выполнение |
| 20 | Accept | Accept action | Принять |
| 21 | Uncomplete | Mark as incomplete | Открыть заново |
| 22 | Follow | Follow record | Привязать к шаблону |
| 23 | Unfollow | Unfollow record | Отвязать от шаблону |
| 24 | Exclude | Exclude from list | - |
| 25 | Include | Include in list | - |
| 26 | Script | Execute C# script | С# скрипт |
| 27 | Cancel | Cancel action | Остановить процесс |
| 28 | EditDiagram | Edit diagram | - |
| 29 | UserEvent | Execute custom scenario (use "Trigger scenario") | Вызвать событие «Нажата кнопка» |

### Usage Examples

```python
# Default: Trigger custom scenario
edit_or_create_button.invoke({
    "operation": "create",
    "application_system_name": "FacilityManagement",
    "template_system_name": "MaintenancePlans",
    "button_system_name": "run_maintenance_check",
    "name": "Run Maintenance Check",
    "kind": "Trigger scenario",  # Maps to UserEvent in API
})

# Explicit API term also works
edit_or_create_button.invoke({
    "operation": "create",
    "kind": "UserEvent",  # Direct API term
})

# Other common kinds
edit_or_create_button.invoke({
    "operation": "create",
    "kind": "Archive",  # Archive button
})

# Case-insensitive variants work
edit_or_create_button.invoke({
    "operation": "create",
    "kind": "trigger_scenario",  # Snake_case → UserEvent
})
```

### Validation
- The validator is case-insensitive and handles variants like `trigger_scenario`, `TRIGGER SCENARIO`
- Invalid kinds (e.g., `Test`) are rejected with clear error messages
- All 29 API enum values are validated

### Edit Behavior
- `kind` parameter is optional in edit operations
- If omitted, the existing kind is preserved
- If provided, the kind is updated (even if changing to UserEvent)
- Always verify changes via `get_button` API, not just the UI (UI may cache stale values)

## List and Edit Datasets

```python
from tools.templates_tools.tools_dataset import list_datasets, get_dataset, edit_or_create_dataset

datasets = list_datasets.invoke({
    "application_system_name": "<app>",
    "template_system_name": "<template>"
})

edit_or_create_dataset.invoke({
    "operation": "edit",
    "application_system_name": "<app>",
    "template_system_name": "<template>",
    "dataset_system_name": "<dataset>",
    "name": "<New Name>",
    "toolbar_system_name": "<toolbar>",
    "columns": {
        "<column>": {"Name": "<New Label>"},
        "<columnToHide>": {"isHidden": True},
    }
})
```

### Toolbar-Dataset Linking
- Datasets link to toolbars via the `toolbar_system_name` parameter
- Toolbars link back to datasets via `IsDefaultForLists` flag (set on the toolbar)
- When a toolbar has `IsDefaultForLists: true`, it becomes the default for all datasets in that template
- To make a toolbar dataset-specific, set `IsDefaultForLists: false` and link it explicitly via the dataset's `toolbar_system_name`

## Edit Form Widgets

```python
from tools.templates_tools.tools_form import get_form, edit_or_create_form

form = get_form.invoke({
    "application_system_name": "<app>",
    "template_system_name": "<template>",
    "form_system_name": "<form>"
})

edit_or_create_form.invoke({
    "operation": "edit",
    "application_system_name": "<app>",
    "template_system_name": "<template>",
    "form_system_name": "<form>",
    "widgets": [
        {"system_name": "<widget>", "label": "<New Label>"},
    ]
})
```

---

→ See also: [tool_inventory.md](tool_inventory.md)
→ See also: [workflow_sequences.md](workflow_sequences.md)
