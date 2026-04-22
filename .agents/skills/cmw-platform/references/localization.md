# Comindware Platform Application Localization

Russian→English translation guide for Comindware Platform business applications.

## Overview

Localizing Comindware Platform applications involves translating UI strings from Russian (RU) to English (EN) while preserving system names, aliases, and technical identifiers.

## Core Rule

**ALWAYS update the localization CSV with any new terms discovered during translation.**

## What to Localize (UI Strings)

| Item Type | Examples | JSON Fields |
|-----------|----------|-------------|
| Template Names | "Здания" → "Buildings" | `"Name"` |
| Attribute Names | "Название" → "Name" | `"Name"` |
| Form Labels | "Основная форма" → "Main Form" | `"Text"`, `"DisplayName"` |
| Dataset Columns | "Статус" → "Status" | `"Name"` |
| UserCommands (Buttons) | "Создать" → "Create" | `"Name"` |
| Toolbar Names/Items | "Область кнопок" → "Toolbar" | `"Name"` |
| Enum Variants | "Свободно" → "Vacant" | `"Ru"`/`"En"` in LocalizedTextModel |
| Workspace Navigation | "Рабочий стол" → "Dashboard" | `"Name"` |
| Widget Display Names | "Новости" → "News" | `"Name"` |
| Role Names | "Руководитель" → "Manager" | `"Name"` |
| Route Names | "пуш" → "push" | `"Name"`, `"MessageName"` |

## What NOT to Localize

- System aliases (e.g., `Zdaniya`, `Pomescheniya`)
- JSON paths and property references
- Expression logic in calculated attributes
- GUIDs/UUIDs and internal IDs
- Technical field names (e.g., `Alias`, `GlobalAlias`)
- Mathematical or code expressions

## Workflow

### Step 1: Harvest Strings

Extract all Russian strings from JSON files (outputs JSON):

```bash
python .agents/skills/cmw-platform/scripts/harvest_strings.py \
    "path/to/Workspaces" --output harvested.json
```

### Step 2: Build Translation Dictionary

Create placeholder JSON for LLM to fill:

```bash
python .agents/skills/cmw-platform/scripts/build_translations.py \
    harvested.json --output translations.json
```

Then edit translations.json manually or use LLM to translate.

### Step 3: Apply Translations

Replace Russian strings with English:

```bash
python .agents/skills/cmw-platform/scripts/apply_translations.py \
    "path/to/Workspaces" translations.json
```

### Step 4: Update CSV Reference

Append new terms to localization CSV:

```bash
python .agents/skills/cmw-platform/scripts/update_csv.py \
    translations.json translations.csv
```

## File Structure Reference

```
Application/
├── RecordTemplates/          # Business entities
│   └── TemplateName/
│       ├── Attributes/         # Field definitions
│       ├── Forms/              # UI forms
│       ├── Datasets/           # List views
│       ├── UserCommands/      # Actions (Create, Edit, etc.)
│       └── Toolbars/          # Button bars
├── WidgetConfigs/              # Dashboard widgets
├── Workspaces/                # Role workspaces
├── Roles/                     # Role configurations
├── Routes/                    # Communication routes
└── Pages/                     # Pages
```

## CSV Format

```
исходное название (RU);Системное имя (RU);Английское название (EN);Системное имя (EN);Исходный JSON-Path
```

---

*End of localization.md*