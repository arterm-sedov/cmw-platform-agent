# Localization Script Update Plan

**Date:** 20260427
**Status:** Planning complete - awaiting implementation

## Schema.json Requirements

- 10-step pipeline (0-10) with date prefix `yyyyddMM-HHmmss`
- JSON fields: `aliasOriginal`, `aliasRenamed`, `displayNameOriginal`, `displayNameRenamed`, `jsonPathOriginal`, `jsonPathRenamed`, `expressions[]`
- Interactive step-by-step workflow (1 alias at a time)
- State saving after each alias
- Resume from last processed

## Script Changes Planned

### 1. Date Timestamp Function
Add:
```python
from datetime import datetime

def get_timestamp() -> str:
    return datetime.now().strftime("%Y%d%m-%H%M%S")
```

### 2. Data Structure Update

| Old Field | New Field | Notes |
|----------|-----------|-------|
| systemName | aliasOriginal | Per schema |
| renamedSystemName | aliasRenamed | Per schema |
| — | displayNameOriginal | Collect Name field |
| — | displayNameRenamed | Filled in step 5 |
| jsonPath | jsonPathOriginal | Per schema |
| — | jsonPathRenamed | With suffix tracking |
| — | expressions[] | Array of {jsonPathOriginal, jsonPathRenamed, expressionOriginal, expressionRenamed} |

### 3. Enhanced Folder Recursion

CTF folder levels to scan:
- Root: RecordTemplates/, ProcessTemplates/ (template-level json)
- Subfolder: Datasets/, Attributes/, Forms/, Toolbars/, UserCommands/ (template children)
- Recursive: ALL nested folders

### 4. CLI Arguments (Human-Readable)

| Argument | Type | Purpose |
|---------|------|---------|
| --app | str | Application name (e.g., "Volga1") |
| --step | int | Resume from step (0-10) |
| --resume-alias | str | Resume from specific alias |

### 5. State Management
- Save after processing each alias
- Store last processed alias ID
- Enable resume from interruption