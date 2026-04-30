# Plan: Refactor `get_platform_entity_url` Tool

## Goal
Replace the restrictive 5-type enum with a flexible, LLM-friendly tool that resolves entity IDs to `#Resolver/{id}` URLs and looks up entities by system name across all applications.

## Behavior Contract

### Input Schema
```python
class GetPlatformEntityUrlSchema(BaseModel):
    entity_id: str | None = None       # Direct: "oa.193", "event.15199", "sln.13"
    system_name: str | None = None     # Lookup: "MaintenancePlans"
    application: str | None = None     # Disambiguation: "CustomerPortal"
```

### Validation Rules
- `entity_id` only → resolve by ID via GetAxioms
- `system_name` only → lookup across all apps via TemplateService/List
- `system_name` + `application` → filtered lookup
- `entity_id` + `system_name` → resolve both, verify they match
- Neither → error: "Provide entity_id, system_name, or both"
- Empty strings → error: "Value must be non-empty"

### Response Format
```python
{
    "success": bool,
    "entity_id": str | None,
    "entity_url": str | None,
    "entity_type": str | None,       # "Record Template", "Button", "Application"
    "system_name": str | None,       # "MaintenancePlans"
    "name": str | None,              # "Maintenance Plans"
    "parent_system_name": str | None,  # Direct parent (template or app), None for apps
    "application": str | None,       # Always the app system name
    "matches": list | None           # Only for system_name lookup with multiple results
}
```

### Match Item Format (for `matches` list)
```python
{
    "entity_id": str,
    "entity_url": str,
    "entity_type": str,
    "system_name": str,
    "name": str,
    "parent_system_name": str | None,
    "application": str,
}
```

## Resolution Logic

### Path 1: `entity_id` provided
1. Call `_resolve_entity_id(entity_id)` from `platform_entity_resolver.py`
2. If `success=False` → error: "Entity not found"
3. If `success=True` → return URL + metadata from resolver result
4. URL format: `{base_url}/#Resolver/{entity_id}`

### Path 2: `system_name` provided (no `entity_id`)
1. Call TemplateService/List for all entity types (Record, Role, Process, OrgStructure, Undefined)
2. Filter items where `alias == system_name`
3. For each match:
   - Resolve solution → app_alias
   - Build URL + metadata
4. If `application` provided → filter matches by app
5. Return all matches (even single match goes into `matches` list for consistency)

### Path 3: Both `entity_id` + `system_name` provided
1. Resolve `entity_id` via GetAxioms
2. Check if resolved `alias == system_name`
3. If match → return URL + metadata
4. If mismatch → error: "entity_id and system_name do not match"

## Entity Type Mapping (TemplateService/List "Type" values)
```python
_ENTITY_TYPES = ["Record", "Role", "Process", "OrgStructure", "Undefined"]
```

## ID Prefix → Entity Type Mapping (reused from platform_entity_resolver.py)
```python
"oa": "Template", "pa": "ProcessTemplate", "ra": "Template", "os": "Template",
"sln": "Application", "event": "Button", "form": "Form", "card": "Card",
"tb": "Toolbar", "lst": "Table", "ds": "Table", "diagram": "ProcessDiagram",
"role": "Role", "workspace": "NavigationSection"
```

## Files to Modify

### 1. `tools/applications_tools/tool_platform_entity_url.py` — Rewrite
- New schema with optional `entity_id`, `system_name`, `application`
- New tool function with 3 resolution paths
- Clean docstring with usage examples
- Reuse `_resolve_entity_id` from `platform_entity_resolver.py`

### 2. `tools/_tests/test_platform_entity_url.py` — New test file
- TDD: Write tests first, then implement

### 3. `tools/tool_utils.py` — Remove `GET_URL_TYPE_MAPPING`
- No longer needed after refactor
- Verify no other code references it

## Test Plan (TDD — write tests FIRST)

### Test File: `tools/_tests/test_platform_entity_url.py`

#### Test Class: `TestEntityIdResolution`
- [ ] `test_entity_id_valid` — valid ID returns URL + metadata
- [ ] `test_entity_id_application` — app ID returns URL, parent=None
- [ ] `test_entity_id_button` — button ID returns URL + parent template
- [ ] `test_entity_id_not_found` — invalid ID returns error
- [ ] `test_entity_id_empty_string` — empty string returns error

#### Test Class: `TestSystemNameLookup`
- [ ] `test_system_name_unique_match` — unique name returns single match
- [ ] `test_system_name_multiple_matches` — duplicate names return all matches
- [ ] `test_system_name_with_application` — filtered by app returns single match
- [ ] `test_system_name_no_match` — unknown name returns empty matches
- [ ] `test_system_name_empty_string` — empty string returns error

#### Test Class: `TestCombinedResolution`
- [ ] `test_both_matching` — matching ID + name returns URL
- [ ] `test_both_mismatching` — mismatching ID + name returns error
- [ ] `test_neither_provided` — no params returns error

#### Test Class: `TestEdgeCases`
- [ ] `test_whitespace_trimming` — handles leading/trailing whitespace
- [ ] `test_case_insensitive_application` — app name case handling

## Implementation Order

1. **Write tests** — `tools/_tests/test_platform_entity_url.py` (TDD)
2. **Implement tool** — `tools/applications_tools/tool_platform_entity_url.py`
3. **Remove dead code** — `GET_URL_TYPE_MAPPING` from `tools/tool_utils.py`
4. **Run tests** — `python -m pytest tools/_tests/test_platform_entity_url.py -v`
5. **Run lint** — `ruff check tools/applications_tools/tool_platform_entity_url.py`
6. **Run full test suite** — `python -m pytest tools/_tests/ -v` (verify no breakage)
7. **Commit** — when asked

## Backward Compatibility

- Tool name stays `get_platform_entity_url`
- Old `type` + `system_name` required schema is **replaced** (breaking change for direct callers, but LLM agents adapt via new schema)
- `GET_URL_TYPE_MAPPING` removed after verifying no other references

## DRY Principles

- Reuse `_resolve_entity_id()` from `platform_entity_resolver.py` — no duplicate GetAxioms logic
- Reuse `_ID_PREFIX_MAP` from `platform_entity_resolver.py` — single source of truth
- Extract `_build_entity_url(base_url, entity_id)` helper
- Extract `_format_match(item, base_url, app_alias)` helper for match formatting

## Error Handling

- No silent exceptions — always log with `logger.warning/exception`
- Safe defaults: `None` for optional fields, `[]` for matches
- Validate external data: check GetAxioms response structure
- Handle multiple response formats: dict vs object, different field names
