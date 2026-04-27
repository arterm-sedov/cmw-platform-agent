# Platform Entity URL Resolver — TDD Implementation Plan

**Date:** 2026-04-27
**Type:** New feature — LangChain tool for agent_ng
**Files:** `tools/platform_entity_resolver.py`, `tools/_tests/test_platform_entity_resolver.py`

---

## Problem

When a user pastes a CMW Platform URL like `https://platform.example.com/#RecordType/oa.193/Operation/event.15199`, the agent cannot resolve it to API-ready entity objects. Existing tools require `application_system_name + template_system_name + entity_system_name` — no bridge exists from raw internal IDs to those parameters.

## Solution

A single `resolve_entity` tool that:
1. Parses any CMW Platform URL to extract entity IDs
2. Resolves template IDs (`oa.*`, `pa.*`, `ra.*`, `os.*`) via `Solution/TemplateService/List`
3. Lists candidate entities for buttons/forms/toolbars/datasets using parent template context
4. Returns full entity data ready for API manipulation

## Verified API Contracts

| Endpoint | Works? | Returns Internal IDs? |
|----------|--------|----------------------|
| `POST Solution/TemplateService/List` (Record/Process/Role/OrgStructure) | ✅ | ✅ `id`, `alias`, `solution`, `solutionName` |
| `POST Solution/TemplateService/List` (Form/Dataset/Toolbar/UserCommand) | ❌ 500 | N/A |
| `POST ResolveObjectInfo` | ⚠️ Minimal | `{"id": "...", "name": "..."}` — useless |
| `POST ResolveObjectApp` | ❌ 500 | N/A |
| `POST GetAxioms` | ⚠️ Empty | `{}` for entity IDs |
| `GET webapi/{Entity}/List/Template@{app}.{tpl}` | ✅ | ❌ No internal IDs |

**Key finding:** `TemplateService/List` is the ONLY endpoint that maps internal IDs to system names + application context. For buttons/forms/toolbars/datasets, internal IDs are NOT exposed — we list candidates and let the agent match by name/context.

## Entity ID Prefix Registry

| Prefix | Entity Type | Resolution |
|--------|-------------|------------|
| `oa.{N}` | Record Template | `TemplateService/List` (Type: Record) |
| `pa.{N}` | Process Template | `TemplateService/List` (Type: Process) |
| `ra.{N}` | Role Template | `TemplateService/List` (Type: Role) |
| `os.{N}` | OrgStructure Template | `TemplateService/List` (Type: OrgStructure) |
| `sln.{N}` | Solution | Match `solution` field in TemplateService results |
| `role.{N}` | Role | `TemplateService/List` (Type: Role) |
| `event.{N}` | Button | List via `UserCommand/List` with parent context |
| `form.{N}` / `card.{N}` | Form | List via `Form/List` with parent context |
| `tb.{N}` | Toolbar | List via `Toolbar/List` with parent context |
| `lst.{N}` / `ds.{N}` | Dataset | List via `Dataset/List` with parent context |
| `diagram.{N}` | Diagram | `Process/DiagramService/ResolveDiagram` |
| `workspace.{N}` | Workspace | Metadata only (no known API) |
| Plain `{N}` | Record | `GET webapi/Record/{recordId}` |

## Architecture

```
resolve_entity(url_or_id, fetch_full=True)
  │
  ├─ _parse_url(url_or_id)
  │   └─ Extract all entity IDs + types from URL hash + query params
  │
  ├─ _resolve_templates(template_ids)
  │   └─ POST TemplateService/List (4 types in parallel) → cache by ID
  │
  ├─ _resolve_solutions(solution_ids, template_cache)
  │   └─ Match sln.* against template_cache solution field
  │
  ├─ _resolve_roles(role_ids, template_cache)
  │   └─ Filter Role templates from cache by role.{N} ID
  │
  ├─ _list_entity_candidates(entity_type, parent_context)
  │   └─ GET webapi/{Entity}/List/Template@{app}.{tpl} → return all
  │
  └─ Assemble resolved[] with full_data (if fetch_full=True)
```

## URL Patterns Supported

| Pattern | Example | Entities |
|---------|---------|----------|
| `#desktop/` | `#desktop/` | None |
| `#solutions` | `#solutions` | None |
| `#solutions/sln.{N}/...` | `#solutions/sln.23/Administration` | Solution |
| `#solutions/sln.{N}/roles/role.{M}` | `#solutions/sln.23/roles/role.83` | Solution + Role |
| `#solutions/sln.{N}/Workspaces/workspace.{M}` | `#solutions/sln.2/Workspaces/workspace.41` | Solution + Workspace |
| `#RecordType/oa.{N}/...` | `#RecordType/oa.3/Operation/event.454` | Template + Button |
| `#RecordType/oa.{N}/Forms/form.{M}` | `#RecordType/oa.3/Forms/form.80` | Template + Form |
| `#RecordType/oa.{N}/Toolbar/Settings/tb.{M}` | `#RecordType/oa.3/Toolbar/Settings/tb.228` | Template + Toolbar |
| `#RecordType/oa.{N}/Lists/lst.{M}` | `#RecordType/oa.3/Lists/lst.81` | Template + List |
| `#RecordType/oa.{N}/Card/Settings/card.{M}` | `#RecordType/oa.3/Card/Settings/card.148` | Template + Form |
| `#RecordType/ra.{N}/...` | `#RecordType/ra.23/Administration` | Role Template |
| `#RecordType/os.{N}/...` | `#RecordType/os.23/Administration` | OrgStructure Template |
| `#ProcessTemplate/pa.{N}/...` | `#ProcessTemplate/pa.77/Operation/event.15193` | Process + Button |
| `#ProcessTemplate/pa.{N}/Designer/Revision/diagram.{M}` | `#ProcessTemplate/pa.77/.../diagram.315` | Process + Diagram |
| `#ProcessTemplate/pa.{N}/Toolbar/tb.{M}` | `#ProcessTemplate/pa.77/Toolbar/tb.8215` | Process + Toolbar |
| `#ProcessTemplate/pa.{N}/Lists/lst.{M}` | `#ProcessTemplate/pa.77/Lists/lst.2741` | Process + List |
| `#data/{tpl}/lst.{M}/...` | `#data/oa.26/lst.137/s=ds.5615...` | Template + List + Dataset |
| `#form/{tpl}/form.{M}/{recordId}` | `#form/oa.3/form.80/55` | Template + Form + Record |
| `#app/{App}/list/{Tpl}` | `#app/FM/list/MP` | App + Template (system name) |
| `#app/{App}/view/{Tpl}/{recordId}` | `#app/FM/view/MP/15199` | App + Template + Record |
| `#Settings/globalSecurity/role.{N}` | `#Settings/globalSecurity/role.9` | Role |
| `#Settings/*` | `#Settings/Groups` | Page type only |
| `#Resolver/{id}` | `#Resolver/oa.193` | Single entity |
| Full URL with hash | `https://host/#RecordType/...` | Same as hash |
| Raw ID only | `oa.193` | Single entity |

---

## Tasks

### Task 1: URL Parser — Tests

**File:** `tools/_tests/test_platform_entity_resolver.py`

Write tests for `_parse_url()` — the function that extracts all entity IDs from a URL.

```python
# Test cases (each is a separate test function):

def test_parse_desktop_url():
    """#desktop/ → no entities"""

def test_parse_solutions_url():
    """#solutions → no entities"""

def test_parse_solution_administration_url():
    """#solutions/sln.23/Administration → [{type: Solution, id: sln.23}]"""

def test_parse_solution_roles_url():
    """#solutions/sln.23/roles/role.83/privileges → [Solution(sln.23), Role(role.83)]"""

def test_parse_solution_workspace_url():
    """#solutions/sln.2/Workspaces/workspace.41 → [Solution(sln.2), Workspace(workspace.41)]"""

def test_parse_record_type_operation_url():
    """#RecordType/oa.3/Operation/event.454 → [Template(oa.3), Button(event.454)]"""

def test_parse_record_type_forms_url():
    """#RecordType/oa.3/Forms/form.80 → [Template(oa.3), Form(form.80)]"""

def test_parse_record_type_toolbar_url():
    """#RecordType/oa.3/Toolbar/Settings/tb.228 → [Template(oa.3), Toolbar(tb.228)]"""

def test_parse_record_type_lists_url():
    """#RecordType/oa.3/Lists/lst.81 → [Template(oa.3), List(lst.81)]"""

def test_parse_record_type_card_url():
    """#RecordType/oa.3/Card/Settings/card.148 → [Template(oa.3), Form(card.148)]"""

def test_parse_role_template_url():
    """#RecordType/ra.23/Administration → [Template(ra.23)]"""

def test_parse_orgstructure_template_url():
    """#RecordType/os.23/Administration → [Template(os.23)]"""

def test_parse_process_template_operation_url():
    """#ProcessTemplate/pa.77/Operation/event.15193 → [ProcessTemplate(pa.77), Button(event.15193)]"""

def test_parse_process_template_diagram_url():
    """#ProcessTemplate/pa.77/Designer/Revision/diagram.315 → [ProcessTemplate(pa.77), Diagram(diagram.315)]"""

def test_parse_process_template_toolbar_url():
    """#ProcessTemplate/pa.77/Toolbar/tb.8215 → [ProcessTemplate(pa.77), Toolbar(tb.8215)]"""

def test_parse_process_template_lists_url():
    """#ProcessTemplate/pa.77/Lists/lst.2741 → [ProcessTemplate(pa.77), List(lst.2741)]"""

def test_parse_data_view_url():
    """#data/oa.26/lst.137/s%3Dds.5615 → [Template(oa.26), List(lst.137), Dataset(ds.5615)]"""

def test_parse_form_view_url():
    """#form/oa.3/form.80/55 → [Template(oa.3), Form(form.80), Record(55)]"""

def test_parse_app_list_url():
    """#app/FacilityManagement/list/MaintenancePlans → [App(FacilityManagement), Template(MaintenancePlans)]"""

def test_parse_app_view_url():
    """#app/FacilityManagement/view/MaintenancePlans/15199 → [App(FM), Template(MP), Record(15199)]"""

def test_parse_settings_role_url():
    """#Settings/globalSecurity/role.9/privileges → [Role(role.9)]"""

def test_parse_settings_page_url():
    """#Settings/Groups → no entities"""

def test_parse_resolver_url():
    """#Resolver/oa.193 → [Template(oa.193)]"""

def test_parse_full_url():
    """https://platform.example.com/#RecordType/oa.3/Operation/event.454 → same as hash-only"""

def test_parse_raw_id():
    """oa.193 → [Template(oa.193)]"""

def test_parse_empty_string():
    """'' → error"""
```

### Task 2: URL Parser — Implementation

**File:** `tools/platform_entity_resolver.py`

Implement `_parse_url(url_or_id: str) -> ParsedUrl`:

```python
@dataclass
class ParsedEntity:
    entity_type: str  # "Template", "Button", "Form", "Toolbar", "Dataset", "Diagram", "Role", "Solution", "Workspace", "Record", "App"
    entity_id: str    # The raw ID (e.g., "oa.193", "event.454")

@dataclass
class ParsedUrl:
    original: str
    hash_path: str          # The part after # (e.g., "RecordType/oa.3/Operation/event.454")
    entities: list[ParsedEntity]
    query_params: dict      # Decoded query params
    page_type: str          # "desktop", "solutions", "RecordType", "ProcessTemplate", "data", "form", "app", "Settings", "Resolver", "unknown"
```

**Implementation approach:**
- Strip base URL, extract hash
- Regex patterns for each prefix: `r'(oa|pa|ra|os)\.(\d+)'`, `r'event\.(\d+)'`, `r'form\.(\d+)'`, `r'card\.(\d+)'`, `r'tb\.(\d+)'`, `r'lst\.(\d+)'`, `r'ds\.(\d+)'`, `r'diagram\.(\d+)'`, `r'role\.(\d+)'`, `r'sln\.(\d+)'`, `r'workspace\.(\d+)'`
- URL-decode query params, extract IDs from filter expressions
- Classify page type from hash path segments
- Determine parent context: template/process/role IDs are parents for child entities

**DRY principle:** Single regex pass over hash path + query params, classify by prefix mapping dict.

### Task 3: TemplateService Resolver — Tests

```python
def test_resolve_template_by_id_record():
    """Mock TemplateService response for oa.193 → {alias: MaintenancePlans, solution: sln.23, solutionName: ...}"""

def test_resolve_template_by_id_process():
    """Mock TemplateService response for pa.77 → {alias: SomeProcess, solution: sln.23, ...}"""

def test_resolve_template_by_id_role():
    """Mock TemplateService response for ra.23 / role.83 → {alias: SomeRole, solution: sln.23, ...}"""

def test_resolve_template_by_id_orgstructure():
    """Mock TemplateService response for os.23 → {alias: SomeOrg, solution: sln.XX, ...}"""

def test_resolve_template_not_found():
    """ID not in TemplateService results → return None"""

def test_resolve_solution_by_id():
    """sln.23 matched against template_cache → {solutionName: ...}"""

def test_resolve_solution_not_found():
    """sln.999 not in any template's solution field → return metadata only"""
```

### Task 4: TemplateService Resolver — Implementation

```python
def _resolve_templates(
    template_ids: list[str],
    requests_module,
) -> dict[str, dict]:
    """
    Fetch all templates via TemplateService/List for 4 types.
    Returns {id: {alias, solution, solutionName, type, name, ...}}.
    """
    # 4 parallel POST calls: Record, Process, Role, OrgStructure
    # Merge results into single dict keyed by id
    # Return empty dict for failed calls (log warning)

def _resolve_solutions(
    solution_ids: list[str],
    template_cache: dict[str, dict],
) -> dict[str, dict]:
    """
    Match sln.* IDs against template_cache solution field.
    Returns {sln_id: {solutionName, ...}}.
    """
```

**DRY principle:** Single helper `_fetch_template_type(type_name)` called 4 times. Cache built once, reused for all entity resolution.

### Task 5: Entity Lister — Tests

```python
def test_list_button_candidates():
    """Mock UserCommand/List → return all buttons with system_name, name, kind"""

def test_list_form_candidates():
    """Mock Form/List → return all forms with system_name, name"""

def test_list_toolbar_candidates():
    """Mock Toolbar/List → return all toolbars with system_name, name"""

def test_list_dataset_candidates():
    """Mock Dataset/List → return all datasets with system_name, name"""

def test_list_entity_unknown_type():
    """Unknown entity type → return empty candidates list"""

def test_list_entity_no_parent_context():
    """No parent template → cannot list candidates → return error"""
```

### Task 6: Entity Lister — Implementation

```python
def _list_entity_candidates(
    entity_type: str,
    app_system_name: str,
    template_system_name: str,
    requests_module,
) -> list[dict]:
    """
    List all entities of given type for a template.
    Returns list of {system_name, name, kind (for buttons), api_endpoint, full_data}.
    """
    endpoint_map = {
        "Button": f"webapi/UserCommand/List/Template@{app}.{tpl}",
        "Form": f"webapi/Form/List/Template@{app}.{tpl}",
        "Toolbar": f"webapi/Toolbar/List/Template@{app}.{tpl}",
        "Dataset": f"webapi/Dataset/List/Template@{app}.{tpl}",
    }
    # GET endpoint → parse response → extract fields → return list
```

**DRY principle:** Single function with endpoint routing dict. No per-entity-type duplication.

### Task 7: Diagram Resolver — Tests

```python
def test_resolve_diagram_success():
    """Mock ResolveDiagram → return diagram string"""

def test_resolve_diagram_failure():
    """Mock 500 response → return error"""
```

### Task 8: Diagram Resolver — Implementation

```python
def _resolve_diagram(
    diagram_id: str,
    requests_module,
) -> dict:
    """
    POST Process/DiagramService/ResolveDiagram with serverId.
    Returns {diagram_id, diagram_data, success, error}.
    """
```

### Task 9: Main Tool — Tests (End-to-End)

```python
def test_resolve_record_type_operation_url():
    """Full pipeline: #RecordType/oa.3/Operation/event.454 → template + button candidates"""

def test_resolve_process_template_diagram_url():
    """Full pipeline: #ProcessTemplate/pa.77/.../diagram.315 → process + diagram"""

def test_resolve_solution_roles_url():
    """Full pipeline: #solutions/sln.23/roles/role.83 → solution + role"""

def test_resolve_data_view_url():
    """Full pipeline: #data/oa.26/lst.137/s=ds.5615 → template + list + dataset"""

def test_resolve_form_view_url():
    """Full pipeline: #form/oa.3/form.80/55 → template + form + record"""

def test_resolve_fetch_full_false():
    """fetch_full=False → skip webapi fetches, return metadata only"""

def test_resolve_raw_template_id():
    """Input: 'oa.193' → resolve template only"""

def test_resolve_unsupported_url():
    """Input: '#Settings/Groups' → page type only, no entities"""

def test_resolve_empty_input():
    """Input: '' → error"""

def test_resolve_network_error():
    """Mock requests failure → return error with status_code"""
```

### Task 10: Main Tool — Implementation

```python
class ResolveEntitySchema(BaseModel):
    url_or_id: str = Field(
        description="CMW Platform URL or entity ID to resolve. "
            "Examples: 'https://host/#RecordType/oa.3/Operation/event.454', "
            "'oa.193', '#Resolver/event.15199'. "
            "RU: URL платформы или ID сущности",
    )
    fetch_full: bool = Field(
        default=True,
        description="Fetch full entity data from API. "
            "Set False to get only resolved metadata. "
            "RU: Загружать полные данные сущности",
    )

@tool("resolve_entity", return_direct=False, args_schema=ResolveEntitySchema)
def resolve_entity(
    url_or_id: str,
    fetch_full: bool = True,
) -> dict[str, Any]:
    """
    Resolve a CMW Platform URL or entity ID to API-ready entity objects.

    Parses the URL to extract all entity IDs, resolves template IDs to
    application + system names via TemplateService, and fetches full entity
    data for buttons, forms, toolbars, datasets, and diagrams.

    Returns:
        dict: {
            "success": bool,
            "status_code": int,
            "error": str|None,
            "url_parsed": {...},
            "resolved": [
                {
                    "entity_type": str,
                    "id": str,
                    "system_name": str,
                    "application_system_name": str,
                    "api_endpoint": str,
                    "full_data": dict|None,
                    "candidates": list|None,
                    "note": str|None,
                },
                ...
            ]
        }
    """
    # 1. Parse URL
    # 2. Resolve templates (TemplateService)
    # 3. Resolve solutions
    # 4. Resolve roles
    # 5. For each non-template entity: list candidates or fetch directly
    # 6. Assemble response
```

### Task 11: Register Tool in tools.py

Add import and expose in `tools/tools.py`:

```python
from .platform_entity_resolver import resolve_entity
```

The tool is automatically available to agent_ng via the `@tool` decorator.

### Task 12: Lint + Test + Verify

```powershell
.venv\Scripts\Activate.ps1
ruff check tools/platform_entity_resolver.py
ruff format tools/platform_entity_resolver.py
ruff check tools/_tests/test_platform_entity_resolver.py
ruff format tools/_tests/test_platform_entity_resolver.py
python -m pytest tools/_tests/test_platform_entity_resolver.py -v
```

---

## Design Decisions

1. **Single TemplateService call per type** — 4 parallel calls (Record, Process, Role, OrgStructure), unified cache by ID. No repeated calls.
2. **Parent context propagation** — Child entities inherit template/solution context from URL path. `event.454` in `#RecordType/oa.3/Operation/event.454` gets context from `oa.3`.
3. **Candidate matching** — Internal IDs for buttons/forms/toolbars/datasets are NOT exposed by any API. Return all candidates with full data; agent matches by name/context.
4. **`fetch_full: bool = True`** — When False, skip webapi fetches. Returns only resolved metadata (saves API calls).
5. **Query param extraction** — URL-decode `s%3Dds.5615` → `ds.5615`. Extract `sln.*` from filter expressions.
6. **Follows existing patterns** — Uses `tools.requests_` harness, Pydantic models, LangChain `@tool` decorator, `AttributeResult`-style response format.
7. **No external deps** — Pure stdlib + existing project deps.
8. **Error handling per AGENTS.md** — No silent exceptions, safe defaults, validate external data, log warnings.

## File Structure

```
tools/
  platform_entity_resolver.py           # New: resolve_entity tool + helpers
  _tests/
    test_platform_entity_resolver.py    # New: TDD tests (25+ cases)
```

## Dependencies

- `tools.requests_` — HTTP request harness (existing)
- `tools.models.AttributeResult` — Response model (existing, or create new `ResolveEntityResult`)
- `pydantic.BaseModel, Field` — Schema validation (existing)
- `langchain_core.tools.tool` — Tool decorator (existing)
- `re`, `urllib.parse`, `dataclasses` — stdlib (no new deps)
