# Plan: Workspace/Navigation Section Management Tools

**Date:** 2026-04-23  
**Status:** Analysis Complete - Awaiting Decision

---

## Executive Summary

**Finding:** CMW Platform workspaces (navigation sections) have **data models defined** in the OpenAPI specs but **no dedicated CRUD API endpoints** exposed in the documented APIs.

**Recommendation:** Cannot implement typical CRUD toolset without API endpoints. Alternative approaches available.

---

## Analysis Results

### 1. Terminology Mapping

| User Term | CMW Platform API Term | Context |
|-----------|----------------------|---------|
| Workspace | RoleWorkspace | Navigation section visible to specific roles |
| Navigation Section | RoleWorkspace | Same as workspace |
| Navigation Item | NavigationItem | Individual menu item within a workspace |

### 2. Data Models Found (web_api_v1.json)

**Primary Models:**

- `Comindware.Solution.Api.Data.RoleWorkspaceConfiguration` - Full workspace config
  - Properties: `id`, `name`, `alias`, `solution`, `isDefault`, `description`, `iconClass`, `roles[]`, `navigationItems[]`, `mobile`
  
- `Comindware.Solution.Api.Data.RoleWorkspaceView` - Workspace view/summary
  - Properties: `id`, `name`, `alias`, `solution`, `solutionName`, `isDefault`, `description`

- `Comindware.Mobile.Api.Data.Workspaces.MyRoleWorkspaceView` - User's workspace view
  - Properties: `key`, `id`, `name`, `solution`, `iconClass`, `itemType`, `navigationItems[]`

- `Comindware.Mobile.Api.Data.Workspaces.NavigationItem` - Navigation menu item
  - Properties: `key`, `alias`, `id`, `name`, `description`, `container`, `priority`, `itemType`, `customURL`, `iconClass`, `target`, `targetUrl`, `severity`, `userCommand`, `childrenItems[]`

- `Comindware.Mobile.Api.Data.Workspaces.WorkspaceQuery` - Query filter
  - Properties: `type` (Portal/Studio/Console/Architect), `searchString`, `selectedWorkspaces[]`, `navigationItemId`

**Navigation Item Types (200+ enum values):**
- `RoleWorkspace`, `RoleWorkspaces`, `SolutionRoleWorkspace`
- `RecordTemplateList`, `ProcessTemplateList`, `DashboardPage`, `DesktopPage`
- `Group`, `Splitter`, `Action`, `Link`, `ExternalPage`
- Plus 190+ other template/entity types

### 3. API Endpoints Search Results

**Searched in:**
- `web_api_v1.json` (113 endpoints) - ❌ No workspace endpoints
- `system_core_api.json` - ❌ No workspace CRUD endpoints (only navigation reference services)
- `solition_api.json` (846 endpoints) - ❌ No workspace endpoints (only RolesCatalog for roles)

**What I found:**
- `/Volga_RolesCatalog` - Manages **roles**, not workspaces
- `/Base/EncryptedNavigationReferenceService/*` - Navigation reference encryption (not CRUD)
- No `/webapi/Workspace/*`, `/webapi/RoleWorkspace/*`, or similar patterns

### 4. File-Based Configuration Evidence

From `localization.md` reference:

```
Application/
├── RecordTemplates/
├── WidgetConfigs/
├── Workspaces/          ← Workspaces stored as JSON files
├── Roles/
├── Routes/
└── Pages/
```

Workspaces appear to be managed as **file-based configurations** within application exports, not via REST API.

---

## Why No API Endpoints?

**Hypothesis:** Workspaces are likely:

1. **Configuration-level entities** - Managed through the platform UI designer, not runtime API
2. **Part of application structure** - Exported/imported with application packages
3. **Tied to deployment** - Modified during development, not runtime operations
4. **Security-sensitive** - Navigation structure changes affect access control

Similar to how you don't typically have API endpoints to modify your application's routing configuration at runtime.

---

## Alternative Approaches

### Option 1: File-Based Tools (Recommended for Localization Workflow)

**What:** Tools that work with workspace JSON files in application exports

**Use Case:** Localization, bulk editing, version control

**Tools to implement:**
- `read_workspace_config(app_export_path, workspace_name)` - Read workspace JSON
- `list_workspaces(app_export_path)` - List all workspaces in export
- `validate_workspace_config(workspace_json)` - Validate structure
- `extract_workspace_strings(workspace_json)` - For localization

**Pros:**
- Works with existing export/import workflow
- Integrates with localization skill
- Version control friendly
- No API dependency

**Cons:**
- Requires application export/import cycle
- Not real-time
- Doesn't work on live platform

### Option 2: Query/Read-Only Tools

**What:** Tools to query workspace information from the platform

**Possible endpoints to investigate:**
- Mobile API endpoints (the models suggest mobile app uses workspace data)
- Solution API endpoints (RoleWorkspaceConfiguration suggests solution-level access)
- Custom endpoint discovery (check if newer API versions exist)

**Tools to implement:**
- `get_my_workspaces()` - Get current user's workspaces
- `query_workspaces(solution, search_string)` - Search workspaces
- `get_workspace_navigation_items(workspace_id)` - Get menu structure

**Pros:**
- Read current state from live platform
- Useful for documentation/analysis
- No modification risk

**Cons:**
- Still need to find the actual endpoints
- Read-only (no create/edit/delete)

### Option 3: Investigate Undocumented APIs

**What:** Reverse-engineer the platform UI to find workspace management endpoints

**Approach:**
1. Open CMW Platform UI in browser
2. Open DevTools Network tab
3. Create/edit a workspace in the UI
4. Capture the API calls
5. Document the endpoints and payloads

**Pros:**
- Might discover full CRUD API
- Could enable complete workspace management

**Cons:**
- Time-consuming
- Undocumented APIs may change
- May not be officially supported
- Requires platform access

### Option 4: Use Platform Export/Import API

**What:** Manage workspaces through application export/import

**Workflow:**
1. Export application (includes Workspaces/)
2. Modify workspace JSON files
3. Import application back

**Tools to implement:**
- Wrapper around existing export/import tools
- Workspace-specific extraction/modification
- Validation before import

**Pros:**
- Uses official API
- Complete control over workspace config
- Integrates with existing tools

**Cons:**
- Heavy operation (full app export/import)
- Potential for breaking changes
- Requires careful validation

---

## Recommendation

**I recommend Option 1 (File-Based Tools) + Option 2 (Read-Only Query Tools) as a hybrid approach:**

### Phase 1: Read-Only Tools (Quick Win)

Implement tools to query and understand workspace structure:

1. **Investigate Mobile/Solution APIs** - The models exist, endpoints might too
2. **`list_my_workspaces()`** - Get user's accessible workspaces
3. **`get_workspace_structure(workspace_id)`** - Get navigation tree
4. **`search_navigation_items(query)`** - Find specific menu items

### Phase 2: File-Based Management (For Localization)

Integrate with localization workflow:

1. **`extract_workspace_config(app_export_path)`** - Extract from export
2. **`list_workspace_navigation_items(workspace_json)`** - Parse structure
3. **`validate_workspace_json(workspace_json)`** - Validate before import
4. **Integration with localization skill** - Harvest/apply translations

### Phase 3: Full CRUD (If Endpoints Found)

If we discover actual CRUD endpoints:

1. **`create_workspace(solution, name, description, roles, navigation_items)`**
2. **`edit_workspace(workspace_id, updates)`**
3. **`delete_workspace(workspace_id)`**
4. **`add_navigation_item(workspace_id, item_config)`**
5. **`reorder_navigation_items(workspace_id, item_order)`**

---

## Next Steps

**Decision Required:**

1. **Do you want read-only query tools?** (Requires finding the endpoints first)
2. **Do you want file-based tools for localization workflow?** (Can implement immediately)
3. **Should I investigate undocumented APIs via browser DevTools?** (Time investment)
4. **Is there a specific use case driving this request?** (Helps prioritize approach)

**If you proceed, I'll need:**
- Access to a CMW Platform instance to test endpoint discovery
- Confirmation of which approach aligns with your workflow
- Sample workspace JSON files (if going file-based route)

---

## References

- **OpenAPI Specs:** `cmw_open_api/web_api_v1.json`, `system_core_api.json`, `solition_api.json`
- **Localization Skill:** `.agents/skills/cmw-platform/references/localization.md`
- **Existing Tool Patterns:** `tools/templates_tools/`, `tools/attributes_tools/`
- **Navigation Item Types:** 200+ enum values in `ComindwarePlatformApiDataNavigationItemType`
