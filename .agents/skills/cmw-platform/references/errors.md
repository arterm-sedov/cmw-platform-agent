# Error Handling Playbook

## Response Structure

All tools return this structure:
```python
{
    "success": bool,      # False on any error
    "status_code": int,   # HTTP status code
    "data": None,         # None on error
    "error": str|dict     # Error details
}
```

## HTTP Status Classification

| Status | Meaning | Recovery |
|--------|---------|----------|
| 200 | Success | Continue |
| 400 | Bad request | Validate input parameters |
| 401 | Unauthorized | Check configuration credentials |
| 408 | Request timeout | Reduce `limit` parameter, increase timeout |
| 500 | Server error | Retry with exponential backoff |

## Common Error Patterns

### 401 Unauthorized
```
{"success": false, "error": "API operation failed: Bad credentials"}
```
**Fix:** Verify configuration has correct credentials

### 400 Bad Request
```
{"success": false, "error": "API operation failed: Invalid parameter"}
```
**Fix:** Check attribute names match template schema exactly

### 500 Server Error
```
{"success": false, "status_code": 500, "error": "API operation failed: ..."}
```
**Fix:** Retry with backoff, check platform status

## Retry Pattern

```python
import time

def retry_with_backoff(func, payload, max_retries=3, delay=1):
    for attempt in range(max_retries):
        result = func.invoke(payload)
        
        if result["success"]:
            return result
            
        # Retry on transient errors
        status = result.get("status_code")
        if status in (500, 503, 408):
            wait = delay * (2 ** attempt)
            time.sleep(wait)
            continue
            
        # Non-retryable error
        return result
        
    return {"success": False, "error": "Max retries exceeded"}
```

## Diagnostic Command

Test connectivity before debugging operations:
```bash
python .agents/skills/cmw-platform/scripts/diagnose_connection.py
```

Exit code 0 = all checks passed, 1 = some checks failed

## Error Message Patterns

| Pattern | Meaning |
|---------|---------|
| "API operation failed: Bad credentials" | Auth failure - check configuration |
| "API operation failed: ..." | Platform returned error |
| "Request timeout" | Increase timeout or reduce limit |
| "Connection error" | Platform unreachable - check URL |
| "Max retries exceeded" | Transient errors persisted |

## List API cast failures (single bad row)

When `GET webapi/Records/Template@…` fails for a page (serializer/cast error on one record), do not treat the template as empty:

1. Narrow with `limit`/`offset` or note the failing id from the error.
2. Fall back to **`GET webapi/Record/{id}`** or per-record property read for other rows.
3. Fix or exclude the bad row, then retry the list.

→ [platform_usage_discoveries.md](platform_usage_discoveries.md#list-api-cast-errors--per-record-fallback)

## PascalCase alias no-op (HTTP 200, field unchanged)

`create_edit_record` and raw Record PUT return **200** when **`values` keys use lowercase** aliases that do not match Attribute/List — the platform **silently ignores** them.

| Step | Action |
|------|--------|
| 1 | `list_attributes` on the template — note **PascalCase** `alias` values |
| 2 | PUT with exact PascalCase keys (e.g. `WorkStatus_calc`, `Assignee_calc`) |
| 3 | GET verify — do not trust status_code alone |

→ [platform_usage_discoveries.md](platform_usage_discoveries.md#maintenanceexecution_calc-refs-lookups-and-api-writes)

## Stale source ids in dataset filters

List datasets imported from `{source_host}` may filter on **catalog record ids that do not exist on `{target_host}`** — rows look populated in Records API but **filtered lists show zero**. Remap via GET → merge `filter.value` → PUT ([edit_or_create.md](edit_or_create.md#dataset-filters-account-id-literals-and-other-filter-json)).

→ [platform_usage_discoveries.md](platform_usage_discoveries.md#list-dataset-filters-with-stale-source-catalog-ids) · [Dataset PUT endpoint and globalAlias](platform_usage_discoveries.md#dataset-put-endpoint-globalalias-and-full-body) · [Composite filters](platform_usage_discoveries.md#composite-list-filters-technicalsystem--spacefloor-or) · [MyMaintenance USER() lists](platform_usage_discoveries.md#mymaintenance-and-user-scoped-maintenance-lists)

## 405 Dataset PUT wrong endpoint

`PUT webapi/Dataset/{app}/Dataset@{template}.{dataset}` returns **405 Method Not Allowed** on many hosts.

| Step | Action |
|------|--------|
| 1 | GET via `Dataset@{template}.{dataset}` (read path) |
| 2 | Merge filter / expression changes; inject **`globalAlias`** if GET returned null |
| 3 | **`PUT webapi/Dataset/{app}`** with the **full** merged body |

→ [edit_or_create.md](edit_or_create.md#dataset-filters-account-id-literals-and-other-filter-json) · [platform_usage_discoveries.md — Dataset PUT](platform_usage_discoveries.md#dataset-put-endpoint-globalalias-and-full-body)

## Assignee: staff row id vs platform account

PUT **`Assignee_calc`** with a **Staff row id** (or stale `{source_host}` account id) may return **200** while **`GET webapi/Record/account.{id}`** is **404** — USER()-scoped lists show zero rows.

| Step | Action |
|------|--------|
| 1 | Resolve platform login via **Administration / AccountService** on `{target_host}` |
| 2 | **IncludeInContainer** (or Attach account) to link login ↔ Staff row |
| 3 | PUT **`Record/account.{target_AccountService_id}`**; verify GET Record |

→ [platform_usage_discoveries.md — Staff row id vs platform account](platform_usage_discoveries.md#staff-row-id-vs-platform-account-includeincontainer) · [cmw-platform-staff-account-link](../../cmw-platform-staff-account-link/SKILL.md)

## Legacy app expression on calculated fields (advisory)

Calculated attributes whose expressions call **`OBJECT('{legacy_app}', …)`** evaluate **null** on `{target_host}` after app rename/migration — form tabs show **"–"** while sibling writable fields (e.g. **`Status_calc`**) persist correctly. Treat as **expression localization debt**, not a failed PUT.

→ [platform_usage_discoveries.md — Inspections StatusBoost](platform_usage_discoveries.md#inspections_calc-empty-form-ui-vs-populated-api)

## Validation Before Create/Edit

Always verify attribute names exist in template before create/edit:
```python
# Get schema first
schema = list_attributes.invoke({
    "application_system_name": app,
    "template_system_name": tmpl
})

# Check attribute exists
attr_names = {a["Attribute system name"] for a in schema["data"]}
invalid = set(values.keys()) - attr_names
if invalid:
    print(f"Unknown attributes: {invalid}")
```