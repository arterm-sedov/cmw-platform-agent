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