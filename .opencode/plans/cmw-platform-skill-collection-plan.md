# cmw-platform Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a production-ready autonomous platform skill using TDD methodology (RED→GREEN→REFACTOR cycle).

**Architecture:** Single-skill (`cmw-platform`) with progressive disclosure. Skill imports tools from `cmw-platform-agent/tools/` and orchestrates via LLM reasoning. Supplementary scripts for diagnostics in `scripts/`.

**Tech Stack:** Python 3.11+, existing `tools/` modules, `requests`, `python-dotenv`.

---

## TDD SKILL CREATION: RED→GREEN→REFACTOR

**This plan applies TDD to skill creation:**

| Phase | Purpose | Deliverable |
|-------|---------|-------------|
| **RED** | Baseline test WITHOUT skill | Document how agent fails without skill |
| **GREEN** | Write minimal skill | SKILL.md + references addressing baseline failures |
| **REFACTOR** | Close loopholes | Bulletproof skill against rationalizations |

---

## PART 0: RED PHASE - Baseline Testing (Watch Agent Fail)

**Before writing ANY skill content, run baseline scenarios WITHOUT the skill.**

### Task 0.1: Create Pressure Test Scenarios

**Files:**
- Create: `cmw-platform-agent/cmw-platform-workspace/red-phase/baseline-scenarios.md`

- [ ] **Step 1: Document baseline scenario templates**

Pressure scenario template (from writing-skills):
```markdown
IMPORTANT: This is a real scenario. You must choose and act.

[Realistic task description with specific platform context]
Real file paths, real constraints, real consequences

Options:
A) [Correct approach]
B) [Rationalized shortcut]
C) [Partial compliance]

Choose A, B, or C. Be honest.
```

- [ ] **Step 2: Create 3 baseline scenarios**

**Scenario 1 - Time Pressure + Complex Query:**
```markdown
IMPORTANT: This is a real scenario. You must choose and act.

User needs to find all tenants (Arendator) with debt over 30 days 
in the Volga application. Deadline: 5 minutes. User is waiting.

Options:
A) Run explore workflow properly: list_templates → list_attributes → 
   list_template_records with filters
B) Just call list_template_records directly without understanding schema
C) Ask user for SQL or more details

Choose A, B, or C.
```

**Scenario 2 - Sunk Cost + Manual Approach:**
```markdown
IMPORTANT: This is a real scenario. You must choose and act.

You spent 2 hours trying to create a record via direct API calls.
Keep getting 400 errors. You have working credentials. User is waiting.

Options:
A) Use the existing create_edit_record tool which handles type coercion
B) Continue debugging the direct API approach (almost working)
C) Tell user platform API is broken

Choose A, B, or C.
```

**Scenario 3 - Exhaustion + Simplification:**
```markdown
IMPORTANT: This is a real scenario. You must choose and act.

User wants to explore 99 templates in Volga application. 
It's 6pm, you're tired. User just wants "a quick overview."

Options:
A) Run systematic explore: list_templates, then batch list_attributes
B) Just call list_applications and give up
C) Guess which templates are important based on names

Choose A, B, or C.
```

- [ ] **Step 3: Run WITHOUT skill, document verbatim rationalizations**

Run each scenario WITHOUT the skill loaded. Document:
- Which option agent chose
- Exact rationalizations used
- Patterns in failures

- [ ] **Step 4: Create RED phase report**

```markdown
# RED Phase Baseline Report

## Scenario 1: Complex Query
**Chosen:** [A/B/C]
**Rationalizations:** 
- "..." verbatim
- "..." verbatim

## Scenario 2: Manual Approach  
**Chosen:** [A/B/C]
**Rationalizations:**
- "..." verbatim

## Scenario 3: Exhaustion
**Chosen:** [A/B/C]
**Rationalizations:**
- "..." verbatim

## Failure Patterns Identified:
1. [Pattern 1]
2. [Pattern 2]
3. [Pattern 3]
```

- [ ] **Step 5: Commit RED phase**

```bash
git add cmw-platform-agent/cmw-platform-workspace/red-phase/
git commit -m "test(skill): RED phase baseline scenarios and failures"
```

---

## PART 1: GREEN PHASE - Write Minimal Skill

### Task 1.1: Create Skill Directory Structure

**Files:**
- Create: `~/.agents/skills/cmw-platform/SKILL.md`
- Create: `~/.agents/skills/cmw-platform/references/`
- Create: `~/.agents/skills/cmw-platform/scripts/`

- [ ] **Step 1: Create directories**

```bash
mkdir -p "~/.agents/skills/cmw-platform/references"
mkdir -p "~/.agents/skills/cmw-platform/scripts"
```

- [ ] **Step 2: Create SKILL.md with frontmatter**

```markdown
---
name: cmw-platform
description: Use when working with Comindware Platform — connecting to platform, listing applications, exploring templates, managing records, querying data, creating or editing attributes, or any task requiring autonomous platform interaction. Triggers on platform operations, CMW queries, tenant management, rental lot operations, debt tracking, or record CRUD.
---

# cmw-platform Skill

## Overview
Enables autonomous interaction with Comindware Platform using tools from `cmw-platform-agent`.

## Quick Start

1. Activate venv: `.venv\Scripts\Activate.ps1`
2. Import tool: `from tools.<category>.<tool_file> import <tool_function>`
3. Invoke: `<tool>.invoke({...})`
4. Check: `result["success"]`

## Tool Invocation Pattern

```python
from tools.applications_tools.tool_list_applications import list_applications
result = list_applications.invoke({})
```

## Response Structure
```python
{
    "success": bool,
    "status_code": int,
    "data": list|dict,
    "error": str|null
}
```

## Core Tools

| Tool | Import | Use |
|------|--------|-----|
| list_applications | tools.applications_tools.tool_list_applications | List apps |
| list_templates | tools.applications_tools.tool_list_templates | List templates |
| list_attributes | tools.templates_tools.tool_list_attributes | Get schema |
| list_template_records | tools.templates_tools.tool_list_records | Query records |
| create_edit_record | tools.templates_tools.tool_create_edit_record | CRUD records |

## Workflow Patterns

### Explore Application (Systematic)
1. list_applications → find app system name
2. list_templates → get template list  
3. list_attributes (per template) → get schema

### Query with Filters
1. list_template_records.invoke({"filters": {"FieldName": value}})
2. Check result["success"], extract result["data"]

### Create Record
1. list_attributes → understand schema
2. create_edit_record.invoke({"operation": "create", "values": {...}})

## Error Handling

| Status | Meaning | Action |
|--------|---------|--------|
| 401 | Bad credentials | Check .env |
| 408 | Query timeout | Reduce limit |
| 500 | Server error | Retry with backoff |

## References

- Tool inventory: [references/tool_inventory.md](references/tool_inventory.md)
- API endpoints: [references/api_endpoints.md](references/api_endpoints.md)  
- Workflows: [references/workflow_sequences.md](references/workflow_sequences.md)
- Errors: [references/errors.md](references/errors.md)

## Diagnostic Script
Run: `python ~/.agents/skills/cmw-platform/scripts/diagnose_connection.py`
```

- [ ] **Step 3: Commit SKILL.md**

```bash
git add "~/.agents/skills/cmw-platform/SKILL.md"
git commit -m "feat(skill): initial cmw-platform SKILL.md"
```

---

### Task 1.2: Create references/tool_inventory.md

**Files:**
- Create: `~/.agents/skills/cmw-platform/references/tool_inventory.md`

- [ ] **Step 1: Write complete tool inventory**

```markdown
# Tool Inventory

## Import Convention
`from tools.<category>.<tool_file> import <tool_function>`

## Applications Tools

### list_applications
- **Import:** `from tools.applications_tools.tool_list_applications import list_applications`
- **Signature:** `list_applications.invoke({})`
- **Returns:** `{"success": bool, "data": [{"Application system name": str, "Name": str}]}`

### list_templates
- **Import:** `from tools.applications_tools.tool_list_templates import list_templates`
- **Signature:** `list_templates.invoke({"application_system_name": str})`
- **Returns:** `{"success": bool, "data": [{"Template system name": str, "Name": str}]}`

## Templates Tools

### list_attributes
- **Import:** `from tools.templates_tools.tool_list_attributes import list_attributes`
- **Signature:** `list_attributes.invoke({"application_system_name": str, "template_system_name": str})`
- **Returns:** `{"success": bool, "data": [{"Attribute system name": str, "Attribute type": str, "Name": str}]}`

### list_template_records
- **Import:** `from tools.templates_tools.tool_list_records import list_template_records`
- **Params:** application_system_name, template_system_name, attributes, filters, limit (max 100), offset, sort_by, sort_desc
- **Returns:** `{"success": bool, "data": [record, ...]}`

### create_edit_record
- **Import:** `from tools.templates_tools.tool_create_edit_record import create_edit_record`
- **Params:** operation ("create"|"edit"), application_system_name, template_system_name, values, record_id
- **Returns:** `{"success": bool, "data": {...}}`

## Attributes Tools

Pattern: `from tools.attributes_tools.tool_<type>_attribute import create_or_edit_<type>_attribute`

Types: text, decimal, enum, boolean, datetime, document, image, drawing, role, record, account
```

- [ ] **Step 2: Commit**

```bash
git add "~/.agents/skills/cmw-platform/references/tool_inventory.md"
git commit -m "feat(skill): add tool_inventory.md"
```

---

### Task 1.3: Create references/api_endpoints.md

**Files:**
- Create: `~/.agents/skills/cmw-platform/references/api_endpoints.md`

- [ ] **Step 1: Write API reference**

```markdown
# API Endpoints

## Base URL
`{CMW_BASE_URL}/webapi/{Endpoint}`

## Authentication
HTTP Basic: `Authorization: Basic {base64(login:password)}`

## Core Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| webapi/Solution | GET | List applications |
| webapi/Attribute/List/Template@{app}.{tmpl} | GET | List attributes |
| webapi/Records/Template@{app}.{tmpl} | GET | List records |
| webapi/Records/Template@{app}.{tmpl} | POST | Create record |
| webapi/Records/Template@{app}.{tmpl}/{id} | PUT | Update record |

## Response Format
```json
{"response": {...}, "success": true, "error": null}
```

## Error Format
```json
{"response": null, "success": false, "error": {"message": "...", "type": "..."}}
```

## API Schemas
- `cmw-platform-agent/cmw_open_api/web_api_v1.json`
- `cmw-platform-agent/cmw_open_api/solition_api.json`
- `cmw-platform-agent/cmw_open_api/system_core_api.json`
```

- [ ] **Step 2: Commit**

```bash
git add "~/.agents/skills/cmw-platform/references/api_endpoints.md"
git commit -m "feat(skill): add api_endpoints.md"
```

---

### Task 1.4: Create references/workflow_sequences.md

**Files:**
- Create: `~/.agents/skills/cmw-platform/references/workflow_sequences.md`

- [ ] **Step 1: Write workflow patterns**

```markdown
# Workflow Sequences

## 1. Explore Application Structure

```python
from tools.applications_tools.tool_list_templates import list_templates
from tools.templates_tools.tool_list_attributes import list_attributes

def explore_app(app_name: str):
    templates = list_templates.invoke({"application_system_name": app_name})
    if not templates["success"]:
        return {"error": templates["error"]}
    
    for tmpl in templates["data"][:5]:
        attrs = list_attributes.invoke({
            "application_system_name": app_name,
            "template_system_name": tmpl["Template system name"]
        })
    return templates
```

## 2. Query with Filters

```python
from tools.templates_tools.tool_list_records import list_template_records

def find_tenants_with_debt(app_name: str, min_days: int = 30):
    result = list_template_records.invoke({
        "application_system_name": app_name,
        "template_system_name": "Arendator",
        "filters": {"Debt": {"$gt": 0}},
        "limit": 100
    })
    return result
```

## 3. Create Record

```python
from tools.templates_tools.tool_create_edit_record import create_edit_record

def create_rent_lot(app_name: str, name: str, area: float):
    result = create_edit_record.invoke({
        "operation": "create",
        "application_system_name": app_name,
        "template_system_name": "RentLots", 
        "values": {"Nazvanie": name, "Ploschad": area}
    })
    return result
```

## 4. Pagination

```python
def fetch_all(app_name: str, template: str, page_size: int = 100):
    all_records = []
    offset = 0
    while True:
        result = list_template_records.invoke({
            "application_system_name": app_name,
            "template_system_name": template,
            "limit": page_size,
            "offset": offset
        })
        if not result["success"] or not result["data"]:
            break
        all_records.extend(result["data"])
        if len(result["data"]) < page_size:
            break
        offset += page_size
    return all_records
```
```

- [ ] **Step 2: Commit**

```bash
git add "~/.agents/skills/cmw-platform/references/workflow_sequences.md"
git commit -m "feat(skill): add workflow_sequences.md"
```

---

### Task 1.5: Create references/errors.md

**Files:**
- Create: `~/.agents/skills/cmw-platform/references/errors.md`

- [ ] **Step 1: Write error playbook**

```markdown
# Error Handling Playbook

## Error Response Structure
```python
{"success": False, "status_code": int, "error": str|dict, "data": None}
```

## HTTP Status Classification

| Status | Meaning | Recovery |
|--------|---------|----------|
| 200 | Success | Continue |
| 400 | Bad request | Validate input |
| 401 | Unauthorized | Check .env credentials |
| 408 | Timeout | Reduce limit |
| 500 | Server error | Retry with backoff |

## Retry Pattern

```python
import time

def retry_with_backoff(func, max_retries=3, delay=1):
    for attempt in range(max_retries):
        result = func.invoke({})
        if result["success"]:
            return result
        if result.get("status_code") in (500, 503, 408):
            time.sleep(delay * (2 ** attempt))
            continue
        return result
    return {"success": False, "error": "Max retries exceeded"}
```

## Diagnostic Command
```bash
python ~/.agents/skills/cmw-platform/scripts/diagnose_connection.py
```
```

- [ ] **Step 2: Commit**

```bash
git add "~/.agents/skills/cmw-platform/references/errors.md"
git commit -m "feat(skill): add errors.md"
```

---

### Task 1.6: Create scripts/diagnose_connection.py

**Files:**
- Create: `~/.agents/skills/cmw-platform/scripts/diagnose_connection.py`

- [ ] **Step 1: Write complete diagnostic script**

```python
#!/usr/bin/env python3
"""
Connection diagnostics for cmw-platform skill.

Usage: python diagnose_connection.py
Exit: 0 = pass, 1 = fail
"""

import sys
import os
from pathlib import Path

def check_env():
    print("=" * 60)
    print("CHECKING .env FILE")
    print("=" * 60)
    
    repo_root = Path("cmw-platform-agent")
    env_path = repo_root / ".env"
    
    if not env_path.exists():
        print(f"FAIL: .env not found at {env_path}")
        return False
    
    print(f"OK: .env found")
    
    required = ["CMW_BASE_URL", "CMW_LOGIN", "CMW_PASSWORD"]
    with open(env_path, encoding="utf-8") as f:
        found = [k for k in required if any(k in line for line in f)]
    
    missing = set(required) - set(found)
    if missing:
        print(f"FAIL: Missing {missing}")
        return False
    
    print("OK: All required keys present")
    return True

def check_connection():
    print("\n" + "=" * 60)
    print("TESTING API CONNECTION")
    print("=" * 60)
    
    try:
        from dotenv import load_dotenv
        load_dotenv(Path("cmw-platform-agent/.env"))
    except ImportError:
        print("FAIL: python-dotenv not installed")
        return False
    
    base_url = os.environ.get("CMW_BASE_URL", "").rstrip("/")
    login = os.environ.get("CMW_LOGIN", "")
    password = os.environ.get("CMW_PASSWORD", "")
    
    if not all([base_url, login, password]):
        print("FAIL: Missing credentials")
        return False
    
    print(f"  URL: {base_url}")
    print(f"  Login: {login}")
    
    import base64, requests
    creds = base64.b64encode(f"{login}:{password}".encode()).decode()
    headers = {"Authorization": f"Basic {creds}", "Content-Type": "application/json"}
    
    try:
        resp = requests.get(f"{base_url}/webapi/Solution", headers=headers, timeout=30)
        print(f"  Status: {resp.status_code}")
        
        if resp.status_code == 200:
            print("OK: Connection successful")
            return True
        elif resp.status_code == 401:
            print("FAIL: Authentication failed")
            return False
        else:
            print(f"WARN: Unexpected status")
            return False
    except Exception as e:
        print(f"FAIL: {e}")
        return False

def main():
    print("cmw-platform Connection Diagnostics\n")
    env_ok = check_env()
    conn_ok = check_connection()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if env_ok and conn_ok:
        print("ALL CHECKS PASSED")
        sys.exit(0)
    else:
        print("SOME CHECKS FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test script**

```bash
.venv\Scripts\Activate.ps1 && python "~/.agents/skills/cmw-platform/scripts/diagnose_connection.py"
```

Expected: Diagnostics output with PASS/FAIL summary

- [ ] **Step 3: Commit**

```bash
git add "~/.agents/skills/cmw-platform/scripts/diagnose_connection.py"
git commit -m "feat(skill): add diagnose_connection.py"
```

---

## PART 2: REFACTOR PHASE - Close Loopholes

### Task 2.1: GREEN Phase Verification

- [ ] **Step 1: Re-run baseline scenarios WITH skill**

Run the same 3 pressure scenarios with the skill loaded. Document:
- Which option agent chose
- Did they cite skill sections?
- Any new rationalizations?

- [ ] **Step 2: If violations found, add explicit counters**

If agent chose wrong option despite skill:

```markdown
## Rationalization Counter (Add to SKILL.md)

| Excuse | Reality |
|--------|---------|
| "Just this once" | Systematically wrong approach |
| "Working directly is faster" | create_edit_record handles type coercion |
```

- [ ] **Step 3: Update description with violation symptoms**

```yaml
description: Use when working with Comindware Platform... 
Also triggers on: trying direct API calls, ignoring schema exploration,
using manual approaches when tools exist.
```

---

## PART 3: TEST PHASE - Verify Skill Works

### Task 3.1: Create Test Evals

**Files:**
- Create: `cmw-platform-agent/cmw-platform-workspace/evals/evals.json`

- [ ] **Step 1: Create evaluation prompts**

```json
{
  "skill_name": "cmw-platform",
  "evals": [
    {
      "id": 1,
      "prompt": "Explore the Volga application. List all templates and their key attributes.",
      "expected_output": "Systematic explore: list_templates → list_attributes per template",
      "files": []
    },
    {
      "id": 2,
      "prompt": "Find all tenants (Arendator) in Volga with debt. Show IDs and names.",
      "expected_output": "list_template_records with filters on Debt field",
      "files": []
    },
    {
      "id": 3,
      "prompt": "Run the connection diagnostic to check if the platform is accessible.",
      "expected_output": "Output from diagnose_connection.py showing pass/fail",
      "files": []
    }
  ]
}
```

- [ ] **Step 2: Run evals and verify outputs**

- [ ] **Step 3: Commit evals**

```bash
git add cmw-platform-agent/cmw-platform-workspace/evals/
git commit -m "test(skill): add evaluation prompts"
```

---

## Self-Review Checklist

Before claiming completion, verify:

### SKILL.md Quality
- [ ] Under 500 lines
- [ ] Description starts with "Use when..."
- [ ] Third person, specific triggers
- [ ] Unix-style paths (forward slashes)
- [ ] All references one level deep

### References Quality
- [ ] tool_inventory.md: Complete tool catalog
- [ ] api_endpoints.md: Core endpoints documented
- [ ] workflow_sequences.md: Working code patterns
- [ ] errors.md: Recovery patterns

### Scripts Quality
- [ ] diagnose_connection.py: Complete, runnable, self-documenting

### TDD Compliance
- [ ] RED phase: Baseline failures documented
- [ ] GREEN phase: Skill addresses baseline failures
- [ ] REFACTOR phase: Loopholes closed
- [ ] Test evals: Created and run

### Code Quality
- [ ] No placeholders (TBD, TODO, "...")
- [ ] Complete code in every file
- [ ] Exact commands with expected output
- [ ] DRY: No repo code duplicated

---

## Final Deliverables

| File | Lines | Status |
|------|-------|--------|
| SKILL.md | ~120 | Complete |
| references/tool_inventory.md | ~80 | Complete |
| references/api_endpoints.md | ~40 | Complete |
| references/workflow_sequences.md | ~120 | Complete |
| references/errors.md | ~60 | Complete |
| scripts/diagnose_connection.py | ~100 | Complete |
| evals/evals.json | ~30 | Complete |

---

## Execution Options

**Plan complete. Two execution approaches:**

**1. Subagent-Driven (recommended)**
Dispatch fresh subagent per task, review between tasks.

**2. Inline Execution**  
Execute tasks in this session using executing-plans skill.

Which approach?
