# Button Kind Regression Fix - Summary

**Date:** 2026-04-27  
**Status:** ✅ COMPLETED  
**Duration:** ~1.5 hours

---

## Problem Fixed

**Regression introduced in commit 8d9e16c (Apr 22, 2026):**
- Changed button `kind` default from `"UserEvent"` to `"Trigger scenario"` for better LLM UX
- **Forgot to add validator** to map LLM-friendly term to API term
- Result: Tool sent `"Trigger scenario"` literally to API, which rejected it (not in enum)

---

## Solution Implemented

### 1. Added `@field_validator("kind")` in `tools/templates_tools/tools_button.py`
- Maps LLM-friendly terms to API enum values
- Handles 29 valid button kinds from CMW Platform API
- Case-insensitive mapping (trigger_scenario, TRIGGER SCENARIO → UserEvent)
- Validates against complete API enum
- Clear error messages for invalid kinds

### 2. Updated Field Description
- Comprehensive documentation of common kinds
- References skill documentation for complete list
- Removed invalid "Test" kind

### 3. Fixed Guard Logic
- Changed from `if kind != "Trigger scenario"` to `if kind != "UserEvent"`
- Correctly handles default after validator mapping

### 4. Comprehensive Documentation in `.agents/skills/cmw-platform/SKILL.md`
- Added "Button Kinds (Action Types)" section
- Documented all 29 valid kinds with descriptions
- Usage examples for common scenarios
- Validation behavior explanation

### 5. Complete Test Coverage
- 17 unit tests covering all mapping scenarios
- Integration tests on real platform
- Verified existing buttons still work

---

## Test Results

### Unit Tests (17/17 passing)
✅ Trigger scenario → UserEvent mapping  
✅ Case-insensitive variants (trigger_scenario, TRIGGER SCENARIO)  
✅ All 29 API kinds validated  
✅ Common kinds (Create, Edit, Delete, Archive, Script)  
✅ Invalid kinds rejected (Test, RandomInvalidKind)  
✅ Default behavior preserved  
✅ Edge cases (whitespace, mixed case, underscores)

### Integration Tests
✅ "Trigger scenario" → UserEvent on real platform  
✅ Direct API term "UserEvent" works  
✅ Snake_case variant "trigger_scenario" works  
✅ Common kinds (Edit, Delete, Archive, Script) work  
✅ Existing buttons (schedule_maintenance) still work  

### Linting & Formatting
✅ Ruff checks pass  
✅ Code formatted correctly  
✅ No type errors

---

## Files Changed

1. **tools/templates_tools/tools_button.py**
   - Added `_normalize_kind` validator (100 lines)
   - Updated `kind` field description
   - Fixed guard logic in edit mode

2. **.agents/skills/cmw-platform/SKILL.md**
   - Added "Button Kinds (Action Types)" section
   - Documented all 29 valid kinds
   - Usage examples and validation notes

3. **tools/templates_tools/_tests/test_button_kind_mapping.py** (NEW)
   - 17 comprehensive unit tests
   - Edge case coverage
   - Documentation validation

4. **.opencode/plans/20260427_button_kind_regression_fix.md** (NEW)
   - Detailed implementation plan
   - TDD approach documented
   - Checkpoints tracked

---

## API Enum Values (29 Total)

1. Undefined
2. Create
3. Edit
4. Delete
5. Archive
6. Unarchive
7. ExportObject
8. ExportList
9. CreateRelated
10. CreateToken
11. RetryTokens
12. Migrate
13. StartCase
14. StartLinkedCase
15. StartProcess
16. StartLinkedProcess
17. CompleteTask
18. ReassignTask
19. Defer
20. Accept
21. Uncomplete
22. Follow
23. Unfollow
24. Exclude
25. Include
26. Script
27. Cancel
28. EditDiagram
29. **UserEvent** ← Default for custom scenarios

---

## Success Criteria (All Met)

✅ LLM sees "Trigger scenario" as default  
✅ API receives "UserEvent" when LLM passes "Trigger scenario"  
✅ All 29 button kinds validated and documented  
✅ Case-insensitive mapping works  
✅ Invalid kinds rejected with clear error messages  
✅ Existing buttons continue to work  
✅ Tests pass (unit + integration)  
✅ Linting and type checking pass  
✅ Documentation complete and accurate  

---

## Key Learnings

1. **Always add validators when changing terminology** - Don't just update defaults
2. **TDD approach works** - Write tests first, implement, verify
3. **OpenAPI spec is source of truth** - Always check actual API enum values
4. **Comprehensive documentation matters** - All 29 kinds now documented
5. **Integration testing catches real issues** - Unit tests alone aren't enough

---

## Next Steps

Ready to commit with message:
```
fix(cmw-platform): add button kind validator to map LLM terms to API

- Add @field_validator("kind") to map "Trigger scenario" → "UserEvent"
- Support all 29 valid button kinds from CMW Platform API
- Case-insensitive mapping (trigger_scenario, TRIGGER SCENARIO)
- Validate against API enum, reject invalid kinds (e.g., "Test")
- Update skill documentation with all 29 button kinds
- Add 17 comprehensive unit tests
- Fix guard logic: check "UserEvent" not "Trigger scenario"
- Remove invalid "Test" kind from documentation

Fixes regression from commit 8d9e16c where terminology change
lacked validator, causing API to reject "Trigger scenario" literal.

Tested: Unit tests (17/17), integration tests on real platform,
existing buttons verified working.
```
