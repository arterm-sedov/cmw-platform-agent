# CMW Form Handling Tools Plan

## Decision

Use `webapi/Form` and `cmw_open_api/web_api_v1.json` for form operations. Do not use Solution API for forms.

## OpenAPI Audit Result

Local `cmw_open_api/web_api_v1.json` exposes these form paths:

- `/webapi/Form/{solutionAlias}`: `PUT`, `POST`
- `/webapi/Form/{solutionAlias}/{formGlobalAlias}`: `GET`, `DELETE`
- `/webapi/Form/FormRules/{solutionAlias}/{formGlobalAlias}`: `GET`, `PUT`
- `/webapi/Form/List/{templateGlobalAlias}`: `GET`

Relevant model terms include `Comindware.Platform.Contracts.FormModel`, `VerticalLayoutModel`, `FormComponentModel`, and component enum values such as `VerticalLayout`, `PanelModel`, `LayoutModel`, and `FieldComponent`.

## Problem

`edit_or_create_form.widgets` only edits existing nodes whose aliases are already present. Empty forms can return API success while still containing no visible `FieldComponent` nodes.

## Goal

Make form operations create, edit, copy, upsert, verify, and document visible `FieldComponent`-based forms.

## Scope

- `tools/templates_tools/tools_form.py`
- Form helpers under `tools/templates_tools/`
- Regression tests under `tools/_tests/`
- Generic CMW form documentation under `.agents/skills/cmw-platform`

## Non-goals

- No Solution API for forms
- No blind PUT-only structural form replacement
- No tenant-specific data in committed fixtures or skills

## Definition of Done

- Empty form plus `widgets` creates visible fields or fails clearly.
- Existing widget edits still work.
- Create-from-attributes excludes system attributes.
- Copy-form rewrites source tokens and preserves `FieldComponent` count.
- DELETE + POST replacement is backed up and rollback-capable.
- Ruff, relevant pytest checks, lint, and type checks are run or documented if blocked.
- Skill docs describe the reusable form workflow.
