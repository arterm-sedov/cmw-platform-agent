# CMW Form Handling

## Source of Truth

Use `webapi/Form` and `cmw_open_api/web_api_v1.json` for forms. Do not use Solution API for form structure changes.

Relevant endpoints:

- `GET webapi/Form/{app}/Form@{template}.{form}`
- `DELETE webapi/Form/{app}/Form@{template}.{form}`
- `POST webapi/Form/{app}`
- `PUT webapi/Form/{app}`
- `GET webapi/Form/List/Template@{app}.{template}`

## Critical Rule

Treat `widgets` as desired visible fields to upsert, not as edit-only metadata. A form operation is not complete until `get_form` returns `FieldComponent` nodes.

## Empty Forms

If a form has no `FieldComponent` nodes, editing existing widgets cannot make fields visible. Build or replace the full `root` tree.

Working tree shape:

```text
VerticalLayout
  PanelModel
    VerticalLayout
      LayoutModel.components
        FieldComponent
```

## Structural Replacement

Use this flow for root/widgets structural changes:

```text
GET current -> save BEFORE -> DELETE -> POST -> GET AFTER -> verify
```

If POST fails after DELETE, restore the previous form body with POST and report whether rollback succeeded.

## Verification

API verification:

- Count `FieldComponent` nodes.
- Inspect `propertyPath` aliases.
- Confirm referenced attributes exist when a known attribute set is available.

UI verification:

- Use browser only after API verification.
- Hard refresh if the platform UI shows stale form structure.
- Capture a screenshot for visual proof when needed.
