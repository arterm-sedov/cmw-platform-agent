import json
from pathlib import Path

from tools.templates_tools.form_structure import count_field_components
import tools.templates_tools.tools_form as tools_form_module
from tools.templates_tools.tools_form import EditOrCreateFormSchema, edit_or_create_form

FIXTURES = Path(__file__).parent / "fixtures" / "forms"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_widgets_accept_json_string():
    schema = EditOrCreateFormSchema(
        operation="edit",
        application_system_name="App",
        template_system_name="TargetTemplate",
        form_system_name="defaultForm",
        widgets='{"BusinessName": {"label": "Business name"}}',
    )

    assert schema.widgets == {"BusinessName": {"label": "Business name"}}


def test_edit_or_create_form_widgets_on_empty_form_builds_root(monkeypatch):
    empty_form = load_fixture("form_empty.json")
    captured = {}

    def fake_get_empty_form(*_args):
        return empty_form

    monkeypatch.setattr(tools_form_module, "get_form_raw", fake_get_empty_form)

    def fake_replace(**kwargs):
        captured.update(kwargs)
        return {
            "success": True,
            "status_code": 200,
            "data": kwargs["new_form"],
            "error": None,
        }

    monkeypatch.setattr(tools_form_module, "replace_form_delete_post", fake_replace)

    result = edit_or_create_form.invoke(
        {
            "operation": "edit",
            "application_system_name": "App",
            "template_system_name": "TargetTemplate",
            "form_system_name": "defaultForm",
            "widgets": {"BusinessName": {"label": "Business name"}},
        }
    )

    assert result["success"] is True
    assert count_field_components(captured["new_form"]) == 1


def test_edit_or_create_form_rejects_saved_empty_form(monkeypatch):
    def fake_get_missing_form(*_args):
        return None

    def fake_empty_replace(**_kwargs):
        return {
            "success": True,
            "status_code": 200,
            "data": {"root": {"rows": []}},
            "error": None,
        }

    monkeypatch.setattr(tools_form_module, "get_form_raw", fake_get_missing_form)
    monkeypatch.setattr(
        tools_form_module,
        "replace_form_delete_post",
        fake_empty_replace,
    )

    result = edit_or_create_form.invoke(
        {
            "operation": "edit",
            "application_system_name": "App",
            "template_system_name": "TargetTemplate",
            "form_system_name": "defaultForm",
            "widgets": {"BusinessName": {"label": "Business name"}},
        }
    )

    assert result["success"] is False
    assert result["status_code"] == 422
    assert "FieldComponent" in result["error"]


def test_existing_widget_edit_and_missing_widget_append(monkeypatch):
    form = load_fixture("form_with_fields.json")
    captured = {}

    def fake_get_existing_form(*_args):
        return form

    monkeypatch.setattr(tools_form_module, "get_form_raw", fake_get_existing_form)

    def fake_replace(**kwargs):
        captured.update(kwargs)
        return {
            "success": True,
            "status_code": 200,
            "data": kwargs["new_form"],
            "error": None,
        }

    monkeypatch.setattr(tools_form_module, "replace_form_delete_post", fake_replace)

    result = edit_or_create_form.invoke(
        {
            "operation": "edit",
            "application_system_name": "App",
            "template_system_name": "TargetTemplate",
            "form_system_name": "defaultForm",
            "widgets": {
                "BusinessName": {"label": "Renamed"},
                "Comment": {"label": "Comment"},
            },
        }
    )

    assert result["success"] is True
    assert count_field_components(captured["new_form"]) == 2
    assert "Renamed" in json.dumps(captured["new_form"], ensure_ascii=False)
