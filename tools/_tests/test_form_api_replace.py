from tools.templates_tools import form_api


def test_replace_form_delete_post_rolls_back_after_post_failure(monkeypatch, tmp_path):
    old_form = {"globalAlias": {"alias": "defaultForm"}, "root": {"rows": []}}
    calls = []

    def fake_get_form_raw(*_args):
        return old_form

    def fake_delete_form(*_args):
        return {"success": True, "status_code": 200, "data": None, "error": None}

    monkeypatch.setattr(form_api, "get_form_raw", fake_get_form_raw)
    monkeypatch.setattr(
        form_api,
        "delete_form",
        fake_delete_form,
    )

    def fake_post(app, body):
        calls.append(body)
        if len(calls) == 1:
            return {"success": False, "status_code": 500, "data": None, "error": "boom"}
        return {"success": True, "status_code": 200, "data": body, "error": None}

    monkeypatch.setattr(form_api, "post_form", fake_post)

    result = form_api.replace_form_delete_post(
        app="App",
        template="Template",
        form="defaultForm",
        new_form={"root": {"rows": [{"type": "FieldComponent"}]}},
        workspace=tmp_path,
    )

    assert result["success"] is False
    assert result["rollback_success"] is True
    assert calls[1] == old_form
    assert "before" in result["data"]["snapshots"]
