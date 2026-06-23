"""HTTP helpers for CMW form operations."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from tools import requests_

FORM_ENDPOINT = "webapi/Form"
DEFAULT_SNAPSHOT_DIR = Path("cmw-platform-workspace") / "form_snapshots"


def _response_payload(result: dict[str, Any]) -> Any:
    raw_response = result.get("raw_response")
    if isinstance(raw_response, dict) and "response" in raw_response:
        return raw_response["response"]
    return raw_response


def _result(
    *,
    success: bool,
    status_code: int,
    data: Any = None,
    error: Any = None,
    **extra: Any,
) -> dict[str, Any]:
    payload = {
        "success": success,
        "status_code": status_code,
        "data": data,
        "error": error,
    }
    payload.update(extra)
    return payload


def get_form_raw(app: str, template: str, form: str) -> dict[str, Any] | None:
    """Fetch a raw form model."""
    endpoint = f"{FORM_ENDPOINT}/{app}/Form@{template}.{form}"
    result = requests_._get_request(endpoint)
    if not result.get("success"):
        return None
    payload = _response_payload(result)
    return payload if isinstance(payload, dict) else None


def list_forms_raw(app: str, template: str) -> list[dict[str, Any]]:
    """Fetch raw forms for a template."""
    endpoint = f"{FORM_ENDPOINT}/List/Template@{app}.{template}"
    result = requests_._get_request(endpoint)
    if not result.get("success"):
        return []
    payload = _response_payload(result)
    return payload if isinstance(payload, list) else []


def post_form(app: str, body: dict[str, Any]) -> dict[str, Any]:
    """POST a form body."""
    result = requests_._post_request(body, f"{FORM_ENDPOINT}/{app}")
    return _result(
        success=bool(result.get("success")),
        status_code=int(result.get("status_code") or 0),
        data=_response_payload(result),
        error=result.get("error"),
        raw_response=result.get("raw_response"),
        body=result.get("body"),
    )


def delete_form(app: str, template: str, form: str) -> dict[str, Any]:
    """DELETE a form by global alias."""
    endpoint = f"{FORM_ENDPOINT}/{app}/Form@{template}.{form}"
    result = requests_._delete_request(endpoint)
    return _result(
        success=bool(result.get("success")),
        status_code=int(result.get("status_code") or 0),
        data=_response_payload(result),
        error=result.get("error"),
        raw_response=result.get("raw_response"),
    )


def write_form_snapshot(
    *,
    app: str,
    template: str,
    form: str,
    stage: str,
    data: dict[str, Any],
    workspace: Path | None = None,
) -> Path:
    """Write a form snapshot for rollback/audit."""
    root = workspace or DEFAULT_SNAPSHOT_DIR
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = root / f"{timestamp}_{app}_{template}_{form}_{stage}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def replace_form_delete_post(
    *,
    app: str,
    template: str,
    form: str,
    new_form: dict[str, Any],
    workspace: Path | None = None,
) -> dict[str, Any]:
    """Replace a form by GET/snapshot/DELETE/POST with rollback on POST failure."""
    current = get_form_raw(app, template, form)
    snapshots: dict[str, str] = {}
    if current is not None:
        snapshots["before"] = str(
            write_form_snapshot(
                app=app,
                template=template,
                form=form,
                stage="before",
                data=current,
                workspace=workspace,
            )
        )
    snapshots["attempted"] = str(
        write_form_snapshot(
            app=app,
            template=template,
            form=form,
            stage="attempted",
            data=new_form,
            workspace=workspace,
        )
    )

    if current is not None:
        deleted = delete_form(app, template, form)
        if not deleted.get("success"):
            return _result(
                success=False,
                status_code=int(deleted.get("status_code") or 0),
                data={"snapshots": snapshots},
                error=deleted.get("error") or "Failed to delete existing form",
            )

    posted = post_form(app, deepcopy(new_form))
    if not posted.get("success"):
        rollback_result = None
        rollback_success = False
        if current is not None:
            rollback_result = post_form(app, current)
            rollback_success = bool(rollback_result.get("success"))
        return _result(
            success=False,
            status_code=int(posted.get("status_code") or 0),
            data={"post": posted, "rollback": rollback_result, "snapshots": snapshots},
            error=posted.get("error") or "Failed to post replacement form",
            rollback_success=rollback_success,
        )

    after = get_form_raw(app, template, form)
    if isinstance(after, dict):
        snapshots["after"] = str(
            write_form_snapshot(
                app=app,
                template=template,
                form=form,
                stage="after",
                data=after,
                workspace=workspace,
            )
        )
    return _result(
        success=True,
        status_code=int(posted.get("status_code") or 200),
        data=after,
        error=None,
        snapshots=snapshots,
    )
