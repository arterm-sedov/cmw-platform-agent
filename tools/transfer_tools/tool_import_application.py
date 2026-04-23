from __future__ import annotations

import json

from ..requests_ import _check_response_for_errors
from ..requests_models import HTTPResponse
from ..tool_utils import *
from .transfer_models import ImportApplicationSchema

TRANSFER_ENDPOINT = "webapi/Transfer"


@tool("import_application", return_direct=False, args_schema=ImportApplicationSchema)
def import_application(
    application_system_name: str,
    ctf_data: str | None = None,
    ctf_file_path: str | None = None,
) -> dict[str, Any]:
    """
    Import a CMW Platform application from CTF (Comindware Transfer Format).

    The import process consists of two steps:
    1. Upload the CTF data to get a file ID
    2. Import the application using the file ID

    Args:
        application_system_name: System name (alias) for the application.
        ctf_data: Base64-encoded CTF data. Required if ctf_file_path not provided.
        ctf_file_path: Path to a local CTF file. If provided, CTF data will be read from it.
                      Takes precedence over ctf_data if both are provided.

    Returns:
        dict: {
            "success": bool - True if import was successful,
            "status_code": int - HTTP response status code,
            "file_id": str - Uploaded file ID,
            "validation_errors": list - Validation errors if import failed,
            "error": str|None - Error message if operation failed
        }
    """
    from .transfer_utils import TransferUtils

    if ctf_file_path:
        ctf_data = TransferUtils.read_ctf_from_file(ctf_file_path)
        if not ctf_data:
            return {
                "success": False,
                "status_code": 400,
                "error": f"Failed to read CTF data from file: {ctf_file_path}",
            }
    elif not ctf_data:
        return {
            "success": False,
            "status_code": 400,
            "error": "Either ctf_data or ctf_file_path must be provided",
        }

    upload_result = _upload_ctf(ctf_data)
    if not upload_result.get("success"):
        return upload_result

    file_id = upload_result.get("file_id", "")
    if not file_id:
        return {
            "success": False,
            "status_code": 500,
            "error": "No file ID returned from upload",
        }

    return _execute_import(application_system_name, file_id)


def _upload_ctf(ctf_data: str) -> dict[str, Any]:
    """Upload CTF data and return file ID. Sends base64 string as JSON body."""
    import base64
    import os

    import requests

    base_url = os.environ.get("CMW_BASE_URL", "").rstrip("/")
    login = os.environ.get("CMW_LOGIN", "")
    password = os.environ.get("CMW_PASSWORD", "")

    if not base_url or not login or not password:
        return {
            "success": False,
            "status_code": 500,
            "error": "Missing CMW credentials",
        }

    creds = base64.b64encode(f"{login}:{password}".encode()).decode()
    headers = {"Authorization": f"Basic {creds}", "Content-Type": "application/json"}

    url = f"{base_url}/webapi/Transfer/Upload"

    try:
        response = requests.post(
            url,
            headers=headers,
            json=ctf_data,
            timeout=60,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "status_code": 500,
            "error": f"Upload failed: {e!s}",
        }

    try:
        raw_response = response.json()
    except Exception:
        raw_response = response.text

    http_response = HTTPResponse(
        success=response.status_code == 200,
        status_code=response.status_code,
        raw_response=raw_response,
        error=None,
        base_url=url,
    )

    if http_response.success and isinstance(raw_response, dict):
        api_error = _check_response_for_errors(response.text)
        if api_error:
            http_response.error = api_error
            http_response.success = False

    if not http_response.success:
        error_msg = http_response.error or "Unknown error"
        return {
            "success": False,
            "status_code": http_response.status_code,
            "error": f"CTF upload failed: {error_msg}",
        }

    response_data = http_response.raw_response
    if isinstance(response_data, dict):
        file_id = response_data.get("response", "") or response_data.get("fileId", "")
    else:
        file_id = str(response_data) if response_data else ""

    return {
        "success": True,
        "status_code": http_response.status_code,
        "file_id": file_id,
    }


def _execute_import(
    application_system_name: str, file_id: str
) -> dict[str, Any]:
    """Execute the import operation using file ID."""
    endpoint = (
        f"{TRANSFER_ENDPOINT}/{application_system_name}/{file_id}/true/ApplyNew"
    )

    try:
        result = requests_._post_request({}, endpoint)
    except Exception as e:
        return {
            "success": False,
            "status_code": 500,
            "error": f"Import execution failed: {e!s}",
        }

    if not result.get("success", False):
        error_msg = result.get("error", "Unknown error")
        return {
            "success": False,
            "status_code": result.get("status_code", 500),
            "error": f"Application import failed: {error_msg}",
        }

    raw_response = result.get("raw_response", {})
    response_data = raw_response if isinstance(raw_response, dict) else {}
    response_inner = response_data.get("response", response_data)
    validation_errors: list[dict[str, Any]] | None = (
        response_inner.get("validationErrors") or response_inner.get("errors")
    )

    return {
        "success": True,
        "status_code": result.get("status_code", 200),
        "file_id": file_id,
        "validation_errors": validation_errors,
    }
