# tool_export_application.py - Export application to CTF format
from __future__ import annotations

from ..tool_utils import *
from .transfer_models import ExportApplicationSchema

TRANSFER_ENDPOINT = "webapi/Transfer"


@tool("export_application", return_direct=False, args_schema=ExportApplicationSchema)
def export_application(
    application_system_name: str,
    save_to_file: bool = True,
) -> dict[str, Any]:
    """
    Export a CMW Platform application to CTF (Comindware Transfer Format).

    Exports the entire application including all templates, attributes, workflows,
    and other configurations. The export is returned as Base64-encoded CTF data.

    Args:
        application_system_name: System name (alias) of the application to export.
        save_to_file: If True, saves the CTF file to disk and returns the file path.
                      If False, returns the Base64-encoded CTF data directly.

    Returns:
        dict: {
            "success": bool - True if export was successful,
            "status_code": int - HTTP response status code,
            "ctf_data": str - Base64-encoded CTF data,
            "ctf_file_path": str - Path to saved CTF file (if save_to_file=True),
            "error": str|None - Error message if operation failed
        }
    """
    import base64

    from .transfer_utils import TransferUtils

    endpoint = f"{TRANSFER_ENDPOINT}/{application_system_name}"

    try:
        result = requests_._get_request(endpoint)
    except Exception as e:
        return {
            "success": False,
            "status_code": 500,
            "error": f"Export failed: {e!s}",
        }

    if not result.get("success", False):
        error_msg = result.get("error", "Unknown error")
        return {
            "success": False,
            "status_code": result.get("status_code", 500),
            "error": f"API export failed: {error_msg}",
        }

    raw_response = result.get("raw_response")
    if isinstance(raw_response, dict):
        response_data = raw_response.get("response", {})
        if isinstance(response_data, dict):
            ctf_data = response_data.get("data", "") or response_data.get("ctf", "")
            summary = response_data.get("summary", {})
            result_message = summary.get("resultMessage", "Export completed")
        else:
            ctf_data = str(response_data) if response_data else ""
            result_message = "Export completed"
    elif isinstance(raw_response, str):
        ctf_data = raw_response
        result_message = "Export completed"
    else:
        ctf_data = str(raw_response) if raw_response else ""
        result_message = "Export completed"

    response: dict[str, Any] = {
        "success": True,
        "status_code": result.get("status_code", 200),
        "ctf_data": ctf_data,
        "result_message": result_message,
    }

    if save_to_file and ctf_data:
        file_path = TransferUtils.save_ctf(ctf_data, application_system_name)
        response["ctf_file_path"] = file_path

    return response
