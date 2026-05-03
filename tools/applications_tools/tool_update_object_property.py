# tool_update_object_property.py - Update object property by predicate
from __future__ import annotations

import requests

from ..tool_utils import *

try:
    from pydantic import Field
except ImportError:
    from tools.models import Field


TYPE_PREDICATE_MAPPING: dict[str, str] = {
    "RecordTemplate": "cmw.container.alias",
    "ProcessTemplate": "cmw.container.alias",
    "RoleTemplate": "cmw.container.alias",
    "AccountTemplate": "cmw.container.alias",
    "OrgStructureTemplate": "cmw.container.alias",
    "MessageTemplate": "cmw.message.type.alias",
    "Workspace": "cmw.alias",
    "Page": "cmw.desktopPage.alias",
    "Attribute": "cmw.object.alias",
    "Dataset": "cmw.alias",
    "Toolbar": "cmw.alias",
    "Form": "cmw.alias",
    "UserCommand": "cmw.alias",
    "Card": "cmw.alias",
    "Cart": "cmw.cart.alias",
    "Trigger": "cmw.trigger.alias",
    "Role": "cmw.role.alias",
    "WidgetConfig": "cmw.form.alias",
}


class UpdateObjectPropertySchema(BaseModel):
    object_id: str = Field(
        description="Object ID (e.g., 'form.338', 'oa.230')"
    )
    object_type: str = Field(
        description=f"Object type. Supported: {list(TYPE_PREDICATE_MAPPING.keys())}"
    )
    new_value: str = Field(
        description="New value for the property"
    )


@tool("update_object_property", return_direct=False, args_schema=UpdateObjectPropertySchema)
def update_object_property(
    object_id: str,
    object_type: str,
    new_value: str,
) -> dict[str, Any]:
    """
    Update an object property by predicate using POST /Base/OntologyService/AddStatement.

    Args:
        object_id: Object ID (e.g., 'form.338', 'oa.230')
        object_type: Object type (RecordTemplate, Form, Dataset, etc.)
        new_value: New value for the property

    Returns:
        dict: {
            "success": bool,
            "object_id": str,
            "predicate": str,
            "new_value": str,
            "error": str|None
        }
    """
    if object_type not in TYPE_PREDICATE_MAPPING:
        return {
            "success": False,
            "object_id": object_id,
            "error": f"Unknown type: {object_type}. Supported: {list(TYPE_PREDICATE_MAPPING.keys())}",
        }

    predicate = TYPE_PREDICATE_MAPPING[object_type]

    cfg = requests_._load_server_config()
    base_url = cfg.base_url.rstrip("/")
    headers = requests_._basic_headers()

    endpoint = f"{base_url}/api/public/system/Base/OntologyService/AddStatement"

    request_body: dict[str, Any] = {
        "subject": object_id,
        "predicate": predicate,
        "value": new_value,
    }

    try:
        resp = requests.post(endpoint, headers=headers, json=request_body, timeout=cfg.timeout)
        if resp.status_code != 200:
            return {
                "success": False,
                "object_id": object_id,
                "predicate": predicate,
                "new_value": new_value,
                "error": f"HTTP {resp.status_code}: {resp.text}",
            }

        data = resp.json()
        return {
            "success": True,
            "object_id": object_id,
            "object_type": object_type,
            "predicate": predicate,
            "new_value": new_value,
            "response": data,
        }

    except requests.RequestException as e:
        return {
            "success": False,
            "object_id": object_id,
            "predicate": predicate,
            "new_value": new_value,
            "error": f"Request error: {e}",
        }
    except Exception as e:
        return {
            "success": False,
            "object_id": object_id,
            "predicate": predicate,
            "new_value": new_value,
            "error": f"Error: {e}",
        }


if __name__ == "__main__":
    result = update_object_property.invoke({
        "object_id": "form.338",
        "object_type": "Form",
        "new_value": "Migration Form",
    })
    print(result)
