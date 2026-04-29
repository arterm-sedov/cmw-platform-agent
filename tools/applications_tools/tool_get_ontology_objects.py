# tool_get_ontology_objects.py - Get all objects by types for localization
from __future__ import annotations

import re

import requests

from ..tool_utils import *

try:
    from pydantic import Field
except ImportError:
    from tools.models import Field


TYPE_PREDICATE_MAPPING: dict[str, dict[str, str]] = {
    "RecordTemplate": {"alias": "cmw.container.alias"},
    "AccountTemplate": {"alias": "cmw.container.alias"},
    "ProcessTemplate": {"alias": "cmw.container.alias"},
    "RoleTemplate": {"alias": "cmw.container.alias"},
    "OrgStructureTemplate": {"alias": "cmw.container.alias"},
    "MessageTemplate": {"alias": "cmw.message.type.alias"},
    "Workspace": {"alias": "cmw.alias"},
    "Page": {"alias": "cmw.desktopPage.alias"},
    "Attribute": {"alias": "cmw.object.alias"},
    "Dataset": {"alias": "cmw.alias"},
    "Toolbar": {"alias": "cmw.alias"},
    "Form": {"alias": "cmw.alias"},
    "UserCommand": {"alias": "cmw.alias"},
    "Card": {"alias": "cmw.alias"},
    "Cart": {"alias": "cmw.cart.alias"},
    "Trigger": {"alias": "cmw.trigger.alias"},
    "Routes": {"alias": "cmw.procedure.name"},
    "Role": {"alias": "cmw.role.alias", "aliasProperty": "cmw.role.aliasProperty"},
    "WidgetConfig": {"alias": "cmw.form.alias"},
    "DesktopWidgetConfig": {"alias": "cmw.desktopPage.widget.config.alias"},
    "ExportTemplate": {"alias": "cmw.alias"},
    "DesktopComponent": {"alias": "cmw.desktopPage.component.alias"},
}

TYPE_PREFIX_MAPPING: dict[str, list[str]] = {
    "MessageTemplate": ["msgt."],
    "Workspace": ["workspace."],
    "Dataset": ["lst."],
    "Toolbar": ["tb."],
    "UserCommand": ["event."],
    "Form": ["form."],
    "Card": ["card."],
    "RecordTemplate": ["oa."],
    "ProcessTemplate": ["pa."],
    "RoleTemplate": ["ra."],
    "AccountTemplate": ["aa."],
    "OrgStructureTemplate": ["os."],
    "Cart": ["cart."],
    "Trigger": ["trigger."],
    "Role": ["role."],
    "WidgetConfig": ["fw."],
    "DesktopWidgetConfig": ["dwc."],
    "ExportTemplate": ["exportTemplate."],
    "DesktopComponent": ["component."],
}

DEFAULT_PARAMETER = "alias"


class GetOntologyObjectsSchema(BaseModel):
    application_system_name: str = Field(
        description="System name of the application to query"
    )
    types: list[str] = Field(
        description=f"Object types to fetch. Supported: {list(TYPE_PREDICATE_MAPPING.keys())}"
    )
    parameter: str = Field(
        default=DEFAULT_PARAMETER,
        description="Parameter to search by (e.g., 'alias', 'name'). Must exist in type mapping."
    )
    min_count: int = Field(
        default=1,
        description="Minimum number of objects to return per type"
    )
    max_count: int = Field(
        default=10000,
        description="Maximum number of objects to return per type"
    )


def extract_id(item_id: str) -> str:
    if match := re.match(r"^([\w.]+)\s*:", item_id):
        return match.group(1)
    return item_id


def extract_system_name(item: Any) -> str:
    if isinstance(item, dict):
        value = item.get("value", [])
        if isinstance(value, list) and value:
            return value[0]
    return ""


def get_type_by_prefix(item_id: str, type_prefixes: dict[str, list[str]]) -> str | None:
    for obj_type, prefixes in type_prefixes.items():
        for prefix in prefixes:
            if item_id.startswith(prefix):
                return obj_type
    return None


def get_axioms_by_predicate(object_id: str, predicate: str) -> list[str]:
    """
    Call /Base/OntologyService/GetAxiomsByPredicate to resolve property values.

    Used for Role objects where cmw.role.aliasProperty contains an attribute ID
    that needs to be resolved to get the actual alias value.

    Example:
        Request: {"id": "role.2", "predicate": "op.2"}
        Response: ["Администратор"]

    Args:
        object_id: Object ID (e.g., "role.2")
        predicate: Predicate/attribute ID (e.g., "op.2")

    Returns:
        List of values, empty list on error
    """
    cfg = requests_._load_server_config()
    base_url = cfg.base_url.rstrip("/")
    headers = requests_._basic_headers()

    endpoint = f"{base_url}/api/public/system/Base/OntologyService/GetAxiomsByPredicate"

    request_body = {
        "id": object_id,
        "predicate": predicate
    }

    try:
        resp = requests.post(endpoint, headers=headers, json=request_body, timeout=cfg.timeout)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                return data
        return []
    except Exception:
        return []


@tool("get_ontology_objects", return_direct=False, args_schema=GetOntologyObjectsSchema)
def get_ontology_objects(
    application_system_name: str,
    types: list[str],
    parameter: str = DEFAULT_PARAMETER,
    min_count: int = 1,
    max_count: int = 10000,
) -> dict[str, Any]:
    """
    Get all object IDs by their types for localization workflow.

    Uses POST /api/public/system/Base/OntologyService/GetWithMultipleValues endpoint.
    Results are filtered by type-specific prefixes and deduplicated.

    Args:
        application_system_name: System name of the application
        types: List of object types to fetch
        parameter: Parameter to search by. Default: 'alias'. For Routes use 'name'.
        min_count: Minimum number of objects per type (default: 1)
        max_count: Maximum number of objects per type (default: 10000)

    Returns:
        dict: {
            "success": bool,
            "data": [
                {"type": "Form", "id": "form.2024", "systemName": "form.2024"},
                ...
            ],
            "errors": {},
            "total_count": int
        }
    """
    cfg = requests_._load_server_config()
    base_url = cfg.base_url.rstrip("/")
    headers = requests_._basic_headers()

    endpoint = f"{base_url}/api/public/system/Base/OntologyService/GetWithMultipleValues"

    raw_results: list[dict[str, Any]] = []
    errors: dict[str, str] = {}

    for obj_type in types:
        if obj_type not in TYPE_PREDICATE_MAPPING:
            errors[obj_type] = f"Unknown type: {obj_type}. Supported: {list(TYPE_PREDICATE_MAPPING.keys())}"
            continue

        type_mapping = TYPE_PREDICATE_MAPPING[obj_type]
        if parameter not in type_mapping:
            errors[obj_type] = f"Parameter '{parameter}' not available for {obj_type}. Available: {list(type_mapping.keys())}"
            continue

        # Special handling for Role type with both alias and aliasProperty
        if obj_type == "Role" and parameter == "alias":
            # Query cmw.role.alias
            predicate_alias = type_mapping["alias"]
            request_body_alias: dict[str, Any] = {
                "predicate": predicate_alias,
                "min": min_count,
                "max": max_count,
            }

            try:
                resp = requests.post(endpoint, headers=headers, json=request_body_alias, timeout=cfg.timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                item_id = item.get("key", "") or item.get("id", "")
                                clean_id = extract_id(item_id)
                                system_name = extract_system_name(item)
                                raw_results.append({
                                    "id": clean_id,
                                    "systemName": system_name,
                                    "original_type": obj_type,
                                })
                            elif isinstance(item, str):
                                clean_id = extract_id(item)
                                raw_results.append({
                                    "id": clean_id,
                                    "systemName": clean_id,
                                    "original_type": obj_type,
                                })
                    elif isinstance(data, dict):
                        items = data.get("items", []) or data.get("data", []) or data.get("results", [])
                        for item in items:
                            if isinstance(item, dict):
                                item_id = item.get("key", "") or item.get("id", "")
                                clean_id = extract_id(item_id)
                                system_name = extract_system_name(item)
                                raw_results.append({
                                    "id": clean_id,
                                    "systemName": system_name,
                                    "original_type": obj_type,
                                })
                            elif isinstance(item, str):
                                clean_id = extract_id(item)
                                raw_results.append({
                                    "id": clean_id,
                                    "systemName": clean_id,
                                    "original_type": obj_type,
                                })
            except requests.RequestException as e:
                errors[f"{obj_type}_alias"] = str(e)
            except Exception as e:
                errors[f"{obj_type}_alias"] = f"Parse error: {e}"

            # Query cmw.role.aliasProperty
            predicate_alias_property = type_mapping["aliasProperty"]
            request_body_alias_property: dict[str, Any] = {
                "predicate": predicate_alias_property,
                "min": min_count,
                "max": max_count,
            }

            try:
                resp = requests.post(endpoint, headers=headers, json=request_body_alias_property, timeout=cfg.timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    items_to_process = []

                    if isinstance(data, list):
                        items_to_process = data
                    elif isinstance(data, dict):
                        items_to_process = data.get("items", []) or data.get("data", []) or data.get("results", [])

                    for item in items_to_process:
                        if isinstance(item, dict):
                            item_id = item.get("key", "") or item.get("id", "")
                            clean_id = extract_id(item_id)
                            attribute_id = extract_system_name(item)

                            if attribute_id:
                                # Resolve alias via GetAxiomsByPredicate
                                resolved_values = get_axioms_by_predicate(clean_id, attribute_id)
                                if resolved_values:
                                    raw_results.append({
                                        "id": clean_id,
                                        "systemName": resolved_values[0],
                                        "original_type": obj_type,
                                    })
                                # Skip if resolution failed (as per requirements)
            except requests.RequestException as e:
                errors[f"{obj_type}_aliasProperty"] = str(e)
            except Exception as e:
                errors[f"{obj_type}_aliasProperty"] = f"Parse error: {e}"

            continue

        # Standard handling for all other types
        predicate = type_mapping[parameter]

        request_body: dict[str, Any] = {
            "predicate": predicate,
            "min": min_count,
            "max": max_count,
        }

        try:
            resp = requests.post(endpoint, headers=headers, json=request_body, timeout=cfg.timeout)
            if resp.status_code != 200:
                errors[obj_type] = f"HTTP {resp.status_code}: {resp.text}"
                continue

            data = resp.json()

            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        item_id = item.get("key", "") or item.get("id", "")
                        clean_id = extract_id(item_id)
                        system_name = extract_system_name(item)
                        raw_results.append({
                            "id": clean_id,
                            "systemName": system_name,
                            "original_type": obj_type,
                        })
                    elif isinstance(item, str):
                        clean_id = extract_id(item)
                        raw_results.append({
                            "id": clean_id,
                            "systemName": clean_id,
                            "original_type": obj_type,
                        })
            elif isinstance(data, dict):
                items = data.get("items", []) or data.get("data", []) or data.get("results", [])
                for item in items:
                    if isinstance(item, dict):
                        item_id = item.get("key", "") or item.get("id", "")
                        clean_id = extract_id(item_id)
                        system_name = extract_system_name(item)
                        raw_results.append({
                            "id": clean_id,
                            "systemName": system_name,
                            "original_type": obj_type,
                        })
                    elif isinstance(item, str):
                        clean_id = extract_id(item)
                        raw_results.append({
                            "id": clean_id,
                            "systemName": clean_id,
                            "original_type": obj_type,
                        })

        except requests.RequestException as e:
            errors[obj_type] = str(e)
        except Exception as e:
            errors[obj_type] = f"Parse error: {e}"

    results: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for item in raw_results:
        item_id = item["id"]
        original_type = item["original_type"]
        stored_system_name = item.get("systemName", "")

        if original_type in TYPE_PREFIX_MAPPING:
            prefixes = TYPE_PREFIX_MAPPING[original_type]
            matched_type = None
            for prefix in prefixes:
                if item_id.startswith(prefix):
                    matched_type = original_type
                    break

            if matched_type is None:
                continue
        else:
            matched_type = original_type

        if item_id in seen_ids:
            continue
        seen_ids.add(item_id)

        results.append({
            "type": matched_type,
            "id": item_id,
            "systemName": stored_system_name if stored_system_name else item_id,
        })

    return {
        "success": len(errors) == 0,
        "data": results,
        "errors": errors if errors else {},
        "total_count": len(results),
    }


def get_references(
    object_id: str = Field(..., description="Object ID to get references for"),
    predicate: str = Field(..., description="Predicate to query (e.g., cmw.solution.cart)"),
    application_system_name: str = Field(default="", description="Application system name"),
) -> dict:
    """
    Get references for an object by predicate.

    Used to verify object ownership (e.g., verify Cart belongs to a Solution).

    Args:
        object_id: Object ID (e.g., "cart.1")
        predicate: Predicate to query (e.g., "cmw.solution.cart")
        application_system_name: Optional application system name

    Returns:
        Dict with references, e.g., {"cmw.solution.cart": ["sln.1"]}
    """
    base_url = os.environ.get("CMW_BASE_URL", "")
    if not base_url:
        return {"success": False, "error": "CMW_BASE_URL not set"}

    session = get_session()
    url = f"{base_url}/api/public/system/Base/OntologyService/GetReferences"

    payload = {
        "id": object_id,
        "predicate": predicate,
    }

    if application_system_name:
        payload["applicationSystemName"] = application_system_name

    try:
        response = session.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()

        return {
            "success": True,
            "data": data,
            "status_code": response.status_code,
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": str(e),
            "status_code": getattr(response, "status_code", 0) if hasattr(response, "status_code") else 0,
        }


if __name__ == "__main__":
    result = get_ontology_objects.invoke({
        "application_system_name": "supportTest",
        "types": ["Form", "Toolbar", "Dataset", "UserCommand", "RecordTemplate"],
        "parameter": "alias",
        "min_count": 1,
        "max_count": 10000,
    })
    print(result)
