import ast
import logging
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from tools import requests_
from tools.platform_entity_resolver import _ID_PREFIX_MAP, _resolve_entity_id
from tools.tool_utils import APPLICATION_RESPONSE_MAPPING, TEMPLATE_RESPONSE_MAPPING

logger = logging.getLogger(__name__)

_ENTITY_TYPES = ["Record", "Role", "Process", "OrgStructure", "Undefined"]


class GetPlatformEntityUrlSchema(BaseModel):
    """Schema for getting platform entity URL by ID or system name."""

    entity_id: str | None = Field(
        default=None,
        description=(
            "Platform entity ID to resolve. "
            "Examples: 'oa.193', 'event.15199', 'sln.13', '12345' (record). "
            "RU: ID сущности платформы для получения URL"
        ),
    )
    system_name: str | None = Field(
        default=None,
        description=(
            "System name of the entity to look up across all applications. "
            "Works for templates, applications, and roles. "
            "For attributes, buttons, forms, datasets, toolbars — use entity_id instead. "
            "Examples: 'ServiceRequests', 'approve_request'. "
            "RU: Системное имя сущности для поиска"
        ),
    )
    application: str | None = Field(
        default=None,
        description=(
            "Application system name to filter system_name lookup. "
            "Example: 'CustomerPortal'. "
            "RU: Системное имя приложения для фильтрации"
        ),
    )

    @field_validator("entity_id", "system_name", "application", mode="before")
    @classmethod
    def non_empty_str(cls, v: Any) -> Any:
        if isinstance(v, str) and v.strip() == "":
            raise ValueError("Value must be a non-empty string")
        return v


@tool("get_platform_entity_url", return_direct=False, args_schema=GetPlatformEntityUrlSchema)
def get_platform_entity_url(
    entity_id: str | None = None,
    system_name: str | None = None,
    application: str | None = None,
) -> dict[str, Any]:
    """
    Get a #Resolver URL for a platform entity by ID or system name.

    Resolves entity IDs via GetAxioms (works for all entity types) or looks
    up by system name across all applications. Returns entity metadata
    (type, name, parent, application) alongside the URL.

    Resolvable entities (have standalone #Resolver/{id} pages):
    - Templates (oa.*, pa.*, ra.*, os.*)
    - Applications (sln.*)
    - Buttons (event.*)
    - Toolbars (tb.*)
    - Cards (card.*)
    - Forms (form.*)
    - Tables (lst.*, ds.*)
    - Roles (role.*)
    - Navigation sections / workspaces (workspace.*)
    - Records (plain numeric IDs)
    - Process diagrams (diagram.*)

    Non-resolvable (modal-only, no standalone URL):
    - Attributes (op.*) — opened within template editor
    - Groups — opened within template editor

    system_name lookup: works for templates, applications, and roles only.
    For child entities (buttons, forms, toolbars, tables), provide entity_id.

    Usage:
        # By entity ID (fastest, works for all resolvable types)
        get_platform_entity_url.invoke({"entity_id": "oa.193"})
        get_platform_entity_url.invoke({"entity_id": "event.15199"})
        get_platform_entity_url.invoke({"entity_id": "12345"})  # record

        # By system name (templates, apps, roles only)
        get_platform_entity_url.invoke({"system_name": "ServiceRequests"})

        # By system name filtered by application
        get_platform_entity_url.invoke({
            "system_name": "schedule_maintenance",
            "application": "CustomerPortal"
        })

        # Both (verifies they match)
        get_platform_entity_url.invoke({
            "entity_id": "oa.193",
            "system_name": "ServiceRequests"
        })

    Returns:
        dict: {
            "success": bool,
            "entity_id": str|None,
            "entity_url": str|None,
            "entity_type": str|None,
            "system_name": str|None,
            "name": str|None,
            "parent_system_name": str|None,
            "application": str|None,
            "matches": list|None  # For system_name lookup results
        }
    """
    try:
        cfg = requests_._load_server_config()
        base_url = cfg.base_url.rstrip("/")
        if not base_url:
            return _error("Base URL not found in server configuration")

        if not entity_id and not system_name:
            return _error("Provide entity_id, system_name, or both")

        if entity_id and system_name:
            return _resolve_both(entity_id, system_name, base_url)

        if entity_id:
            return _resolve_by_id(entity_id, base_url)

        return _resolve_by_name(system_name, application, base_url)

    except Exception as e:
        logger.exception("get_platform_entity_url failed")
        return _error(f"Error generating entity URL: {str(e)}")


def _resolve_by_id(entity_id: str, base_url: str) -> dict[str, Any]:
    """Resolve entity by ID via GetAxioms."""
    result = _resolve_entity_id(entity_id)
    if not result["success"]:
        return _error(f"Entity '{entity_id}' not found")

    entity_type = _get_entity_type_from_id(entity_id)
    parent = result.get("container")
    app = result.get("app_alias")

    return {
        "success": True,
        "entity_id": entity_id,
        "entity_url": f"{base_url}/#Resolver/{entity_id}",
        "entity_type": entity_type,
        "system_name": result.get("alias"),
        "name": result.get("name"),
        "parent_system_name": parent,
        "application": app,
        "matches": None,
    }


def _resolve_by_name(
    system_name: str,
    application: str | None,
    base_url: str,
) -> dict[str, Any]:
    """Resolve entity by system name via TemplateService/List."""
    all_templates = _fetch_all_templates()
    system_name = system_name.strip()
    application = application.strip() if application else None

    matches = []
    for item in all_templates:
        alias = item.get("alias", "").strip()
        if alias != system_name:
            continue

        if application:
            item_app = (item.get("solutionAlias") or "").strip()
            if item_app != application:
                continue

        entity_id = item.get("id", "")
        entity_type = _get_entity_type_from_template(item)
        app_alias = item.get("solutionAlias") or ""
        parent = _get_parent_from_template(item)

        matches.append({
            "entity_id": entity_id,
            "entity_url": f"{base_url}/#Resolver/{entity_id}",
            "entity_type": entity_type,
            "system_name": alias,
            "name": item.get("name", ""),
            "parent_system_name": parent,
            "application": app_alias,
        })

    if not matches:
        return _error(
            f"No entity found with system_name '{system_name}'"
            + (f" in application '{application}'" if application else "")
            + ". For attributes, buttons, forms, datasets, toolbars — use entity_id instead."
        )

    return {
        "success": True,
        "entity_id": None,
        "entity_url": None,
        "entity_type": None,
        "system_name": system_name,
        "name": None,
        "parent_system_name": None,
        "application": application,
        "matches": matches,
    }


def _resolve_both(
    entity_id: str,
    system_name: str,
    base_url: str,
) -> dict[str, Any]:
    """Resolve both entity_id and system_name, verify they match."""
    result = _resolve_entity_id(entity_id)
    if not result["success"]:
        return _error(f"Entity '{entity_id}' not found")

    resolved_alias = result.get("alias") or ""
    if resolved_alias.strip() != system_name.strip():
        return _error(
            f"entity_id '{entity_id}' resolves to '{resolved_alias}', "
            f"does not match system_name '{system_name}'"
        )

    entity_type = _get_entity_type_from_id(entity_id)
    parent = result.get("container")
    app = result.get("app_alias")

    return {
        "success": True,
        "entity_id": entity_id,
        "entity_url": f"{base_url}/#Resolver/{entity_id}",
        "entity_type": entity_type,
        "system_name": resolved_alias,
        "name": result.get("name"),
        "parent_system_name": parent,
        "application": app,
        "matches": None,
    }


def _fetch_all_templates() -> list[dict[str, Any]]:
    """Fetch all templates from TemplateService/List for all entity types."""
    all_items = []
    for entity_type in _ENTITY_TYPES:
        try:
            result = requests_._post_request(
                {"Type": entity_type},
                "api/public/system/Solution/TemplateService/List",
            )
            if result.get("success"):
                raw = result["raw_response"]
                items = ast.literal_eval(raw) if isinstance(raw, str) else raw
                all_items.extend(items)
        except Exception as e:
            logger.warning("Failed to fetch templates for type %s: %s", entity_type, e)

    return all_items


def _get_entity_type_from_id(entity_id: str) -> str:
    """Determine entity type from ID prefix."""
    if "." in entity_id:
        prefix = entity_id.split(".")[0]
        return _ID_PREFIX_MAP.get(prefix, "Unknown")
    return "Unknown"


def _get_entity_type_from_template(item: dict[str, Any]) -> str:
    """Determine entity type from template item."""
    entity_type = item.get("type", "")
    type_map = {
        "Record": "Template",
        "Role": "Role",
        "Process": "ProcessTemplate",
        "OrgStructure": "Template",
        "Undefined": "Application",
    }
    return type_map.get(entity_type, "Unknown")


def _get_parent_from_template(item: dict[str, Any]) -> str | None:
    """Get parent system name from template item."""
    entity_type = item.get("type", "")
    if entity_type == "Undefined":
        return None
    return item.get("solutionAlias")


def _error(message: str) -> dict[str, Any]:
    """Return a standardized error response."""
    return {
        "success": False,
        "entity_id": None,
        "entity_url": None,
        "entity_type": None,
        "system_name": None,
        "name": None,
        "parent_system_name": None,
        "application": None,
        "matches": None,
        "error": message,
    }


if __name__ == "__main__":
    results = get_platform_entity_url.invoke({
        "entity_id": "oa.193",
    })
    print(results)
