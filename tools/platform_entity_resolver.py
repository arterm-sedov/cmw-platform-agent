"""
Platform Entity URL Resolver - Universal GetAxioms-based resolver for CMW Platform.

Parses CMW Platform URLs (hash-based SPA routing) and resolves entity IDs to
system names via OntologyService/GetAxioms. Returns system_name and
application_system_name that agent_ng can use with other tools.

Key Features:
- Recursive GetAxioms resolution with caching
- Extracts cmw.alias / cmw.container.alias → system_name
- Extracts cmw.solution.alias → application_system_name
- No API endpoints (agent uses tools, not direct API calls)

Usage:
    from tools.platform_entity_resolver import resolve_entity

    # Full URL with template + button
    result = resolve_entity.invoke({
        "url_or_id": "https://host/#RecordType/oa.193/Operation/event.15199"
    })

    # Single entity ID
    result = resolve_entity.invoke({"url_or_id": "oa.193"})
    result = resolve_entity.invoke({"url_or_id": "event.15199"})

Output (agent_ng can use system_name + application_system_name with tools):
    {
        "success": True,
        "resolved": [
            {
                "entity_type": "Template",
                "id": "oa.193",
                "system_name": "ServiceRequests",
                "application_system_name": "CustomerPortal",
                "name": "Service Requests"
            },
            {
                "entity_type": "Button",
                "id": "event.15199",
                "system_name": "approve_request",
                "application_system_name": "CustomerPortal",
                "name": "Approve Request",
                "kind": "Trigger scenario"
            }
        ]
    }
"""

import ast
from dataclasses import dataclass, field
import logging
import os
import re
from typing import Any
import urllib.parse

from dotenv import load_dotenv
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import requests

from tools import requests_

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ParsedEntity:
    """A single entity extracted from a URL."""

    entity_type: str
    entity_id: str


@dataclass
class ParsedUrl:
    """Complete parse result for a CMW Platform URL."""

    original: str
    hash_path: str = ""
    entities: list[ParsedEntity] = field(default_factory=list)
    query_params: dict[str, str] = field(default_factory=dict)
    page_type: str = "unknown"


# ---------------------------------------------------------------------------
# ID prefix → entity type mapping (human-friendly terms per system_prompt.json)
# ---------------------------------------------------------------------------

_ID_PREFIX_MAP: dict[str, str] = {
    "oa": "Template",
    "pa": "ProcessTemplate",
    "ra": "Template",
    "os": "Template",
    "sln": "Application",  # API: solution → human: application
    "event": "Button",  # API: user command → human: button
    "form": "Form",
    "card": "Card",
    "tb": "Toolbar",
    "lst": "Table",
    "ds": "Table",
    "diagram": "ProcessDiagram",  # API: diagram/scheme → human: process diagram
    "role": "Role",
    "workspace": "NavigationSection",  # API: workspace → human: navigation section
    "op": "Attribute",  # API: property → human: attribute
}

# Compiled regex: matches prefix.number pattern (e.g., "oa.193", "event.454")
_ID_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(p) for p in _ID_PREFIX_MAP) + r")\.(\d+)"
)

# Template type prefixes — these are the "parent" entities
_TEMPLATE_PREFIXES = {"oa", "pa", "ra", "os"}

# ---------------------------------------------------------------------------
# URL parser
# ---------------------------------------------------------------------------


def _parse_url(url_or_id: str) -> ParsedUrl:
    """
    Parse a CMW Platform URL or raw entity ID.

    Extracts all entity IDs from the URL hash path and query parameters.
    Classifies the page type from the first hash segment.

    Args:
        url_or_id: Full URL, hash-only string, or raw entity ID.

    Returns:
        ParsedUrl with all extracted entities and metadata.
    """
    original = url_or_id.strip()
    if not original:
        return ParsedUrl(original=original)

    # Extract hash portion
    hash_part = ""
    if "#" in original:
        hash_part = original.split("#", 1)[1].rstrip("/")
    elif "/" not in original and "." in original:
        # Raw ID like "oa.193"
        return _parse_raw_id(original)
    elif original.isdigit():
        # Plain numeric record ID
        return ParsedUrl(
            original=original,
            entities=[ParsedEntity("Record", original)],
        )

    if not hash_part:
        return ParsedUrl(original=original)

    # Determine page type from first segment
    first_segment = hash_part.split("/")[0]
    page_type = first_segment

    # Decode query params from hash path (some URLs embed them after /s=)
    query_params: dict[str, str] = {}
    if "?" in hash_part or "%3D" in hash_part or "&" in hash_part:
        # Find query-like portion (after first ? or embedded in path)
        query_str = ""
        if "?" in hash_part:
            query_str = hash_part.split("?", 1)[1]
        else:
            # Look for segments that look like query params
            for segment in hash_part.split("/"):
                if "%3D" in segment or "%26" in segment or "=" in segment:
                    query_str = segment
                    break
        if query_str:
            query_params = _decode_query_params(query_str)

    # Extract entities from hash path (before query params)
    path_part = hash_part.split("?")[0] if "?" in hash_part else hash_part
    # Also handle embedded query-like segments
    for seg in path_part.split("/"):
        if "%3D" in seg or "%26" in seg:
            path_part = path_part.replace(seg, "")
            break
    path_part = path_part.rstrip("/")

    entities = _extract_ids_from_path(path_part)

    # Extract additional IDs from query params
    entities.extend(_extract_ids_from_query(query_params))

    # Handle app/{App}/list/{Tpl} and app/{App}/view/{Tpl}/{recordId} patterns
    if page_type == "app":
        segments = path_part.split("/")
        if len(segments) >= 2 and segments[1]:
            entities.append(ParsedEntity("App", segments[1]))
        if len(segments) >= 4 and segments[3]:
            entities.append(ParsedEntity("Template", segments[3]))
        if len(segments) >= 5 and segments[4].isdigit():
            entities.append(ParsedEntity("Record", segments[4]))

    # Handle form/{tpl}/form.{M}/{recordId} pattern
    if page_type == "form":
        segments = path_part.split("/")
        if len(segments) >= 2 and segments[1]:
            # First segment after 'form' is template ID
            tpl_match = _ID_PATTERN.match(segments[1])
            if tpl_match:
                prefix, num = tpl_match.groups()
                entity_type = _ID_PREFIX_MAP.get(prefix, "Template")
                entities.append(ParsedEntity(entity_type, f"{prefix}.{num}"))
        # Last segment may be a record ID
        if len(segments) >= 4 and segments[3].isdigit():
            entities.append(ParsedEntity("Record", segments[3]))

    # Task and myTasks page patterns
    if page_type == "task":
        segments = path_part.split("/")
        if len(segments) >= 2 and segments[1].isdigit():
            entities.append(ParsedEntity("Task", segments[1]))

    if page_type == "myTasks":
        pass  # No specific entity IDs, just page type

    # Deduplicate entities
    seen = set()
    unique_entities = []
    for e in entities:
        key = (e.entity_type, e.entity_id)
        if key not in seen:
            seen.add(key)
            unique_entities.append(e)

    return ParsedUrl(
        original=original,
        hash_path=hash_part,
        entities=unique_entities,
        query_params=query_params,
        page_type=page_type,
    )


def _parse_raw_id(raw_id: str) -> ParsedUrl:
    """Parse a raw entity ID like 'oa.193' or 'event.454'."""
    match = _ID_PATTERN.fullmatch(raw_id)
    if match:
        prefix, _num = match.groups()
        entity_type = _ID_PREFIX_MAP.get(prefix, "Template")
        return ParsedUrl(
            original=raw_id,
            entities=[ParsedEntity(entity_type, raw_id)],
        )
    # Plain numeric → record ID
    if raw_id.isdigit():
        return ParsedUrl(
            original=raw_id,
            entities=[ParsedEntity("Record", raw_id)],
        )
    return ParsedUrl(original=raw_id)


def _extract_ids_from_path(path: str) -> list[ParsedEntity]:
    """Extract all entity IDs from a URL path segment."""
    entities: list[ParsedEntity] = []
    for match in _ID_PATTERN.finditer(path):
        prefix, num = match.groups()
        entity_type = _ID_PREFIX_MAP.get(prefix, "Template")
        entities.append(ParsedEntity(entity_type, f"{prefix}.{num}"))
    return entities


def _extract_ids_from_query(query_params: dict[str, str]) -> list[ParsedEntity]:
    """Extract entity IDs from decoded query parameter values."""
    entities: list[ParsedEntity] = []
    for value in query_params.values():
        for match in _ID_PATTERN.finditer(value):
            prefix, num = match.groups()
            entity_type = _ID_PREFIX_MAP.get(prefix, "Template")
            entities.append(ParsedEntity(entity_type, f"{prefix}.{num}"))
    return entities


def _decode_query_params(query_str: str) -> dict[str, str]:
    """URL-decode a query string and return key-value pairs."""
    decoded = urllib.parse.unquote(query_str)
    params: dict[str, str] = {}
    # Split on & (may be literal or encoded)
    for part in re.split(r"[&]", decoded):
        if "=" in part:
            key, _, value = part.partition("=")
            params[key.strip()] = value.strip()
        elif part.strip():
            params[part.strip()] = ""
    return params


# ---------------------------------------------------------------------------
# TemplateService resolver
# ---------------------------------------------------------------------------

_TEMPLATE_SERVICE_ENDPOINT = "api/public/system/Solution/TemplateService/List"


def _resolve_templates(
    template_ids: list[str],
    requests_module: Any = requests_,
    fetch_all_types: bool = False,
) -> dict[str, dict[str, Any]]:
    """
    Resolve template IDs via TemplateService/List.

    Fetches all templates for Record, Process, Role, and OrgStructure types,
    then returns a lookup dict keyed by internal ID.

    Args:
        template_ids: List of template IDs to resolve (e.g., ["oa.193", "pa.77"]).
        requests_module: The requests module to use (default: tools.requests_).
        fetch_all_types: If True, fetch all 4 types regardless of template_ids.

    Returns:
        Dict mapping template ID to resolved data.
    """
    if not template_ids and not fetch_all_types:
        return {}

    type_map = {
        "oa": "Record",
        "pa": "Process",
        "ra": "Role",
        "os": "OrgStructure",
    }

    # Determine which types to fetch based on IDs
    types_to_fetch: set[str] = set()
    if fetch_all_types:
        types_to_fetch = set(type_map.values())
    else:
        for tid in template_ids:
            prefix = tid.split(".")[0]
            if prefix in type_map:
                types_to_fetch.add(type_map[prefix])

    if not types_to_fetch:
        return {}

    # Fetch each type
    all_templates: dict[str, dict[str, Any]] = {}
    for tpl_type in types_to_fetch:
        try:
            result = requests_module._post_request(
                {"Type": tpl_type},
                _TEMPLATE_SERVICE_ENDPOINT,
            )
            if not result.get("success"):
                logger.warning(
                    "TemplateService/List failed for type %s: %s",
                    tpl_type,
                    result.get("error"),
                )
                continue

            raw = result.get("raw_response")
            if isinstance(raw, str):
                try:
                    items = ast.literal_eval(raw)
                except (ValueError, SyntaxError):
                    logger.warning(
                        "Failed to parse TemplateService response for %s", tpl_type
                    )
                    continue
            elif isinstance(raw, list):
                items = raw
            else:
                logger.warning(
                    "Unexpected TemplateService response type for %s", tpl_type
                )
                continue

            for item in items:
                if isinstance(item, dict) and "id" in item:
                    all_templates[item["id"]] = item

        except Exception:
            logger.warning(
                "Exception fetching templates for type %s", tpl_type, exc_info=True
            )

    return all_templates


def _resolve_templates_single(template_id: str) -> dict[str, Any]:
    """Resolve a single template ID via TemplateService/List."""
    prefix = template_id.split(".")[0]
    type_map = {"oa": "Record", "pa": "Process", "ra": "Role", "os": "OrgStructure"}
    tpl_type = type_map.get(prefix)
    if not tpl_type:
        return {}
    result = requests_._post_request({"Type": tpl_type}, _TEMPLATE_SERVICE_ENDPOINT)
    if not result.get("success"):
        return {}
    raw = result.get("raw_response", "")
    if isinstance(raw, str):
        try:
            items = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return {}
    elif isinstance(raw, list):
        items = raw
    else:
        return {}
    for item in items:
        if isinstance(item, dict) and item.get("id") == template_id:
            return item
    return {}


# ---------------------------------------------------------------------------
# Entity lister
# ---------------------------------------------------------------------------


def _list_entity_candidates(
    entity_type: str,
    app_system_name: str,
    template_system_name: str,
    requests_module: Any = requests_,
) -> list[dict[str, Any]]:
    """
    List all entities of a given type for a template.

    Args:
        entity_type: "Button", "Form", "Toolbar", or "Dataset".
        app_system_name: Application system name.
        template_system_name: Template system name.
        requests_module: The requests module to use.

    Returns:
        List of entity dicts with system_name, name, kind (for buttons), etc.
    """
    endpoint_map = {
        "Button": f"webapi/UserCommand/List/Template@{app_system_name}.{template_system_name}",
        "Form": f"webapi/Form/List/Template@{app_system_name}.{template_system_name}",
        "Toolbar": f"webapi/Toolbar/List/Template@{app_system_name}.{template_system_name}",
        "Dataset": f"webapi/Dataset/List/Template@{app_system_name}.{template_system_name}",
    }

    endpoint = endpoint_map.get(entity_type)
    if not endpoint:
        logger.warning("Unknown entity type for listing: %s", entity_type)
        return []

    try:
        result = requests_module._get_request(endpoint)
        if not result.get("success"):
            logger.warning("Failed to list %s: %s", entity_type, result.get("error"))
            return []

        raw = result.get("raw_response")
        if not isinstance(raw, dict):
            return []

        response = raw.get("response", [])
        if not isinstance(response, list):
            return []

        candidates = []
        for item in response:
            if not isinstance(item, dict):
                continue

            ga = item.get("globalAlias", {})
            alias = ga.get("alias", "") if isinstance(ga, dict) else ""

            candidate: dict[str, Any] = {
                "system_name": alias,
                "name": item.get("name", ""),
            }
            if entity_type == "Button":
                candidate["kind"] = item.get("kind", "")
            candidate["api_endpoint"] = (
                f"webapi/{_get_api_entity_type(entity_type)}/"
                f"{app_system_name}/"
                f"{_get_api_entity_type(entity_type)}@{template_system_name}.{alias}"
            )
            candidate["full_data"] = item
            candidates.append(candidate)

        return candidates

    except Exception:
        logger.warning(
            "Exception listing %s for %s/%s",
            entity_type,
            app_system_name,
            template_system_name,
            exc_info=True,
        )
        return []


def _get_api_entity_type(entity_type: str) -> str:
    """Map internal entity type to API endpoint type."""
    mapping = {
        "Button": "UserCommand",
        "Form": "Form",
        "Toolbar": "Toolbar",
        "Dataset": "Dataset",
    }
    return mapping.get(entity_type, entity_type)


# ---------------------------------------------------------------------------
# Internal ID resolver via GetAxioms
# ---------------------------------------------------------------------------


def _resolve_entity_id(entity_id: str, _cache: dict[str, dict] | None = None) -> dict[str, Any]:
    """
    Resolve any entity ID via OntologyService/GetAxioms.

    Recursively resolves container → solution chain to get full context.
    Uses cache to avoid duplicate GetAxioms calls for same ID.

    Returns cmw.alias, cmw.object.name, container, solution, app_alias, and
    type-specific fields (cmw.eventTrigger.kind for buttons).
    """
    if _cache is None:
        _cache = {}

    if entity_id in _cache:
        return _cache[entity_id]

    load_dotenv()
    base_url = os.environ.get("CMW_BASE_URL", "")
    if not base_url:
        return {"success": False, "note": "CMW_BASE_URL not configured"}

    url = f"{base_url}/api/public/system/Base/OntologyService/GetAxioms"
    try:
        session = requests.Session()
        session.auth = (
            os.environ.get("CMW_LOGIN", ""),
            os.environ.get("CMW_PASSWORD", ""),
        )
        response = session.post(
            url,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            data=entity_id,
            timeout=30,
        )
        if response.status_code != 200:
            result = {"success": False, "note": f"HTTP {response.status_code}"}
            _cache[entity_id] = result
            return result
        data = response.json()
    except Exception as e:
        result = {"success": False, "note": str(e)}
        _cache[entity_id] = result
        return result

    if not data:
        result = {"success": False, "note": "Empty response"}
        _cache[entity_id] = result
        return result

    alias = (data.get("cmw.alias") or data.get("cmw.container.alias") or data.get("cmw.solution.alias") or [""])[0]
    name = (
        data.get("cmw.object.name")
        or data.get("cmw.eventTrigger.name")
        or data.get("cmw.container.alias")
        or data.get("cmw.solution.name")
        or [""]
    )[0]
    container = (
        data.get("cmw.eventTrigger.container") or data.get("cmw.form.container") or [""]
    )[0]
    solution = (data.get("cmw.solution") or [""])[0]
    rdf_type = (data.get("http://www.w3.org/1999/02/22-rdf-syntax-ns#type") or [""])[0]
    kind_raw = (data.get("cmw.eventTrigger.kind") or [""])[0]
    kind = None
    if kind_raw:
        kind = kind_raw.replace("cmw.eventTrigger.", "")
        kind = {
            "UserEvent": "Trigger scenario",
            "Create": "Create",
            "Edit": "Edit",
            "Delete": "Delete",
            "Archive": "Archive",
            "Unarchive": "Unarchive",
            "StartProcess": "Start process",
            "StartCase": "Start case",
            "CompleteTask": "Complete task",
            "ReassignTask": "Reassign task",
            "Defer": "Defer",
            "Accept": "Accept",
            "Uncomplete": "Uncomplete",
            "Follow": "Follow",
            "Unfollow": "Unfollow",
            "Exclude": "Exclude",
            "Include": "Include",
            "Script": "Script",
            "Cancel": "Cancel",
            "EditDiagram": "Edit diagram",
            "CreateRelated": "Create related",
            "ExportObject": "Export object",
            "ExportList": "Export list",
            "CreateToken": "Create token",
            "RetryTokens": "Retry tokens",
            "Migrate": "Migrate",
            "StartLinkedCase": "Start linked case",
            "StartLinkedProcess": "Start linked process",
        }.get(kind, kind)

    # Recursively resolve container → solution chain to get app alias
    app_alias = None
    if solution:
        sol_result = _resolve_entity_id(solution, _cache)
        if sol_result["success"]:
            app_alias = sol_result["alias"]
    elif container:
        cont_result = _resolve_entity_id(container, _cache)
        if cont_result["success"] and cont_result.get("solution"):
            sol_result = _resolve_entity_id(cont_result["solution"], _cache)
            if sol_result["success"]:
                app_alias = sol_result["alias"]

    result = {
        "success": True,
        "alias": alias or None,
        "name": name or None,
        "container": container or None,
        "solution": solution or None,
        "app_alias": app_alias,
        "rdf_type": rdf_type,
        "kind": kind,
        "raw": data,
    }
    _cache[id] = result
    return result


# ---------------------------------------------------------------------------
# Diagram resolver
# ---------------------------------------------------------------------------


def _resolve_diagram(
    diagram_id: str,
    requests_module: Any = requests_,
) -> dict[str, Any]:
    """
    Resolve a diagram by ID via Process/DiagramService/ResolveDiagram.

    Args:
        diagram_id: Diagram internal ID (e.g., "diagram.315").
        requests_module: The requests module to use.

    Returns:
        Dict with diagram_id, diagram_data, success, error.
    """
    try:
        result = requests_module._post_request(
            {"serverId": diagram_id},
            "api/public/system/Process/DiagramService/ResolveDiagram",
        )
        if result.get("success"):
            return {
                "diagram_id": diagram_id,
                "diagram_data": result.get("raw_response"),
                "success": True,
                "error": None,
            }
        return {
            "diagram_id": diagram_id,
            "diagram_data": None,
            "success": False,
            "error": result.get("error"),
        }
    except Exception as e:
        logger.warning("Exception resolving diagram %s", diagram_id, exc_info=True)
        return {
            "diagram_id": diagram_id,
            "diagram_data": None,
            "success": False,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Main tool
# ---------------------------------------------------------------------------


class ResolveEntitySchema(BaseModel):
    """Schema for the resolve_entity tool."""

    url_or_id: str = Field(
        description=(
            "CMW Platform URL or entity ID to resolve. "
            "Returns system_name and application_system_name for use with other tools. "
            "Examples: 'https://host/#RecordType/oa.3/Operation/event.454', "
            "'oa.193', 'event.15199', '#Resolver/event.15199'. "
            "RU: URL платформы или ID сущности для получения системных имён"
        ),
    )
    fetch_full: bool = Field(
        default=True,
        description=(
            "Include full raw data from GetAxioms in response. "
            "Default True. Set False for minimal output. "
            "RU: Включить полные данные из GetAxioms"
        ),
    )


@tool("resolve_entity", return_direct=False, args_schema=ResolveEntitySchema)
def resolve_entity(
    url_or_id: str,
    fetch_full: bool = True,
) -> dict[str, Any]:
    """
    Resolve CMW Platform URL or entity ID to system names for use with other tools.

    When user pastes a platform URL, this tool extracts entity IDs and resolves them
    via GetAxioms to get system_name and application_system_name. Agent can then use
    these names with edit_or_create_* tools.

    Examples:
        User: "Edit this button https://host/#RecordType/oa.193/Operation/event.15199"

        Agent calls: resolve_entity(url_or_id="...")

        Returns: {
            "resolved": [
                {"entity_type": "Template", "system_name": "ServiceRequests",
                 "application_system_name": "CustomerPortal"},
                {"entity_type": "Button", "system_name": "approve_request",
                 "application_system_name": "CustomerPortal", "kind": "Trigger scenario"}
            ]
        }

        Agent then calls: edit_or_create_button(
            application_system_name="CustomerPortal",
            template_system_name="ServiceRequests",
            button_system_name="approve_request",
            ...
        )

    Returns:
        dict: {
            "success": bool,
            "status_code": int,
            "error": str|None,
            "url_parsed": {
                "original": str,
                "page_type": str,
                "entities_found": [{"type": str, "id": str}, ...]
            },
            "resolved": [
                {
                    "entity_type": str,  # Template, Button, Form, Toolbar, Dataset, etc.
                    "id": str,  # oa.193, event.15199, etc.
                    "system_name": str,  # ServiceRequests, approve_request
                    "application_system_name": str|None,  # CustomerPortal
                    "name": str,  # Display name (Планы техобслуживания)
                    "kind": str|None,  # Button kind (Trigger scenario, Create, etc.)
                    "full_data": dict|None,  # Raw GetAxioms response if fetch_full=True
                    "note": str|None,  # Error message if resolution failed
                },
                ...
            ]
        }
    """
    if not url_or_id or not url_or_id.strip():
        return {
            "success": False,
            "status_code": 400,
            "error": "Empty URL or ID provided",
            "url_parsed": None,
            "resolved": [],
        }

    try:
        # Step 1: Parse URL
        parsed = _parse_url(url_or_id)

        if not parsed.entities:
            return {
                "success": True,
                "status_code": 200,
                "error": None,
                "url_parsed": {
                    "original": parsed.original,
                    "page_type": parsed.page_type,
                    "entities_found": [],
                },
                "resolved": [],
                "note": "No entity IDs found in URL. This may be a settings or landing page.",
            }

# Step 2: Resolve all entities via GetAxioms (with shared cache)
        resolved: list[dict[str, Any]] = []
        parent_context: dict[str, str] = {}
        cache: dict[str, dict] = {}

        for entity in parsed.entities:
            eid = entity.entity_id
            ax = _resolve_entity_id(eid, cache)

            if not ax["success"]:
                resolved.append({
                    "entity_type": entity.entity_type,
                    "id": eid,
                    "system_name": None,
                    "application_system_name": None,
                    "name": None,
                    "note": ax.get("note"),
                })
                continue

            alias = ax["alias"]
            name = ax["name"]
            app = ax["app_alias"] or parent_context.get("app", "")

            # Update parent context for subsequent entities
            if app and alias:
                parent_context["app"] = app
                if ax["container"]:
                    cont_ax = cache.get(ax["container"])
                    if cont_ax and cont_ax.get("alias"):
                        parent_context["template"] = cont_ax["alias"]

            etype = entity.entity_type

            resolved.append({
                "entity_type": etype,
                "id": eid,
                "system_name": alias,
                "application_system_name": app or None,
                "name": name,
                "kind": ax.get("kind") if etype == "Button" else None,
                "full_data": ax.get("raw") if fetch_full else None,
            })

        return {
            "success": True,
            "status_code": 200,
            "error": None,
            "url_parsed": {
                "original": parsed.original,
                "page_type": parsed.page_type,
                "entities_found": [
                    {"type": e.entity_type, "id": e.entity_id} for e in parsed.entities
                ],
            },
            "resolved": resolved,
        }

    except Exception as e:
        logger.exception("resolve_entity failed for %s", url_or_id)
        return {
            "success": False,
            "status_code": 500,
            "error": f"Tool execution failed: {e!s}",
            "url_parsed": None,
            "resolved": [],
        }


def _find_app_for_solution(
    solution_id: str,
    template_cache: dict[str, dict[str, Any]],
) -> str:
    """
    Find the application system name for a solution ID.

    Uses list_applications to map solution display name to system name.
    Falls back to solutionName from template cache if list_applications fails.
    """
    # Collect unique solutionNames for this solution
    names = set()
    for tpl in template_cache.values():
        if tpl.get("solution") == solution_id:
            name = tpl.get("solutionName", "")
            if name:
                names.add(name)

    if not names:
        return ""

    solution_name = max(
        names,
        key=lambda n: sum(
            1
            for t in template_cache.values()
            if t.get("solution") == solution_id and t.get("solutionName") == n
        ),
    )

    # Try to map display name to system name via list_applications
    try:
        from tools.applications_tools.tool_list_applications import (  # noqa: PLC0415
            list_applications,
        )

        result = list_applications.invoke({})
        apps = result.get("data", [])
        for app in apps:
            if app.get("Name") == solution_name:
                return app.get("Application system name", "")
    except Exception:
        logger.debug("list_applications failed, falling back to solutionName")

    # Fallback: return display name (may not work as app system name in API)
    return solution_name
