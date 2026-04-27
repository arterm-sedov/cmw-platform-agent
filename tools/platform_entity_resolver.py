"""
Platform Entity URL Resolver Tool for CMW Platform agent_ng.

Parses CMW Platform URLs to extract entity IDs, resolves template IDs to
application + system names via TemplateService, and fetches full entity data
for buttons, forms, toolbars, datasets, and diagrams.

Usage:
    from tools.platform_entity_resolver import resolve_entity

    result = resolve_entity.invoke({
        "url_or_id": "https://bububu.bau.cbap.ru/#RecordType/oa.193/Operation/event.15199",
        "fetch_full": True,
    })
"""

import ast
from dataclasses import dataclass, field
import logging
import re
from typing import TYPE_CHECKING, Any
import urllib.parse

if TYPE_CHECKING:
    from types import ModuleType

from langchain_core.tools import tool
from pydantic import BaseModel, Field

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


def _resolve_solutions(
    solution_ids: list[str],
    template_cache: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    Resolve solution IDs by matching against template cache.

    Args:
        solution_ids: List of solution IDs (e.g., ["sln.23"]).
        template_cache: Cache from _resolve_templates.

    Returns:
        Dict mapping solution ID to resolved data.
    """
    solutions: dict[str, dict[str, Any]] = {}
    for sid in solution_ids:
        for tpl in template_cache.values():
            if tpl.get("solution") == sid:
                solutions[sid] = {
                    "id": sid,
                    "solutionName": tpl.get("solutionName", ""),
                    "solution": sid,
                }
                break
    return solutions


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
            "Examples: "
            "'https://host/#RecordType/oa.3/Operation/event.454', "
            "'oa.193', '#Resolver/event.15199'. "
            "RU: URL платформы или ID сущности"
        ),
    )
    fetch_full: bool = Field(
        default=True,
        description=(
            "Fetch full entity data from API. "
            "Set False to get only resolved metadata. "
            "RU: Загружать полные данные сущности"
        ),
    )


@tool("resolve_entity", return_direct=False, args_schema=ResolveEntitySchema)
def resolve_entity(
    url_or_id: str,
    fetch_full: bool = True,
) -> dict[str, Any]:
    """
    Resolve a CMW Platform URL or entity ID to API-ready entity objects.

    Parses the URL to extract all entity IDs, resolves template IDs to
    application + system names via TemplateService, and fetches full entity
    data for buttons, forms, toolbars, datasets, and diagrams.

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
                    "entity_type": str,
                    "internal_id": str,
                    "system_name": str,
                    "application_system_name": str|None,
                    "api_endpoint": str|None,
                    "full_data": dict|None,
                    "candidates": list|None,
                    "note": str|None,
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

        # Step 2: Collect template IDs and application IDs, then resolve templates
        template_ids = [
            e.entity_id
            for e in parsed.entities
            if e.entity_type in ("Template", "ProcessTemplate")
        ]
        application_ids = [
            e.entity_id for e in parsed.entities if e.entity_type == "Application"
        ]

        # Always fetch templates if we have application IDs (need cache for app mapping)
        # or template IDs (need cache for resolution)
        need_template_cache = bool(template_ids) or bool(application_ids)
        template_cache = (
            _resolve_templates(template_ids, fetch_all_types=bool(application_ids))
            if need_template_cache
            else {}
        )

        # Step 3: Build resolved entities
        resolved: list[dict[str, Any]] = []
        parent_context: dict[str, str] = {}  # app + template from resolved templates

        for entity in parsed.entities:
            eid = entity.entity_id
            etype = entity.entity_type

            if etype in ("Template", "ProcessTemplate"):
                tpl_data = template_cache.get(eid, {})
                alias = tpl_data.get("alias", "")
                solution = tpl_data.get("solution", "")
                solution_name = tpl_data.get("solutionName", "")

                # Determine app system name from solution
                app_name = (
                    _find_app_for_solution(solution, template_cache) if solution else ""
                )

                if alias and app_name:
                    parent_context["app"] = app_name
                    parent_context["template"] = alias

                api_endpoint = None
                if alias:
                    api_type = (
                        "RecordTemplate" if etype == "Template" else "ProcessTemplate"
                    )
                    api_endpoint = (
                        f"webapi/{api_type}/{app_name}/{alias}" if app_name else None
                    )

                resolved_entry: dict[str, Any] = {
                    "entity_type": "Template" if etype == "ProcessTemplate" else etype,
                    "internal_id": eid,
                    "system_name": alias or None,
                    "application_system_name": app_name or None,
                    "solution_id": solution or None,
                    "solution_name": solution_name or None,
                    "name": tpl_data.get("name"),
                    "api_endpoint": api_endpoint,
                    "full_data": tpl_data if fetch_full and tpl_data else None,
                    "candidates": None,
                    "note": None,
                }
                resolved.append(resolved_entry)

            elif etype == "Application":
                solution_data = _resolve_solutions([eid], template_cache)
                sol_info = solution_data.get(eid, {})
                resolved.append(
                    {
                        "entity_type": "Application",
                        "internal_id": eid,
                        "system_name": sol_info.get("solutionName"),
                        "application_system_name": None,
                        "api_endpoint": "webapi/Solution",
                        "full_data": sol_info if fetch_full and sol_info else None,
                        "candidates": None,
                        "note": None,
                    }
                )

            elif etype == "Role":
                # Try to find in template cache (Role templates)
                role_data = template_cache.get(eid, {})
                alias = role_data.get("alias", "")
                solution = role_data.get("solution", "")
                app_name = (
                    _find_app_for_solution(solution, template_cache) if solution else ""
                )

                if alias and app_name:
                    api_endpoint = f"webapi/RoleTemplate/{app_name}/{alias}"
                else:
                    api_endpoint = None

                resolved.append(
                    {
                        "entity_type": "Role",
                        "internal_id": eid,
                        "system_name": alias or None,
                        "application_system_name": app_name or None,
                        "api_endpoint": api_endpoint,
                        "full_data": role_data if fetch_full and role_data else None,
                        "candidates": None,
                        "note": None
                        if alias
                        else "Role ID not found in template cache.",
                    }
                )

            elif etype in ("Button", "Form", "Toolbar", "Dataset"):
                # Need parent context to list candidates
                app = parent_context.get("app", "")
                tpl = parent_context.get("template", "")

                if app and tpl:
                    candidates = (
                        _list_entity_candidates(etype, app, tpl) if fetch_full else []
                    )
                    resolved.append(
                        {
                            "entity_type": etype,
                            "internal_id": eid,
                            "candidates": candidates,
                            "note": (
                                "Internal entity IDs are not exposed by the API. "
                                "Match by name, system_name, or context from the URL."
                            ),
                            "full_data": None,
                            "system_name": None,
                            "application_system_name": app,
                            "api_endpoint": (
                                f"webapi/{_get_api_entity_type(etype)}/"
                                f"{app}/{_get_api_entity_type(etype)}@{tpl}.<system_name>"
                            ),
                        }
                    )
                else:
                    resolved.append(
                        {
                            "entity_type": etype,
                            "internal_id": eid,
                            "candidates": None,
                            "note": "Cannot list candidates: no parent template context in URL.",
                            "full_data": None,
                            "system_name": None,
                            "application_system_name": None,
                            "api_endpoint": None,
                        }
                    )

            elif etype == "Diagram":
                diagram_data = _resolve_diagram(eid) if fetch_full else {}
                resolved.append(
                    {
                        "entity_type": "ProcessDiagram",
                        "internal_id": eid,
                        "full_data": diagram_data.get("diagram_data")
                        if fetch_full
                        else None,
                        "system_name": None,
                        "application_system_name": parent_context.get("app"),
                        "api_endpoint": "api/public/system/Process/DiagramService/ResolveDiagram",
                        "candidates": None,
                        "note": diagram_data.get("error")
                        if not diagram_data.get("success")
                        else None,
                    }
                )

            elif etype == "Task":
                task_data = None
                task_error = None
                if fetch_full:
                    try:
                        task_result = requests_._post_request(
                            eid,
                            "api/public/system/TeamNetwork/UserTaskService/Get",
                        )
                        if task_result.get("success"):
                            task_data = task_result.get("raw_response")
                        else:
                            task_error = task_result.get("error")
                    except Exception as e:
                        task_error = str(e)

                resolved.append(
                    {
                        "entity_type": "Task",
                        "internal_id": eid,
                        "full_data": task_data,
                        "system_name": None,
                        "application_system_name": None,
                        "api_endpoint": "api/public/system/TeamNetwork/UserTaskService/Get",
                        "candidates": None,
                        "note": task_error
                        or "Task resolved by ID. Full data available if UserTaskService is accessible.",
                    }
                )

            elif etype == "NavigationSection":
                resolved.append(
                    {
                        "entity_type": "NavigationSection",
                        "internal_id": eid,
                        "full_data": None,
                        "system_name": None,
                        "application_system_name": None,
                        "api_endpoint": None,
                        "candidates": None,
                        "note": "Navigation section (workspace) — no direct API endpoint.",
                    }
                )

            elif etype == "Record":
                record_endpoint = f"webapi/Record/{eid}"
                record_data = None
                record_error = None
                if fetch_full:
                    try:
                        rec_result = requests_._get_request(record_endpoint)
                        if rec_result.get("success"):
                            raw = rec_result.get("raw_response")
                            if isinstance(raw, dict):
                                resp = raw.get("response", {})
                                if isinstance(resp, dict) and resp.get("success"):
                                    record_data = resp
                                else:
                                    record_error = (
                                        resp.get("error", {}).get("message")
                                        if isinstance(resp.get("error"), dict)
                                        else str(resp.get("error", ""))
                                    )
                        else:
                            record_error = rec_result.get("error")
                    except Exception as e:
                        record_error = str(e)

                resolved.append(
                    {
                        "entity_type": "Record",
                        "internal_id": eid,
                        "full_data": record_data,
                        "system_name": None,
                        "application_system_name": parent_context.get("app"),
                        "api_endpoint": record_endpoint,
                        "candidates": None,
                        "note": record_error,
                    }
                )

            elif etype == "App":
                resolved.append(
                    {
                        "entity_type": "App",
                        "internal_id": eid,
                        "system_name": eid,
                        "application_system_name": eid,
                        "api_endpoint": "webapi/Solution",
                        "full_data": None,
                        "candidates": None,
                        "note": "App identified by system name. Use list_applications to get details.",
                    }
                )

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
