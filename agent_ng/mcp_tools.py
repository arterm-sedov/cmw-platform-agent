"""
External MCP tool registry and loader (HTTP-first).

Registry: config/mcp_servers.yaml (fixed path). String values may use
``${ENV_NAME}`` placeholders; missing variables expand to an empty string so
secrets stay in ``.env`` rather than git-tracked YAML.

Env toggles: CMW_MCP_ENABLED (default off), allowlists, tool cap, host allowlist.
"""

from __future__ import annotations

import asyncio
from fnmatch import fnmatch
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Any
from urllib.parse import urlparse

import yaml

try:
    from .logging_config import _parse_bool
except ImportError:
    from agent_ng.logging_config import _parse_bool

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MCP_SERVERS_FILE = _REPO_ROOT / "config" / "mcp_servers.yaml"

_mcp_tools_cache: list[Any] | None = None
_mcp_load_attempted = False

_HTTP_TRANSPORTS = frozenset({"http", "streamable_http", "streamable-http"})
_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env_vars(value: Any) -> Any:
    """Recursively substitute ``${VAR}`` in strings; unset vars become ``""``."""
    if isinstance(value, str):

        def _replace(match: re.Match[str]) -> str:
            return os.environ.get(match.group(1), "")

        return _ENV_VAR_PATTERN.sub(_replace, value)
    if isinstance(value, dict):
        return {key: _expand_env_vars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    return value


def is_mcp_enabled() -> bool:
    """True when external MCP tools should be loaded (default off)."""
    return _parse_bool(os.getenv("CMW_MCP_ENABLED"), False)


def _normalize_transport(transport: str) -> str:
    normalized = transport.strip().lower().replace("-", "_")
    if normalized == "http":
        return "http"
    if normalized in ("streamable_http", "streamablehttp"):
        return "streamable_http"
    return normalized


def _parse_allowed_servers() -> set[str] | None:
    raw = (os.getenv("CMW_MCP_ALLOWED_SERVERS") or "").strip()
    if not raw:
        return None
    return {part.strip() for part in raw.split(",") if part.strip()}


def _parse_allowed_hosts() -> set[str] | None:
    raw = (os.getenv("CMW_MCP_ALLOWED_HOSTS") or "").strip()
    if not raw:
        return None
    return {part.strip().lower() for part in raw.split(",") if part.strip()}


def _host_allowed(url: str, allowed_hosts: set[str] | None) -> bool:
    if allowed_hosts is None:
        return True
    host = (urlparse(url).hostname or "").lower()
    if not host:
        return False
    for pattern in allowed_hosts:
        if pattern.startswith("*."):
            suffix = pattern[1:]
            if host == pattern[2:] or host.endswith(suffix):
                return True
        elif fnmatch(host, pattern) or host == pattern:
            return True
    return False


def load_mcp_registry(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """
    Load and validate MCP server entries from YAML.

    Returns a mapping suitable for MultiServerMCPClient(connections=...).
    """
    registry_path = path or DEFAULT_MCP_SERVERS_FILE
    if not registry_path.is_file():
        msg = f"MCP registry not found: {registry_path}"
        raise FileNotFoundError(msg)

    text = registry_path.read_text(encoding="utf-8")
    if registry_path.suffix.lower() == ".json":
        raw = json.loads(text)
    else:
        raw = yaml.safe_load(text)

    if raw is None:
        return {}
    if not isinstance(raw, dict):
        msg = f"MCP registry root must be a mapping: {registry_path}"
        raise ValueError(msg)

    entries = raw.get("servers", raw)
    if not isinstance(entries, dict):
        msg = f"MCP registry servers must be a mapping: {registry_path}"
        raise ValueError(msg)

    allowed_names = _parse_allowed_servers()
    allowed_hosts = _parse_allowed_hosts()
    if allowed_hosts is None:
        logger.info(
            "CMW_MCP_ALLOWED_HOSTS unset — all HTTP MCP hosts permitted; "
            "set a comma-separated allowlist in production"
        )
    connections: dict[str, dict[str, Any]] = {}

    for name, block in entries.items():
        if not isinstance(block, dict):
            msg = f"MCP server {name!r}: connection block must be a mapping"
            raise ValueError(msg)
        if allowed_names is not None and name not in allowed_names:
            continue

        connection = _expand_env_vars(dict(block))
        transport = connection.get("transport")
        if not transport or not isinstance(transport, str):
            msg = f"MCP server {name!r}: missing or invalid 'transport'"
            raise ValueError(msg)

        normalized = _normalize_transport(transport)
        if normalized not in _HTTP_TRANSPORTS:
            msg = (
                f"MCP server {name!r}: Phase 1 supports HTTP MCP only "
                f"(got {transport!r}; use http or streamable_http)"
            )
            raise ValueError(msg)
        connection["transport"] = (
            "streamable_http" if normalized == "streamable_http" else "http"
        )

        url = connection.get("url")
        if isinstance(url, str) and not _host_allowed(url, allowed_hosts):
            msg = (
                f"MCP server {name!r}: host not in CMW_MCP_ALLOWED_HOSTS: "
                f"{urlparse(url).hostname!r}"
            )
            raise ValueError(msg)

        connections[name] = connection

    return connections


def _tool_name_prefix_enabled() -> bool:
    return _parse_bool(os.getenv("CMW_MCP_TOOL_NAME_PREFIX"), True)


def _max_mcp_tools() -> int | None:
    raw = (os.getenv("CMW_MCP_MAX_TOOLS") or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid CMW_MCP_MAX_TOOLS=%r — ignoring cap", raw)
        return None
    return value if value > 0 else None


async def fetch_mcp_tools_async() -> list[Any]:
    """Load tools from configured MCP servers (async)."""
    if not is_mcp_enabled():
        return []

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError as exc:
        logger.warning("langchain-mcp-adapters not installed — skipping MCP tools: %s", exc)
        return []

    connections = load_mcp_registry()
    if not connections:
        logger.info("MCP enabled but registry has no servers after filtering")
        return []

    client = MultiServerMCPClient(
        connections,
        tool_name_prefix=_tool_name_prefix_enabled(),
        tool_interceptors=[_MCPAuditInterceptor()],
    )
    tools = await client.get_tools()
    cap = _max_mcp_tools()
    if cap is not None and len(tools) > cap:
        tools = tools[:cap]

    logger.info(
        "Loaded %s MCP tool(s) from server(s): %s",
        len(tools),
        ", ".join(sorted(connections)),
    )
    return list(tools)


class _MCPAuditInterceptor:
    """Minimal audit log for MCP tool calls (no argument payloads)."""

    async def __call__(self, request: Any, handler: Any) -> Any:
        started = time.perf_counter()
        ok = False
        try:
            result = await handler(request)
            ok = True
            return result
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.info(
                "mcp_tool_invoke server=%s tool=%s ok=%s duration_ms=%.1f",
                getattr(request, "server_name", "?"),
                getattr(request, "name", "?"),
                ok,
                elapsed_ms,
            )


def _run_async(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    msg = "fetch_mcp_tools_async cannot run inside a running event loop; use preload_mcp_tools()"
    raise RuntimeError(msg)


async def preload_mcp_tools() -> list[Any]:
    """Load MCP tools once and cache (call from async app init)."""
    global _mcp_tools_cache, _mcp_load_attempted
    _mcp_load_attempted = True
    if not is_mcp_enabled():
        _mcp_tools_cache = []
        return _mcp_tools_cache

    try:
        _mcp_tools_cache = await fetch_mcp_tools_async()
    except Exception:
        logger.exception("Failed to load external MCP tools")
        _mcp_tools_cache = []
    return _mcp_tools_cache


def get_cached_mcp_tools() -> list[Any]:
    """Return cached MCP tools, loading synchronously on first access if needed."""
    global _mcp_tools_cache, _mcp_load_attempted
    if not is_mcp_enabled():
        return []
    if _mcp_tools_cache is not None:
        return _mcp_tools_cache
    if _mcp_load_attempted:
        return _mcp_tools_cache or []

    _mcp_load_attempted = True
    try:
        _mcp_tools_cache = _run_async(fetch_mcp_tools_async())
    except RuntimeError:
        logger.debug(
            "MCP preload skipped in running event loop; "
            "await preload_mcp_tools() at startup"
        )
        _mcp_tools_cache = []
    except Exception:
        logger.exception("Failed to load external MCP tools")
        _mcp_tools_cache = []
    return _mcp_tools_cache or []


def reset_mcp_tools_cache() -> None:
    """Clear MCP tool cache (tests)."""
    global _mcp_tools_cache, _mcp_load_attempted
    _mcp_tools_cache = None
    _mcp_load_attempted = False


def merge_tools(native: list[Any], mcp_tools: list[Any]) -> list[Any]:
    """Append MCP tools without name collisions (MCP skipped on duplicate)."""
    if not mcp_tools:
        return native
    names = {getattr(t, "name", None) for t in native}
    merged = list(native)
    for tool in mcp_tools:
        tool_name = getattr(tool, "name", None)
        if tool_name in names:
            logger.warning(
                "Skipping MCP tool %r — name already used by native tool",
                tool_name,
            )
            continue
        merged.append(tool)
        names.add(tool_name)
    return merged
