"""Unit tests for external MCP registry loading (no live MCP servers)."""

from pathlib import Path

import pytest
import yaml

from agent_ng.mcp_tools import (
    DEFAULT_MCP_SERVERS_FILE,
    is_mcp_enabled,
    load_mcp_registry,
    merge_tools,
    reset_mcp_tools_cache,
)


@pytest.fixture(autouse=True)
def _clear_mcp_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CMW_MCP_ENABLED", raising=False)
    monkeypatch.delenv("CMW_MCP_ALLOWED_SERVERS", raising=False)
    monkeypatch.delenv("CMW_MCP_ALLOWED_HOSTS", raising=False)
    reset_mcp_tools_cache()


def test_mcp_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CMW_MCP_ENABLED", raising=False)
    assert is_mcp_enabled() is False


def test_default_registry_path_is_tracked_config() -> None:
    assert DEFAULT_MCP_SERVERS_FILE.name == "mcp_servers.yaml"
    assert DEFAULT_MCP_SERVERS_FILE.parent.name == "config"
    assert DEFAULT_MCP_SERVERS_FILE.resolve() == DEFAULT_MCP_SERVERS_FILE


def test_tracked_registry_file_exists() -> None:
    assert DEFAULT_MCP_SERVERS_FILE.is_file()


def test_tracked_registry_lists_ennoia_producer() -> None:
    connections = load_mcp_registry(DEFAULT_MCP_SERVERS_FILE)
    assert "comindware_kb" in connections
    url = connections["comindware_kb"]["url"]
    assert "ennoia.slickjump.org" in url
    assert "ask_comindware" in url
    assert connections["comindware_kb"]["transport"] == "streamable_http"


def test_allowed_servers_filter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry = tmp_path / "mcp.yaml"
    registry.write_text(
        yaml.dump(
            {
                "alpha": {"transport": "http", "url": "https://a.example/gradio_api/mcp/"},
                "beta": {"transport": "http", "url": "https://b.example/gradio_api/mcp/"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CMW_MCP_ALLOWED_SERVERS", "beta")
    connections = load_mcp_registry(registry)
    assert set(connections) == {"beta"}


def test_invalid_transport_raises(tmp_path: Path) -> None:
    registry = tmp_path / "mcp.yaml"
    registry.write_text(
        yaml.dump({"bad": {"transport": "ftp", "url": "https://x.example/mcp"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Phase 1 supports HTTP MCP only"):
        load_mcp_registry(registry)


def test_merge_tools_skips_name_collisions() -> None:
    class _Tool:
        def __init__(self, name: str) -> None:
            self.name = name

    native = [_Tool("dup"), _Tool("only_native")]
    mcp = [_Tool("dup"), _Tool("mcp_only")]
    merged = merge_tools(native, mcp)
    assert [t.name for t in merged] == ["dup", "only_native", "mcp_only"]


def test_env_var_substitution_in_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = tmp_path / "mcp.yaml"
    registry.write_text(
        yaml.dump(
            {
                "kb": {
                    "transport": "http",
                    "url": "https://example-host/gradio_api/mcp/",
                    "headers": {
                        "Authorization": "Bearer ${REMOTE_MCP_BEARER_TOKEN}",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("REMOTE_MCP_BEARER_TOKEN", "test-token-abc")
    connections = load_mcp_registry(registry)
    assert connections["kb"]["headers"]["Authorization"] == "Bearer test-token-abc"


def test_missing_env_var_expands_to_empty_string(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = tmp_path / "mcp.yaml"
    registry.write_text(
        yaml.dump(
            {
                "kb": {
                    "transport": "http",
                    "url": "https://example-host/gradio_api/mcp/",
                    "headers": {"Authorization": "Bearer ${REMOTE_MCP_BEARER_TOKEN}"},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("REMOTE_MCP_BEARER_TOKEN", raising=False)
    connections = load_mcp_registry(registry)
    assert connections["kb"]["headers"]["Authorization"] == "Bearer "


def test_tracked_registry_never_requires_bearer_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REMOTE_MCP_BEARER_TOKEN", raising=False)
    connections = load_mcp_registry(DEFAULT_MCP_SERVERS_FILE)
    assert "headers" not in connections["comindware_kb"]


def test_servers_wrapper_key(tmp_path: Path) -> None:
    registry = tmp_path / "mcp.yaml"
    registry.write_text(
        yaml.dump(
            {
                "servers": {
                    "wrapped": {
                        "transport": "streamable_http",
                        "url": "https://example-host/gradio_api/mcp/",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    connections = load_mcp_registry(registry)
    assert "wrapped" in connections
    assert connections["wrapped"]["transport"] == "streamable_http"


@pytest.mark.asyncio
async def test_fetch_mcp_tools_mocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import AsyncMock, MagicMock, patch

    from agent_ng import mcp_tools
    from agent_ng.mcp_tools import load_mcp_registry as _load_mcp_registry_impl

    registry = tmp_path / "s.yaml"
    monkeypatch.setenv("CMW_MCP_ENABLED", "true")
    registry.write_text(
        yaml.dump(
            {
                "kb": {
                    "transport": "http",
                    "url": "https://example-host/gradio_api/mcp/",
                }
            }
        ),
        encoding="utf-8",
    )

    mock_tool = MagicMock()
    mock_tool.name = "kb_ask_comindware"
    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[mock_tool])

    def _load_registry(_path=None):
        return _load_mcp_registry_impl(registry)

    with (
        patch.object(mcp_tools, "load_mcp_registry", side_effect=_load_registry),
        patch(
            "langchain_mcp_adapters.client.MultiServerMCPClient",
            return_value=mock_client,
        ),
    ):
        tools = await mcp_tools.fetch_mcp_tools_async()

    assert len(tools) == 1
    assert tools[0].name == "kb_ask_comindware"
