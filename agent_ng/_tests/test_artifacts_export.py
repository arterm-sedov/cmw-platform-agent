"""Registered artifact ZIP export contracts."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from typing import TYPE_CHECKING
import zipfile

from agent_ng.artifacts_export import (
    build_registered_artifacts_zip,
    collect_export_files,
    collect_registered_artifacts,
)

if TYPE_CHECKING:
    from pathlib import Path


class DummyAgent:
    def __init__(self) -> None:
        self.file_registry: dict[tuple[str, str], str] = {}

    def register_file(self, session_id: str, logical_name: str, path: Path) -> None:
        self.file_registry[(session_id, logical_name)] = str(path)


def test_collect_registered_artifacts_filters_current_session(tmp_path: Path) -> None:
    agent = DummyAgent()
    keep = tmp_path / "keep.txt"
    other = tmp_path / "other.txt"
    keep.write_text("keep", encoding="utf-8")
    other.write_text("other", encoding="utf-8")

    agent.register_file("session-a", "keep.txt", keep)
    agent.register_file("session-b", "other.txt", other)

    artifacts = collect_registered_artifacts(agent, "session-a")

    assert [item.logical_name for item in artifacts] == ["keep.txt"]
    assert artifacts[0].source_path == keep.resolve()


def test_collect_registered_artifacts_skips_missing_files(tmp_path: Path) -> None:
    agent = DummyAgent()
    agent.register_file("session-a", "missing.txt", tmp_path / "missing.txt")

    assert collect_registered_artifacts(agent, "session-a") == []


def test_build_registered_artifacts_zip_returns_none_for_empty_registry() -> None:
    assert build_registered_artifacts_zip(DummyAgent(), "session-a") is None


def test_build_registered_artifacts_zip_packages_files_and_manifest(
    tmp_path: Path,
) -> None:
    agent = DummyAgent()
    first = tmp_path / "first.txt"
    second = tmp_path / "second.bin"
    first.write_text("alpha", encoding="utf-8")
    second.write_bytes(b"beta")
    agent.register_file("session-a", "folder/first.txt", first)
    agent.register_file("session-a", "second.bin", second)

    zip_path = build_registered_artifacts_zip(
        agent,
        "session-a",
        exported_at=datetime(2026, 6, 3, 12, 0, 0, tzinfo=UTC),
    )

    assert zip_path is not None
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        assert names == {
            "artifacts/first.txt",
            "artifacts/second.bin",
            "manifest.json",
        }
        assert zf.read("artifacts/first.txt") == b"alpha"
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))

    assert manifest["session_id"] == "session-a"
    assert manifest["artifact_count"] == 2
    assert {item["logical_name"] for item in manifest["artifacts"]} == {
        "folder/first.txt",
        "second.bin",
    }
    assert all(item["sha256"] for item in manifest["artifacts"])


def test_build_registered_artifacts_zip_includes_markdown_and_html_exports(
    tmp_path: Path,
) -> None:
    markdown = tmp_path / "CMW_Copilot_20260604_120000.md"
    html = tmp_path / "CMW_Copilot_20260604_120000.html"
    markdown.write_text("# Export\n\n![img](artifacts/chart.png)", encoding="utf-8")
    html.write_text("<html><body>Export</body></html>", encoding="utf-8")

    zip_path = build_registered_artifacts_zip(
        DummyAgent(),
        "session-a",
        export_files=[
            ("conversation.md", markdown),
            ("conversation.html", html),
        ],
    )

    assert zip_path is not None
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        assert names == {
            "conversations/conversation.md",
            "conversations/conversation.html",
            "manifest.json",
        }
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))

    assert manifest["artifact_count"] == 0
    assert manifest["conversation_count"] == 2
    assert {item["zip_path"] for item in manifest["conversations"]} == {
        "conversations/conversation.md",
        "conversations/conversation.html",
    }


def test_build_registered_artifacts_zip_includes_exports_and_linked_artifacts(
    tmp_path: Path,
) -> None:
    agent = DummyAgent()
    markdown = tmp_path / "conversation.md"
    image = tmp_path / "chart.png"
    markdown.write_text("![chart](chart.png)", encoding="utf-8")
    image.write_bytes(b"png")
    agent.register_file("session-a", "chart.png", image)

    zip_path = build_registered_artifacts_zip(
        agent,
        "session-a",
        export_files=[("conversation.md", markdown)],
    )

    assert zip_path is not None
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        assert names == {
            "conversations/conversation.md",
            "artifacts/chart.png",
            "manifest.json",
        }
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))

    assert manifest["conversation_count"] == 1
    assert manifest["artifact_count"] == 1
    assert manifest["conversations"][0]["zip_path"] == (
        "conversations/conversation.md"
    )
    assert manifest["artifacts"][0]["zip_path"] == "artifacts/chart.png"


def test_collect_export_files_skips_missing_files(tmp_path: Path) -> None:
    existing = tmp_path / "existing.md"
    existing.write_text("ok", encoding="utf-8")

    exports = collect_export_files(
        [
            ("existing.md", existing),
            ("missing.md", tmp_path / "missing.md"),
            ("", existing),
        ]
    )

    assert [item.logical_name for item in exports] == ["existing.md"]
    assert exports[0].source_path == existing.resolve()


def test_build_registered_artifacts_zip_deduplicates_internal_names(
    tmp_path: Path,
) -> None:
    agent = DummyAgent()
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("one", encoding="utf-8")
    second.write_text("two", encoding="utf-8")
    agent.register_file("session-a", "dir/same.txt", first)
    agent.register_file("session-a", "other/same.txt", second)

    zip_path = build_registered_artifacts_zip(agent, "session-a")

    assert zip_path is not None
    with zipfile.ZipFile(zip_path) as zf:
        assert "artifacts/same.txt" in zf.namelist()
        assert "artifacts/same_2.txt" in zf.namelist()
