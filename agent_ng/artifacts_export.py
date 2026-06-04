"""Build downloadable ZIP packages from a session agent's file registry."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
from posixpath import join as posix_join
import re
import tempfile
from typing import TYPE_CHECKING, Any
import zipfile

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegisteredArtifact:
    """A single file registered for a session."""

    logical_name: str
    source_path: Path


@dataclass(frozen=True)
class ExportFile:
    """A generated conversation export to include in the package."""

    logical_name: str
    source_path: Path


def build_registered_artifacts_zip(
    agent: Any,
    session_id: str,
    *,
    exported_at: datetime | None = None,
    export_files: Iterable[tuple[str, str | Path]] | None = None,
) -> str | None:
    """Create a ZIP with exports plus all registered files for ``session_id``.

    Returns the ZIP path, or ``None`` when there are no packageable artifacts.
    """
    artifacts = collect_registered_artifacts(agent, session_id)
    exports = collect_export_files(export_files)
    if not artifacts and not exports:
        return None

    exported_at = exported_at or datetime.now()
    export_dir = Path(tempfile.mkdtemp())
    zip_path = export_dir / f"CMW_Copilot_artifacts_{exported_at:%Y%m%d_%H%M%S}.zip"

    manifest: dict[str, Any] = {
        "exported_at": exported_at.isoformat(timespec="seconds"),
        "session_id": session_id,
        "artifact_count": len(artifacts),
        "conversation_count": len(exports),
        "conversations": [],
        "artifacts": [],
    }
    used_names_by_folder: dict[str, set[str]] = {}

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for index, export_file in enumerate(exports, 1):
            arcname = _unique_file_arcname(
                export_file.logical_name,
                export_file.source_path,
                used_names_by_folder.setdefault("conversations", set()),
                index,
                "conversations",
            )
            try:
                size_bytes = export_file.source_path.stat().st_size
                sha256 = _sha256_file(export_file.source_path)
                zf.write(export_file.source_path, arcname)
            except OSError as exc:
                logger.warning(
                    "Skipping export file %s at %s: %s",
                    export_file.logical_name,
                    export_file.source_path,
                    exc,
                )
                continue

            manifest["conversations"].append(
                {
                    "logical_name": export_file.logical_name,
                    "zip_path": arcname,
                    "source_path": str(export_file.source_path),
                    "size_bytes": size_bytes,
                    "sha256": sha256,
                }
            )

        for index, artifact in enumerate(artifacts, 1):
            arcname = _unique_file_arcname(
                artifact.logical_name,
                artifact.source_path,
                used_names_by_folder.setdefault("artifacts", set()),
                index,
                "artifacts",
            )
            try:
                size_bytes = artifact.source_path.stat().st_size
                sha256 = _sha256_file(artifact.source_path)
                zf.write(artifact.source_path, arcname)
            except OSError as exc:
                logger.warning(
                    "Skipping registered artifact %s at %s: %s",
                    artifact.logical_name,
                    artifact.source_path,
                    exc,
                )
                continue

            manifest["artifacts"].append(
                {
                    "logical_name": artifact.logical_name,
                    "zip_path": arcname,
                    "source_path": str(artifact.source_path),
                    "size_bytes": size_bytes,
                    "sha256": sha256,
                }
            )

        if not manifest["artifacts"] and not manifest["conversations"]:
            return None

        manifest["artifact_count"] = len(manifest["artifacts"])
        manifest["conversation_count"] = len(manifest["conversations"])
        zf.writestr(
            "manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2),
        )

    return str(zip_path)


def collect_registered_artifacts(
    agent: Any,
    session_id: str,
) -> list[RegisteredArtifact]:
    """Return existing regular files from ``agent.file_registry`` for one session."""
    registry = getattr(agent, "file_registry", None)
    if not isinstance(registry, dict):
        return []

    artifacts: list[RegisteredArtifact] = []
    for key, raw_path in registry.items():
        key_session_id, logical_name = _parse_registry_key(key)
        if key_session_id != session_id or not logical_name:
            continue
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue
        source_path = Path(raw_path)
        if not source_path.is_file():
            continue
        artifacts.append(
            RegisteredArtifact(
                logical_name=str(logical_name),
                source_path=source_path.resolve(),
            )
        )

    artifacts.sort(key=lambda item: item.logical_name.lower())
    return artifacts


def collect_export_files(
    export_files: Iterable[tuple[str, str | Path]] | None,
) -> list[ExportFile]:
    """Return existing regular files explicitly added to an export package."""
    if export_files is None:
        return []

    exports: list[ExportFile] = []
    for logical_name, raw_path in export_files:
        if not str(logical_name).strip():
            continue
        source_path = Path(raw_path)
        if not source_path.is_file():
            continue
        exports.append(
            ExportFile(
                logical_name=str(logical_name),
                source_path=source_path.resolve(),
            )
        )
    return exports


def _parse_registry_key(key: Any) -> tuple[str | None, str | None]:
    if (
        isinstance(key, tuple)
        and len(key) == 2
        and isinstance(key[0], str)
        and isinstance(key[1], str)
    ):
        return key[0], key[1]
    return None, None


def _unique_file_arcname(
    logical_name: str,
    source_path: Path,
    used_names: set[str],
    index: int,
    folder: str,
) -> str:
    filename = _safe_filename(logical_name) or _safe_filename(source_path.name)
    if not filename:
        filename = f"artifact_{index}{source_path.suffix}"

    candidate = filename
    stem = Path(filename).stem or f"artifact_{index}"
    suffix = Path(filename).suffix
    counter = 2
    while candidate.lower() in used_names:
        candidate = f"{stem}_{counter}{suffix}"
        counter += 1
    used_names.add(candidate.lower())
    return posix_join(folder, candidate)


def _safe_filename(name: str) -> str:
    basename = Path(str(name).replace("\\", "/")).name.strip()
    basename = re.sub(r"[\x00-\x1f<>:\"|?*]", "_", basename)
    return basename.strip(" .")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
