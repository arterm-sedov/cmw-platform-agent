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
from typing import Any
import zipfile

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegisteredArtifact:
    """A single file registered for a session."""

    logical_name: str
    source_path: Path


def build_registered_artifacts_zip(
    agent: Any,
    session_id: str,
    *,
    exported_at: datetime | None = None,
) -> str | None:
    """Create a ZIP with all existing files registered for ``session_id``.

    Returns the ZIP path, or ``None`` when there are no packageable artifacts.
    """
    artifacts = collect_registered_artifacts(agent, session_id)
    if not artifacts:
        return None

    exported_at = exported_at or datetime.now()
    export_dir = Path(tempfile.mkdtemp())
    zip_path = export_dir / f"CMW_Copilot_artifacts_{exported_at:%Y%m%d_%H%M%S}.zip"

    manifest: dict[str, Any] = {
        "exported_at": exported_at.isoformat(timespec="seconds"),
        "session_id": session_id,
        "artifact_count": len(artifacts),
        "artifacts": [],
    }
    used_names: set[str] = set()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for index, artifact in enumerate(artifacts, 1):
            arcname = _unique_artifact_arcname(
                artifact.logical_name,
                artifact.source_path,
                used_names,
                index,
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

        if not manifest["artifacts"]:
            return None

        manifest["artifact_count"] = len(manifest["artifacts"])
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


def _parse_registry_key(key: Any) -> tuple[str | None, str | None]:
    if (
        isinstance(key, tuple)
        and len(key) == 2
        and isinstance(key[0], str)
        and isinstance(key[1], str)
    ):
        return key[0], key[1]
    return None, None


def _unique_artifact_arcname(
    logical_name: str,
    source_path: Path,
    used_names: set[str],
    index: int,
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
    return posix_join("artifacts", candidate)


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
