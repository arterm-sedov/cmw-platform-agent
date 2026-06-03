"""Shared bootstrap for cmw-platform maintenance CLI scripts."""
from __future__ import annotations

import ast
import base64
import json
import os
import sys
from pathlib import Path
from typing import Any

import requests as http
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent.parent


def repo_root() -> Path:
    return REPO_ROOT


def configure_env(*, base_url: str | None = None) -> None:
    """Load .env from repo root and optional CMW_BASE_URL override."""
    load_dotenv(REPO_ROOT / ".env", override=True)
    os.environ["CMW_USE_DOTENV"] = "true"
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
    if base_url:
        os.environ["CMW_BASE_URL"] = base_url.rstrip("/") + "/"


def require_credentials() -> tuple[str, str, str]:
    base = (os.environ.get("CMW_BASE_URL") or "").strip().rstrip("/")
    login = (os.environ.get("CMW_LOGIN") or "").strip()
    password = (os.environ.get("CMW_PASSWORD") or "").strip()
    if not base:
        raise SystemExit("CMW_BASE_URL is empty — set in .env or pass --base-url")
    if not login or not password:
        raise SystemExit("CMW_LOGIN / CMW_PASSWORD required — see .env.example")
    return base, login, password


def basic_headers() -> dict[str, str]:
    _, login, password = require_credentials()
    token = base64.b64encode(f"{login}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}", "Content-Type": "application/json"}


def system_core_post(path: str, body: dict | None = None) -> Any:
    base, _, _ = require_credentials()
    timeout = int(os.environ.get("CMW_TIMEOUT", "60") or "60")
    resp = http.post(
        f"{base}/api/public/system/{path}",
        headers=basic_headers(),
        json=body or {},
        timeout=timeout,
    )
    if not resp.ok:
        raise RuntimeError(f"{path} HTTP {resp.status_code}: {(resp.text or '')[:500]}")
    text = (resp.text or "").strip()
    if not text:
        return {}
    return json.loads(text)


def parse_raw(raw: object) -> object:
    if isinstance(raw, (list, dict)):
        return raw
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return []
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return ast.literal_eval(raw)
    return raw


def add_repo_to_path() -> None:
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
