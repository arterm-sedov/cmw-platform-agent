import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest


# Ensure project root is on sys.path to import tools
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tools.tools as t  # noqa: E402


def write_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame({
        "A": [1, 2, 3, 4],
        "B": [0.5, 1.5, 2.5, 3.5],
        "C": ["x", "y", "x", "z"],
    })
    p = tmp_path / "sample.csv"
    df.to_csv(p, index=False)
    return p


def write_excel(tmp_path: Path) -> Path:
    df = pd.DataFrame({
        "A": [10, 20, 30, 40],
        "B": [5, 15, 25, 35],
        "C": ["u", "v", "u", "w"],
    })
    p = tmp_path / "sample.xlsx"
    # Rely on default engine; skip test if engine missing
    try:
        df.to_excel(p, index=False)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"Excel engine not available: {e}")
    return p


def parse_tool_response(s: str):
    # Responses are JSON strings created via FileUtils.create_tool_response
    return json.loads(s)


def test_helper_empty_query_preview():
    df = pd.DataFrame({
        "A": [1, 2, 3, 4],
        "B": [0.5, 1.5, 2.5, 3.5],
        "C": ["x", "y", "x", "z"],
    })
    _, payload = t._apply_pandas_query(df, query=None)
    assert payload.get("table_markdown")
    assert payload.get("schema")


def test_helper_expr_query():
    df = pd.DataFrame({
        "A": [1, 2, 3, 4],
        "B": [0.5, 1.5, 2.5, 3.5],
        "C": ["x", "y", "x", "z"],
    })
    _, payload = t._apply_pandas_query(df, query="expr: B > 1.0")
    assert payload.get("table_markdown")


def test_helper_pipeline_query():
    df = pd.DataFrame({
        "A": [1, 2, 3, 4],
        "B": [0.5, 1.5, 2.5, 3.5],
        "C": ["x", "y", "x", "z"],
    })
    pipeline = json.dumps([
        {"op": "query", "expr": "B > 1.0"},
        {"op": "head", "n": 2},
    ])
    _, payload = t._apply_pandas_query(df, query=pipeline)
    assert payload.get("table_markdown")


def test_helper_preview_includes_shape_and_schema():
    df = pd.DataFrame({
        "A": [1, 2, 3, 4],
        "B": [0.5, 1.5, 2.5, 3.5],
    })
    _, payload = t._apply_pandas_query(df, query=None)
    assert "shape" in payload and isinstance(payload["shape"], tuple)
    assert "schema" in payload and isinstance(payload["schema"], dict)


