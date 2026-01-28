"""Utility modules for agent_ng."""

# Re-export functions from parent utils.py to avoid import conflicts
import sys
from pathlib import Path

# Import from parent utils.py if it exists
_parent_utils = Path(__file__).parent.parent / "utils.py"
if _parent_utils.exists():
    import importlib.util
    spec = importlib.util.spec_from_file_location("agent_ng_utils", _parent_utils)
    _utils_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_utils_module)

    # Re-export commonly used functions
    if hasattr(_utils_module, "get_tool_call_count"):
        get_tool_call_count = _utils_module.get_tool_call_count
    if hasattr(_utils_module, "ensure_valid_answer"):
        ensure_valid_answer = _utils_module.ensure_valid_answer
    if hasattr(_utils_module, "parse_env_bool"):
        parse_env_bool = _utils_module.parse_env_bool
    if hasattr(_utils_module, "safe_string"):
        safe_string = _utils_module.safe_string
    if hasattr(_utils_module, "format_cost"):
        format_cost = _utils_module.format_cost
