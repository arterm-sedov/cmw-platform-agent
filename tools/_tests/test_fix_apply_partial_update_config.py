"""
Test that _apply_partial_update uses _load_server_config() instead of os.environ.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tools.tool_utils import _apply_partial_update


class TestApplyPartialUpdateConfigSource:
    """_apply_partial_update must use _load_server_config(), not os.environ."""

    def test_uses_load_server_config_not_os_environ(self):
        """When _load_server_config returns valid config, _apply_partial_update
        must use those credentials, not bypass via os.environ."""
        mock_cfg = MagicMock()
        mock_cfg.base_url = "https://test.example.com"
        mock_cfg.login = "user"
        mock_cfg.password = "pass"

        with patch(
            "tools.tool_utils.requests_._load_server_config", return_value=mock_cfg
        ) as mock_load:
            result = _apply_partial_update(
                "webapi/Form/TestApp",
                {"globalAlias": {"owner": "Tpl", "alias": "defaultForm", "type": "Form"}},
            )
            mock_load.assert_called_once()

    def test_returns_body_when_config_fails(self):
        """When _load_server_config raises, returns request_body unchanged."""
        request_body = {"globalAlias": {"owner": "Tpl", "alias": "test"}}

        with patch(
            "tools.tool_utils.requests_._load_server_config",
            side_effect=RuntimeError("No config"),
        ):
            result = _apply_partial_update("webapi/Form/App", request_body)
            assert result is request_body

    def test_does_not_call_os_environ(self):
        """_apply_partial_update must not directly access os.environ for API creds."""
        import ast
        import inspect

        source = inspect.getsource(_apply_partial_update)
        tree = ast.parse(source)

        os_environ_accesses = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Attribute)
            and getattr(node.value, "attr", None) == "environ"
        ]
        assert not os_environ_accesses, (
            "_apply_partial_update must not access os.environ directly. "
            "Use _load_server_config() instead."
        )
