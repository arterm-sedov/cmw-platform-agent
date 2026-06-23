"""Queue helper contracts: UI listeners default to private API visibility."""

from __future__ import annotations

from agent_ng.queue_manager import (
    apply_concurrency_to_change_event,
    apply_concurrency_to_click_event,
    apply_concurrency_to_submit_event,
    create_queue_manager,
)


def test_apply_click_defaults_private_visibility() -> None:
    qm = create_queue_manager()

    def _noop() -> None:
        return None

    cfg = apply_concurrency_to_click_event(qm, "logs_refresh", _noop, [], [])
    assert cfg["api_visibility"] == "private"


def test_apply_submit_kwargs_override_visibility() -> None:
    qm = create_queue_manager()

    def _noop() -> None:
        return None

    cfg = apply_concurrency_to_submit_event(
        qm,
        "chat",
        _noop,
        [],
        [],
        api_visibility="public",
    )
    assert cfg["api_visibility"] == "public"


def test_apply_change_defaults_private_visibility() -> None:
    qm = create_queue_manager()

    def _noop() -> None:
        return None

    cfg = apply_concurrency_to_change_event(qm, "stats_refresh", _noop, [], [])
    assert cfg["api_visibility"] == "private"
