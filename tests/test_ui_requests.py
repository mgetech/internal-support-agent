"""The My requests tab: waiting requests first, then the history, newest first, each with
a status badge. The tab runs with a fake API answer, so no server and no database is
needed.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from streamlit.testing.v1 import AppTest
from ui.app import MY_ACTIONS_LIMIT, split_actions

APP_FILE = str(Path(__file__).parent.parent / "ui" / "app.py")


def _action(action_id: int, status: str, decided_by: str | None = None) -> dict:
    decided = decided_by is not None
    return {
        "id": action_id,
        "request_id": f"req-{action_id}",
        "tool": "submit_leave_request",
        "payload": {"start_date": "2026-11-02", "end_date": "2026-11-02", "working_days": 1},
        "status": status,
        "decided_by": decided_by,
        "decided_at": "2026-10-01T09:30:00+00:00" if decided else None,
        "created_at": "2026-10-01T08:00:00+00:00",
    }


# the API answers newest first, so the waiting ones are mixed into the history
ACTIONS = [
    _action(6, "executed", "emp_005"),
    _action(5, "pending_approval"),
    _action(4, "rejected", "emp_010"),
    _action(3, "pending_approval"),
    _action(2, "approved", "emp_005"),
    _action(1, "expired"),
]


def test_waiting_actions_come_first_and_each_list_is_newest_first():
    waiting, history = split_actions(ACTIONS)

    assert [a["id"] for a in waiting] == [5, 3]
    assert [a["id"] for a in history] == [6, 4, 2, 1]


def test_split_sorts_by_id_even_if_the_answer_is_not_in_order():
    waiting, history = split_actions(list(reversed(ACTIONS)))

    assert [a["id"] for a in waiting] == [5, 3]
    assert [a["id"] for a in history] == [6, 4, 2, 1]


def test_split_of_no_actions_is_empty():
    assert split_actions([]) == ([], [])


class FakeApi:
    """Stands in for httpx.get. It answers with `actions` and keeps each call."""

    def __init__(self) -> None:
        self.actions = ACTIONS
        self.calls: list[tuple[str, dict]] = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: self.actions)


@pytest.fixture
def api(monkeypatch):
    fake = FakeApi()
    monkeypatch.setattr(httpx, "get", fake.get)
    return fake


def _open_requests_tab() -> AppTest:
    app = AppTest.from_file(APP_FILE, default_timeout=10)
    app.session_state["active_tab"] = "My requests"
    return app.run()


def test_the_tab_asks_for_the_most_the_api_gives(api):
    _open_requests_tab()

    ((url, kwargs),) = api.calls
    assert url.endswith("/me/actions")
    assert kwargs["params"] == {"limit": MY_ACTIONS_LIMIT}
    assert kwargs["headers"] == {"X-Employee-Id": "emp_001"}


def test_the_tab_shows_the_waiting_requests_before_the_history(api):
    app = _open_requests_tab()

    assert not app.exception
    assert [h.value for h in app.subheader] == ["Waiting for approval (2)", "History"]
    titles = [m.value for m in app.markdown if m.value.startswith("**#")]
    assert [t.split(" ")[0] for t in titles] == ["**#5", "**#3", "**#6", "**#4", "**#2", "**#1"]


def test_every_request_has_a_status_badge(api):
    app = _open_requests_tab()

    badges = [m.value for m in app.markdown if "-badge[" in m.value]
    assert badges == [
        ":blue-badge[Waiting for approval]",
        ":blue-badge[Waiting for approval]",
        ":green-badge[Approved and done]",
        ":red-badge[Rejected]",
        ":green-badge[Approved]",
        ":gray-badge[Expired]",
    ]


def test_the_tab_has_no_history_heading_when_nothing_is_decided(api):
    api.actions = [_action(2, "pending_approval"), _action(1, "pending_approval")]

    app = _open_requests_tab()

    assert [h.value for h in app.subheader] == ["Waiting for approval (2)"]


def test_the_tab_has_no_waiting_heading_when_nothing_waits(api):
    api.actions = [_action(2, "rejected", "emp_010"), _action(1, "executed", "emp_005")]

    app = _open_requests_tab()

    assert [h.value for h in app.subheader] == ["History"]


def test_the_tab_says_so_when_there_are_no_requests(api):
    api.actions = []

    app = _open_requests_tab()

    assert [i.value for i in app.info][-1].startswith("You have no requests yet")
    assert len(app.subheader) == 0


def test_the_tab_says_when_the_list_was_cut_at_the_limit(api):
    api.actions = [_action(n, "executed", "emp_005") for n in range(MY_ACTIONS_LIMIT, 0, -1)]

    app = _open_requests_tab()

    assert f"Only the newest {MY_ACTIONS_LIMIT} requests are shown." in [
        c.value for c in app.caption
    ]
